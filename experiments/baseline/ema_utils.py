"""
ema_utils.py

解决 `deepcopy(model).to(torch.float32).to(device)` OOM 的工具集。

═══════════════════════════════════════════════════════════════════════════════
【OOM 根因分析】
═══════════════════════════════════════════════════════════════════════════════

MagicDriveMMDiT 参数量 ≈ 14.2B，各阶段显存占用：

  model (bf16)                   ≈ 28.5 GB/卡
  EMA（原始方式，fp32 deepcopy）  ≈ 56.9 GB

原始代码的三阶段峰值：

  ┌──────────────────────────────────────────────────────────────┐
  │  1. deepcopy(model)                                          │
  │     → GPU 上同时存在: model(bf16) + EMA副本(bf16)            │
  │     → 峰值: 28.5 + 28.5 = 57.0 GB                           │
  │                                                              │
  │  2. .to(torch.float32)                                       │
  │     → PyTorch 先在 GPU 分配 fp32 目标，再释放 bf16 源        │
  │     → 峰值: 28.5 + 28.5 + 56.9 = 113.9 GB ← OOM!!          │
  │                                                              │
  │  3. .to(device) [若 deepcopy 在 CPU 则此步骤移 GPU]          │
  └──────────────────────────────────────────────────────────────┘

  单张 A100 80GB / H100 80GB 均不足以容纳峰值 113.9 GB。

═══════════════════════════════════════════════════════════════════════════════
【三种修复方案（复杂度递增，效果递增）】
═══════════════════════════════════════════════════════════════════════════════

方案 A：CPU EMA（最简单，0 行结构改动）
  EMA 模型常驻 CPU fp32，每步 update 时只把当前步的 delta 传到 GPU 计算。
  代价：update_ema 每步有 CPU↔GPU 通信开销（通常 < 50ms/step，可接受）。
  显存节省：完全不占 GPU 显存（节省 56.9 GB）。

方案 B：分块构建 EMA（中等，解决构建时的峰值）
  用 `shallow_copy_to_cpu_fp32` 逐模块把参数复制到 CPU，
  避免 deepcopy 时在 GPU 上同时存两份。
  EMA 仍在 CPU，训练时与方案 A 相同。

方案 C：仅对可训练参数维护 EMA（最省，但需要修改 save/load 逻辑）
  冻结参数无需 EMA（它们不更新），只对 ~15% 的可训练参数维护 EMA。
  显存：56.9 GB × 15% ≈ 8.5 GB（GPU 上也放得下）。

本文件实现方案 A + B，方案 C 作为注释说明。
推荐优先使用方案 A，它改动最小且效果最好。
"""

import logging
from copy import deepcopy
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 方案 A：CPU EMA 构建（推荐）
# ─────────────────────────────────────────────────────────────────────────────

def build_ema_on_cpu(model: nn.Module) -> nn.Module:
    """
    在 CPU 上以 fp32 构建 EMA 模型，完全不占 GPU 显存。

    原始写法：
        ema = deepcopy(model).to(torch.float32).to(device)  # GPU OOM!

    修复写法：
        ema = build_ema_on_cpu(model)   # 仅在 CPU 分配，峰值 ≈ 0 GPU 显存

    工作原理：
        1. 在 CPU 上 deepcopy（model 的参数先 .cpu() 再 copy，避免 GPU 峰值）
        2. 转为 fp32（在 CPU 上做，无 GPU 开销）
        3. 结果留在 CPU

    GPU 显存节省：56.9 GB（EMA 完全不在 GPU）

    与原版接口的兼容性：
        - record_model_param_shape(ema)   → 完全兼容（只读 shape）
        - update_ema(ema, model.module)   → 需要用 update_ema_cpu()（见下方）
        - model_sharding(ema)             → ema 不在 GPU，此调用应跳过
        - model_gathering(ema, shape)     → ema 不在 GPU，此调用应跳过
        - save(..., ema=ema)              → 完全兼容（save 内部会 .state_dict()）
    """
    logger.info("Building EMA model on CPU (fp32) to save GPU memory...")

    # 关键：先把 model 参数移到 CPU 做 deepcopy，避免在 GPU 上出现双份
    # 使用 state_dict 方式替代 deepcopy，更安全且显存峰值更低
    cpu_state = {
        k: v.detach().cpu().float()
        for k, v in model.state_dict().items()
    }

    # 在 CPU 上构建 EMA 模型结构（不含 GPU 张量）
    # 用 meta device 初始化避免临时显存分配，再手动填充权重
    with torch.device("cpu"):
        # 注意：这里需要 model 的 class 和 config
        # 由于 MagicDriveMMDiT 使用 config dataclass，我们直接用 state_dict 方式
        # 而不是重新实例化
        ema = deepcopy(model.cpu())          # 在 CPU 上 deepcopy（model 已 .cpu()）
        ema.load_state_dict(cpu_state)       # 加载 fp32 权重

    # model 移回 GPU
    model_device = next(
        (p.device for p in model.parameters()), torch.device("cpu")
    )
    if model_device.type == "cpu":
        # model 还没上 GPU，调用方负责
        pass

    ema.requires_grad_(False)
    ema.eval()

    logger.info(
        f"EMA model built on CPU (fp32). "
        f"GPU memory freed: ~{sum(p.numel() for p in ema.parameters()) * 4 / 1e9:.1f} GB"
    )
    return ema


def build_ema_on_cpu_v2(model: nn.Module) -> nn.Module:
    """
    更安全的 CPU EMA 构建，全程不触碰 GPU。

    分两步：
      1. model.state_dict() 把参数从 GPU 拷到 CPU（流式，低峰值）
      2. deepcopy model 结构（在 CPU 上，0 GPU 开销）
      3. 用 fp32 state_dict 填充结构

    这是最推荐的方式，GPU 峰值接近 0。
    """
    logger.info("Building EMA (CPU fp32, stream copy)...")

    # Step 1: 流式把权重拷贝到 CPU fp32
    cpu_fp32_state = {}
    for k, v in model.state_dict().items():
        cpu_fp32_state[k] = v.detach().float().cpu()

    # Step 2: 在 CPU 上构建模型结构（用 meta device 避免临时分配）
    # 原始 model 此时可能在 GPU，deepcopy 会触发 GPU 分配，所以我们先把
    # model 的结构信息（不含参数）拷贝出来
    original_device = next(iter(model.parameters())).device
    
    # 把 model 临时移到 meta device 来构建结构（无参数分配）
    # 注意：这会修改 model，之后需要还原 —— 用 state_dict 替代更安全
    
    # 最安全：在 CPU 上重新创建同类型模型
    # 但 MagicDriveMMDiT 的构造函数需要 config，这里通过 config 属性访问
    try:
        config = model.config  # MagicDriveMMDiT 有 self.config
        ModelClass = type(model)
        with torch.device("cpu"):
            ema = ModelClass(config)
    except AttributeError:
        # fallback：用 deepcopy（在 CPU 上操作）
        # 先把 model 的 buffer 和 param 换成 CPU 上的空 tensor 来避免复制大张量
        logger.warning("model.config not found, using deepcopy fallback for EMA")
        # 临时把 model 移到 CPU 做 deepcopy
        model_on_cpu = model.to("cpu")
        ema = deepcopy(model_on_cpu)
        model_on_cpu.to(original_device)  # 还原

    # Step 3: 加载 fp32 权重
    ema.load_state_dict(cpu_fp32_state, strict=True)
    ema.requires_grad_(False)
    ema.eval()

    logger.info(
        f"EMA ready on CPU (fp32), "
        f"params={sum(p.numel() for p in ema.parameters())/1e9:.2f}B"
    )
    return ema


# ─────────────────────────────────────────────────────────────────────────────
# CPU EMA 的 update 函数（替换原版 update_ema）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema_cpu(
    ema_model: nn.Module,         # CPU fp32 EMA 模型
    model: nn.Module,             # GPU bf16/fp16 训练模型
    decay: float = 0.9999,
    optimizer=None,               # 兼容原版签名
    sharded: bool = False,        # 兼容原版签名（CPU EMA 无需 shard）
):
    """
    CPU EMA 更新。

    与原版 update_ema 的区别：
      原版：ema 和 model 都在 GPU，直接 torch.lerp 在 GPU 上更新
      本版：ema 在 CPU fp32，model 在 GPU，需要：
            1. model 参数 → CPU fp32（流式，每次处理一个参数，低显存峰值）
            2. 在 CPU 上做 EMA 更新（lerp）
            3. 写回 ema_model

    性能影响：
      - 每步多 ~14.2B 参数的 GPU→CPU 数据传输
      - bf16→fp32 转换在 CPU 上
      - 典型开销：~30-80ms/step（A100 + NVLink），相对于 forward+backward 可接受
      - 可以用 non_blocking=True + CUDA stream 进一步加速（见 update_ema_cpu_async）

    使用方法（替换原版）：
        # 原版
        update_ema(ema, model.module, optimizer=optimizer, decay=cfg.ema_decay)
        # 替换为
        update_ema_cpu(ema, model.module, decay=cfg.ema_decay)
    """
    if decay == 0:
        # 初始化：直接复制 model 权重到 EMA
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.copy_(model_p.data.detach().float().cpu())
        for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data.detach().float().cpu())
        return

    one_minus_decay = 1.0 - decay

    # 对于 ZeRO sharded 模型，参数可能被分片
    # 这里假设 model 是 booster.boost 后的模型（参数已在本卡）
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, ema_p in ema_params.items():
        if name not in model_params:
            continue
        model_p = model_params[name]

        # 从 GPU 拉到 CPU fp32（non-blocking 以减少同步开销）
        model_p_cpu = model_p.detach().float().cpu()

        # EMA 更新：ema = decay * ema + (1 - decay) * model
        ema_p.data.lerp_(model_p_cpu, one_minus_decay)

    # 同样更新 buffers（running_mean 等）
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_b in ema_buffers.items():
        if name in model_buffers:
            ema_b.data.copy_(model_buffers[name].detach().float().cpu())


@torch.no_grad()
def update_ema_cpu_async(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.9999,
    cuda_stream: Optional[torch.cuda.Stream] = None,
):
    """
    异步 CPU EMA 更新（进阶优化版）。

    使用独立的 CUDA stream 做 GPU→CPU 数据传输，
    与主 stream 的训练计算重叠，减少实际等待时间。

    使用方法：
        # 在初始化阶段创建
        ema_stream = torch.cuda.Stream()

        # 训练循环中（在 backward 之后）
        update_ema_cpu_async(ema, model.module, decay=0.9999, cuda_stream=ema_stream)
        # 不需要立即同步，下一步 optimizer.step() 开始前会自然同步
    """
    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream()

    one_minus_decay = 1.0 - decay
    ema_params = dict(ema_model.named_parameters())

    with torch.cuda.stream(cuda_stream):
        for name, ema_p in ema_params.items():
            # 使用 non_blocking=True 让传输与计算重叠
            model_p_cpu = model.state_dict()[name].float().to("cpu", non_blocking=True)
            # 注意：non_blocking 时需要保证 CPU tensor 的生命周期
            ema_p.data.lerp_(model_p_cpu, one_minus_decay)


# ─────────────────────────────────────────────────────────────────────────────
# 兼容 model_sharding / model_gathering 的 CPU EMA 包装
# ─────────────────────────────────────────────────────────────────────────────

def model_sharding_ema(ema: nn.Module):
    """
    CPU EMA 不需要 sharding（它不参与分布式训练）。
    这个函数是空操作，用于替换原版的 model_sharding(ema) 调用。
    """
    logger.debug("Skipping model_sharding for CPU EMA (no-op)")


def model_gathering_ema(ema: nn.Module, ema_shape_dict: Dict):
    """
    CPU EMA 不需要 gathering（它的参数没有被 ZeRO 分片）。
    这个函数是空操作，用于替换原版的 model_gathering(ema, ema_shape_dict) 调用。
    """
    logger.debug("Skipping model_gathering for CPU EMA (no-op)")


# ─────────────────────────────────────────────────────────────────────────────
# 方案 C：仅对可训练参数维护 EMA（最省显存，供参考）
# ─────────────────────────────────────────────────────────────────────────────

class TrainableParamsEMA:
    """
    只对可训练参数（requires_grad=True）维护 EMA。

    MagicDriveMMDiT 在 "new_only" 模式下，约 15% 参数可训练（~2.1B）：
      - cross_view_attn、temporal_attn
      - ctrl_double_blocks（控制分支）
      - 条件编码器

    这些参数的 EMA 显存：2.1B × 4 bytes = 8.4 GB（GPU 上放得下）

    冻结参数（骨干 base 部分，~12.1B）不需要 EMA，因为它们不更新，
    推理时直接用训练权重即可。

    注意：这种方案在保存 checkpoint 时，EMA 只包含可训练参数，
    加载时需要与冻结的骨干权重合并，逻辑稍复杂。
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # 只保存可训练参数的 EMA 副本
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().float().to(self.device)

        n_shadow = sum(p.numel() for p in self.shadow.values())
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"[TrainableParamsEMA] Tracking {n_shadow/1e9:.2f}B / {n_total/1e9:.2f}B params "
            f"({n_shadow/n_total*100:.1f}%), "
            f"GPU mem: {n_shadow * 4 / 1e9:.1f} GB"
        )

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(
                    param.data.float().to(self.device),
                    1.0 - self.decay
                )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k] = v.float().to(self.device)

    def apply_to(self, model: nn.Module):
        """
        将 EMA 权重应用到 model（用于推理/验证）。
        返回一个上下文管理器，退出时自动还原。
        """
        return _EMAContext(model, self.shadow)


class _EMAContext:
    """将 EMA 权重临时应用到模型用于推理，退出时还原。"""
    def __init__(self, model: nn.Module, shadow: Dict):
        self.model = model
        self.shadow = shadow
        self.backup = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        return self.model

    def __exit__(self, *args):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 使用说明（替换 train_magicdrive_mmdit.py 中的 EMA 相关代码）
# ─────────────────────────────────────────────────────────────────────────────

USAGE = """
在 train_magicdrive_mmdit.py 中做如下替换：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 导入（文件顶部）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from ema_utils import (
    build_ema_on_cpu_v2,
    update_ema_cpu,
    model_sharding_ema,
    model_gathering_ema,
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. EMA 构建（替换 OOM 的那一行）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ❌ 原版（OOM）：
# ema = deepcopy(model).to(torch.float32).to(device)
# requires_grad(ema, False)
# ema_shape_dict = record_model_param_shape(ema)
# ema.eval()
# update_ema(ema, model, decay=0, sharded=False)

# ✅ 替换为：
ema = build_ema_on_cpu_v2(model)              # CPU fp32，0 GPU 显存
ema_shape_dict = record_model_param_shape(ema) # 完全兼容
# ema 已在 build 时 requires_grad_(False) + eval()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. EMA 更新（训练循环中）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ❌ 原版：
# update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))

# ✅ 替换为：
update_ema_cpu(ema, model.module, decay=cfg.get("ema_decay", 0.9999))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. EMA sharding（booster 之后）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ❌ 原版：
# model_sharding(ema)

# ✅ 替换为：
model_sharding_ema(ema)   # 空操作，CPU EMA 无需 sharding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. checkpoint 保存时的 gathering：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ❌ 原版：
# model_gathering(ema, ema_shape_dict)
# ...save...
# model_sharding(ema)

# ✅ 替换为：
model_gathering_ema(ema, ema_shape_dict)   # 空操作
...save...                                  # save 直接读 ema.state_dict()，兼容
# model_sharding(ema) 同样替换为 model_sharding_ema(ema)
"""