# filename: models/localdpo/multiview_mask.py

import torch
import torch.nn as nn
import numpy as np
import random
from scipy.special import binom
from einops import rearrange


class BezierCurve:
    """Bézier曲线生成平滑运动轨迹"""
    def __init__(self, control_points, num_samples):
        self.control_points = control_points
        self.num_samples = num_samples
        self.n = len(control_points) - 1
        
    def bernstein(self, i, n, t):
        return binom(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def sample(self):
        t = np.linspace(0, 1, self.num_samples)
        curve = np.zeros((self.num_samples, 3))
        for i, point in enumerate(self.control_points):
            curve += np.outer(self.bernstein(i, self.n, t), point)
        return curve


class MultiViewSpatioTemporalMask(nn.Module):
    """
    创新点1：多视图一致的3D时空遮罩
    - 在3D空间中生成运动物体轨迹
    - 投影到6个相机视角
    - 输出时空一致的遮罩序列
    """
    def __init__(self, 
                 img_size=(106, 200),  # nuScenes图像尺寸
                 num_cameras=6,
                 mask_size_range=(0.1, 0.3),  # 遮罩相对大小
                 speed_range=(0.05, 0.15),    # 运动速度范围
                 traj_points=4,               # Bézier控制点数量
                 smoothness=0.3):             # 轨迹平滑度
        super().__init__()
        
        self.img_h, self.img_w = img_size
        self.num_cameras = num_cameras
        self.mask_size_range = mask_size_range
        self.speed_range = speed_range
        self.traj_points = traj_points
        self.smoothness = smoothness
        
        # 可学习的遮罩形状参数（创新：让模型自己学遮罩形状）
        self.mask_shape_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def generate_3d_trajectory(self, batch_size, T):
        """
        生成3D空间中的运动轨迹
        返回: [B, T, 3] 轨迹点序列
        """
        trajectories = []
        for b in range(batch_size):
            # 随机起点：在相机前方3D空间
            start_x = random.uniform(-10, 10)
            start_y = random.uniform(-5, 5) 
            start_z = random.uniform(20, 40)  # 前方20-40米
            
            # 随机速度向量
            vx = random.uniform(*self.speed_range) * random.choice([-1, 1])
            vy = random.uniform(*self.speed_range) * random.choice([-1, 1])
            vz = random.uniform(*self.speed_range) * random.choice([-1, 1])
            
            # 生成Bézier控制点（平滑轨迹）
            control_points = []
            for i in range(self.traj_points):
                t = i / (self.traj_points - 1)
                x = start_x + vx * t * T
                y = start_y + vy * t * T
                z = start_z + vz * t * T
                control_points.append([x, y, z])
            
            # 采样Bézier曲线
            bezier = BezierCurve(np.array(control_points), T)
            trajectory = bezier.sample()
            trajectories.append(trajectory)
        
        return torch.FloatTensor(np.stack(trajectories))
    
    def project_3d_to_multiview(self, points_3d, camera_params):
        """
        将3D点投影到多视图相机
        points_3d: [B, T, 3] 3D轨迹点
        camera_params: nuScenes相机参数
        返回: [B, NC, T, 2] 2D投影点
        """
        B, T, _ = points_3d.shape
        NC = self.num_cameras
        
        # 扩展维度
        points_3d = points_3d.view(B, 1, T, 3, 1)
        points_3d = points_3d.expand(-1, NC, -1, -1, -1)
        
        # 获取相机内外参
        intrinsics = camera_params['intrinsics']  # [NC, 3, 3]
        extrinsics = camera_params['extrinsics']  # [NC, 4, 4]
        
        # 世界坐标转相机坐标
        points_cam = extrinsics[:, :3, :3] @ points_3d + extrinsics[:, :3, 3:4]
        
        # 相机坐标转像素坐标
        points_2d = intrinsics @ points_cam
        points_2d = points_2d[:, :, :, :2, 0] / (points_2d[:, :, :, 2:3, 0] + 1e-8)
        
        return points_2d  # [B, NC, T, 2]
    
    def render_mask(self, points_2d, video_shape):
        """
        根据2D投影点渲染遮罩
        points_2d: [B, NC, T, 2] 投影点
        video_shape: [B, NC, T, H, W] 视频形状
        返回: [B, NC, 1, T, H, W] 遮罩
        """
        B, NC, T, H, W = video_shape
        device = points_2d.device
        
        # 初始化遮罩
        mask = torch.zeros(B, NC, 1, T, H, W, device=device)
        
        # 动态遮罩尺寸（可学习）
        mask_size = self.mask_size_range[0] + self.mask_shape_weight * (
            self.mask_size_range[1] - self.mask_size_range[0]
        )
        radius_h = int(H * mask_size.item() / 2)
        radius_w = int(W * mask_size.item() / 2)
        
        # 为每个视图、每帧渲染圆形遮罩
        for b in range(B):
            for nc in range(NC):
                for t in range(T):
                    x, y = points_2d[b, nc, t]
                    
                    # 检查是否在图像范围内
                    if 0 <= x < W and 0 <= y < H:
                        x1 = max(0, int(x) - radius_w)
                        x2 = min(W, int(x) + radius_w + 1)
                        y1 = max(0, int(y) - radius_h)
                        y2 = min(H, int(y) + radius_h + 1)
                        
                        # 创建圆形遮罩（高斯衰减边缘）
                        yy, xx = torch.meshgrid(
                            torch.arange(y1, y2, device=device),
                            torch.arange(x1, x2, device=device),
                            indexing='ij'
                        )
                        dist = ((xx - x) / radius_w) ** 2 + ((yy - y) / radius_h) ** 2
                        mask[b, nc, 0, t, y1:y2, x1:x2] = torch.exp(-dist * 4).float()
        
        return mask
    
    def forward(self, video_shape, camera_params):
        """
        输入：
            video_shape: [B, NC, T, H, W] 视频形状
            camera_params: 相机参数（包含intrinsics/extrinsics）
        输出：
            mask: [B, NC, 1, T, H, W] 多视图时空一致遮罩
        """
        B, NC, T, H, W = video_shape
        
        # 1. 生成3D轨迹
        trajectory_3d = self.generate_3d_trajectory(B, T)
        trajectory_3d = trajectory_3d.to(video_shape.device)
        
        # 2. 投影到多视图
        points_2d = self.project_3d_to_multiview(trajectory_3d, camera_params)
        
        # 3. 渲染遮罩
        mask = self.render_mask(points_2d, (B, NC, T, H, W))
        
        return mask


class SemanticAwareMaskGenerator(MultiViewSpatioTemporalMask):
    """
    创新点1增强：VGGT语义引导的遮罩生成
    - 使用VGGT特征识别动态物体区域
    - 优先遮罩车辆/行人等移动物体
    """
    def __init__(self, vggt_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vggt = vggt_model
        self.vggt.eval()
        
    def get_semantic_mask_prior(self, first_frame):
        """
        用VGGT提取语义先验
        first_frame: [B, NC, C, H, W] 首帧图像
        返回: [B, NC, 1, H, W] 语义重要度图
        """
        with torch.no_grad():
            # 提取VGGT特征
            features = self.vggt.extract_features(first_frame)
            # 计算特征变化剧烈区域（可能是动态物体）
            semantic_map = features.std(dim=2, keepdim=True)
            semantic_map = (semantic_map - semantic_map.min()) / (
                semantic_map.max() - semantic_map.min() + 1e-8
            )
        return semantic_map
    
    def forward(self, video_shape, camera_params, first_frame=None):
        mask = super().forward(video_shape, camera_params)
        
        if first_frame is not None:
            # 获取语义先验
            semantic_prior = self.get_semantic_mask_prior(first_frame)
            # 扩展时间维度
            semantic_prior = semantic_prior.unsqueeze(3).expand(-1, -1, -1, video_shape[2], -1, -1)
            # 与随机遮罩融合
            mask = mask * (0.7 + 0.3 * semantic_prior)
            
        return mask