import os
import json
import glob
import numpy as np
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline
from PIL import Image

pipe = DifixPipeline.from_pretrained(
    "nvidia/difix_ref",  # 
    trust_remote_code=True
)
pipe.to("cuda")

PROMPT = "remove degradation"
NUM_STEPS = 1
TIMESTEPS = [199]
GUIDANCE = 0.0

lpips_fn = lpips.LPIPS(net="alex").cuda()

def compute_metrics(pred, gt):
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).cuda()
    gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).cuda()
    lpips_val = lpips_fn(pred_t, gt_t).item()
    return psnr_val, ssim_val, lpips_val


def main():
    eval_base = "/baai-cwm-vepfs/cwm/nan.wang/gen.li/workspace/DL3DV/dl3dv-nerfacto-eval-render-output"
    ref_base = "/baai-cwm-vepfs/cwm/nan.wang/gen.li/dataset/dl3dv-difix-test-data"
    output_base = "/baai-cwm-backup/cwm/gen.li/dl3dv-ref-difix-results-nerfacto"
    os.makedirs(output_base, exist_ok=True)

    all_metrics = {"psnr": [], "ssim": [], "lpips": []}

    scene_dirs = sorted([d for d in glob.glob(os.path.join(eval_base, "*")) if os.path.isdir(d)])

    for scene in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = os.path.basename(scene)
        save_dir = os.path.join(output_base, scene_name)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

        scene_metrics = {"psnr": [], "ssim": [], "lpips": []}

        eval_imgs = sorted(glob.glob(os.path.join(scene, "eval_img_*.png")))
        for eval_path in eval_imgs:
            fname = os.path.basename(eval_path)  # eval_img_xxxx.png
            seq_num = fname.replace("eval_img_", "").replace(".png", "")

            full_img = np.array(Image.open(eval_path).convert("RGB"))
            h, w, _ = full_img.shape
            mid = w // 2
            gt_image = full_img[:, :mid, :].astype(np.float32) / 255.0
            input_image = Image.fromarray(full_img[:, mid:, :])

            ref_path = os.path.join(ref_base, scene_name, "ref_image", f"val_step29999_{seq_num}.png")
            if not os.path.exists(ref_path):
                print(f"⚠️ 找不到 ref_image: {ref_path}")
                continue
            ref_image = load_image(ref_path)

            with torch.no_grad():
                output = pipe(
                    PROMPT,
                    image=input_image,
                    ref_image=ref_image,
                    num_inference_steps=NUM_STEPS,
                    timesteps=TIMESTEPS,
                    guidance_scale=GUIDANCE,
                ).images[0]

            output = output.resize((gt_image.shape[1], gt_image.shape[0]))
            save_path = os.path.join(save_dir, "images", fname)
            output.save(save_path)

            pred_img = np.array(output).astype(np.float32) / 255.0

            psnr_val, ssim_val, lpips_val = compute_metrics(pred_img, gt_image)
            scene_metrics["psnr"].append(psnr_val)
            scene_metrics["ssim"].append(ssim_val)
            scene_metrics["lpips"].append(lpips_val)

        scene_avg = {k: float(np.mean(v)) if v else 0.0 for k, v in scene_metrics.items()}
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(scene_avg, f, indent=4)

        for k in all_metrics:
            all_metrics[k].append(scene_avg[k])

    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}
    with open(os.path.join(output_base, "metrics_avg.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)



if __name__ == "__main__":
    main()
