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


pipe = DifixPipeline.from_pretrained(
    "nvidia/difix_ref",  
    trust_remote_code=True
)
pipe.to("cuda")

PROMPT = "remove degradation"
NUM_STEPS = 1
TIMESTEPS = [199]
GUIDANCE = 0.0

lpips_fn = lpips.LPIPS(net="alex").cuda()

def compute_metrics(pred, gt):
    """ pred/gt: numpy [H, W, 3], float32, [0,1] """
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).cuda()
    gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).cuda()
    lpips_val = lpips_fn(pred_t, gt_t).item()
    return psnr_val, ssim_val, lpips_val


def main():
    base_dir = "/baai-cwm-vepfs/cwm/nan.wang/gen.li/workspace/Difix3D/dl3dv-gsplat"
    output_base = "/baai-cwm-backup/cwm/gen.li/dl3dv-gsplat-difix3d+-results-39999"
    os.makedirs(output_base, exist_ok=True)

    all_metrics = {"psnr": [], "ssim": [], "lpips": []}

    scene_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])

    for scene in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = os.path.basename(scene)

        input_dir = os.path.join(scene, "renders", "val", "39999", "Pred")
        ref_dir   = os.path.join(scene, "renders", "novel", "39999", "Ref")
        gt_dir    = os.path.join(scene, "renders", "val", "39999", "GT")  

        if not (os.path.exists(input_dir) and os.path.exists(ref_dir) and os.path.exists(gt_dir)):
            continue

        save_dir = os.path.join(output_base, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

        scene_metrics = {"psnr": [], "ssim": [], "lpips": []}

        for img_path in sorted(glob.glob(os.path.join(input_dir, "*.png"))):
            fname = os.path.basename(img_path)



            input_image = load_image(img_path)
            ref_path = os.path.join(ref_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            if not os.path.exists(ref_path) or not os.path.exists(gt_path):
                continue

            ref_image = load_image(ref_path)
            gt_image = np.array(load_image(gt_path)).astype(np.float32) / 255.0

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
