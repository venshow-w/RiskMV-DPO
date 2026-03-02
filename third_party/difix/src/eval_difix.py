import os
import json
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from model import Difix
from torchvision import transforms
import torchmetrics
import lpips

def compute_metrics(pred, gt):
    """Compute PSNR, SSIM, LPIPS between two PIL images"""
    pred_tensor = transforms.ToTensor()(pred).unsqueeze(0).cuda()
    gt_tensor = transforms.ToTensor()(gt).unsqueeze(0).cuda()

    # PSNR & SSIM
    psnr = torchmetrics.functional.peak_signal_noise_ratio(pred_tensor, gt_tensor, data_range=1.0).item()
    ssim = torchmetrics.functional.structural_similarity_index_measure(pred_tensor, gt_tensor, data_range=1.0).item()

    # LPIPS
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    lpips_val = loss_fn(pred_tensor * 2 - 1, gt_tensor * 2 - 1).item()

    return psnr, ssim, lpips_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, required=True, help='Directory containing .pkl weights')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Test dataset directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--height', type=int, default=544)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    weight_files = sorted(glob(os.path.join(args.weights_dir, "model_*.pkl")))
    weight_files = [wf for wf in weight_files if int(os.path.basename(wf).split("_")[1].split(".")[0]) % 5000 == 1]

    for weight_file in weight_files:
        epoch = int(os.path.basename(weight_file).split("_")[1].split(".")[0])
        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        model = Difix(
            pretrained_name=None,
            pretrained_path=weight_file,
            timestep=args.timestep,
            mv_unet=True
        )
        model.set_eval()

        scene_dirs = [d for d in glob(os.path.join(args.test_data_dir, "*")) if os.path.isdir(d)]
        metrics_all_scenes = {}

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)
            image_dir = os.path.join(scene_dir, "images")
            ref_dir = os.path.join(scene_dir, "ref_image")
            target_dir = os.path.join(scene_dir, "target_image")

            output_scene_dir = os.path.join(epoch_output_dir, scene_name, "images")
            os.makedirs(output_scene_dir, exist_ok=True)

            input_images = sorted([p for p in glob(os.path.join(image_dir, "*.png")) if "step29999" in os.path.basename(p)])
            ref_images = sorted([p for p in glob(os.path.join(ref_dir, "*.png")) if os.path.basename(p) in [os.path.basename(ip) for ip in input_images]])
            target_images = {os.path.basename(p): p for p in glob(os.path.join(target_dir, "*.png"))}

            scene_metrics = []

            for i, input_image_path in enumerate(tqdm(input_images, desc=f"Processing {scene_name}")):
                input_image = Image.open(input_image_path).convert('RGB')
                ref_image = Image.open(ref_images[i]).convert('RGB') if ref_images else None

                output_image = model.sample(
                    input_image,
                    height=args.height,
                    width=args.width,
                    ref_image=ref_image,
                    prompt=args.prompt
                )

                output_path = os.path.join(output_scene_dir, os.path.basename(input_image_path))
                output_image.save(output_path)

                gt_path = target_images.get(os.path.basename(input_image_path), None)
                if gt_path:
                    gt_image = Image.open(gt_path).convert('RGB')
                    psnr, ssim, lpips_val = compute_metrics(output_image, gt_image)
                    scene_metrics.append({"psnr": psnr, "ssim": ssim, "lpips": lpips_val})

            if scene_metrics:
                avg_psnr = np.mean([m["psnr"] for m in scene_metrics])
                avg_ssim = np.mean([m["ssim"] for m in scene_metrics])
                avg_lpips = np.mean([m["lpips"] for m in scene_metrics])
                metrics_all_scenes[scene_name] = {
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                    "lpips": avg_lpips
                }

        with open(os.path.join(epoch_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_all_scenes, f, indent=4)
