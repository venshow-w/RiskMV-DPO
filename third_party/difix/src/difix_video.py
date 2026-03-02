import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import lpips
from PIL import Image
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from model import Difix

MODEL_PATH = "path_to_difix.pkl"
HEIGHT = 304
WIDTH = 528
PROMPT = "remove degradation"
TIMESTEP = 199

device = "cuda"
lpips_fn = lpips.LPIPS(net="alex").to(device)

def compute_metrics(pred, gt):
    """ pred/gt: numpy [H, W, 3], float32, [0,1] """
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    lpips_val = lpips_fn(pred_t, gt_t).item()
    return psnr_val, ssim_val, lpips_val

def process_video(scene_dir):
    video_path = os.path.join(scene_dir, "video.mp4")
    gt_video_path = os.path.join(scene_dir, "gt_video.mp4")
    if not (os.path.exists(video_path) and os.path.exists(gt_video_path)):
        return None

    reader = imageio.get_reader(video_path)
    gt_reader = imageio.get_reader(gt_video_path)
    n_frames = min(reader.count_frames(), gt_reader.count_frames())

    model = Difix(
        pretrained_name=None,
        pretrained_path=MODEL_PATH,
        timestep=TIMESTEP,
        mv_unet=False
    )
    model.set_eval()

    psnr_list, ssim_list, lpips_list = [], [], []
    for idx in tqdm(range(n_frames), desc=f"{os.path.basename(scene_dir)}"):
        frame = reader.get_data(idx)
        gt_frame = gt_reader.get_data(idx)
        frame_pil = Image.fromarray(frame).convert("RGB")
        gt_pil = Image.fromarray(gt_frame).convert("RGB")

        with torch.no_grad():
            pred_pil = model.sample(
                frame_pil,
                height=HEIGHT,
                width=WIDTH,
                ref_image=None,
                prompt=PROMPT
            )
        pred_pil = pred_pil.resize(gt_pil.size)
        pred_np = np.array(pred_pil).astype(np.float32) / 255.0
        gt_np = np.array(gt_pil).astype(np.float32) / 255.0

        psnr_val, ssim_val, lpips_val = compute_metrics(pred_np, gt_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

    reader.close()
    gt_reader.close()
    return {
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
        "lpips": np.mean(lpips_list)
    }

def main():
    A = "vis/xxx"
    scene_dirs = sorted([os.path.join(A, d) for d in os.listdir(A) if os.path.isdir(os.path.join(A, d))])
    all_metrics = {"psnr": [], "ssim": [], "lpips": []}
    for scene_dir in scene_dirs:
        metrics = process_video(scene_dir)
        if metrics is not None:
            print(f"{os.path.basename(scene_dir)}: PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")
            for k in all_metrics:
                all_metrics[k].append(metrics[k])
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}
if __name__ == "__main__":
    main()