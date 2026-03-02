import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from third_party.difix.src.model import Difix

def process_images_with_difix(img_tensor, model_path):

    model = Difix(
        pretrained_path=model_path,
        timestep=199,
        mv_unet=False
    )
    model.set_eval()

    _, orig_h, orig_w = img_tensor.shape
    
    new_h = ((orig_h + 7) // 8) * 8
    new_w = ((orig_w + 7) // 8) * 8
    
    if new_h != orig_h or new_w != orig_w:
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        img_pil = to_pil(img_tensor)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tensor = to_tensor(img_pil)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    input_pil = transforms.ToPILImage()(img_tensor.cpu()) 

    model.sched.set_timesteps(1, device=device)
    model.sched.timesteps = torch.tensor([model.timesteps.item()], device=device)

    
    output_pil = model.sample(
        input_pil,
        width=new_w,
        height=new_h,
        prompt="remove degradation"
    )
    
    output_tensor = transforms.ToTensor()(output_pil)
    
    if new_h != orig_h or new_w != orig_w:
        output_tensor = F.interpolate(
            output_tensor.unsqueeze(0), 
            size=(orig_h, orig_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return output_tensor
