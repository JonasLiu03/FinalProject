import torch
import lpips
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_srresnet_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    return model

def process_image(img_path, target_size=None, downscale_factor=4):
    img = Image.open(img_path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    lr_img = img.resize((img.width // downscale_factor, img.height // downscale_factor), Image.BICUBIC)
    lr_img = ToTensor()(lr_img).unsqueeze(0).to(device)
    return img, lr_img

def generate_sr_image(model, lr_tensor):
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_img = ToPILImage()(sr_tensor.squeeze().cpu().clamp(0, 1))
    return sr_img

def calculate_lpips_distance(img1, img2):
    transform = transforms.Compose([
        ToTensor()
    ])
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)  # 使用AlexNet作为底层网络
    distance = lpips_fn(img1_tensor, img2_tensor)
    return distance.item()

def main():
    model_path = "SrresnetCheckpoint(cropsize64batch8mish)/checkpoint_epoch_42_srresnet.pth.tar"
    hr_image_path = "img/butterfly.png"
    srresnet = load_srresnet_model(model_path)
    hr_img, lr_tensor = process_image(hr_image_path)
    sr_img = generate_sr_image(srresnet, lr_tensor)
    distance = calculate_lpips_distance(hr_img, sr_img)
    print(f"LPIPS distance: {distance}")
    hr_img.show(title="Original HR Image")
    sr_img.show(title="Super Resolved Image")

if __name__ == "__main__":
    main()


