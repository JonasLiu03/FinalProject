from utils import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

srresnet_checkpoint = "SrresnetCheckpoint(cropsize96batch16prelu)/checkpoint_epoch_43_srresnet.pth.tar"
checkpoint = torch.load(srresnet_checkpoint, map_location=device)
srresnet = checkpoint['model'].to(device)
srresnet.eval()

def visualize_attention(img, halve=False):
    hr_img = Image.open(img).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    lr_tensor = convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor, attention_maps = srresnet(lr_tensor)


    attention_map = attention_maps[0].squeeze()
    attention_map = np.mean(attention_map, axis=0)
    attention_map = np.clip(attention_map, 0, 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(hr_img)
    plt.title("Original HR")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(lr_img.resize(hr_img.size, Image.BICUBIC))
    plt.title("Bicubic")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    sr_img = convert_image(sr_tensor.squeeze(0), '[-1, 1]', 'pil')
    plt.imshow(sr_img)
    plt.title("SRResNet")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(attention_map, cmap='hot')
    plt.title("Attention Map")
    plt.axis("off")

    plt.show()

if __name__ == '__main__':
    visualize_attention("D:/A_Refp/img/butterfly.png")
