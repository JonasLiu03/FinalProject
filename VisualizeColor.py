import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet_checkpoint = "SrresnetCheckpoint(cropsize96batch16prelu)/checkpoint_epoch_42_srresnet.pth.tar"
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()

def visualize_spectrum(image, title):
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    plt.figure(figsize=(6, 4))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_sr_modified(img, halve=False):
    hr_img = Image.open(img).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)
    bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)
    output_tuple = srresnet(convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = output_tuple[0].squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, '[-1, 1]', 'pil')
    visualize_spectrum(hr_img, 'HR Image Spectrum')
    visualize_spectrum(bicubic_img, 'Bicubic Image Spectrum')
    visualize_spectrum(sr_img_srresnet, 'SRResNet Image Spectrum')
    # hr_img.show(title="Original HR")
    # bicubic_img.show(title="Bicubic")
    # sr_img_srresnet.show(title="SRResNet")

if __name__ == '__main__':
    visualize_sr_modified("img/samurai.png")
