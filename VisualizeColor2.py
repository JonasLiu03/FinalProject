import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet_checkpoint = "SrresnetCheckpoint/checkpoint_epoch_42_srresnet.pth.tar"
checkpoint = torch.load(srresnet_checkpoint, map_location=device)
srresnet = checkpoint['model'].to(device)
srresnet.eval()


def visualize_spectrum(image, title):
    # Convert the image to grayscale
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    # Apply Fourier transform
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    # Calculate the magnitude spectrum and visualize using a logarithmic scale
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    plt.figure(figsize=(6, 4))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    return fft_shifted
def calculate_spectral_properties(fft_image):
    # Calculate energy in the entire spectrum (sum of squares)
    energy_total = np.sum(np.abs(fft_image) ** 2)
    # Assuming the center of the spectrum is the low-frequency region
    center_x, center_y = fft_image.shape[0] // 2, fft_image.shape[1] // 2
    radius = 20  # Define radius for low-frequency region
    # Create a circular mask for low frequencies
    y, x = np.ogrid[:fft_image.shape[0], :fft_image.shape[1]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    energy_low = np.sum(np.abs(fft_image * mask) ** 2)
    energy_high = energy_total - energy_low  # High frequency energy is the remainder
    return energy_low, energy_high, energy_total


def spectral_loss(fft_image_1, fft_image_2):
    # Calculate the mean squared error between two spectra
    loss = np.mean((np.abs(fft_image_1) - np.abs(fft_image_2)) ** 2)
    return loss
def visualize_sr_modified(img_path, halve=False):
    hr_img = Image.open(img_path).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)

    sr_img_tensor = srresnet(convert_image(lr_img, 'pil', 'imagenet-norm').unsqueeze(0).to(device))
    sr_img_tensor = sr_img_tensor[0].squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_tensor, '[-1, 1]', 'pil')

    # Visualize the frequency domain analysis
    fft_hr = visualize_spectrum(hr_img, 'HR Image Spectrum')
    fft_bicubic = visualize_spectrum(bicubic_img, 'Bicubic Image Spectrum')
    fft_srresnet = visualize_spectrum(sr_img_srresnet, 'SRResNet Image Spectrum')

    # Calculate spectral properties
    energy_low_hr, energy_high_hr, energy_total_hr = calculate_spectral_properties(fft_hr)
    energy_low_bicubic, energy_high_bicubic, energy_total_bicubic = calculate_spectral_properties(fft_bicubic)
    energy_low_srresnet, energy_high_srresnet, energy_total_srresnet = calculate_spectral_properties(fft_srresnet)

    # Calculate spectral loss between HR and SR
    loss_hr_bicubic = spectral_loss(fft_hr, fft_bicubic)
    loss_hr_srresnet = spectral_loss(fft_hr, fft_srresnet)

    print("Spectral Properties (HR Image): Low Energy = {:.2f}, High Energy = {:.2f}, Total Energy = {:.2f}".format(
        energy_low_hr, energy_high_hr, energy_total_hr))
    print(
        "Spectral Properties (Bicubic Image): Low Energy = {:.2f}, High Energy = {:.2f}, Total Energy = {:.2f}".format(
            energy_low_bicubic, energy_high_bicubic, energy_total_bicubic))
    print(
        "Spectral Properties (SRResNet Image): Low Energy = {:.2f}, High Energy = {:.2f}, Total Energy = {:.2f}".format(
            energy_low_srresnet, energy_high_srresnet, energy_total_srresnet))

    print("Spectral Loss (HR vs Bicubic): {:.2f}".format(loss_hr_bicubic))
    print("Spectral Loss (HR vs SRResNet): {:.2f}".format(loss_hr_srresnet))


if __name__ == '__main__':
    visualize_sr_modified("img/butterfly.png")
