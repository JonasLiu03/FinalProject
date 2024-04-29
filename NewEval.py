from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import numpy as np
import torch
import torch.nn.functional as F

# MSE
def compute_mse(hr, sr):
    return np.mean((hr - sr) ** 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "./"
test_data_names = ["Set5", "Set14", "BSDS100"]
srresnet_checkpoint = "SrresnetCheckpoint/checkpoint_epoch_43_srresnet.pth.tar"
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)
    test_dataset = SRDataset(data_folder, split='test', crop_size=0, scaling_factor=4, lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]', test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    PSNRs_srresnet = AverageMeter()
    SSIMs_srresnet = AverageMeter()
    MSEs_srresnet = AverageMeter()
    PSNRs_bicubic = AverageMeter()
    SSIMs_bicubic = AverageMeter()
    MSEs_bicubic = AverageMeter()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            sr_imgs_y = convert_image(sr_imgs[0], source='[-1, 1]', target='y-channel').squeeze(0)
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            psnr_srresnet = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            ssim_srresnet = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            mse_srresnet = compute_mse(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy())
            PSNRs_srresnet.update(psnr_srresnet, lr_imgs.size(0))
            SSIMs_srresnet.update(ssim_srresnet, lr_imgs.size(0))
            MSEs_srresnet.update(mse_srresnet, lr_imgs.size(0))
            hr_imgs_downscaled = F.interpolate(hr_imgs, scale_factor=1/4, mode='bicubic', align_corners=False)
            hr_imgs_upscaled = F.interpolate(hr_imgs_downscaled, scale_factor=4, mode='bicubic', align_corners=False)
            hr_imgs_upscaled_y = convert_image(hr_imgs_upscaled, source='[-1, 1]', target='y-channel').squeeze(0)
            psnr_bicubic = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), hr_imgs_upscaled_y.cpu().numpy(), data_range=255.)
            ssim_bicubic = structural_similarity(hr_imgs_y.cpu().numpy(), hr_imgs_upscaled_y.cpu().numpy(), data_range=255.)
            mse_bicubic = compute_mse(hr_imgs_y.cpu().numpy(), hr_imgs_upscaled_y.cpu().numpy())
            PSNRs_bicubic.update(psnr_bicubic, lr_imgs.size(0))
            SSIMs_bicubic.update(ssim_bicubic, lr_imgs.size(0))
            MSEs_bicubic.update(mse_bicubic, lr_imgs.size(0))
    print('SRResNet PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs_srresnet))
    print('SRResNet SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs_srresnet))
    print(f'SRResNet MSE: {MSEs_srresnet.avg:.3f}')
    print('Bicubic PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs_bicubic))
    print('Bicubic SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs_bicubic))
    print(f'Bicubic MSE: {MSEs_bicubic.avg:.3f}')

print("\n")
