import os
import re
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from datasets import SRDataset
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "./"
test_data_names = ["Set5", "Set14", "BSDS100"]

# Directory where the checkpoints are stored
checkpoint_dir = "D:/A_Refp/SrresnetCheckpoint3"

# Prepare to save the results
results_file = "crop64batch8mishALL.txt"
with open(results_file, 'w') as f:
    f.write("Model,Test Data,PSNR,SSIM,MSE\n")  # Add MSE in the header

epoch_pattern = re.compile(r'checkpoint_epoch_(\d+)_srresnet\.pth\.tar')
valid_checkpoints = [file for file in os.listdir(checkpoint_dir) if epoch_pattern.match(file)]

# Iterate over each valid checkpoint file
for idx, checkpoint_file in enumerate(valid_checkpoints, start=1):
    print(f"Processing {idx}/{len(valid_checkpoints)}: {checkpoint_file}")
    srresnet_checkpoint = os.path.join(checkpoint_dir, checkpoint_file)

    # Load the SRResNet model
    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srresnet.eval()
    model = srresnet

    # Evaluate on each data set
    for test_data_name in test_data_names:
        # Custom dataloader
        test_dataset = SRDataset(data_folder,
                                 split='test',
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                  pin_memory=True)

        # Track PSNR, SSIM, and MSE
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()
        MSEs = AverageMeter()  # Initialize an AverageMeter object for MSE

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = model(lr_imgs)

                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                mse = mean_squared_error(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy())  # Calculate MSE
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))
                MSEs.update(mse, lr_imgs.size(0))  # Update MSE AverageMeter

        # Save the results for this checkpoint and dataset, including MSE
        with open(results_file, 'a') as f:
            f.write(f"{checkpoint_file},{test_data_name},{PSNRs.avg:.3f},{SSIMs.avg:.3f},{MSEs.avg:.3f}\n")

print(f"Results saved to {results_file}")
