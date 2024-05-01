import time
import os
import torch
from torch import nn
from models import SRResNet
from datasets import EnhancedDataset
from utils import AverageMeter, clip_gradient
from torch.optim import Adam
from torch.utils.data import DataLoader

# Constants
DATA_FOLDER = './'
CHECKPOINT_PATH = 'SrresnetCheckpoint(cropsize64batch8prelu)/checkpoint_epoch_19_srresnet.pth.tar'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDNN_BENCHMARK = True
CUDA_LAUNCH_BLOCKING = 1

def load_model(checkpoint_path):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch'] + 1
    else:
        model = SRResNet(large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4)
        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        start_epoch = 0

    return model, optimizer, start_epoch

def save_checkpoint(model, optimizer, epoch, folder):
    path = os.path.join(folder, f'checkpoint_epoch_{epoch}_srresnet.pth.tar')
    os.makedirs(folder, exist_ok=True)
    torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer}, path)

def train_epoch(train_loader, model, criterion, optimizer, epoch, grad_clip=None):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)
        lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        if grad_clip:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % 500 == 0:
            print_status(epoch, i, train_loader, losses, batch_time, data_time)

def print_status(epoch, i, train_loader, losses, batch_time, data_time):
    print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]----'
          f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
          f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
          f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def main():
    model, optimizer, start_epoch = load_model(CHECKPOINT_PATH)
    model.to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
    train_dataset = EnhancedDataset(DATA_FOLDER, split='train', crop_size=64, scaling_factor=4, lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    epochs = 50

    for epoch in range(start_epoch, epochs):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch, 'SrresnetCheckpoint(cropsize64batch8prelu)')

if __name__ == '__main__':
    main()