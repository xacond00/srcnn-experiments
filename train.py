# Normal ResNet/SRSPCN training
# xacond00

import matplotlib.pyplot as plt
from torch import nn
import torch
import ssim
from torch.amp import autocast, GradScaler
from dataset import ImageDataset
from utils import *
import time

def compare_images(ds : ImageDataset, model : nn.Module, device, idx = 0, factor = 4, downsample = 1):
    model.eval()

    lr, hr = ds.load_img(idx, factor, downsample, 1024, False)
    sr_in = lr.unsqueeze(0).to(device, memory_format=torch.channels_last)
    with torch.no_grad():
        sr = model(sr_in).squeeze()
        sr = torch.clip(sr, 0, 1)
    x = sr.unsqueeze(0).to(device, memory_format=torch.channels_last)
    y = hr.unsqueeze(0).to(device,memory_format=torch.channels_last)
    ssim_val = ssim.ssim(x, y, in_channels=3)
    sr = sr.permute(1, 2, 0).cpu().detach().numpy()
    lr = lr.permute(1, 2, 0).numpy()
    hr = hr.permute(1, 2, 0).numpy()

    # Create a subplot for side-by-side display
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # Show LR image
    axes[0].imshow(lr)
    axes[0].axis("off")
    axes[0].set_title("Low-Resolution (LR)")
    # Show HR image
    axes[1].imshow(sr)
    axes[1].axis("off")
    axes[1].set_title("Super-Resolution (SR)")
    # Show HR image
    axes[2].imshow(hr)
    axes[2].axis("off")
    axes[2].set_title("High-Resolution (HR)")

    fig.suptitle(f'SSIM: {ssim_val}')
    plt.tight_layout()
    plt.show()
    
# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
def train(train_loader, model, criterion, optimizer, epoch, grad_clip, print_freq, device, valid_ds = None):
    """
    One epoch's training with mixed precision, channels_last optimization, and performance improvements.
    """
    model.train()  # Enable training mode
    #model.to(memory_format=torch.channels_last)  # Convert model to NHWC format

    gpu_time = AverageMeter()
    cpu_time = AverageMeter()
    losses = AverageMeter()

    # Initialize automatic mixed precision scaler
    scaler = GradScaler()
    t_cpu = time.time()
    tally = t_cpu
    for (lr_imgs, hr_imgs) in train_loader:
        # Move to GPU and convert format to channels_last
        lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        t_gpu = time.time()
        cpu_time.update(t_gpu - t_cpu)
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.float16):
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient clipping (if needed)
        if grad_clip is not None:
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        losses.update(loss.item(), lr_imgs.size(0))
        t_cpu = time.time()
        gpu_time.update(t_cpu - t_gpu)

    # Printing    
    if valid_ds:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(valid_ds[0]), valid_ds[1]).item()

    print(f'Epoch: [{epoch}]----'
        f'GPU tm ({gpu_time.sum:.3f})----'
        f'CPU tm ({cpu_time.sum:.3f})----'
        f'Total tm ({time.time() - tally:.3f})----'
        f'Loss ({losses.avg():.4f})----'
        f'Loss val ({val_loss:.4f})')
    # Free memory
    del lr_imgs, hr_imgs, sr_imgs
    return val_loss if valid_ds is not None else losses.avg()

