import matplotlib.pyplot as plt
from torch import nn
import torch
import pytorch_ssim
from torch.amp import autocast, GradScaler
from dataset import ImageDataset
from utils import *
import time

def compare_images(ds : ImageDataset, model : nn.Module, device, idx = 0, factor = 4):
    model.eval()

    lr, hr = ds.load_img(idx, factor, 1, 1024, False)
    sr_in = lr.unsqueeze(0).to(device, memory_format=torch.channels_last)
    sr = model(sr_in).squeeze()
    ssim = pytorch_ssim.ssim(sr.unsqueeze(0).to(device, memory_format=torch.channels_last), hr.unsqueeze(0).to(device,memory_format=torch.channels_last))
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

    fig.suptitle(f'SSIM: {ssim}')
    plt.tight_layout()
    plt.show()
    
# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
def train(train_loader, model, criterion, optimizer, epoch, grad_clip, print_freq, device):
    """
    One epoch's training with mixed precision, channels_last optimization, and performance improvements.
    """
    model.train()  # Enable training mode
    #model.to(memory_format=torch.channels_last)  # Convert model to NHWC format

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    tally = start
    # Initialize automatic mixed precision scaler
    scaler = GradScaler()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)  # Time taken to load data
        
        # Move to GPU and convert format to channels_last
        lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

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

        batch_time.update(time.time() - start)
        start = time.time()

    tally = (time.time() - tally)
    print(f'Epoch: [{epoch}]----'
        f'Batch Time ({batch_time.avg:.3f})----'
        f'Data Time ({data_time.avg:.3f})----'
        f'Time per iter ({tally:.3f})----'
        f'Loss ({losses.avg:.4f})')

    # Free memory
    del lr_imgs, hr_imgs, sr_imgs
    torch.cuda.empty_cache()
