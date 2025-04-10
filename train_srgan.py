## TODO: EVERYTHING

import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision
import ssim
import signal
import sys
import math
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from layers import SqrtLoss, ConvLayer, ActivLayer
from models import SRCNN, VGG_Loss, freeze_model, unfreeze_model, SRResNet, Discriminator
from dataset import ImageDataset
from torch.amp import autocast, GradScaler
from train import train, compare_images
from utils import *
from contextlib import nullcontext
# Data parameters
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
n_channels = 3  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks

# Learning parameters
test = 0 # Enable test mode (show output images)
checkpoint = True  # Load checkpoint
unfreeze = False # Unfreeze all parameters
srresnet = False # Use referential resnet
srcnn_resnet = True # Use custom resnet
res_blocks = 16 # Number of residual blocks in resnet
nch = 96 # Number of channels in core layers
log2_upscale = False
batch_norm = False
# You can also input non-gan models as base to be retrained
base_model =  None #"base/4x96ssae_c5x2_rc3x16.pth" #"4x96mseganh_rc3x16_d8x32.pth"#"base/4x96ssae_c5x2_rc3x16.pth"
model_name = "auxresnet_maegan.pth" if srresnet else "4x96mseganh_rc3x16_d8x32.pth"
aux_name = "base/c5x4.pth" # Name of auxiliary upscaler network (or classical method like bicubic)
ps_ks = 3 # Pre-Pixel shuffle conv kernel size
last_ks = 0 # Add post shuffle conv layer, or when negative a clip function
del_last = False # Delete last layer
freeze = False # Freeze the backbone when appending shuffle conv layer

vgg_i = 5 # VGG_Loss maxpool index
vgg_j = 4 # VGG_Loss conv index (in a block)
vgg_alpha = 0.0 # Lerp mae with vgg loss
ssim_alpha = 0.5  # Mix mae with vgg
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 2 # Selected loss
## Gan params ##
dis_blocks = 8
dis_ch = 32
# Content loss
cont_min = 0.01 # If balance loss it's minimum contribution else it's total content loss contribution
cont_mul = 10.0 # If balance loss is active, it multiplies content loss weight
cont_adapt = cont_min / cont_mul # Adaptive content multuplier (don't change)

label_smooth = 0.00 # Label smoothing parameter
balance_loss = True
## Training params ##
ds_train = True # Set dataset to training mode (random crop position)
use_fp16 = True
batch_size = 64 # batch size
crop_size = 96 # Crop dimension for training
pre_scale = 1 # Prescale in training
lr = 1e-4 / 3 #/8  # learning rate
lr_disc = 0.1 * lr if balance_loss else lr # Base discriminator loss

try:
    import google.colab
    ds_cache = 'full'
    workers = 12  # number of workers for loading data in the DataLoader

except:
    ds_cache = 1000
    workers = 6  # number of workers for loading data in the DataLoader


min_loss = 1000000.0 # Minimal loss in network
start_epoch = 0  # start at this epoch
iterations = 3600 * 5  # number of training iterations
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 0 # Validation batch
valid_crop = 512 # Validation crop
grad_clip = None  # clip if gradients are exploding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint_ram = {}
checkpoint_safe = {}

def create_checkpoint(epoch, gen, disc, optimizer_d, optimizer_g, min_loss):
    return {'epoch': epoch,
                    'gen': gen,
                    'disc': disc,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d,
                    'loss' : min_loss}

def save_checkpoint():
    if not test and checkpoint_ram:
        print("Saving checkpoint...")
        torch.save(checkpoint_ram, model_name)  

def clear_memory():
    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()  # Free unused GPU memory

def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, min_loss, checkpoint_ram
    # Initialize gen or load checkpoint
    init_model = base_model if base_model and not test and checkpoint else model_name 
    if not checkpoint or not os.path.exists(init_model):
        if srresnet: 
            gen = SRResNet(9, 3, nch, res_blocks, scaling_factor, aux_name, 'lin', batch_norm)
        else:
            if not srcnn_resnet:
                layers = [(nch,5), (nch,5), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3)]#ESPCNN
            else:
                layers = [(nch,5), (nch,5)] # Custom srresnet implementation
                for i in range(res_blocks):
                    layers.append(('res', 3, batch_norm))
                layers.append((nch, 3))

            last_layer = (last_ks, 'clip') if last_ks != 0 else None
            gen = SRCNN(layers, n_channels, ps_ks, scaling_factor, aux_name, "lrelu", log2_upscale, last=last_layer)

        disc = Discriminator(3, dis_ch, dis_blocks, 1024)
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, disc.parameters()),lr=lr)

    else:
        checkpoint = torch.load(init_model, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1

        if('gen' in checkpoint):
            gen = checkpoint['gen']
        elif('model' in checkpoint):
            gen = checkpoint['model']
        if(del_last):
            delattr(gen, 'last_layer')
        # Allow loading models without discriminator
        if('disc' in checkpoint):
            disc = checkpoint['disc']
            optimizer_d = checkpoint['optimizer_d']
        else:
            disc = Discriminator(3, dis_ch, dis_blocks, 1024)
            optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, disc.parameters()),lr=lr)

        min_loss = checkpoint.get('loss', min_loss)
        print("Loaded gen:", init_model, "Loss:", min_loss)
        if last_ks > 0 and not hasattr(gen, 'last_layer'):
            if freeze:
                freeze_model(gen)
            gen.last_layer = ConvLayer(3,3,last_ks,1,1,'clip')
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        elif last_ks < 0 and not hasattr(gen, 'last_layer'):
             gen.last_layer = ActivLayer('clip')
             optimizer_g = checkpoint['optimizer'] if 'optimizer' in checkpoint else checkpoint['optimizer_g']
        elif unfreeze:
            unfreeze_model(gen)
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        elif 'optimizer_g' in checkpoint:
            #optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
            optimizer_g = checkpoint['optimizer_g']
        elif 'optimizer' in checkpoint:
            optimizer_g = checkpoint['optimizer']
        else:
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        

    if not test:
        summary(gen, input_size=(batch_size, n_channels, crop_size // scaling_factor, crop_size // scaling_factor))
    # Move to default device
    gen = gen.to(device, memory_format=torch.channels_last)
    disc = disc.to(device, memory_format=torch.channels_last)
    torch.compile(gen)
    torch.compile(disc)
    if test:
        train_dataset = ImageDataset("DIV2K", False, scaling_factor, pre_scale, test_crop, 0)
    else:
        train_dataset = ImageDataset("DIV2K", ds_train, scaling_factor, pre_scale, crop_size, ds_cache)
    if(test):
        for i in range(50):
            compare_images(train_dataset, gen, device, i + 100, scaling_factor)
            #c = input("Enter E to exit or enter to continue: ")
            #if(c == 'e'): break
        return
    
    # Select loss function
    if(loss_fns[loss_tp] == 'vgg'):
        vgg = VGG_Loss('mse', vgg_i, vgg_j, vgg_alpha)
        vgg_dims = (batch_size, n_channels, crop_size, crop_size)
        vgg_inp = torch.full(vgg_dims, 0, dtype=torch.float32)
        #summary(vgg, input_data=[vgg_inp, vgg_inp])
        vgg = vgg.to(device, memory_format=torch.channels_last)
        torch.compile(vgg)
        vgg.eval()
        criterion = vgg
    elif(loss_fns[loss_tp] == 'mae'):
        criterion = nn.L1Loss(reduction='mean')
    elif(loss_fns[loss_tp] == 'sqrt'):
        criterion = SqrtLoss()
    elif(loss_fns[loss_tp] == 'ssim'):
        criterion = ssim.SSIM(in_channels=3, as_loss=True, mae_alpha=ssim_alpha)
        criterion.to(device, memory_format=torch.channels_last)
    else:
        criterion = nn.MSELoss()
    
    for g in optimizer_g.param_groups:
        g['lr'] = lr
    for g in optimizer_d.param_groups:
        g['lr'] = lr_disc

    adv_criterion = None #nn.BCEWithLogitsLoss().to(device, memory_format=torch.channels_last)
    # Validation batch
    valid_x = []
    valid_y = []
    for idx in range(valid_size):
        x, y = train_dataset.load_img(idx, scaling_factor, pre_scale, valid_crop, False)
        valid_x.append(x)
        valid_y.append(y)
    if valid_size:
        valid_x = torch.stack(valid_x).to(device, memory_format=torch.channels_last)
        valid_y = torch.stack(valid_y).to(device, memory_format=torch.channels_last)
        valid_ds = (valid_x, valid_y)
    else: valid_ds = None
    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, prefetch_factor=2, persistent_workers=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations)
    print("Training for: ", epochs, " epochs")
    # Epochs
    local_epoch = int(0)
    loss = 0.99
    for epoch in range(start_epoch, epochs):
        if(local_epoch % 100 == 0) and loss < 1:
            print("Created backup checkpoint in epoch:", local_epoch)
            checkpoint_safe = create_checkpoint(epoch,gen,disc,optimizer_d,optimizer_g, min_loss)
        # One epoch's training
        loss = train_gan(train_loader=train_loader,
            gen=gen,
            disc=disc,
            criterion=criterion,
            adv_criterion=adv_criterion,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            epoch=epoch,
            grad_clip=grad_clip,
            print_freq=print_freq,
            device=device,
            valid_ds=valid_ds
            )
        if(loss < 5):
            checkpoint_ram = create_checkpoint(epoch,gen,disc,optimizer_d,optimizer_g, min_loss)
            min_loss = min(loss, min_loss)
        else:
            checkpoint_ram = checkpoint_safe
            print("Loss has exploded ! Try tweaking the learning rate")
            break
        local_epoch += 1
        #if(epoch):
        #   compare_images(train_dataset, gen, device, epoch, scaling_factor)
@torch.compile
def d_hinge_loss(real_pred, fake_pred):
    real_loss = torch.mean(F.relu(1. - real_pred))
    fake_loss = torch.mean(F.relu(1. + fake_pred))
    return real_loss + fake_loss

@torch.compile
def g_hinge_loss(fake_pred):
    return -torch.mean(fake_pred)

# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
def train_gan(train_loader, gen, disc, criterion, adv_criterion, optimizer_g, optimizer_d, epoch, grad_clip, print_freq, device, valid_ds = None):
    global lr_disc, cont_adapt
    """
    One epoch's training with mixed precision, channels_last optimization, and performance improvements.
    """
    gen.train()  # Enable training mode
    disc.train()
    #gen.to(memory_format=torch.channels_last)  # Convert gen to NHWC format

    gpu_time = AverageMeter()
    cpu_time = AverageMeter()
    losses_a = AverageMeter()
    losses_c = AverageMeter()
    losses_d = AverageMeter()
    # Initialize automatic mixed precision scaler
    scaler = GradScaler(enabled=use_fp16)
    t_cpu = time.time()
    tally = t_cpu
    autocast_ctx = autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else nullcontext()
    cont_weight = max(cont_adapt * cont_mul, cont_min) if balance_loss else cont_min
    for (lr_imgs, hr_imgs) in train_loader:
        # Move to GPU and convert format to channels_last
        #(lr_imgs, hr_imgs) = next(data_iter)
        lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        t_gpu = time.time()
        cpu_time.update(t_gpu - t_cpu)  # Time taken to load data
        # Mixed precision forward pass
        with autocast_ctx: 
            sr_imgs = gen(lr_imgs)
            sr_disc = disc(sr_imgs)
            g_loss = g_hinge_loss(sr_disc) #adv_criterion(sr_disc, torch.ones_like(sr_disc))
            c_loss = criterion(sr_imgs, hr_imgs)
            p_loss = g_loss + cont_weight * c_loss
        ## Generator update
        optimizer_g.zero_grad(set_to_none=True)
        scaler.scale(p_loss).backward()
              # Gradient clipping (if needed)
        if grad_clip is not None:
            scaler.unscale_(optimizer_g)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
        
        scaler.step(optimizer_g)
        loss_gen = g_loss.item()
        loss_con = c_loss.item()

        with autocast_ctx:
            hr_disc = disc(hr_imgs)
            sr_disc = disc(sr_imgs.detach())
            a_loss = d_hinge_loss(hr_disc, sr_disc) #adv_criterion(sr_disc, torch.zeros_like(sr_disc)) + adv_criterion(hr_disc, torch.full_like(hr_disc, 1.0 - label_smooth))

        optimizer_d.zero_grad(set_to_none=True)
        scaler.scale(a_loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer_d)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        scaler.step(optimizer_d)
        scaler.update()
        # Track loss
        loss_dis = a_loss.item()
        losses_c.update(loss_con, lr_imgs.size(0))
        losses_a.update(loss_gen, lr_imgs.size(0))
        losses_d.update(loss_dis, lr_imgs.size(0))
        t_cpu = time.time()
        gpu_time.update(t_cpu - t_gpu)


    if valid_ds:
        gen.eval()
        with torch.no_grad():
            val_loss = criterion(gen(valid_ds[0]), valid_ds[1]).item()
    else:
        val_loss = losses_c.avg()

    if(balance_loss):
        #ratio = losses_d.sum / losses_a.sum
        l = losses_d.avg()
        cont_adapt = losses_c.avg()
        rate = max(0.01, min(l, 1.0))
        #rate = 1 - 2 * max(0.5 - l, 0)
        #rate = rate * rate
        lr_disc = rate * lr
        #lr_disc = lr * max(min(ratio, 2.0), 0.002)
        for g in optimizer_d.param_groups:
            g['lr'] = lr_disc

    print(f'Epoch: [{epoch}]--'
        f'GPU tm ({gpu_time.sum:.3f})----'
        f'CPU tm ({cpu_time.sum:.3f})----'
        f'Total tm ({time.time() - tally:.3f})----'
        f'Loss cont ({losses_c.avg():.4f})--'
        f'Loss gen ({losses_a.avg():.4f})--'
        f'Loss disc ({losses_d.avg():.4f})--'
        f'Lr disc ({lr_disc:.2e})--'
        f'Wei cont ({cont_weight:.2e})')
    # Free memory
    if 'hr_disc' in locals():
        del lr_imgs, hr_imgs, sr_imgs, sr_disc, hr_disc
    else:
        del lr_imgs, hr_imgs, sr_imgs, sr_disc

    return val_loss if valid_ds is not None else losses_c.avg()



if __name__ == '__main__':
    try: 
        main()
        save_checkpoint()

    except KeyboardInterrupt:
        save_checkpoint()

    clear_memory()

