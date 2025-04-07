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
from torchinfo import summary
from layers import SqrtLoss, ConvLayer
from models import SRCNN, VGG_Loss, freeze_model, unfreeze_model, SRResNet, Discriminator
from dataset import ImageDataset
from torch.amp import autocast, GradScaler
from train import train, compare_images
from utils import *

# Data parameters
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
n_channels = 3  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks

# Learning parameters
checkpoint = True  # Load checkpoint
unfreeze = False # Unfreeze all parameters
test = False# Enable test mode (show output images)
srresnet = False # Use referential resnet
srcnn_resnet = True # Use custom resnet
res_blocks = 16 # Number of residual blocks in resnet
nch = 64 # Number of channels in core layers
batch_norm = False

base_model = None #"final/4x96ssae_c5x2_c3x6.pth" #None #"4x96maegan_c5x2_rc3x16_w1.pth" #"4x96maegan_c5x2_rc3x16_w5.pth" #"4x96maegan_c5x2_rc3x16.pth" #None #"final/4x96ssae_c5x2_rc3x16.pth"
model_name = "auxresnet_ssae_nobn.pth" if srresnet else "4x96maegan_c5x2_rc3x16bn.pth"#"4x96maegan_c5x2_c3x6.pth" #"4x96maegan_c5x2_rc3x16bn_ns.pth"#"4x64maegan_c5x2_rc3x16pu.pth"
aux_name = "base/c5x4.pth" # Name of auxiliary upscaler network (or classical method like bicubic)
ps_ks = 3 # Pre-Pixel shuffle conv kernel size
last_ks = 0 # Add post shuffle conv layer (doesnt improve much)
freeze = False # Freeze the backbone when appending shuffle conv layer

vgg_i = 3 # VGG_Loss maxpool index
vgg_j = 3 # VGG_Loss conv index (in a block)
vgg_alpha = 0.0 # Lerp mae with vgg loss
ssim_alpha = 0.5  # Mix mae with vgg
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 0 # Selected loss
# Gan params
cont_alpha = 0.1 # Weight of content loss
label_smooth = 0.02 # Label smoothing parameter
balance_loss = True
ds_train = True # Set dataset to training mode (random crop position)
batch_size = 16 # batch size
crop_size = 192 # Crop dimension for training
pre_scale = 1 # Prescale in training
lr = 1e-4 / 2 #/8  # learning rate
try:
    import google.colab
    ds_cache = 10000
except:
    ds_cache = 0

min_loss = 1000000.0 # Minimal loss in network
start_epoch = 0  # start at this epoch
iterations = 5000  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 0 # Validation batch
valid_crop = 512 # Validation crop
grad_clip = None  # clip if gradients are exploding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint_saved = False
checkpoint_ram = {}

def store_checkpoint(epoch, gen, disc, optimizer_d, optimizer_g, min_loss):
    global checkpoint_ram
    checkpoint_ram = {'epoch': epoch,
                    'gen': gen,
                    'disc': disc,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d,
                    'loss' : min_loss}

def save_checkpoint_on_exit(signum, frame):
    global checkpoint_saved

    if not checkpoint_saved and not test and checkpoint_ram:
        print("Saving checkpoint...")
        torch.save(checkpoint_ram, model_name)  
        checkpoint_saved = True  
        if torch.cuda.is_available():
            print("Clearing GPU memory...")
            torch.cuda.empty_cache()  # Free unused GPU memory

    sys.exit(0)  # Exit the program gracefully

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

            last_layer = (last_ks, 'clip') if last_ks else None
            gen = SRCNN(layers, n_channels, ps_ks, scaling_factor, aux_name, "lrelu", last=last_layer)

        disc = Discriminator(3, 64, 8, 1024)
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, disc.parameters()),lr=lr)

    else:
        checkpoint = torch.load(init_model, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1

        if('gen' in checkpoint):
            gen = checkpoint['gen']
        elif('model' in checkpoint):
            gen = checkpoint['model']

        # Allow loading models without discriminator
        if('disc' in checkpoint):
            disc = checkpoint['disc']
            optimizer_d = checkpoint['optimizer_d']
        else:
            disc = Discriminator(3, 64, 8, 1024)
            optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, disc.parameters()),lr=lr)

        min_loss = checkpoint.get('loss', min_loss)
        print("Loaded gen:", init_model, "Loss:", min_loss)
        if last_ks > 0 and not hasattr(gen, 'last_layer'):
            if freeze:
                freeze_model(gen)
            gen.last_layer = ConvLayer(3,3,last_ks,1,1,'clip')
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
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
        g['lr'] = lr * 0.5

    adv_criterion = nn.BCEWithLogitsLoss().to(device)
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
                                               pin_memory=True, prefetch_factor=2)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations)
    print("Training for: ", epochs, " epochs")
    # Epochs
    for epoch in range(start_epoch, epochs):
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
        if(loss < 1000 * min_loss):
            min_loss = min(loss, min_loss)
            store_checkpoint(epoch,gen,disc,optimizer_d,optimizer_g, min_loss)
        else:
            print("Loss has exploded ! Try tweaking the learning rate")
            break
        #if(epoch % 20 == 0):
        #    compare_images(train_dataset, gen, device, epoch, scaling_factor)

# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
def train_gan(train_loader, gen, disc, criterion, adv_criterion, optimizer_g, optimizer_d, epoch, grad_clip, print_freq, device, valid_ds = None):
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
    ratios = AverageMeter()
    # Initialize automatic mixed precision scaler
    scaler = GradScaler()
    data_iter = iter(train_loader)
    t_cpu = time.time()
    tally = t_cpu
    for _ in range(len(train_loader)):
        # Move to GPU and convert format to channels_last
        (lr_imgs, hr_imgs) = next(data_iter)
        lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        t_gpu = time.time()
        cpu_time.update(t_gpu - t_cpu)  # Time taken to load data
        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.float16):
            sr_imgs = gen(lr_imgs)
            sr_disc = disc(sr_imgs)
            a_loss = adv_criterion(sr_disc, torch.ones_like(sr_disc))
            if(torch.isnan(a_loss).item() == True):
                print("Loss is NAN !")
                gpu_time.update(start - start_iter)
                ratios.update(0.00001, 0.00001)
                losses_c.update(loss_con, lr_imgs.size(0))
                losses_a.update(loss_gen, lr_imgs.size(0))
                losses_d.update(0.00001, 0.00001)
                continue
                
            c_loss = criterion(sr_imgs, hr_imgs)
            p_loss = a_loss + c_loss * cont_alpha
        ## Generator update
        optimizer_g.zero_grad(set_to_none=True)
        scaler.scale(p_loss).backward()
              # Gradient clipping (if needed)
        if grad_clip is not None:
            scaler.unscale_(optimizer_g)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
        
        loss_con = c_loss.item()
        loss_gen = a_loss.item()
        scaler.step(optimizer_g)

        with autocast(device_type='cuda', dtype=torch.float16):

            hr_disc = disc(hr_imgs)
            sr_disc = disc(sr_imgs.detach())
            a_loss = adv_criterion(sr_disc, torch.zeros_like(sr_disc)) + adv_criterion(hr_disc, torch.full_like(hr_disc, 1.0 - label_smooth))
            if(torch.isnan(a_loss).item() == True):
                print("Loss is NAN !")
                continue
        loss_dis = a_loss.item()
        ratio = loss_gen / loss_dis
        if balance_loss:
            for g in optimizer_d.param_groups:
                g['lr'] = lr * max(min(1 / math.sqrt(ratio), 2.0), 0.1)
        optimizer_d.zero_grad(set_to_none=True)
        scaler.scale(a_loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer_d)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        scaler.step(optimizer_d)

        scaler.update()

        # Track loss
        ratios.update(ratio, lr_imgs.size(0))
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

    print(f'Epoch: [{epoch}]--'
        f'GPU tm ({gpu_time.sum:.3f})----'
        f'CPU tm ({cpu_time.sum:.3f})----'
        f'Total tm ({time.time() - tally:.3f})----'
        f'Loss cont ({losses_c.avg():.4f})--'
        f'Loss adve ({losses_a.avg():.4f})--'
        f'Loss disc ({losses_d.avg():.4f})--'
        f'Ratio ({ratios.avg():.4f})--'
        f'Loss val ({val_loss:.4f})')
    # Free memory
    if 'hr_disc' in locals():
        del lr_imgs, hr_imgs, sr_imgs, sr_disc, hr_disc
    else:
        del lr_imgs, hr_imgs, sr_imgs, sr_disc

    return val_loss if valid_ds is not None else losses_c.avg()

if __name__ == '__main__':
    try: 
        main()
        save_checkpoint_on_exit(0,0)

    except KeyboardInterrupt:
        save_checkpoint_on_exit(0,0)


