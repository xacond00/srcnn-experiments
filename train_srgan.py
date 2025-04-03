## TODO: EVERYTHING

import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision
import ssim

from torch import nn
from torchinfo import summary
from layers import SqrtLoss, ConvLayer
from models import SRCNN, VGG_Loss, freeze_model, unfreeze_model, SRResNet, Discriminator
from dataset import ImageDataset
from train import train, compare_images



# Data parameters
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
n_channels = 3  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks

# Learning parameters
checkpoint = True  # Load checkpoint
unfreeze = False # Unfreeze all parameters
test = False # Enable test mode (show output images)
srresnet = False # Use referential resnet
srcnn_resnet = True # Use custom resnet
res_blocks = 64 # Number of residual blocks in resnet
nch = 64 # Number of channels in core layers

base_model = None #"4x96ssae_c5x2_c3x6.pth"
model_name = "auxresnet_ssae_nobn.pth" if srresnet else "4x64ssae_c5x2_rc3x64.pth"
aux_name = "base/c5x4.pth" # Name of auxiliary upscaler network (or classical method like bicubic)
ps_ks = 3 # Pre-Pixel shuffle conv kernel size
last_ks = 0 # Add post shuffle conv layer (doesnt improve much)
freeze = False # Freeze the backbone when appending shuffle conv layer

vgg_i = 3 # VGG_Loss maxpool index
vgg_j = 3 # VGG_Loss conv index (in a block)
vgg_alpha = 0.0 # Lerp mae with vgg loss
ssim_alpha = 0.5  # Mix mae with vgg
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 4 # Selected loss

ds_train = True # Set dataset to training mode (random crop position)
batch_size = 8 # batch size
crop_size = 384 # Crop dimension for training
pre_scale = 1 # Prescale in training
lr = 2e-4 / 8 #/8  # learning rate

min_loss = 1000000.0 # Minimal loss in network
start_epoch = 0  # start at this epoch
iterations = 2000  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 8 # Validation batch
valid_crop = 512 # Validation crop
grad_clip = None  # clip if gradients are exploding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, min_loss

    # Initialize gen or load checkpoint
    init_model = base_model if base_model and not test and checkpoint else model_name 
    if not checkpoint or not os.path.exists(init_model):
        if srresnet: 
            gen = SRResNet(9, 3, nch, res_blocks, scaling_factor, aux_name, 'lin', False)
        else:
            if not srcnn_resnet:
                layers = [(nch,5), (nch,5), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3)]#ESPCNN
            else:
                layers = [(nch,5), (nch,5)] # Custom srresnet implementation
                for i in range(res_blocks):
                    layers.append(('res', 3))
                layers.append((nch, 3))

            last_layer = (last_ks, 'clip') if last_ks else None
            gen = SRCNN(layers, n_channels, ps_ks, scaling_factor, aux_name, "lrelu", last=last_layer)
            disc = Discriminator(3, 64, 8, 1024)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),
                                     lr=lr)
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, disc.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(init_model, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        gen = checkpoint['gen']
        disc = checkpoint['disc']
        min_loss = checkpoint.get('loss', min_loss)
        print("Loaded gen:", init_model, "Loss:", min_loss)
        
        if last_ks > 0 and not hasattr(gen, 'last_layer'):
            if freeze:
                freeze_model(gen)
            gen.last_layer = ConvLayer(3,3,last_ks,1,1,'clip')
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        elif unfreeze:
            unfreeze_model(gen)
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
        else:
            #optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),lr=lr)
            optimizer = checkpoint['optimizer']
        
        optimizer_d = checkpoint['optimizer_d']


    if not test:
        summary(gen, input_size=(batch_size, n_channels, crop_size // scaling_factor, crop_size // scaling_factor))
    # Move to default device
    gen = gen.to(device, memory_format=torch.channels_last)
    disc = disc.to(device, memory_format=torch.channels_last)
    if test:
        train_dataset = ImageDataset("DIV2K", False, scaling_factor, pre_scale, test_crop, 0)
    else:
        train_dataset = ImageDataset("DIV2K", ds_train, scaling_factor, pre_scale, crop_size)
    if(test):
        for i in range(50):
            compare_images(train_dataset, gen, device, i + 20, scaling_factor)
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
    
    for g in optimizer.param_groups:
        g['lr'] = lr

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
                                               pin_memory=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations)
    print("Training for: ", epochs, " epochs")
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        loss = train(train_loader=train_loader,
              gen=gen,
              disc=disc,
              criterion=criterion,
              optimizer=optimizer,
              optimizer_d=optimizer_d,
              epoch=epoch,
              grad_clip=grad_clip,
              print_freq=print_freq,
              device=device,
              valid_ds=valid_ds
              )
        if(loss < 5 * min_loss):
            min_loss = min(loss, min_loss)
        # Save checkpoint
            torch.save({'epoch': epoch,
                        'gen': gen,
                        'disc': disc,
                        'optimizer': optimizer,
                        'loss' : min_loss},
                    model_name)
        else:
            print("Loss has exploded ! Try tweaking the learning rate")
            break
        #if(epoch % 20 == 0):
        #    compare_images(train_dataset, gen, device, epoch, scaling_factor)
        

if __name__ == '__main__':
    main()


# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
def train_gan(train_loader, gen, disc, criterion, optimizer, optimizer_d, epoch, grad_clip, print_freq, device, valid_ds = None):
    """
    One epoch's training with mixed precision, channels_last optimization, and performance improvements.
    """
    gen.train()  # Enable training mode
    #gen.to(memory_format=torch.channels_last)  # Convert gen to NHWC format

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
            sr_imgs = gen(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient clipping (if needed)
        if grad_clip is not None:
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        losses.update(loss.item(), lr_imgs.size(0))

        batch_time.update(time.time() - start)
        start = time.time()
    if valid_ds:
        gen.eval()
        with torch.no_grad():
            val_loss = criterion(gen(valid_ds[0]), valid_ds[1]).item()

    tally = (time.time() - tally)
    print(f'Epoch: [{epoch}]----'
        f'Batch Time ({batch_time.avg:.3f})----'
        f'Data Time ({data_time.avg:.3f})----'
        f'Time per iter ({tally:.3f})----'
        f'Loss ({losses.avg:.4f})----'
        f'Val loss ({val_loss:.4f})')
    # Free memory
    del lr_imgs, hr_imgs, sr_imgs
    torch.cuda.empty_cache()
    return val_loss if valid_ds is not None else losses.avg

