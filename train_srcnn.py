# Normal ResNet/SRSPCN training
# xacond00

import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision
import ssim

from torch import nn
from torchinfo import summary
from layers import SqrtLoss, ConvLayer
from baseline_models import SRCNN, VGG_Loss, freeze_model, unfreeze_model, SRResNet
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
res_blocks = 24 # Number of residual blocks in resnet
nch = 96 # Number of channels in core layers
log2_upscale = False
batch_norm = False
output_activ = 'linear'

if srresnet:
    model_name = "auxresnet.pth"
else:
    model_name = "gan_base/4x96_rc3x16.pth"

"""
4x96ssae_c5x2_rc3x16.pth = 
Velikost Kanaly Loss _ Konvoluce Kernel x Pocet _ ResKonvoluce Kernel x Pocet . pth
"""

base_model = None  # "4x96ssae_c5x2_c3x6.pth"
aux_name = "base/c5x4.pth"  # Name of auxiliary upscaler network (or classical method like bicubic)
ps_ks = 3 # Pre-Pixel shuffle conv kernel size
last_ks = 0  # Add post shuffle conv layer (doesnt improve much)
freeze = False  # Freeze the backbone when appending shuffle conv layer

vgg_i = 3 # VGG_Loss maxpool index
vgg_j = 3 # VGG_Loss conv index (in a block)
vgg_alpha = 0.0 # Lerp mae with vgg loss
ssim_alpha = 0.5  # Mix mae with vgg
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 0 # Selected loss

ds_train = True # Set dataset to training mode (random crop position)
batch_size = 64 # batch size
crop_size = 128 # Crop dimension for training
pre_scale = 1 # Prescale in training
lr = 1e-4 #/8  # learning rate
try:
    import google.colab
    ds_cache = 'full'
    workers = 12  # number of workers for loading data in the DataLoader

except:
    ds_cache = 1000
    workers = 6  # number of workers for loading data in the DataLoader


min_loss = 1000000.0 # Minimal loss in network
start_epoch = 0  # start at this epoch
iterations = 2000  # number of training iterations
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 8 # Validation batch
valid_crop = 512 # Validation crop
grad_clip = None  # clip if gradients are exploding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint_saved = False
checkpoint_ram = {}

def save_checkpoint_on_exit(signum, frame):
    global checkpoint_saved

    if not checkpoint_saved and not test and checkpoint_ram:
        print("Saving checkpoint...")
        torch.save(checkpoint_ram, model_name)  
        checkpoint_saved = True
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
            gen = SRResNet(9, 3, nch, res_blocks, scaling_factor, aux_name, output_activ, batch_norm)
        else:
            if not srcnn_resnet:  # ESPCNN
                layers = [(nch,5), (nch,5), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3)]
            else:  # Custom srresnet implementation
                layers = [(nch,5), (nch,5)]
                for i in range(res_blocks):
                    layers.append(('res', 3, batch_norm))
                layers.append((nch, 3))

            last_layer = (last_ks, 'clip') if last_ks else None
            gen = SRCNN(layers, n_channels, ps_ks, scaling_factor, aux_name, "lrelu", log2_upscale, last=last_layer, output_activ=output_activ)

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(init_model, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        gen = checkpoint['gen'] if 'gen' in checkpoint else checkpoint['model']
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

    if not test:
        summary(gen, input_size=(batch_size, n_channels, crop_size // scaling_factor, crop_size // scaling_factor))
    # Move to default device
    gen = gen.to(device, memory_format=torch.channels_last)
    if test:
        train_dataset = ImageDataset("DIV2K", False, scaling_factor, pre_scale, test_crop, 0)
    else:
        train_dataset = ImageDataset("DIV2K", ds_train, scaling_factor, pre_scale, crop_size, ds_cache)
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
                                               pin_memory=True, prefetch_factor=2, persistent_workers=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations)
    print("Training for: ", epochs, " epochs")
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        loss = train(train_loader=train_loader,
              model=gen,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              grad_clip=grad_clip,
              print_freq=print_freq,
              device=device,
              valid_ds=valid_ds
              )
        if(loss < 5 * min_loss):
            min_loss = min(loss, min_loss)
        # Save checkpoint
            checkpoint_ram = {'epoch': epoch,
                        'gen': gen,
                        'optimizer': optimizer,
                        'loss' : min_loss}
        else:
            print("Loss has exploded ! Try tweaking the learning rate")
            break
        #if(epoch % 20 == 0):
        #    compare_images(train_dataset, gen, device, epoch, scaling_factor)
        


if __name__ == '__main__':
    try: 
        main()
        save_checkpoint_on_exit(0,0)

    except KeyboardInterrupt:
        save_checkpoint_on_exit(0,0)

