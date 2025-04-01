import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision
import ssim

from torch import nn
from torchinfo import summary
from layers import SqrtLoss, ConvLayer
from models import SRCNN, VGG_Loss, freeze_model, unfreeze_model, SRResNet
from dataset import ImageDataset
from train import train, compare_images



# Data parameters
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
n_channels = 3  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks

# Learning parameters
checkpoint = True  # Load checkpoint
unfreeze = False # Unfreeze all parameters
test = False # Enable test mode (show output images)
resnet = False
res_blocks = 16
base_model = None #"ssresnet_mae.pth" #"4x64ssim_c5x2_c3x5.pth" #"4x64vge_c5x2_c3x5.pth" #"4x_c5x96x2_c3x96x5.pth"   #"8x_c5x256x2_c3x256x5.pth" #"base/c5x64x2_c3x64x5.pth" #"c5x64x2_rc3x5c3_s3.pth"
model_name = "auxresnet_ssae_nobn.pth"#"4x64ssim_c5x2_c3x5.pth" #"c5x64x2_c3x64x5_ssim.pth"
aux_name = "base/c5x4.pth"
ps_ks = 3 # Pre-Pixel shuffle conv kernel size
last_ks = 0 # Add post shuffle conv layer
nch = 64
freeze = False # Freeze the backbone when appending shuffle conv layer

vgg_i = 3 # VGG_Loss maxpool index
vgg_j = 3 # VGG_Loss conv index (in a block)
vgg_alpha = 0.0 # Lerp mae with vgg loss
ssim_alpha = 0.5  # Mix mae with vgg
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 4

ds_train = True # Set dataset to training mode (random crop position)
batch_size = 8 # batch size
crop_size = 256
pre_scale = 1   
lr = 3e-4  # learning rate

start_epoch = 0  # start at this epoch
iterations = 2000  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 8
valid_crop = 512
grad_clip = None  # clip if gradients are exploding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    init_model = base_model if base_model and not test and checkpoint else model_name 
    if not checkpoint or not os.path.exists(init_model):
        if res_blocks > 0: 
            model = SRResNet(9, 3, nch, res_blocks, scaling_factor, aux_name, 'lin', False)
        else:
            if not resnet:
                layers = [(nch,5), (nch,5), (nch,3), (nch,3), (nch,3), (nch,3), (nch,3)]#ESPCNN
            else:
                layers = [(nch,5), (nch,5), ('res',3), ('res',3), ('res',3), ('res',3), ('res',3), (nch, 3)] # Resnet
            last_layer = (last_ks, 'clip') if last_ks else None
            model = SRCNN(layers, n_channels, ps_ks, scaling_factor, aux_name, "lrelu", last=last_layer)

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        print("Loaded model: ", init_model)
        checkpoint = torch.load(init_model, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

        if last_ks > 0 and not hasattr(model, 'last_layer'):
            if freeze:
                freeze_model(model)
            model.last_layer = ConvLayer(3,3,last_ks,1,1,'clip')
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
        elif unfreeze:
            unfreeze_model(model)
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
        else:
            #optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
            optimizer = checkpoint['optimizer']

    if not test:
        summary(model, input_size=(batch_size, n_channels, crop_size // scaling_factor, crop_size // scaling_factor))
    # Move to default device
    model = model.to(device, memory_format=torch.channels_last)
    if test:
        train_dataset = ImageDataset("DIV2K", False, scaling_factor, pre_scale, test_crop)
    else:
        train_dataset = ImageDataset("DIV2K", ds_train, scaling_factor, pre_scale, crop_size)
    if(test):
        for i in range(50):
            compare_images(train_dataset, model, device, i + 20, scaling_factor)
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
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              grad_clip=grad_clip,
              print_freq=print_freq,
              device=device,
              valid_ds=valid_ds
              )

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   model_name)
        #if(epoch % 20 == 0):
        #    compare_images(train_dataset, model, device, epoch, scaling_factor)
        

if __name__ == '__main__':
    main()
