import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision
import pytorch_ssim
from torch import nn
from torchinfo import summary
from layers import ShufConvLayer, SqrtLoss, ConvLayer
from models import SRCNN, VGG_Loss, freeze_model, unfreeze_model
from dataset import ImageDataset
from train import train, compare_images



# Data parameters
scaling_factor = 8  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
kernel_size = 9
pre_scale = 1
n_channels = 3  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = True  # Load checkpoint
test = True # Enable test mode (show output images)
base_model = None #f"c{kernel_size}x{scaling_factor}.pth" #"c5x64x2_rc3x5c3_s3.pth"
model_name = f"c{kernel_size}x{scaling_factor}.pth"

vgg_i = 3 # VGG_Loss maxpool index
vgg_j = 3 # VGG_Loss conv index (in a block)
vgg_alpha = 0.5 # Lerp mae with vgg loss
loss_fns = ['mae', 'vgg', 'mse', 'sqrt', 'ssim']
loss_tp = 0

ds_train = True # Set dataset to training mode (random crop position)
batch_size = 8 # batch size
crop_size = 1024
lr = 1e-4  # learning rate

start_epoch = 0  # start at this epoch
iterations = 2000  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 1000  # print training status once every __ batches
test_crop = 1024 # Crop of test mode images
valid_size = 16
valid_crop = 1024
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
        model = ShufConvLayer(n_channels, n_channels, kernel_size, scaling_factor, 1, 'clip')
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)
    else:
        print("Loaded model: ", init_model)
        checkpoint = torch.load(model_name, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
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
        for i in range(5):
            compare_images(train_dataset, model, device, i + 105, scaling_factor)
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
        criterion = pytorch_ssim.SSIM(window_size = 11)
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
