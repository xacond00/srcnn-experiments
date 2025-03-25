import os
import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from layers import ShufConvLayer
from models import SRCNN
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
model_name = f"c{kernel_size}x{scaling_factor}.pth"
#"5c64x2_3c64x2_x4.pth"
batch_size = 64  # batch size
start_epoch = 0  # start at this epoch
iterations = 1000  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
crop_size = 1024
lr = 1e-2  # learning rate
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if not checkpoint or not os.path.exists(model_name):
        model = ShufConvLayer(n_channels, n_channels, kernel_size, scaling_factor, 'clip')
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(model_name, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    print(model)
    torch.backends.cudnn.benchmark = True
    # Move to default device
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.L1Loss(reduction="mean")
    for g in optimizer.param_groups:
        g['lr'] = lr
    # Custom dataloaders
    train_dataset = ImageDataset("DIV2K", False, scaling_factor, pre_scale, crop_size)
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
              device=device
              )

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   model_name)
        if(epoch % 20 == 0):
            compare_images(train_dataset, model, device, epoch, scaling_factor)
        

if __name__ == '__main__':
    main()
