# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
import math
from layers import *

"""
def freeze_model(model):
    try:
        # Attempt to freeze all parameters of the current model
        for param in model.parameters():
            param.requires_grad = False
    except Exception as e:
        print(f"Warning: Skipping parameter freezing due to error: {e}")
    
    # Recursively freeze all sub-modules (for nested models)
    try:
        for child in model.children():
            try:
                freeze_model(child)
            except Exception as e:
                print(f"Warning: Skipping child module freezing due to error: {e}") 
    except Exception as e:
        print(f"Warning: Skipping child module freezing due to error: {e}")
"""
def freeze_model(model):
    for layer in model.modules():  # Iterate through all layers
        if any(p.requires_grad for p in layer.parameters(recurse=False)):  
            # Only update if the layer has trainable parameters
            for param in layer.parameters(recurse=False):
                param.requires_grad = False
def unfreeze_model(model):
    for layer in model.modules():  # Iterate through all layers
        if any(p.requires_grad is False for p in layer.parameters(recurse=False)):  
            # Only update if the layer has parameters
            for param in layer.parameters(recurse=False):
                param.requires_grad = True

class SRCNN(nn.Module):

    def __init__(self, layers : list, n_channels = 3, out_ks = 3, scaling_factor=4, aux_upscaler = None, activation = "lrelu", last = None):
        super().__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        conv_layers = []
        prev_ch = n_channels
        for i in range(len(layers)):
            use_batch = len(layers[i]) > 2 and layers[i][2] == True
            if(layers[i][0] == 'res'): # Residual block
                layer = ResLayer(layers[i][1], prev_ch, use_batch)
            else:
                layer = ConvLayer(prev_ch, layers[i][0], layers[i][1], 1, 1, activation, use_batch)
                prev_ch = layers[i][0]
            conv_layers.append(layer)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.upsc_layer = ShufConvLayer(prev_ch, n_channels, out_ks, scaling_factor, 1, "linear")
        #self.last_layer = None
        if(last):
            self.last_layer = ConvLayer(n_channels, n_channels, last[0], 1, 1, last[1])
        if(aux_upscaler):
            au = aux_upscaler
            if(au in {'nearest', 'bilinear', 'bicubic'}):
                self.aux_upscaler = nn.Upsample(scale_factor=scaling_factor, mode=au, align_corners=(au != "nearest"))
            else:
                self.aux_upscaler = torch.load(au, weights_only=False)['model']
                freeze_model(self.aux_upscaler)

    def forward(self, lr_imgs):
        output = self.conv_layers(lr_imgs)  # (N, 3, w, h)
        output = self.upsc_layer(output)
        if hasattr(self, 'aux_upscaler') and self.aux_upscaler is not None:
            output = output + self.aux_upscaler(lr_imgs)
        if hasattr(self, 'last_layer'):
            output = self.last_layer(output)
        return output


class VGG_Loss(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    param: alpha - alpha * mae_loss + (1 - alpha) * vgg_loss
    """

    def __init__(self, loss_fn = "mse", i = 1, j = 2, alpha = 0.0):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super().__init__()
        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.L1Loss(reduction='mean')
        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(weights= 'VGG19_Weights.DEFAULT')
        self.alpha = alpha
        self.mae = nn.L1Loss(reduction='mean')
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, x, y):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        xf = self.truncated_vgg19(x)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        yf = self.truncated_vgg19(y)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        if(self.alpha > 0.0):
            return self.alpha * self.mae(x,y) + (1 - self.alpha) * self.loss_fn(xf, yf)
        else:
            return self.loss_fn(xf, yf)



class SRCNN_Orig(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2,2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


class SRResNet(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4, aux_upscaler = None, last_activ = 'tanh', batch_norm = True):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super().__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        self.conv_block1 = ConvLayer(3, n_channels, large_kernel_size, 1, 1, 'prelu', False)
        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[ResLayer(small_kernel_size, n_channels, batch_norm, 'prelu') for i in range(n_blocks)])
        
        # Another convolutional block
        self.conv_block2 = ConvLayer(n_channels, n_channels, small_kernel_size, 1, 1, None, batch_norm)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[ShufConvLayer2(n_channels, n_channels, small_kernel_size, 2, 1, None) for i
              in range(n_subpixel_convolution_blocks)])

        # The last convolutional block
        self.conv_block3 = ConvLayer(n_channels, 3, large_kernel_size, 1, 1, last_activ, False)
        if(aux_upscaler):
            au = aux_upscaler
            if(au in {'nearest', 'bilinear', 'bicubic'}):
                self.aux_upscaler = nn.Upsample(scale_factor=scaling_factor, mode=au, align_corners=(au != "nearest"))
            else:
                self.aux_upscaler = torch.load(au, weights_only=False)['model']
                freeze_model(self.aux_upscaler)

    def forward(self, lr_imgs):
        """
        Forward prop.

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)
        if hasattr(self, 'aux_upscaler') and self.aux_upscaler is not None:
            sr_imgs = sr_imgs + self.aux_upscaler(lr_imgs)
        return sr_imgs

class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvLayer(in_channels, out_channels, kernel_size, 1 if i % 2 == 0 else 2, 1, 'lrelu', i != 0)
                )
            in_channels = out_channels
         
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.

        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit