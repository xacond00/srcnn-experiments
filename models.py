# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

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
        super(SRCNN, self).__init__()

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
        else:
            self.aux_upscaler = None
    def forward(self, lr_imgs):
        output = self.conv_layers(lr_imgs)  # (N, 3, w, h)
        output = self.upsc_layer(output)
        if(self.aux_upscaler):
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
        super(VGG_Loss, self).__init__()
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