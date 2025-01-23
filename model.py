import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import kornia
import torch.nn.functional as F
import math

class VGG(nn.Module):
    def __init__(self, features):
        """
        Initialize the VGG model.

        Args:
            features: A sequential module representing the convolutional layers.
        """
        super(VGG,self).__init__()
        self.features = features
        
        # Weight initialization for convolutional layers
        # Iterate through all modules in the model
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2. / n))
                # Bias initialization to zero
                layer.bias.data.zero_()

        # Fully connected layers for final classification
        self.classifier = nn.Sequential(
            nn.Dropout(), # Dropout for regularization
            nn.Linear(512,512), # Fully connected layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,47)
        )

    # Function to create layers based on configuration
    def forward(self, inputs):
        """
        Forward pass through the VGG model.

        Args:
            x : Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        inputs = self.features(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        output = self.classifier(inputs)
        return output
    

def make_layers(configuration):
    """
    Function to create layers for VGG-16 based on configuration
    :param configuration: configuration
    """
    layers = []
    in_channels =3

    # Iterate through the configuration to build layers
    for layer_config  in configuration:
        # 'M' represents a Max Pooling layer
        if layer_config  == 'Max':
            layers += [nn.MaxPool2d(kernel_size = 2, stride =2)]
        else:
            conv2d = nn.Conv2d(in_channels, layer_config , kernel_size = 3, padding =1) # Convolutional layer
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace = True))
            # Update the input channels for the next layer
            in_channels = layer_config 
        # Return the sequential model of layers
    return nn.Sequential(*layers)

# Two Conv layers (number of filters , number of filters) + Max Pool
configuration = [
64, 64, 'Max',
128, 128, 'Max',
256, 256, 256, 'Max',
512, 512, 512, 'Max',
512, 512, 512, 'Max']

def vgg16():
    """
    Function to create a VGG-16 model
    """
    return VGG(make_layers(configuration))