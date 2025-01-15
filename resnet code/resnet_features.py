import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import kornia
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50_inat': 'pretrained_models/BBN.iNaturalist2017.res50.90epoch.best_model.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_dir = './pretrained_models'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class filter_denoising_block(torch.nn.Module):
    ''' Simple filters as denoising block'''
    def __init__(self, in_planes, ksize, filter_type): 
        super().__init__()
        self.in_planes = in_planes
        self.ksize = ksize
        self.filter_type = filter_type 
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        if self.filter_type == 'Median':
            x_denoised = kornia.filters.median_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Mean':
            x_denoised = kornia.filters.box_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Gaussian':
            x_denoised = kornia.filters.gaussian_blur2d(x, (self.ksize, self.ksize), (0.3 * ((x.shape[3] - 1) * 0.5 - 1) + 0.8, 0.3 * ((x.shape[2] - 1) * 0.5 - 1) + 0.8))
        new_x = x + self.conv(x_denoised)
        return new_x


class non_local_denoising_block(torch.nn.Module): 
    def __init__(self, in_channels, inter_channels = None, mode = 'embedded', bn_layer = True):
        
        """ in_channels: original channel size (1024 in the paper) #depends on the test dataset
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
            """
        
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.mode = mode 
        self.inter_channels = inter_channels
        
         # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        
        #dimension=2 for images
        conv_nd = nn.Conv2d 
        max_pool_layer = nn.MaxPool2d(kernel_size=(2,2))
        bn = nn.BatchNorm2d
        
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            
            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)
            
            
        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        if self.mode == "nonlocal_gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "nonlocal_gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # only conv with possibly not 1 stride
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, 3, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, 1, 0]

        return block_kernel_sizes, block_strides, block_paddings
    

def get_filter_block(denoise, in_channels, ksize):
    # add a denoising layer
    if denoise is None or denoise == 'None':
        return None
    if denoise == "Mean" or denoise == "Median" or denoise == "Gaussian":
        return filter_denoising_block(in_planes = in_channels, ksize = ksize, filter_type=denoise)
    
    elif denoise == "concatenate" or denoise == "dot" or denoise == "embedded" or denoise == "nonlocal_gaussian" :
       return non_local_denoising_block(in_channels = in_channels,  mode=denoise) 
    else:
        raise ValueError('Unknown filter type!')


class ResNet_features(nn.Module):
    '''
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    '''

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, filter=None, filter_layer=None):
        super(ResNet_features, self).__init__()
        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AvgPool2d(kernel_size=7)

        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers


        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.dn_layer1 = None
        if filter_layer in [1,9]:
            self.dn_layer1 = get_filter_block(filter, 64, 3)


        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.dn_layer2 = None
        if filter_layer in [2,9]:
            self.dn_layer2 = get_filter_block(filter, 128, 3)


        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.dn_layer3 = None
        if filter_layer in [3,9]:
            self.dn_layer3 = get_filter_block(filter, 256, 3)


        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)
        self.dn_layer4 = None
        if filter_layer in [4,9]:
            self.dn_layer4 = get_filter_block(filter, 512, 3)

        self.fc = nn.Linear(512, 10)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.dn_layer1 is not None:
            x = self.dn_layer1(x)
        x = self.layer2(x)
        if self.dn_layer2 is not None:
            x = self.dn_layer2(x)
        x = self.layer3(x)
        if self.dn_layer3 is not None:
            x = self.dn_layer3(x)
        x = self.layer4(x)
        if self.dn_layer4 is not None:
            x = self.dn_layer4(x)
        x = self.global_pool(x)
        x = x.reshape(-1, 512)

        x = self.fc(x)
        return x
    
    def forward_all(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        all_feas = []
        for i in range(4):
            x = getattr(self, 'layer{}'.format(i + 1))(x)
            all_feas.append(x)

        return x, all_feas

    def forward_shallow(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(1):
            x = getattr(self, 'layer{}'.format(i + 1))(x)
        
        return x

    def forward_shallow_to_deep(self, shallow_feas):
        x = shallow_feas
        for i in range(1, 4):
            x = getattr(self, 'layer{}'.format(i + 1))(x)
        
        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network, not counting the number
        of bypass layers
        '''

        return (self.block.num_layers * self.layers[0]
              + self.block.num_layers * self.layers[1]
              + self.block.num_layers * self.layers[2]
              + self.block.num_layers * self.layers[3]
              + 1)


    # def __repr__(self):
    #     template = 'resnet{}_features'
    #     return template.format(self.num_layers() + 1)

def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

def resnet50_inat_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a pre-trained model
        inat (bool): If True, returns the version pre-trained on iNaturalist
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 4], **kwargs)
    
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet50_inat'], model_dir=model_dir)
        
        my_dict.pop('module.classifier.weight')
        my_dict.pop('module.classifier.bias')
        for key in list(my_dict.keys()):
            my_dict[key.replace('module.backbone.', '').replace('cb_block', 'layer4.2').replace('rb_block','layer4.3' )] = my_dict.pop(key)
        model.load_state_dict(my_dict, strict=False)


    return model

def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_features(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet101'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_features(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet152'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model