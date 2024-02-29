""" Models definitions archictures """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import DWT, IWT, DWTone
import common
import segmentation_models_pytorch as smp
import os.path as osp
import numpy as np
from transformers import SegformerForSemanticSegmentation
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torchinfo import summary

#eden
from .LANet import LANet


try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):

    def __init__(self, n_class=6):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)



class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        #self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_1 = nn.Conv2d(self.n_features*8 if self.pool_type == 'gap' else self.n_features*self.multiplier*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self,x):
        x = self.conv_in(x)
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += residual
        return out


class SegNet_resblock(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_resblock, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        number_of_kernels = 128
        #self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        self.res_block_1 = ResidualBlock(self.n_features, number_of_kernels)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        self.res_block_2 = ResidualBlock(self.n_features*2, number_of_kernels)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        self.res_block_3 = ResidualBlock(self.n_features*4, number_of_kernels)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)
        self.res_block_4 = ResidualBlock(self.n_features*8, number_of_kernels)

        self.conv5_1 = nn.Conv2d(self.n_features*8 if self.pool_type == 'gap' else self.n_features*self.multiplier*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)
        self.res_block_5 = ResidualBlock(self.n_features*8, number_of_kernels)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*8 + self.n_features*2, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*2, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features*2, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(conv5)

        # Decoder block 5
        conv5 = self.res_block_5(conv5)
        x = self.unpool(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        conv4 = self.res_block_4(conv4)
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        conv3 = self.res_block_3(conv3)
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        conv2 = self.res_block_2(conv2)
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        conv1 = self.res_block_1(conv1)
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)
        
        
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        #x1 = torch.zeros_like(x1)
        #x2 = torch.zeros(6, 128, 128, 128)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)
        
        
        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x) 
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)
        
        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools_test(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools_test, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWTone()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*3, self.n_features*2, 1)
        #self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*6, self.n_features*4, 1)
        #self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*12, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*24, self.n_features*8, 1)
        #self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*24, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*24, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*6, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask1_2 = self.pool2(x)

        # Encoder block 2
        x1_h = self.conv2_0_1(x1_h)
        #x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask2_2 = self.pool2(x)
        
        
        # Encoder block 3
        x1_h = self.conv3_0_1(x1_h)
        #x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask3_2 = self.pool2(x)
        
        # Encoder block 4
        x1_h = self.conv4_0_1(x1_h)
        #x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0] 
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1_h = self.conv5_0_1(x1_h)
        #x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask5_2 = self.pool2(x)
        
        # Decoder block 5
        x1_h = self.conv5_0_1_D(x1_h)
        #x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x), dim=1)
        return x
    
    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass



class SegNet_two_pools_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*4 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)
        
        
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_2 = self.pool2(conv2)
        
        
        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(conv4) 
        x2, mask4_2 = self.pool2(conv4)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(conv5)
        x2, mask5_2 = self.pool2(conv5)
        
        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass




class SegNet_two_pools_2(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*1 + self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*2 + self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*4 + self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_1 = nn.Conv2d(self.n_features*8 + self.n_features*32, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_2 = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_3 = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*16)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*16 + self.n_features*4, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_2_D = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*16)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*16, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        
        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x2, mask4_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Decoder block 5
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        
        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class UNet(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "up", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(UNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)

        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)

        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)


        conv3_2_D_in_sz = 8 if self.pool_type == 'up' else 2 
        #self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(self.n_features*int(conv3_2_D_in_sz) + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != 'up' else 4
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == 'up' else 2 
        #self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(self.n_features*int(conv2_2_D_in_sz) + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != 'up' else 2
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))

        #self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == 'up' else 1
        self.conv1_2_D = nn.Conv2d(self.n_features*int(conv1_2_D_in_sz) + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()
    


    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        
        # Decoder block 3
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        
        # Decoder block 2
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        #x = self.conv1_1_D_F(x)
        x = F.log_softmax(self.conv1_1_D_F(x))
        return x
    



    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class UNet_two_pools(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "up", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(UNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool1 = DWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool1 = IWT()
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.pool = DWT() if self.pool_type == "dwt" else 
        #self.unpool = IWT() if self.pool_type == "dwt" else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)

        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)

        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)


        conv3_2_D_in_sz = 8 if self.pool_type == 'up' else 2 
        #self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(self.n_features*int(conv3_2_D_in_sz) + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != 'up' else 4
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == 'up' else 2 
        #self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(self.n_features*int(conv2_2_D_in_sz) + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != 'up' else 2
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))

        #self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == 'up' else 1
        self.conv1_2_D = nn.Conv2d(self.n_features*int(conv1_2_D_in_sz) + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()
    


    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)
           
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_1 = self.pool2(conv2)


        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        
        # Decoder block 3
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        
        # Decoder block 2
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        #x = self.conv1_1_D_F(x)
        x = F.log_softmax(self.conv1_1_D_F(x))
        return x
    



    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass



''' 
class SegNet_two_pools_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*1 + self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*2 + self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*4 + self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        
        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*8 + self.n_features*2, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)
        x = torch.cat([x1, x2], dim=1)
        
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_2 = self.pool2(conv2)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(conv4)
        x2, mask4_2 = self.pool2(conv4)
        x = torch.cat([x1, x2], dim=1)
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass
           
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    """
    Recurrent Block
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out

class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)


    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "up", n_features = 64, kernel_size = 3, t=2):
        super(R2U_Net, self).__init__()

        filters = [n_features, n_features * 2, n_features * 4, n_features * 8, n_features * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(in_channels, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        #self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        #self.Up5 = up_conv(filters[4], filters[3])
        #self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        #e5 = self.Maxpool3(e4)
        #e5 = self.RRCNN5(e5)

        #d5 = self.Up5(e5)
        #d5 = torch.cat((e4, d5), dim=1)
        #d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        out = F.log_softmax(out)
      # out = self.active(out)

        return out


    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass
'''
'''
class MWCNN(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3, conv=common.default_conv):
        super(MWCNN, self).__init__()
        n_feats = n_features
        nColor = in_channels
        self.scale_idx = 0

        act = nn.ReLU(True)

        self.DWT = DWT()
        self.IWT = IWT()

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, out_channels, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1, mask1 = self.DWT(x0)
        x1 = self.d_l1(x1)
        x2, mask2 = self.DWT(x1)
        x2 = self.d_l2(x2)
        x_, mask_ = self.DWT(x2) 
        x_ = self.IWT(self.pro_l3(x_), mask_) + x2
        x_ = self.IWT(self.i_l2(x_), mask2) + x1
        x_ = self.IWT(self.i_l1(x_), mask1) + x0
        x = self.tail(self.i_l0(x_))

        x = F.log_softmax(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
'''


def build_model(model_name: str, params: list):
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['n_classes'],        # model output channels (number of classes in your dataset)
            activation=None,
        )
    elif model_name == 'unet_bn':
        model = smp.Unet(
            encoder_name='efficientnet-b0',  # Nome do encoder, como 'resnet34', 'resnet50', etc.
            encoder_weights='imagenet',  # Carregar os pesos pré-treinados do encoder
            in_channels=3,  # Número de canais de entrada da imagem (por exemplo, 3 para imagens coloridas RGB)
            classes=params['n_classes'],  # Número de classes para segmentação (por exemplo, 1 para segmentação binária)
            activation=None,  # Função de ativação da camada de saída (por exemplo, 'sigmoid' para segmentação binária ou 'softmax' para segmentação multiclasse)
            decoder_use_batchnorm=True, # Usar BatchNorm após cada camada do decoder
            encoder_depth=5,  # Profundidade do encoder (número de camadas)
            decoder_channels=[256, 128, 64, 32, 16],  # Número de canais nas camadas do decoder
            decoder_attention_type=None,  # Tipo de atenção (opcional, pode ser None)
        )
    elif model_name == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['n_classes'],        # model output channels (number of classes in your dataset)
            activation=None,
        )
    elif model_name == 'efficientnet':
        model = EfficientNet.from_pretrained(model_name='efficientnet-b0', in_channels=3, num_classes=params['n_classes'])
    elif model_name == 'segformer':
        df = pd.read_csv('classes.csv')
        classes = df['name']
        palette = df[[' r', ' g', ' b']].values
        id2label = classes.to_dict()
        label2id = {v: k for k, v in id2label.items()}
        num_labels=len(label2id)
        params['id2label'] = id2label
        params['palette'] = palette

        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True,
                    num_labels=len(id2label), id2label=id2label, label2id=label2id,
                    reshape_last_stage=True)
    elif model_name == 'segnet_modificada':
        # model = SegNet(in_channels = 3, out_channels = params['n_classes'])
        model = SegNet_two_pools_test(in_channels = 3, out_channels = params['n_classes'], pretrained = True, pool_type = 'dwt')
    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['n_classes'],        # model output channels (number of classes in your dataset)
            activation=None,
        )
    elif model_name == 'lanet':
        model = LANet(in_channels = 3, num_classes = params['n_classes'], pretrained = False)
    else:
        raise Exception("{} -> invalid model name.".format(model_name))


    model.to(params['device'])
    summary(model, (8,3,256,256))
    return model
