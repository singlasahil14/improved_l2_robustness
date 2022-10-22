import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from conv_layers import *
from custom_activations import *
from last_layers import *

class LipBlock(nn.Module):
    def __init__(self, conv_module, act_module, pool_module, in_channels, out_channels, 
                 kernel_size, stride, downsample=True, **kwargs):
        super(LipBlock, self).__init__()
        self.ortho = conv_module(in_channels, out_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=kernel_size//2, **kwargs)
        if downsample:
            self.act = pool_module(out_channels)
        else:
            self.act = act_module(out_channels)

    def forward(self, x):
        x = self.ortho(x)
        x = self.act(x)
        return x
    
class Normalize(nn.Module):
    def __init__(self, conv_module, act_module, pool_module, in_channels, out_channels, 
                 kernel_size, stride, downsample=True, **kwargs):
        super(LipBlock, self).__init__()
        self.ortho = conv_module(in_channels, out_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=kernel_size//2, **kwargs)
        if downsample:
            self.act = pool_module(out_channels)
        else:
            self.act = act_module(out_channels)

    def forward(self, x):
        x = self.ortho(x)
        x = self.act(x)
        return x
            
class LipConvNet(nn.Module):
    def __init__(self, conv_name='soc', activation_name='maxmin', pooling_name='max', last_name='ortho', init_channels=32, 
                 block_size=1, num_hidden=None, num_classes=10, input_size=32, fast_train=True, attack_radius=0., 
                 cifar_mean = (0.4914, 0.4822, 0.4465), cifar_std = (0.2507, 0.2507, 0.2507)):
        super(LipConvNet, self).__init__()
        self.register_buffer('shift', torch.tensor(cifar_mean).view(3, 1, 1))
        self.register_buffer('scale', torch.tensor(cifar_std).view(3, 1, 1))
                
        self.lip_const = torch.reciprocal(torch.min(self.scale)).item()
        
        if last_name == 'crc_full':
            last_downsample = False
        else:
            last_downsample = True
        
        if conv_name == 'soc':
            self.kwargs = {'fast_train': fast_train}
        else:
            self.kwargs = {}
        
        self.conv_module = conv_mapping[conv_name]
        self.act_module = GNP_activation_mapping[activation_name]
        self.pool_module = pool_mapping[pooling_name]
        
        self.in_channels = 3
        
        self.init_layer = self._init_layer(out_channels=init_channels)
        self.block1 = self._make_block(block_size, stride=2, kernel_size=3)
        self.block2 = self._make_block(block_size, stride=2, kernel_size=3)
        self.block3 = self._make_block(block_size, stride=2, kernel_size=3)
        self.block4 = self._make_block(block_size, stride=2, kernel_size=3)
        self.block5 = self._make_block(block_size, stride=2, kernel_size=1, 
                                       is_last=True, downsample=last_downsample)
        
        if last_name == 'crc_ortho':
            self.features_layer = self.conv_module(self.in_channels, self.in_channels, kernel_size=1, 
                                                   stride=1, **self.kwargs)
        else:
            self.features_layer = nn.Identity()
            
        if num_hidden == None:
            num_hidden = self.in_channels
            
        if last_name == 'crc_full':
            self.logits_layer = last_mapping[last_name](self.in_channels, num_classes, num_hidden=num_hidden, 
                                                        attack_radius=attack_radius * self.lip_const)
        else:
            if last_name == 'ortho':
                last_name = conv_name + '_' + last_name
            self.logits_layer = last_mapping[last_name](self.in_channels, num_classes)
            
        self._input_sizes_dict(input_size)
        
    def _init_layer(self, out_channels):
        layer = LipBlock(self.conv_module, self.act_module, self.pool_module, self.in_channels,
                         out_channels, kernel_size=3, stride=1, downsample=False, **self.kwargs)
        self.in_channels = out_channels
        return layer
    
    def _helper(self, stride_list, ksize_list, downsample=True):
        downsample_list = [False] * (len(stride_list) - 1) + [downsample]
        layers = []
        for stride, kernel_size, downsample in zip(stride_list, ksize_list, downsample_list):
            layers.append(LipBlock(self.conv_module, self.act_module, self.pool_module, self.in_channels, 
                          self.in_channels * stride * stride, kernel_size, stride, downsample, **self.kwargs))            
            self.in_channels = self.in_channels * stride * stride
            if downsample:
                self.in_channels = self.in_channels // 2
                
        return nn.Sequential(*layers)
        
    def _make_block(self, num_blocks, stride=2, kernel_size=3, is_last=False, downsample=True):
        if is_last:
            strides = [stride] + [1]*(num_blocks-1)
            kernel_sizes = [kernel_size]*(num_blocks)
        else:
            strides = [1]*(num_blocks-1) + [stride]
            kernel_sizes = [3]*(num_blocks-1) + [kernel_size]

        block = self._helper(strides, kernel_sizes, downsample)
        return block
    
    def extract_features(self, x):
        x = (x - self.shift) / self.scale
        x = self.init_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.features_layer(x)        
        return x

    def forward(self, x, y=None, only_features=False, return_cert=False):
        features = self.extract_features(x)
        if only_features:
            return features

        if return_cert:
            logits, certs = self.logits_layer.certificates(features, y)
            return logits, certs/(self.lip_const)
        else:
            logits = self.logits_layer(features)
            return logits
        
    def _input_sizes_dict(self, input_size):
        self.input_sizes = {'init_layer': input_size, 'features_layer': 1, 'logits_layer': 1}
        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential):
                block_index = int(name[-1]) - 1
                block_input_size = input_size // (2**block_index)
                self.input_sizes[name] = block_input_size

    def lipschitz_constant(self, exact=False):
        L = torch.reciprocal(torch.min(self.scale)).item()
        for name, module in self.named_modules():
            if name:
                name = name.split('.')[0]
                input_size = self.input_sizes[name]
                
                if exact and hasattr(module, 'norm'):
                    sigma = module.norm(input_size)
                elif hasattr(module, 'norm_bound'):
                    sigma = module.norm_bound(input_size)
                else:
                    sigma = 1.
            else:
                sigma = 1.
            L = L * sigma
        self.lip_const = L
        return L
    
    def update_sigma(self):
        for name, module in self.named_modules():
            if name and hasattr(module, 'update_sigma'):
                module.update_sigma()
