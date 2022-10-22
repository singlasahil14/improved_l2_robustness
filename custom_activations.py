import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

class MaxMin(nn.Module):
    def __init__(self, channels=None):
        super(MaxMin, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=axis)
    
class HouseHolder(nn.Module):
    def __init__(self, channels):
        super(HouseHolder, self).__init__()
        assert (channels % 2) == 0
        eff_channels = channels // 2
        
        self.theta = nn.Parameter(
                0.5 * np.pi * torch.ones(1, eff_channels, 1, 1).cuda(), requires_grad=True)

    def forward(self, z, axis=1):
        theta = self.theta
        x, y = z.split(z.shape[axis] // 2, axis)
                    
        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))
        
        a_2 = x*torch.cos(theta) + y*torch.sin(theta)
        b_2 = x*torch.sin(theta) - y*torch.cos(theta)
        
        a = (x * (selector <= 0) + a_2 * (selector > 0))
        b = (y * (selector <= 0) + b_2 * (selector > 0))
        return torch.cat([a, b], dim=axis)
    
class HouseHolder_Order_2(nn.Module):
    def __init__(self, channels):
        super(HouseHolder_Order_2, self).__init__()
        assert (channels % 2) == 0
        self.num_groups = channels // 2
        
        self.theta0 = nn.Parameter(
                (np.pi * torch.rand(self.num_groups)).cuda(), 
                requires_grad=True)
        self.theta1 = nn.Parameter(
                (np.pi * torch.rand(self.num_groups)).cuda(), 
                requires_grad=True)
        self.theta2 = nn.Parameter(
                (np.pi * torch.rand(self.num_groups)).cuda(), 
                requires_grad=True)

    def forward(self, z, axis=1):
        theta0 = torch.clamp(self.theta0.view(1, -1, 1, 1), 0., 2 * np.pi)

        x, y = z.split(z.shape[axis] // 2, axis)
        z_theta = (torch.atan2(y, x) - (0.5 * theta0)) % (2 * np.pi)
        
        theta1 = torch.clamp(self.theta1.view(1, -1, 1, 1), 0., 2 * np.pi)
        theta2 = torch.clamp(self.theta2.view(1, -1, 1, 1), 0., 2 * np.pi)
        theta3 = 2 * np.pi - theta1
        theta4 = 2 * np.pi - theta2
        
        ang1 = 0.5 * (theta1)
        ang2 = 0.5 * (theta1 + theta2)
        ang3 = 0.5 * (theta1 + theta2 + theta3)
        ang4 = 0.5 * (theta1 + theta2 + theta3 + theta4)
        
        select1 = torch.logical_and(z_theta >= 0, z_theta < ang1)
        select2 = torch.logical_and(z_theta >= ang1, z_theta < ang2)
        select3 = torch.logical_and(z_theta >= ang2, z_theta < ang3)
        select4 = (z_theta >= ang3)
        
        a1 = x
        b1 = y

        a2 = x*torch.cos(theta0 + theta1) + y*torch.sin(theta0 + theta1)
        b2 = x*torch.sin(theta0 + theta1) - y*torch.cos(theta0 + theta1)
        
        a3 = x*torch.cos(theta2) + y*torch.sin(theta2)
        b3 = -x*torch.sin(theta2) + y*torch.cos(theta2)
        
        a4 = x*torch.cos(theta0) + y*torch.sin(theta0)
        b4 = x*torch.sin(theta0) - y*torch.cos(theta0)

        a = (a1 * select1) + (a2 * select2) + (a3 * select3) + (a4 * select4)
        b = (b1 * select1) + (b2 * select2) + (b3 * select3) + (b4 * select4)
        
        z = torch.cat([a, b], dim=axis)
        return z
        
class LipPool(nn.Module):
    def __init__(self, channels, stride=2, theta=None):
        super(LipPool, self).__init__()
        assert stride == 2, stride
        assert (channels % stride) == 0
        self.num_groups = channels // stride
        
        if theta is None:
            self.theta = nn.Parameter(np.pi * torch.rand(self.num_groups).cuda(), requires_grad=True)
        else:
            self.theta = nn.Parameter(theta * torch.ones(self.num_groups).cuda(), requires_grad=True)

    def forward(self, z, axis=1, verbose=False):
        x, y = z.split(z.shape[axis] // 2, axis)        
        z_theta = torch.atan2(y, x) % (2 * np.pi)
        
        theta_shape = ([1] * axis) + [self.num_groups] + ([1] * (z.ndim - axis - 1))        
        theta = torch.clamp(self.theta, 0., np.pi).view(theta_shape)

        out1 = (-x*torch.sin(0.5 * theta)) + (y*torch.cos(0.5 * theta))
        out2 = torch.stack([x, y], dim=1).norm(dim=1)
        out3 = (-x*torch.sin(0.5 * theta)) - (y*torch.cos(0.5 * theta))
        
        select1 = (z_theta < 0.5 * (theta + np.pi))
        select2 = torch.logical_and(z_theta >= 0.5 * (theta + np.pi), z_theta < 0.5 * (3 * np.pi - theta))
        select3 = torch.logical_and(~select1, ~select2)
        
        dists = (select1 * out1) + (select2 * out2) + (select3 * out3)
        return dists
    
class MaxPool(nn.Module):
    def __init__(self, channels=None):
        super(MaxPool, self).__init__()

    def forward(self, z, axis=1):
        x, y = z.split(z.shape[axis] // 2, axis)
        return torch.max(x, y)
    
def activation_mapping(activation_name, channels=None):
    if activation_name in GNP_activations:
        activation_module = GNP_activation_mapping[activation_name]
        activation_func = activation_module(channels)
    else:
        activation_func = activation_dict[activation_name]
    return activation_func
    
activation_dict = {
    'relu': F.relu,
    'swish': F.silu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'softplus': F.softplus
}

GNP_activation_mapping = {
    'maxmin': MaxMin,
    'hh1': HouseHolder,
    'hh2': HouseHolder_Order_2
}

pool_mapping = {
    'max': MaxPool,
    'lip1': LipPool
}
