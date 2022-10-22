import argparse
import copy
import logging
import os
import time
import math
from shutil import copyfile
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet
from utils import *

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training specifications
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--test-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clip-norm', default=1., type=float)
    parser.add_argument('--gamma', default=0., type=float, help='gamma for certificate regularization')
    parser.add_argument('--beta', default=0., type=float, help='beta for curvature regularization')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    
    # Model architecture specifications
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--fast-train', action='store_true', help='make backward pass of SOC faster during training')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--num-hidden', default=4096, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], help='Activation function')
    parser.add_argument('--pooling', default='max', choices=['max', 'lip1'], help='Pooling layer')
    parser.add_argument('--num-layers', default=5, type=int, choices=[5, 10, 15, 20, 25, 30, 35, 40], 
                        help='number of layers per block in the LipConvnet network')
    parser.add_argument('--last-layer', default='lln', choices=['ortho', 'linear', 'lln', 'crc_full'], 
                        help='last layer that maps features to logits')
    
    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset used for training')
    
    # Other specifications
    parser.add_argument("--attack-radius", default=0., type=float, help="l2 attack radius")
    parser.add_argument("--radii-list", nargs=6, help="list of radii to evaluate certified robust accuracy", 
                        type=float, default=[36/255, 72/255, 108/255, 0.5, 1.0, 1.5])
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--log-epochs', default=10, type=int, help='Number of epochs after which to compute certificate')
    parser.add_argument('--num-other', default=9, type=int, help='Number of classes to be used for computing the certificate')
    return parser.parse_args()

def init_model(args):
    block_size = args.num_layers // 5
    model = LipConvNet(args.conv_layer, args.activation, args.pooling, args.last_layer, init_channels=args.init_channels, 
                       block_size=block_size, num_classes=args.num_classes, num_hidden=args.num_hidden,
                       fast_train=args.fast_train, attack_radius=args.attack_radius)
    return model

def evaluate_final_model(args, model_path, evaluation_func, test_loader, epoch_name):
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(model_path))
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    losses_arr, correct_arr, certs_arr, curvs_arr = evaluation_func(test_loader, model_test, exact=True, return_curv=True)
    test_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_cert_acc_list = robust_statistics(
        losses_arr, correct_arr, certs_arr, radii_list=args.radii_list)
    
    test_curv = curvs_arr.mean().item()
    print('{:s},{:.1f},{:d},{:d},{:d},{:d},{:d},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f},{:.3f}'.format(
        epoch_name, test_time, -1, -1, -1, -1, -1, test_loss, test_acc, test_cert_acc_list[0], test_cert_acc_list[1], 
        test_cert_acc_list[2], test_cert_acc_list[3], test_cert_acc_list[4], test_cert_acc_list[5], test_cert, test_curv)
         )

def main():
    args = get_args()
    
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')
    if args.fast_train and not(args.conv_layer == 'soc'):
        raise ValueError('fast training is only compatible with SOC')
    
    li = []
    for num_layers in [5, 10, 15, 20, 25, 30, 35, 40]:
        out_dir = args.out_dir
        out_dir += '_' + str(args.dataset) 

        out_dir += '_' + str(num_layers)
        if args.fast_train:
            out_dir += '_fast' + str(args.conv_layer)
        else:
            out_dir += '_' + str(args.conv_layer)
        out_dir += '_' + str(args.activation)
        out_dir += '_' + str(args.pooling)
        out_dir += '_' + str(args.gamma)
        out_dir += '_' + str(args.beta)
        out_dir += '_' + str(args.attack_radius)
        out_dir += '_' + str(args.last_layer)

        output_file = os.path.join(out_dir, 'output.log')
        df = pd.read_csv(output_file, skiprows=1)
        last_row = df.iloc[-1]
        
        curr_tuple = last_row['Test Acc'], last_row['Test Cert Acc (0.141)'], last_row['(0.282)'], last_row['(0.424)']
        print('LipConvnet-' + str(num_layers), curr_tuple)
        
        li.append(curr_tuple)
        
if __name__ == "__main__":
    main()
