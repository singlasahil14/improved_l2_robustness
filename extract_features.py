import argparse
import copy
import logging
import os
import time
import math
from shutil import copyfile
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet
from utils import *

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model specifications
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--gamma', default=0., type=float, help='gamma for certificate regularization')
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--fast-train', action='store_true', help='make backward pass of SOC faster during training')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], help='Activation function')
    parser.add_argument('--pooling', default='max', choices=['max', 'lip1'], help='Pooling layer')
    parser.add_argument('--num-layers', default=5, type=int, choices=[5, 10, 15, 20, 25, 30, 35, 40], 
                        help='number of layers per block in the LipConvnet network')
    parser.add_argument('--last-layer', default='ortho', choices=['ortho', 'lln', 'crc_ortho', 'crc_full'], 
                        help='last layer that maps features to logits')
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    parser.add_argument('--out-dir', default='extract', type=str, help='Model directory')
    return parser.parse_args()

def init_model(args):
    block_size = args.num_layers // 5
    model = LipConvNet(args.conv_layer, args.activation, args.pooling, args.last_layer, init_channels=args.init_channels, 
                       block_size=block_size, num_classes=args.num_classes, fast_train=args.fast_train)
    return model

def save_arrays(features, logits, labels, model_dir, prefix):
    np.save(os.path.join(model_dir, prefix + '_features.npy'), features)
    np.save(os.path.join(model_dir, prefix + '_logits.npy'), logits)
    np.save(os.path.join(model_dir, prefix + '_labels.npy'), labels)

def save_metadata(model, model_dir, prefix, train_loader, test_loader):
    model_path = os.path.join(model_dir, prefix + '.pth')
    
    # Evaluation at best model (early stopping)
    model.load_state_dict(torch.load(model_path))
    model.float()
    model.eval()

    train_features, train_logits, train_labels = extract_features(train_loader, model)
    test_features, test_logits, test_labels = extract_features(test_loader, model)
    
    train_preds = np.argmax(train_logits, axis=1)
    train_acc = (train_preds == train_labels).sum()/len(train_labels)

    test_preds = np.argmax(test_logits, axis=1)
    test_acc = (test_preds == test_labels).sum()/len(test_labels)
    
    print(train_acc, test_acc)

    model_dir = os.path.join(model_dir, 'extracted')
    os.makedirs(model_dir, exist_ok=True)
    save_arrays(train_features, train_logits, train_labels, model_dir, prefix + '_train')
    save_arrays(test_features, test_logits, test_labels, model_dir, prefix + '_test')
    

def main():
    args = get_args()
    
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')
    if args.fast_train and not(args.conv_layer == 'soc'):
        raise ValueError('fast training is only compatible with SOC')

    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_' + str(args.num_layers) 
    if args.fast_train:
        args.out_dir += '_fast' + str(args.conv_layer)
    else:
        args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.activation)
    args.out_dir += '_' + str(args.pooling)
    args.out_dir += '_cr' + str(args.gamma)
    args.out_dir += '_' + str(args.last_layer)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset, shuffle=False)
    if args.dataset == 'cifar10':
        args.num_classes = 10    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise Exception('Unknown dataset')
    
    # Evaluation at best model (early stopping)
    model = init_model(args).cuda()
    save_metadata(model, args.out_dir, 'best', train_loader, test_loader)
    save_metadata(model, args.out_dir, 'last', train_loader, test_loader)
    
    
if __name__ == "__main__":
    main()


