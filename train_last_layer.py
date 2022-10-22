import argparse
import copy
import logging
import os
import time
import math
from shutil import copyfile
import json

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet
from utils import *
from last_layers import CRC_Full

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model specifications
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    
    parser.add_argument('--gamma', default=0., type=float, help='gamma for certificate regularization')
    parser.add_argument('--beta', default=0., type=float, help='beta for curvature regularization')
    parser.add_argument('--alpha', default=0., type=float, help='alpha for curvature regularization')
    
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--fast-train', action='store_true', help='make backward pass of SOC faster during training')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], help='Activation function')
    parser.add_argument('--pooling', default='lip1', choices=['max', 'lip1'], help='Pooling layer')
    parser.add_argument('--num-layers', default=5, type=int, choices=[5, 10, 15, 20, 25, 30, 35, 40], 
                        help='number of layers per block in the LipConvnet network')
    parser.add_argument('--last-layer', default='crc_full', choices=['ortho', 'lln', 'crc_full'], 
                        help='last layer that maps features to logits')
    parser.add_argument('--num-hidden', default=4096, type=int, help='Number of hidden')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    parser.add_argument('--out-dir', default='extract', type=str, help='Model directory')
    
    parser.add_argument('--update-freq', default=10, type=int, help='frequency of updating sigma')
    parser.add_argument('--model-name', default='last', type=str, help='Model name')
    parser.add_argument("--radii-list", nargs=6, help="list of radii (rho) to evaluate certified robust accuracy", 
                        type=float, default=[36/255, 72/255, 108/255, 0.5, 1.0, 1.5])
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def load_metadata(model_dir, prefix):
    train_features = np.load(os.path.join(model_dir, 'extracted', prefix + '_train_features.npy'))
    train_labels = np.load(os.path.join(model_dir, 'extracted', prefix + '_train_labels.npy'))

    test_features = np.load(os.path.join(model_dir, 'extracted', prefix + '_test_features.npy'))
    test_labels = np.load(os.path.join(model_dir, 'extracted', prefix + '_test_labels.npy'))
    
    return train_features, train_labels, test_features, test_labels

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
    
    logfile = os.path.join(args.out_dir, 'output_last.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output_last.log'))
    logger.info(args)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset, shuffle=False)
    if args.dataset == 'cifar10':
        args.num_classes = 10    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise Exception('Unknown dataset')
    
    # Evaluation at best model (early stopping)
    train_features, train_labels, test_features, test_labels = load_metadata(args.out_dir, 'last')
    
    # convert numpy arrays to pytorch tensors
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)

    # create dataset and dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    
    
    # convert numpy arrays to pytorch tensors
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels)

    # create dataset and dataloaders
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise Exception('Unknown dataset')

    num_features = train_features.shape[1]
    model = CRC_Full(num_features, args.num_classes, args.num_hidden).cuda()
    
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1)
    
    best_model_path = os.path.join(args.out_dir, 'best_logits_layer.pth')
    last_model_path = os.path.join(args.out_dir, 'last_logits_layer.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')
    
    # Training
    prev_test_acc = 0.
    start_train_time = time.time()
    
    header = 'Epoch,Seconds,LR,Train Loss,Train Acc,Train Cert,Train Curv,Test Loss,Test Acc,'
    header += 'Test Cert Acc ({:.3g}),({:.3g}),'.format(args.radii_list[0], args.radii_list[1])
    header += '({:.3g}),({:.3g}),'.format(args.radii_list[2], args.radii_list[3])
    header += '({:.3g}),({:.3g}),Test Cert'.format(args.radii_list[4], args.radii_list[5])
    logger.info(header)

    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_curv = 0
        train_cert = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            logits, certs, curvs = model.certificates(X, y, return_curv=True)            
            logits_cert = compute_logits_cert(y, certs)
            
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y)
            
            certs, _ = torch.min(certs, dim=1)
            certs = F.relu(certs * correct)
            
            ce_loss = criterion(logits, y)
            ce_loss_cert = criterion(logits_cert, y)
            loss = ce_loss + (args.beta * ce_loss_cert) + (args.alpha * curvs.mean())

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += ce_loss.item() * y.size(0)
            train_cert += certs.sum().item()
            train_curv += curvs.sum().item()
            train_acc += correct.sum().item()
            train_n += y.size(0)
            scheduler.step()
            
        model.update_sigma()
            
        losses_arr, correct_arr, certificates_arr = evaluate_certificates_last(test_loader, model)        
        test_loss, test_acc, test_cert, test_cert_acc_list = robust_statistics(
            losses_arr, correct_arr, certificates_arr, radii_list=args.radii_list)
        
        if (test_acc >= prev_test_acc):
            torch.save(model.state_dict(), best_model_path)
            prev_test_acc = test_acc
            best_epoch = epoch
        
        lr = scheduler.get_last_lr()[0]
        train_loss = train_loss/train_n
        train_cert = train_cert/train_n
        train_curv = train_curv/train_n
        train_acc = train_acc/train_n
        
        epoch_time = time.time() - start_epoch_time


        logger.info('%d,%.1f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f',
                    epoch, epoch_time, lr, train_loss, train_acc, train_cert, train_curv, test_loss, test_acc, 
                    test_cert_acc_list[0], test_cert_acc_list[1], test_cert_acc_list[2], 
                    test_cert_acc_list[3], test_cert_acc_list[4], test_cert_acc_list[5], test_cert)

        torch.save(model.state_dict(), last_model_path)
        
        trainer_state_dict = {'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)
        
    train_time = time.time()
    
    # Evaluation at best model (early stopping)
    model_test = CRC_Full(num_features, args.num_classes, args.num_hidden).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.eval()
        
    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates_last(test_loader, model_test)
    test_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_cert_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr, radii_list=args.radii_list)
    
    logger.info('%d,%.1f,%d,%d,%d,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f',
                best_epoch, test_time, -1, -1, -1, -1, -1, test_loss, test_acc, 
                test_cert_acc_list[0], test_cert_acc_list[1], test_cert_acc_list[2], 
                test_cert_acc_list[3], test_cert_acc_list[4], test_cert_acc_list[5], test_cert)
    

    # Evaluation at last model
    model_test = CRC_Full(num_features, args.num_classes, args.num_hidden).cuda()
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.eval()

    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates_last(test_loader, model_test)
    total_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_cert_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr, radii_list=args.radii_list)
    
    logger.info('%d,%.1f,%d,%d,%d,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f',
                epoch, test_time, -1, -1, -1, -1, -1, test_loss, test_acc, 
                test_cert_acc_list[0], test_cert_acc_list[1], test_cert_acc_list[2], 
                test_cert_acc_list[3], test_cert_acc_list[4], test_cert_acc_list[5], test_cert)

        
    
if __name__ == "__main__":
    main()


