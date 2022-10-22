import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
from apex import amp

def print_stats(tensor, prefix=''):
    print(prefix, tensor.shape, tensor.detach().min().item(), tensor.detach().mean().item(), tensor.detach().max().item())

def l2_normalize(tensor):
    tensor = F.normalize(tensor.flatten(start_dim=0), dim=0).view_as(tensor)
    return tensor

def l2_project(tensor, radius):
    tensor_flat = tensor.flatten(start_dim=1)
    tensor_norm = torch.norm(tensor_flat, dim=1, keepdim=True)
    tensor_flat = tensor_flat/tensor_norm
    tensor_flat *= torch.clamp(tensor_norm, max=radius)
    return tensor_flat.view_as(tensor)

def linf_project(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_loaders(dir_, batch_size, test_batch_size=None, dataset_name='cifar10', shuffle=True):
    if test_batch_size is None:
        test_batch_size = batch_size
        
    if dataset_name == 'cifar10':
        dataset_func = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_func = datasets.CIFAR100
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
        
    num_workers = 4
    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

def attack_pgd_l2(model, X, y, radius, step_size=None, random_steps=2, attack_iters=50, restarts=10, opt=None):
    if step_size is None:
        step_size = 2 * (radius / attack_iters)
        
    max_loss = torch.zeros_like(y)
    max_delta = torch.zeros_like(X)
    
    lower_bounds = 0. - X
    upper_bounds = 1. - X
    for _ in range(restarts):
        delta = torch.randn_like(X)
        delta = l2_project(delta, random_steps * step_size)
        
#         delta_flat = delta.flatten(start_dim=1)
#         delta_norm = delta_flat.norm(dim=1)
#         print('delta_init: ', radius, delta.shape, delta_flat.shape, delta_norm.shape)
#         print(delta_norm.detach().min().item(), delta_norm.detach().mean().item(), delta_norm.detach().max().item())


        
#         delta = linf_project(delta, lower_bounds, upper_bounds)
        delta.requires_grad = True
        for _ in range(attack_iters):
            logits = model(X + delta)
            preds = torch.argmax(logits, dim=1)
            
            indices_y = torch.nonzero(preds == y)
            if len(indices_y) == 0:
                break
                
            loss = F.cross_entropy(logits, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            grad = delta.grad.detach()
            delta_y = delta[indices_y]
            grad_y = grad[indices_y]
            grad_y = F.normalize(grad_y.flatten(start_dim=1), dim=1).view_as(grad_y)
            
            delta_y = delta_y + step_size * grad_y
            delta_y = l2_project(delta_y, radius)
#             delta_y = linf_project(delta_y, lower_bounds, upper_bounds)
    
#             grad_flat = grad_y.flatten(start_dim=1)
#             grad_norm = grad_flat.norm(dim=1)
#             print('grad: ', step_size, grad_y.shape, grad_flat.shape, grad_norm.shape, grad_norm.detach().min().item(), 
#                   grad_norm.detach().mean().item(), grad_norm.detach().max().item())
    
#             delta_flat = delta_y.flatten(start_dim=1)
#             delta_norm = delta_flat.norm(dim=1)
#             print('delta: ', radius, delta_y.shape, delta_flat.shape, delta_norm.shape)
#             print(delta_norm.detach().min().item(), delta_norm.detach().mean().item(), delta_norm.detach().max().item())
            
            delta.data[indices_y, :, :, :] = delta_y
            delta.grad.zero_()
            
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        
#         quit()
    return max_delta

def evaluate_pgd_l2(test_loader, model, radius, step_size=None, attack_iters=50, 
                    restarts=10, opt=None, limit_n=float("inf")):
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        delta = attack_pgd_l2(model, X, y, radius, step_size, attack_iters, restarts, opt=opt)
        with torch.no_grad():
#             test_perturbation(model, X, delta)
            
            logits = model(X + delta)
            preds = torch.argmax(logits, dim=1)
            
            loss = F.cross_entropy(logits, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (preds == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss/n, pgd_acc/n

def test_perturbation(model, x, d):
#     model = model.float()

    x = x.half()
    d = F.normalize(d.flatten(start_dim=1), dim=1).view_as(x)

    x_n = x + d
    x_n = x_n.half()
    f, f_n = model.extract_feat(x, x_n)

    x_d = torch.norm((x_n - x).flatten(start_dim=1), dim=1)
    f_d = torch.norm((f_n - f).flatten(start_dim=1), dim=1)
    
    print(x_d)
    print(f_d)
    
    print((x_d - f_d).min().detach().item())
    
    print(torch.all(f_d <= x_d))




    quit()

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def extract_inputs(data_loader):
    images_list = []
    labels_list = []

    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            images_list.append(X.flatten(start_dim=1).cpu().numpy())
            labels_list.append(y.cpu().numpy())
            
        images_array = np.concatenate(images_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)
    return images_array, labels_array

def other_classes(y, num_classes):
    if y.ndim > 1:
        y = y.squeeze(dim=1)
    
    batch_size = y.shape[0]
    batch_idxs = torch.arange(batch_size, device=y.device)

    onehot = y.new_empty((batch_size, num_classes)).fill_(0.)
    onehot[batch_idxs, y] = 1.
    
    class_idxs = torch.arange(num_classes, device=y.device)
    class_idxs = class_idxs.expand(batch_size, -1)
    other = class_idxs[onehot == 0]
    other = other.view(batch_size, num_classes - 1)
    return other

def topk_classes(y, logits, k):
    batch_size = y.shape[0]
    batch_idxs = torch.arange(batch_size)

    logits_other = logits.clone()
    logits_other[batch_idxs, y] -= float('inf')
    _, other = torch.topk(logits_other, k=k, dim=1)
    return other

def compute_logits_cert(y, certs):
    batch_size = y.shape[0]
    num_classes = certs.shape[1] + 1
    other = other_classes(y, num_classes)
    
    batch_idxs = torch.arange(batch_size).unsqueeze(1)
    logits_cert = certs.new_empty((batch_size, num_classes)).fill_(0.)
    logits_cert[batch_idxs, other] = -certs
    return logits_cert

def evaluate_certificates(test_loader, model, exact=False, return_curv=False):
    losses_list = []
    certs_list = []
    correct_list = []
    curvs_list = []
    model.eval()

    model.lipschitz_constant(exact=exact)
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            
            logits, certs = model(X, y, return_cert=True)
            
            if return_curv:
                if hasattr(model.logits_layer, 'curvature_bounds'):
                    m, M = model.logits_layer.curvature_bounds(y, logits, certs.shape[1], tight=False)
                    curvs = torch.max(m, M)
                else:
                    curvs = torch.zeros_like(certs)
                    
                curvs_list.append(curvs)
            
            losses = F.cross_entropy(logits, y, reduction='none')
            losses_list.append(losses)

            preds = torch.argmax(logits, dim=1)
            correct = (preds==y)
            
            certs, _ = torch.min(certs, dim=1)            
            certs = F.relu(certs * correct)
            
            certs_list.append(certs)
            correct_list.append(correct)
            
        losses_tensor = torch.cat(losses_list, dim=0)
        correct_tensor = torch.cat(correct_list, dim=0)
        certs_tensor = torch.cat(certs_list, dim=0)
        curvs_tensor = torch.cat(curvs_list, dim=0)
    if return_curv:
        return losses_tensor, correct_tensor, certs_tensor, curvs_tensor
    else:
        return losses_tensor, correct_tensor, certs_tensor

def evaluate_certificates_precise(test_loader, model, exact=False, return_curv=False):
    losses_list = []
    certs_list = []
    correct_list = []
    curvs_list = []
    
    model.eval()

    lip_const = model.lipschitz_constant(exact=exact)
    model.logits_layer.float()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            
            features = model(X, only_features=True)
            logits, certs = model.logits_layer.certificates(features, y)
            certs = certs/lip_const
            
            if return_curv:
                if hasattr(model.logits_layer, 'curvature_bounds'):
                    m, M = model.logits_layer.curvature_bounds(y, logits, certs.shape[1], tight=True)
                    curvs = torch.max(m, M)
                else:
                    curvs = torch.zeros_like(certs)
                curvs_list.append(curvs)
            
            losses = F.cross_entropy(logits, y, reduction='none')
            losses_list.append(losses)

            preds = torch.argmax(logits, dim=1)
            correct = (preds==y)
            
            certs, _ = torch.min(certs, dim=1)            
            certs = F.relu(certs * correct)
            
            certs_list.append(certs)
            correct_list.append(correct)
            
        losses_tensor = torch.cat(losses_list, dim=0)
        correct_tensor = torch.cat(correct_list, dim=0)
        certs_tensor = torch.cat(certs_list, dim=0)
        
    model.logits_layer.half()
    if return_curv:
        curvs_tensor = torch.cat(curvs_list, dim=0)
        return losses_tensor, correct_tensor, certs_tensor, curvs_tensor
    else:
        return losses_tensor, correct_tensor, certs_tensor

def robust_statistics(losses_tensor, correct_tensor, certs_tensor, radii_list):
    mean_loss = losses_tensor.mean()
    mean_acc = correct_tensor.sum()/len(correct_tensor)
    mean_certs = (certs_tensor * correct_tensor).sum()/correct_tensor.sum()
    
    robust_acc_list = []
    for radius in radii_list:
        robust_correct_tensor = torch.logical_and(certs_tensor > radius, correct_tensor)
        robust_acc = robust_correct_tensor.sum()/robust_correct_tensor.shape[0]
        robust_acc_list.append(robust_acc)
    return mean_loss, mean_acc, mean_certs, robust_acc_list

def parameter_lists(model):
    weight_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and not('logits_layer.linear' in name):
                weight_params.append(param)
            else:
                other_params.append(param)
    return weight_params, other_params