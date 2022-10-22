import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import time
import numpy as np
from utils import l2_normalize, print_stats, other_classes, topk_classes

from conv_layers import *

class Softplus(nn.Module):
    def __init__(self, beta=1., threshold=20.):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.hessian_bound = 0.25 * beta
            
    def forward(self, x):
        x = self.beta * x

        pos_x = (x >= 0)
        abs_x = torch.abs(x)
        small_x = (abs_x < self.threshold)
        eval_neg_x = small_x * F.softplus(-abs_x)

        eval_x = pos_x * (x + eval_neg_x) + (~ pos_x) * eval_neg_x
        return eval_x/self.beta 
    
    def gradient(self, x):
        x = self.beta * x
        return torch.sigmoid(x)
    
    def hessian(self, x):
        x = self.beta * x
        sigm_x = torch.sigmoid(x)
        return self.beta * sigm_x * (1. - sigm_x)
    
    def local_hessian_bound(self, min_x, max_x):
        sigm_min_x = torch.sigmoid(self.beta * min_x)
        min_x_curv = self.beta * sigm_min_x * (1. - sigm_min_x)
        
        sigm_max_x = torch.sigmoid(self.beta * max_x)
        max_x_curv = self.beta * sigm_max_x * (1. - sigm_max_x)
        
        sel = torch.logical_and((min_x < 0), (0 < max_x))
        max_curv = (sel * 0.25 * self.beta) + (~sel * torch.maximum(min_x_curv, max_x_curv))
        min_curv = torch.minimum(min_x_curv, max_x_curv)
        return min_curv, max_curv
    

class OrthoLinear(nn.Module):
    def __init__(self):
        super(OrthoLinear, self).__init__()
        
    def forward(self, features):
        logits = self.ortho(features)
        logits = torch.flatten(logits, start_dim=1)
        return logits
    
    def certificates(self, features, y=None):
        logits = self.forward(features)
        if y is None:
            y = torch.argmax(logits, dim=1)

        batch_size = logits.shape[0]
        batch_idxs = torch.arange(batch_size)
        
        onehot = torch.zeros_like(logits)
        onehot[batch_idxs, y] = 1.

        num_classes = logits.shape[1]
        class_idxs = torch.arange(num_classes).expand(batch_size, -1)
        other = class_idxs[onehot == 0]
        other = other.view(batch_size, num_classes - 1)
        
        logits_y = logits[batch_idxs, y].unsqueeze(1)
        logits_other = logits[batch_idxs.unsqueeze(1), other]

        logits_diff = logits_y - logits_other
        certs = logits_diff/np.sqrt(2)
        return logits, certs
    
class Cayley_Cert(OrthoLinear):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__()
        self.ortho = CayleyLinear(num_features, num_classes, **kwargs)
        
class SOC_Cert(OrthoLinear):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__()
        self.ortho = SOC(num_features, num_classes, kernel_size=1, 
                         stride=1, padding=0, **kwargs)

class BCOP_Cert(OrthoLinear):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__()
        self.ortho = BCOP(num_features, num_classes, kernel_size=1, 
                          stride=1, padding=0, **kwargs)

class LLN_Linear(nn.Linear):
    def compute_weight(self):
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        return self.weight/weight_norm
        
    def forward(self, features):
        features = torch.flatten(features, start_dim=1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self._weight_n = self.compute_weight()
        logits = F.linear(features, self._weight_n, self.bias)
        return logits
    
    def certificates(self, features, y=None, num_other=9):
        logits = self.forward(features)
        if y is None:
            y = torch.argmax(logits, dim=1)

        batch_size = logits.shape[0]
        batch_idxs = torch.arange(batch_size)
        other = topk_classes(y, logits, num_other)
        
        logits_y = logits[batch_idxs, y]
                
        logits_other = logits[batch_idxs.unsqueeze(1), other]        
        logits_diff = logits_y.unsqueeze(1) - logits_other
        
        weight = self._weight_n
        weight_pdists = 2 * (1 - weight.mm(weight.T))
        
        norm_sq_diff = weight_pdists[y.unsqueeze(1), other]
        norm_diff = torch.sqrt(norm_sq_diff)

        certs = logits_diff/norm_diff
        return logits, certs
        
class CRC_Full(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=None, act_beta=1., act_thresh=20., attack_radius=0., attack_iters=5, 
                 step_multiplier=1.5, eps=1e-6, grad_tol=1e-6, min_curv=1e-4, outer_iters=5, inner_iters=5, taylor_iters=10, 
                 tolerance_multiplier=0.1, init_iters=50, update_iters=1, fast_iters=2, eval_iters=20, num_other=9):
        super(CRC_Full, self).__init__()
        if num_hidden is None:
            num_hidden = num_features
            
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.num_other = num_other
            
        self.linear1 = nn.Linear(num_features, num_hidden)
        self.activation = Softplus(beta = act_beta, threshold = act_thresh)        
        self.linear2 = nn.Linear(num_hidden, num_classes)
        
        self.attack_radius = attack_radius
        self.attack_iters = attack_iters
        self.step_multiplier = step_multiplier
        
        self.eps = eps
        self.min_curv = min_curv
        self.grad_tol = grad_tol
        self.tolerance_multiplier = tolerance_multiplier
        
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters        
        self.taylor_iters = taylor_iters
        
        self.init_iters = init_iters
        self.update_iters = update_iters
        self.fast_iters = fast_iters
        self.eval_iters = eval_iters
        
        self._initialize_singular_vectors()
        self.update_curvatures()
        
    def _initialize_singular_vectors(self):
        num_hidden = self.num_hidden
        num_features = self.num_features
        w1 = self.linear1.weight
        
        u = w1.new_empty((num_features)).normal_(0, 1)
        self.register_buffer('_u', F.normalize(u, dim=0))
        
        v = w1.new_empty((num_hidden)).normal_(0, 1)
        self.register_buffer('_v', F.normalize(v, dim=0))
                
    @torch.autograd.no_grad()
    def _power_method(self, num_iters):
        w1 = self.linear1.weight
        for i in range(num_iters):
            self._v = F.normalize(torch.mv(w1, self._u), dim=0)
            self._u = F.normalize(torch.mv(w1.T, self._v), dim=0)
        
    def _train_update_sigma(self):
        if self.training:
            self._power_method(self.update_iters)            
                                
    def curvature_bounds(self, y, logits, num_other, tight=False):
        other = topk_classes(y, logits, num_other)
        
        if tight:
            m, M = self._tight_curvature_bounds(y.unsqueeze(1), other, self.eval_iters)
        else:
            m, M = self._tight_curvature_bounds(y.unsqueeze(1), other, self.fast_iters, self._u)
        return m, M
    
    def _fast_curvature_bounds(self, y, other):
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        hess = self.activation.hessian_bound

        sigma = torch.dot(self._v, torch.mv(w1, self._u))

        w2_y = w2[y]
        w2_other = w2[other]
        w2_diffs = w2_y - w2_other
        
        diag_neg, _ = torch.min(w2_diffs * (w2_diffs < 0), dim=2)
        diag_pos, _ = torch.max(w2_diffs * (w2_diffs > 0), dim=2)
        
        m = hess * sigma * sigma * torch.abs(diag_neg)
        M = hess * sigma * sigma * torch.abs(diag_pos)
        return m, M
    
    @torch.autograd.no_grad()
    def _curvature_power_method(self, w1, w2_diffs_neg, w2_diffs_pos, num_iters, u=None):
        total_size = w2_diffs_neg.shape[0]
        num_features = self.num_features
        if u is None:
            u_neg = w1.new_empty((total_size, num_features)).normal_(0, 1)
            u_pos = w1.new_empty((total_size, num_features)).normal_(0, 1)

            u_neg = F.normalize(u_neg, dim=1)
            u_pos = F.normalize(u_pos, dim=1)
        else:
            u_neg = u.unsqueeze(0).repeat(total_size, 1)
            u_pos = u.unsqueeze(0).repeat(total_size, 1)

        for i in range(num_iters):
            u_neg = u_neg.mm(w1.T)
            u_neg = u_neg * w2_diffs_neg
            u_neg = u_neg.mm(w1)
            u_neg = F.normalize(u_neg, dim=1)
            
            u_pos = u_pos.mm(w1.T)
            u_pos = u_pos * w2_diffs_pos
            u_pos = u_pos.mm(w1)
            u_pos = F.normalize(u_pos, dim=1)

        return u_neg, u_pos

    def _tight_curvature_bounds(self, y, other, num_iters, u=None):
        batch_size, num_other = other.shape
        num_features = self.num_features
        
        w1 = self.linear1.weight        
        w2 = self.linear2.weight
        hess = self.activation.hessian_bound
        
        w2_y = w2[y]
        w2_other = w2[other]
        w2_diffs = w2_y - w2_other
        
        w2_diffs_neg = (w2_diffs < 0) * w2_diffs
        w2_diffs_neg = w2_diffs_neg.view(batch_size * num_other, -1)
        w2_diffs_pos = (w2_diffs > 0) * w2_diffs
        w2_diffs_pos = w2_diffs_pos.view(batch_size * num_other, -1)

        u_neg, u_pos = self._curvature_power_method(w1, w2_diffs_neg, w2_diffs_pos, num_iters, u)
            
        u = u_neg.mm(w1.T)
        m = hess * torch.abs((u * u * w2_diffs_neg).sum(dim=1))
        m = m.view_as(other)
        
        u = u_pos.mm(w1.T)
        M = hess * torch.abs((u * u * w2_diffs_pos).sum(dim=1))
        M = M.view_as(other)
        return m, M
                    
    def update_curvatures(self):
        self._power_method(num_iters=self.init_iters)
        
    def _intermediate(self, x):
        x = torch.flatten(x, start_dim=1)
        x_lin1 = self.linear1(x)
        x_act = self.activation(x_lin1)
        return x_lin1, x_act
        
    def forward(self, features):
        self._train_update_sigma()
        
        _, x_act = self._intermediate(features)
        logits = self.linear2(x_act)
        return logits    
    
    def _gradient(self, x_lin1, w_last):
        w1 = self.linear1.weight
        
        grad = w_last
        act_grad = self.activation.gradient(x_lin1)
        act_grad = act_grad.view_as(w_last)
        
        grad = (grad * act_grad)
        grad = grad.mm(w1)
        return grad
    
    def _inv_hess1p(self, W, diag, vec):
        d = vec
        for i in range(self.taylor_iters):
            W_vec = d.mm(W.T)
            W_vec = W_vec * diag
            W_vec = W_vec.mm(W)
            
            d = vec - W_vec
        return d
    
    @torch.autograd.no_grad()
    def _crc_certificates(self, x, eta_min, eta_max, w2_diff, b2_diff, logits_diff):
        w1 = self.linear1.weight
        grad_tol = self.tolerance_multiplier * self.grad_tol
        
        delta = torch.zeros_like(x)                
        for i in range(self.outer_iters):
            eta = 0.5 * (eta_min + eta_max)
            eta_mul = eta[:, None]
            
            for j in range(self.inner_iters + 1):
                x_n = (x + delta)
                x_n_lin1, x_n_act = self._intermediate(x_n)

                grad = self._gradient(x_n_lin1, w2_diff)
                grad_dual = delta + (eta_mul * grad)
                
                grad_dual_norm = torch.norm(grad_dual, dim=1)
                
                if (j == self.inner_iters) or torch.all(grad_dual_norm < grad_tol):
                    break

                hess = self.activation.hessian(x_n_lin1)
                delta = delta - self._inv_hess1p(w1, eta_mul * w2_diff * hess, grad_dual)
    
            logits_diff_n = torch.sum(x_n_act * w2_diff, dim=1) + b2_diff
            
            ge_indicator = (logits_diff_n > 0)
            eta_min[ge_indicator] = eta[ge_indicator]
            eta_max[~ge_indicator] = eta[~ge_indicator]
            
        return eta, delta, grad_dual_norm
    
    def _tight_certificates(self, features, y, num_other):
        num_features = self.num_features
        num_hidden = self.num_hidden

        w2 = self.linear2.weight
        b2 = self.linear2.bias
        x_lin1, x_act = self._intermediate(features)
        logits = self.linear2(x_act)
        
        other = topk_classes(y.squeeze(1), logits, num_other)
        
        batch_size = y.shape[0]
        batch_idxs = torch.arange(batch_size).unsqueeze(1)

        logits_y = logits[batch_idxs, y]
        logits_other = logits[batch_idxs, other]
        logits_diff = logits_y - logits_other
        
        logits_sign = torch.sign(logits_diff)
        
        w2_y = w2[y]
        w2_other = w2[other]
        w2_diff = (w2_y - w2_other)
        
        b2 = self.linear2.bias
        b2_y = b2[y]
        b2_other = b2[other]
        b2_diff = (b2_y - b2_other)
        
        x = torch.flatten(features, start_dim=1)
        x = x.unsqueeze(1)
        x = x.repeat(1, num_other, 1)
        x = x.view(-1, num_features)
        
        m, M = self._tight_curvature_bounds(y, other, self.eval_iters)
        curv_bounds = torch.max(m, M)
        
        m = torch.clamp(curv_bounds.flatten(), min=self.min_curv)
        M = torch.clamp(curv_bounds.flatten(), min=self.min_curv)
        
        logits_diff = logits_diff.flatten()
        
        eta_min = (logits_diff < 0) * (-torch.reciprocal(M))
        eta_max = (logits_diff > 0) * torch.reciprocal(m)
        
        w2_diff = w2_diff.view(-1, num_hidden)
        b2_diff = b2_diff.view(-1)
                        
        w2_diff_norm = torch.norm(w2_diff, dim=1)

        eta, delta, grad_dual_norm = self._crc_certificates(x, eta_min, eta_max, w2_diff, b2_diff, logits_diff)
        
        x_n = (x + delta)
        x_n_lin1, x_n_act = self._intermediate(x_n)
        logits_diff_n = torch.sum(x_n_act * w2_diff, dim=1) + b2_diff
        
        certs = (delta * delta).sum(dim=1) + (2 * eta * logits_diff_n)
        certs = certs * (grad_dual_norm < self.grad_tol)
        
        is_certs_large = (certs > self.eps)
        certs = is_certs_large * torch.sqrt(torch.clamp(certs, min=self.eps))
        certs = certs.view(-1, num_other)
        certs = certs * logits_sign
        return certs
        
    def _fast_certificates(self, features, y, num_other):
        batch_size = features.shape[0]
        batch_idxs = torch.arange(batch_size).unsqueeze(1)
        num_features = self.num_features
        num_hidden = self.num_hidden
        
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        b2 = self.linear2.bias
        
        x_lin1, x_act = self._intermediate(features)
        logits = self.linear2(x_act)
        
        other = topk_classes(y.squeeze(1), logits, num_other)
        m, M = self._fast_curvature_bounds(y, other)        
                
        logits_y = logits[batch_idxs, y]
        logits_other = logits[batch_idxs, other]
        
        logits_diff = logits_y - logits_other
        logits_sign = torch.sign(logits_diff)
        curv_bound = (m * (logits_sign > 0)) + (M * (logits_sign < 0))
        curv_bound = torch.abs(curv_bound)
                
        grad = (w2[y] - w2[other])
        act_grad = self.activation.gradient(x_lin1)
        grad = grad * act_grad[:, None, :]
        grad = torch.matmul(grad, w1)
        grad_norm = torch.norm(grad, dim=2)
                
        logits_diff = torch.abs(logits_diff)
        
        curv_i, curv_j = torch.nonzero(curv_bound, as_tuple=True)
        nocurv_i, nocurv_j = torch.where(curv_bound == 0)

        logits_diff_nocurv = logits_diff[nocurv_i, nocurv_j]
        grad_norm_nocurv = grad_norm[nocurv_i, nocurv_j]
        certs_nocurv = (logits_diff_nocurv/grad_norm_nocurv)        
        logits_diff_curv = logits_diff[curv_i, curv_j]
        m_curv = curv_bound[curv_i, curv_j]
        
        grad_norm_curv = grad_norm[curv_i, curv_j]
        certs_curv = (2 * logits_diff_curv * m_curv) + (grad_norm_curv * grad_norm_curv)
        
        is_certs_large = (certs_curv > self.eps)
        certs_curv = torch.clamp(certs_curv, min=self.eps)

        certs_curv = torch.sqrt(certs_curv)
        certs_curv = is_certs_large * (certs_curv - grad_norm_curv) 
            
        certs_curv = certs_curv / m_curv

        certs = torch.empty_like(curv_bound)
        certs[curv_i, curv_j] = certs_curv
        certs[nocurv_i, nocurv_j] = certs_nocurv
        certs = certs * logits_sign        
        return certs
    
    @torch.autograd.no_grad()
    def _l2_attack(self, features, y, num_other):
        batch_size = features.shape[0]
        batch_idxs = torch.arange(batch_size).unsqueeze(1)
        
        step_size = (self.step_multiplier * self.attack_radius) / self.attack_iters
        
        w2 = self.linear2.weight
        b2 = self.linear2.bias
        
        x = torch.flatten(features, start_dim=1)
        _, x_act = self._intermediate(x)
        logits = self.linear2(x_act)
        other = topk_classes(y.squeeze(1), logits, num_other)
        
        delta = torch.randn_like(x)
        delta = step_size * F.normalize(delta, dim=1)
        
        for i in range(self.attack_iters):
            x_n = x + delta
            x_n_lin1, x_n_act = self._intermediate(x_n)
            
            logits = self.linear2(x_n_act)
            
            logits_y = logits[batch_idxs, y]
            logits_other = logits[batch_idxs, other]
            logits_diff = logits_y - logits_other

            other_idxs = torch.argmin(logits_diff, dim=1, keepdims=True)
            attack_idxs = other[batch_idxs, other_idxs]
            
            w2_diff_idx = w2[y] - w2[attack_idxs]
            w2_diff_idx = w2_diff_idx[:, 0, :]
                        
            grad = self._gradient(x_n_lin1, w2_diff_idx)
            grad = F.normalize(grad, dim=1)
            
            delta = delta - (step_size * grad)
            
            delta_norm = torch.norm(delta, dim=1, keepdim=True)
            delta_normalized = delta/delta_norm
            delta_norm_clamped = torch.clamp(delta_norm, max=self.attack_radius)
            delta = delta_normalized * delta_norm_clamped
            
            if torch.all(delta_norm >= self.attack_radius):
                break
            
        delta = delta.view_as(features)
        return delta
        
    def certificates(self, features, y=None, num_other=None):
        if y is None:
            logits = self.forward(features)
            y = torch.argmax(logits, dim=1)
            
        if num_other is None:
            num_other = self.num_other
            
        y = y.unsqueeze(1)
        if self.training or (features.dtype == torch.float16):
            certs = self._fast_certificates(features, y, num_other)
        else:
            certs = self._tight_certificates(features, y, num_other)

        if (self.attack_radius > 0) and self.training:
            delta = self._l2_attack(features, y, num_other)
            logits = self.forward(features + delta)
        else:
            logits = self.forward(features)
            
        return logits, certs
    
last_mapping = {
    'soc_ortho': SOC_Cert,
    'bcop_ortho': BCOP_Cert,
    'cayley_ortho': Cayley_Cert,
    'lln': LLN_Linear,
    'crc_full': CRC_Full
}
