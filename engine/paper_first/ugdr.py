import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False


class UncertaintyCalculator:
    
    @staticmethod
    def calculate_entropy(distribution, eps=1e-8):

        probs = F.softmax(distribution, dim=-1)  # (N, reg_max+1)
        
        log_probs = torch.log(probs + eps)
        entropy = -(probs * log_probs).sum(dim=-1)  # (N,)
        
        return entropy
    
    @staticmethod
    def calculate_variance(distribution):
        probs = F.softmax(distribution, dim=-1)  # (N, reg_max+1)
        
        bins = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        mean = (probs * bins).sum(dim=-1)  # (N,)
        
        variance = (probs * ((bins - mean.unsqueeze(-1)) ** 2)).sum(dim=-1)  # (N,)
        
        return variance
    
    @staticmethod
    def calculate_uncertainty(distribution, mode='entropy+variance'):
        original_shape = distribution.shape
        if len(original_shape) == 4:  # (B, Q, 4, reg_max+1)
            B, Q, corners, bins = original_shape
            distribution = distribution.reshape(-1, bins)  # (B*Q*4, reg_max+1)
        
        if mode == 'entropy':
            uncertainty = UncertaintyCalculator.calculate_entropy(distribution)
            max_entropy = math.log(distribution.shape[-1])
            uncertainty = uncertainty / max_entropy
            
        elif mode == 'variance':
            uncertainty = UncertaintyCalculator.calculate_variance(distribution)
            # 归一化到 [0, 1]
            max_variance = (distribution.shape[-1] ** 2) / 12.0  
            uncertainty = uncertainty / max_variance
            
        elif mode == 'entropy+variance':
            entropy = UncertaintyCalculator.calculate_entropy(distribution)
            variance = UncertaintyCalculator.calculate_variance(distribution)
            
            max_entropy = math.log(distribution.shape[-1])
            max_variance = (distribution.shape[-1] ** 2) / 12.0
            entropy_norm = entropy / max_entropy
            variance_norm = variance / max_variance
            
            uncertainty = (entropy_norm + variance_norm) / 2.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if len(original_shape) == 4:
            uncertainty = uncertainty.reshape(B, Q, corners)
        
        return uncertainty


class UGDRLoss(nn.Module):
   
    
    def __init__(
        self,
        reg_max=15,
        beta_schedule='linear',
        beta_start=1.0,
        beta_end=0.1,
        uncertainty_mode='entropy+variance'
    ):
        
        super().__init__()
        self.reg_max = reg_max
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.uncertainty_mode = uncertainty_mode
        
    def get_beta(self, epoch, max_epochs):
       
        progress = epoch / max_epochs  # [0, 1]
        
        if self.beta_schedule == 'constant':
            beta = self.beta_start
        elif self.beta_schedule == 'linear':
            beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        elif self.beta_schedule == 'cosine':
            beta = self.beta_end + (self.beta_start - self.beta_end) * \
                   (1 + math.cos(math.pi * progress)) / 2
        else:
            raise ValueError(f"Unknown schedule: {self.beta_schedule}")
        
        return beta
    
    def forward(
        self,
        pred_corners,
        target_corners,
        weight_right,
        weight_left,
        iou_weights=None,
        epoch=0,
        max_epochs=160,
        base_loss_fn=None
    ):
        
        uncertainty = UncertaintyCalculator.calculate_uncertainty(
            pred_corners,
            mode=self.uncertainty_mode
        )  # (N,)
        
        beta = self.get_beta(epoch, max_epochs)
        
       
        uncertainty_weight = beta + (1 - beta) * (1 - uncertainty)  # (N,)
        
        if base_loss_fn is not None:
            base_loss = base_loss_fn(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight=iou_weights
            )
        else:
            target_corners_clamped = target_corners.clamp(min=0, max=self.reg_max)
            target_one_hot = F.one_hot(target_corners_clamped.long(), num_classes=self.reg_max + 1).float()
            base_loss = F.cross_entropy(
                pred_corners,
                target_corners_clamped.long(),
                reduction='none'
            )
        
        weighted_loss = base_loss * uncertainty_weight
        
        if iou_weights is not None:
            weighted_loss = weighted_loss * iou_weights
        
        # 7. 归约
        loss = weighted_loss.mean()
        
        return loss, uncertainty