import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from engine.deim.deim_criterion import DEIMCriterion
from .ugdr import UGDRLoss
from engine.core import register


@register()
class CriterionWithUGDR(nn.Module):
    __share__ = ['num_classes']
    __inject__ = ['matcher']  
    
    def __init__(
        self,
        matcher, 
        base_criterion='DEIMCriterion', 
        enable_ugdr: bool = False,
        ugdr_weight: float = 1.0,
        beta_schedule: str = 'linear',
        beta_start: float = 1.0,
        beta_end: float = 0.1,
        uncertainty_mode: str='entropy+variance',
        **base_criterion_kwargs  
    ):
        super().__init__()
        
        if isinstance(base_criterion, str):
            if base_criterion == 'DEIMCriterion':
                self.base_criterion = DEIMCriterion(matcher=matcher, **base_criterion_kwargs)
            else:
                raise ValueError(f"Unsupported base_criterion: {base_criterion}, only 'DEIMCriterion' is supported")
        elif isinstance(base_criterion, nn.Module):
            self.base_criterion = base_criterion
        else:
            raise TypeError(f"base_criterion should be str or nn.Module, got {type(base_criterion)}")
        
        self.enable_ugdr = enable_ugdr
        self.ugdr_weight = ugdr_weight
        
        if enable_ugdr:
            self.ugdr = UGDRLoss(
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                uncertainty_mode=uncertainty_mode
            )
            self.current_epoch = 0
            self.total_epochs = 160  
        else:
            self.ugdr = None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        **kwargs  
    ) -> Dict[str, torch.Tensor]:
        loss_dict = self.base_criterion(outputs, targets, **kwargs)
        
        if not self.enable_ugdr:
            return loss_dict
        
        if 'loss_fgl' in loss_dict and 'pred_corners' in outputs:
            beta = self.ugdr.get_beta(self.current_epoch, self.total_epochs)
            
            outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
            num_queries_list = outputs.get('num_queries_list', None)
            epoch = kwargs.get('epoch', 0)
            
            with torch.no_grad():
                indices = self.base_criterion.matcher(
                    outputs_without_aux, 
                    targets, 
                    epoch=epoch, 
                    num_queries_list=num_queries_list
                )['indices']
            
            if indices is not None:
                idx = self.base_criterion._get_src_permutation_idx(indices)
                
                pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.base_criterion.reg_max + 1))
                
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                
                from ..deim.box_ops import box_iou, box_cxcywh_to_xyxy
                from ..deim.dfine_utils import bbox2distance
                
                ref_points = outputs['ref_points'][idx].detach()
                with torch.no_grad():
                    target_corners, weight_right, weight_left = bbox2distance(
                        ref_points, 
                        box_cxcywh_to_xyxy(target_boxes),
                        self.base_criterion.reg_max, 
                        outputs['reg_scale'], 
                        outputs['up']
                    )
                    
                    target_corners = target_corners.clamp(min=0, max=self.base_criterion.reg_max)
                
                ious = torch.diag(box_iou(
                    box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), 
                    box_cxcywh_to_xyxy(target_boxes)
                )[0])
                iou_weights = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                
                ugdr_loss, uncertainty = self.ugdr(
                    pred_corners,
                    target_corners, 
                    weight_right,
                    weight_left,
                    iou_weights=iou_weights,
                    epoch=self.current_epoch,
                    max_epochs=self.total_epochs
                )
                
                loss_dict['loss_ugdr'] = ugdr_loss * self.ugdr_weight
                loss_dict['ugdr_beta'] = torch.tensor(beta, device=outputs['pred_logits'].device)
                loss_dict['ugdr_uncertainty_mean'] = uncertainty.mean()
        
        return loss_dict
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_criterion, name)


def create_criterion_with_ugdr(
    base_criterion_config: Dict,
    enable_ugdr: bool = False,
    ugdr_weight: float = 1.0,
    beta_schedule: str = 'linear',
    beta_start: float = 1.0,
    beta_end: float = 0.1
) -> CriterionWithUGDR:
    base_criterion = DEIMCriterion(**base_criterion_config)
    
    criterion = CriterionWithUGDR(
        base_criterion=base_criterion,
        enable_ugdr=enable_ugdr,
        ugdr_weight=ugdr_weight,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end
    )
    
    return criterion
