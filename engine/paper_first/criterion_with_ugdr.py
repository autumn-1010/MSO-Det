"""
Criterion with UGDR (Uncertainty-Guided Distribution Refinement) Integration
向后兼容的wrapper实现，enable_ugdr=False时行为与base criterion完全一致
"""
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
    """
    Criterion wrapper with optional UGDR support
    
    核心设计原则：
    1. 非侵入式wrapper：包装任意base criterion，不修改其代码
    2. 向后兼容：enable_ugdr=False时，直接委托给base_criterion
    3. 可选UGDR：enable_ugdr=True时，在loss_dict中添加loss_ugdr
    
    Args:
        matcher: HungarianMatcher 实例（通过依赖注入）
        base_criterion: 基础损失函数（如SetCriterion）
        enable_ugdr: 是否启用UGDR（默认False保持向后兼容）
        ugdr_weight: UGDR损失的权重（仅enable_ugdr=True时使用）
        beta_schedule: Beta调度策略 'linear' | 'cosine' | 'constant'
        beta_start: Beta起始值（不确定性权重）
        beta_end: Beta结束值
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']  # 使用依赖注入机制
    
    def __init__(
        self,
        matcher,  # 通过依赖注入提供
        base_criterion='DEIMCriterion',  # 默认使用DEIMCriterion
        enable_ugdr: bool = False,
        ugdr_weight: float = 1.0,
        beta_schedule: str = 'linear',
        beta_start: float = 1.0,
        beta_end: float = 0.1,
        **base_criterion_kwargs  # 传递给base_criterion的参数
    ):
        super().__init__()
        
        # 创建base_criterion
        if isinstance(base_criterion, str):
            # 字符串名称，直接创建DEIMCriterion（唯一支持的base）
            if base_criterion == 'DEIMCriterion':
                # 将 matcher 传递给 DEIMCriterion
                self.base_criterion = DEIMCriterion(matcher=matcher, **base_criterion_kwargs)
            else:
                raise ValueError(f"Unsupported base_criterion: {base_criterion}, only 'DEIMCriterion' is supported")
        elif isinstance(base_criterion, nn.Module):
            # 已经是模块实例
            self.base_criterion = base_criterion
        else:
            raise TypeError(f"base_criterion should be str or nn.Module, got {type(base_criterion)}")
        
        self.enable_ugdr = enable_ugdr
        self.ugdr_weight = ugdr_weight
        
        # 如果启用UGDR，创建UGDR模块
        if enable_ugdr:
            self.ugdr = UGDRLoss(
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end
            )
            self.current_epoch = 0
            self.total_epochs = 160  # 默认值，可通过set_epoch更新
        else:
            self.ugdr = None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        **kwargs  # 接受额外参数如 epoch, indices 等
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 完全兼容 DEIMCriterion 接口
        
        Args:
            outputs: 模型输出
            targets: 真值标签
            **kwargs: 其他参数（如 epoch, indices 等）
                
        Returns:
            loss_dict: 损失字典
        """
        # 1. 调用base criterion获取基础损失（传递所有kwargs）
        loss_dict = self.base_criterion(outputs, targets, **kwargs)
        
        if not self.enable_ugdr:
            # 向后兼容模式：直接返回base损失
            return loss_dict
        
        # 2. UGDR模式：计算不确定性引导的损失
        # 仅在loss_fgl存在时计算UGDR（避免在没有bbox回归的情况下报错）
        if 'loss_fgl' in loss_dict and 'pred_corners' in outputs:
            # 计算当前beta值（课程学习调度）
            beta = self.ugdr.get_beta(self.current_epoch, self.total_epochs)
            
            # 重新计算匹配indices（与base_criterion使用相同的matcher）
            # 注意：必须使用outputs_without_aux，与DEIMCriterion保持一致
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
                # 获取匹配的索引
                idx = self.base_criterion._get_src_permutation_idx(indices)
                
                # 提取匹配的预测corners
                pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.base_criterion.reg_max + 1))
                
                # 提取匹配的目标boxes
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                
                # 重新计算target_corners和weights（必须与pred_corners对齐）
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
                    
                    # 保护性clamp：确保target_corners在合法范围内 [0, reg_max]
                    target_corners = target_corners.clamp(min=0, max=self.base_criterion.reg_max)
                
                # 计算IoU weights
                ious = torch.diag(box_iou(
                    box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), 
                    box_cxcywh_to_xyxy(target_boxes)
                )[0])
                iou_weights = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                
                # 计算UGDR损失
                ugdr_loss, uncertainty = self.ugdr(
                    pred_corners,
                    target_corners, 
                    weight_right,
                    weight_left,
                    iou_weights=iou_weights,
                    epoch=self.current_epoch,
                    max_epochs=self.total_epochs
                )
                
                # 3. 添加UGDR损失到loss_dict
                loss_dict['loss_ugdr'] = ugdr_loss * self.ugdr_weight
                loss_dict['ugdr_beta'] = torch.tensor(beta, device=outputs['pred_logits'].device)
                loss_dict['ugdr_uncertainty_mean'] = uncertainty.mean()
        
        return loss_dict
    
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """
        设置当前epoch（用于课程学习调度）
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数（可选）
        """
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def __getattr__(self, name):
        """代理访问base_criterion的属性"""
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
    """
    工厂函数：从配置创建CriterionWithUGDR
    
    Args:
        base_criterion_config: base criterion的配置字典
            - weight_dict: 损失权重
            - losses: 损失类型列表
            - alpha, gamma: focal loss参数
        enable_ugdr: 是否启用UGDR
        其他参数: 见CriterionWithUGDR
        
    Returns:
        criterion: CriterionWithUGDR实例
        
    Example:
        >>> criterion = create_criterion_with_ugdr(
        ...     base_criterion_config={
        ...         'weight_dict': {'loss_vfl': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
        ...         'losses': ['vfl', 'boxes'],
        ...         'alpha': 0.75,
        ...         'gamma': 2.0
        ...     },
        ...     enable_ugdr=True,
        ...     ugdr_weight=1.0,
        ...     beta_schedule='linear',
        ...     beta_start=1.0,
        ...     beta_end=0.1
        ... )
    """
    # 创建base criterion
    base_criterion = DEIMCriterion(**base_criterion_config)
    
    # 包装为CriterionWithUGDR
    criterion = CriterionWithUGDR(
        base_criterion=base_criterion,
        enable_ugdr=enable_ugdr,
        ugdr_weight=ugdr_weight,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end
    )
    
    return criterion
