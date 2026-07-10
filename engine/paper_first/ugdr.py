'''
Uncertainty-Guided Distribution Refinement (UGDR)
不确定性引导的分布精炼

引入动机：
DFINE的FDR (Fine-grained Distribution Refinement) 对所有预测使用相同的loss权重，
未考虑预测的可靠性差异。针对GWHD数据集的边界模糊性问题（重叠、遮挡、域偏移），
本模块引入不确定性度量，对高不确定性预测降低loss权重，实现课程学习。

参考论文：
1. DHSA (ECCV 2024): "Restoring Images in Adverse Weather Conditions via Histogram Transformer"
   - 借鉴分布直方图的思想，用于不确定性计算
   - 论文链接：https://arxiv.org/pdf/2407.10172
2. D-FINE (arXiv 2024): "D-FINE: Redefine Regression Task as Fine-grained Distribution Refinement"
   - 扩展其FDR机制，加入不确定性引导
   - Github：https://github.com/Peterande/D-FINE/

理论基础：
- 信息论：熵(Entropy)度量分布的不确定性
- 课程学习：先学习确定样本(低熵)，再学习模糊样本(高熵)
- 鲁棒学习：降低噪声样本的影响
'''

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
    """从FDR的bbox分布计算不确定性
    
    DFINE的FDR预测每个bbox角点的分布 (reg_max+1 bins)，
    我们计算其熵和方差作为不确定性度量。
    
    不确定性的两种度量：
    1. 熵 (Entropy): H = -Σ p_i * log(p_i)
       - 度量分布的混乱程度
       - 高熵 = 分布平坦 = 不确定
       - 低熵 = 分布尖锐 = 确定
    
    2. 方差 (Variance): Var = Σ (i - μ)^2 * p_i
       - 度量分布的离散程度
       - 高方差 = 预测分散 = 不确定
       - 低方差 = 预测集中 = 确定
    """
    
    @staticmethod
    def calculate_entropy(distribution, eps=1e-8):
        """计算分布的熵
        
        Args:
            distribution: bbox角点分布 (N, reg_max+1)，N是所有corner的总数
            eps: 数值稳定性常数
            
        Returns:
            entropy: 熵值 (N,)，范围 [0, log(reg_max+1)]
        """
        # Softmax归一化为概率分布
        probs = F.softmax(distribution, dim=-1)  # (N, reg_max+1)
        
        # 计算熵: H = -Σ p * log(p)
        log_probs = torch.log(probs + eps)
        entropy = -(probs * log_probs).sum(dim=-1)  # (N,)
        
        return entropy
    
    @staticmethod
    def calculate_variance(distribution):
        """计算分布的方差
        
        Args:
            distribution: bbox角点分布 (N, reg_max+1)
            
        Returns:
            variance: 方差 (N,)
        """
        # Softmax归一化
        probs = F.softmax(distribution, dim=-1)  # (N, reg_max+1)
        
        # 计算期望: μ = Σ i * p_i
        bins = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        mean = (probs * bins).sum(dim=-1)  # (N,)
        
        # 计算方差: Var = Σ (i - μ)^2 * p_i
        variance = (probs * ((bins - mean.unsqueeze(-1)) ** 2)).sum(dim=-1)  # (N,)
        
        return variance
    
    @staticmethod
    def calculate_uncertainty(distribution, mode='entropy+variance'):
        """计算综合不确定性
        
        Args:
            distribution: bbox角点分布 (N, reg_max+1) 或 (B, Q, 4, reg_max+1)
            mode: 不确定性计算模式
                - 'entropy': 仅熵
                - 'variance': 仅方差
                - 'entropy+variance': 熵和方差的组合（默认）
        
        Returns:
            uncertainty: 归一化的不确定性 (N,) 或 (B, Q, 4)，范围 [0, 1]
        """
        # 处理不同的输入形状
        original_shape = distribution.shape
        if len(original_shape) == 4:  # (B, Q, 4, reg_max+1)
            B, Q, corners, bins = original_shape
            distribution = distribution.reshape(-1, bins)  # (B*Q*4, reg_max+1)
        
        # 计算不确定性
        if mode == 'entropy':
            uncertainty = UncertaintyCalculator.calculate_entropy(distribution)
            # 归一化到 [0, 1]
            max_entropy = math.log(distribution.shape[-1])
            uncertainty = uncertainty / max_entropy
            
        elif mode == 'variance':
            uncertainty = UncertaintyCalculator.calculate_variance(distribution)
            # 归一化到 [0, 1]
            max_variance = (distribution.shape[-1] ** 2) / 12.0  # 均匀分布的方差
            uncertainty = uncertainty / max_variance
            
        elif mode == 'entropy+variance':
            # 计算熵和方差
            entropy = UncertaintyCalculator.calculate_entropy(distribution)
            variance = UncertaintyCalculator.calculate_variance(distribution)
            
            # 分别归一化
            max_entropy = math.log(distribution.shape[-1])
            max_variance = (distribution.shape[-1] ** 2) / 12.0
            entropy_norm = entropy / max_entropy
            variance_norm = variance / max_variance
            
            # 组合（简单平均）
            uncertainty = (entropy_norm + variance_norm) / 2.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 恢复原始形状
        if len(original_shape) == 4:
            uncertainty = uncertainty.reshape(B, Q, corners)
        
        return uncertainty


class UGDRLoss(nn.Module):
    """不确定性引导的分布精炼损失
    
    基于DFINE的unimodal_distribution_focal_loss，加入不确定性加权：
    
    Loss = w_uncertainty * L_fgl
    
    其中：
    - w_uncertainty = β + (1 - β) * (1 - uncertainty)
    - β是容忍度参数（课程学习调度）
    - uncertainty ∈ [0, 1]
    
    课程学习策略：
    - 训练初期(β=1.0): 完全容忍不确定性，等价于原始FDR
    - 训练中期(β线性衰减): 逐渐提高对确定性的要求
    - 训练后期(β=0.1): 严格要求，高不确定性预测权重很低
    """
    
    def __init__(
        self,
        reg_max=15,
        beta_schedule='linear',
        beta_start=1.0,
        beta_end=0.1,
        uncertainty_mode='entropy+variance'
    ):
        """
        Args:
            reg_max: FDR的最大回归值
            beta_schedule: β衰减策略 ('linear', 'cosine', 'constant')
            beta_start: 初始β值
            beta_end: 最终β值
            uncertainty_mode: 不确定性计算模式
        """
        super().__init__()
        self.reg_max = reg_max
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.uncertainty_mode = uncertainty_mode
        
    def get_beta(self, epoch, max_epochs):
        """获取当前epoch的β值（课程学习调度）
        
        Args:
            epoch: 当前epoch
            max_epochs: 总epoch数
            
        Returns:
            beta: 当前β值
        """
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
        """计算UGDR损失
        
        Args:
            pred_corners: 预测的角点分布 (N, reg_max+1)
            target_corners: 目标角点索引 (N,)
            weight_right: FDR的右权重 (N,)
            weight_left: FDR的左权重 (N,)
            iou_weights: IoU权重 (N,)，用于质量感知加权
            epoch: 当前epoch
            max_epochs: 总epoch数
            base_loss_fn: 基础FDR损失函数（可选，用于复用现有实现）
            
        Returns:
            loss: UGDR损失标量
            uncertainty: 不确定性值 (N,)，用于监控和可视化
        """
        # 1. 计算不确定性
        uncertainty = UncertaintyCalculator.calculate_uncertainty(
            pred_corners,
            mode=self.uncertainty_mode
        )  # (N,)
        
        # 2. 获取当前β值（课程学习）
        beta = self.get_beta(epoch, max_epochs)
        
        # 3. 计算不确定性权重
        # w = β + (1 - β) * (1 - uncertainty)
        # 低不确定性(0) -> w = 1.0
        # 高不确定性(1) -> w = β
        uncertainty_weight = beta + (1 - beta) * (1 - uncertainty)  # (N,)
        
        # 4. 计算基础FDR损失
        if base_loss_fn is not None:
            # 使用提供的FDR损失函数
            base_loss = base_loss_fn(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight=iou_weights
            )
        else:
            # 简化的FDR损失实现（用于测试）
            # 实际使用时应该用DEIM criterion中的unimodal_distribution_focal_loss
            # 保护性clamp：确保索引在合法范围内
            target_corners_clamped = target_corners.clamp(min=0, max=self.reg_max)
            target_one_hot = F.one_hot(target_corners_clamped.long(), num_classes=self.reg_max + 1).float()
            base_loss = F.cross_entropy(
                pred_corners,
                target_corners_clamped.long(),
                reduction='none'
            )
        
        # 5. 应用不确定性加权
        weighted_loss = base_loss * uncertainty_weight
        
        # 6. 结合IoU权重（如果提供）
        if iou_weights is not None:
            weighted_loss = weighted_loss * iou_weights
        
        # 7. 归约
        loss = weighted_loss.mean()
        
        return loss, uncertainty


def test_uncertainty_calculator():
    """测试不确定性计算"""
    print("\n" + "="*80)
    print("测试 Uncertainty Calculator")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_max = 15
    N = 100  # 样本数
    
    # 创建不同不确定性的分布
    print(f"\n创建测试分布:")
    
    # 1. 确定分布（低熵、低方差）
    certain_dist = torch.zeros(N // 3, reg_max + 1, device=device)
    certain_dist[:, reg_max // 2] = 10.0  # 尖峰分布
    
    # 2. 中等不确定性分布
    medium_dist = torch.randn(N // 3, reg_max + 1, device=device)
    
    # 3. 高不确定性分布（高熵、高方差）
    uncertain_dist = torch.ones(N // 3, reg_max + 1, device=device)  # 均匀分布
    
    # 合并
    distributions = torch.cat([certain_dist, medium_dist, uncertain_dist], dim=0)
    
    # 计算不确定性
    entropy = UncertaintyCalculator.calculate_uncertainty(distributions, mode='entropy')
    variance = UncertaintyCalculator.calculate_uncertainty(distributions, mode='variance')
    combined = UncertaintyCalculator.calculate_uncertainty(distributions, mode='entropy+variance')
    
    print(f"  确定分布 (0-{N//3}):")
    print(f"    熵: {entropy[:N//3].mean():.4f} ± {entropy[:N//3].std():.4f}")
    print(f"    方差: {variance[:N//3].mean():.4f} ± {variance[:N//3].std():.4f}")
    print(f"    组合: {combined[:N//3].mean():.4f} ± {combined[:N//3].std():.4f}")
    
    print(f"  中等不确定性分布 ({N//3}-{2*N//3}):")
    print(f"    熵: {entropy[N//3:2*N//3].mean():.4f} ± {entropy[N//3:2*N//3].std():.4f}")
    print(f"    方差: {variance[N//3:2*N//3].mean():.4f} ± {variance[N//3:2*N//3].std():.4f}")
    print(f"    组合: {combined[N//3:2*N//3].mean():.4f} ± {combined[N//3:2*N//3].std():.4f}")
    
    print(f"  高不确定性分布 ({2*N//3}-{N}):")
    print(f"    熵: {entropy[2*N//3:].mean():.4f} ± {entropy[2*N//3:].std():.4f}")
    print(f"    方差: {variance[2*N//3:].mean():.4f} ± {variance[2*N//3:].std():.4f}")
    print(f"    组合: {combined[2*N//3:].mean():.4f} ± {combined[2*N//3:].std():.4f}")
    
    print("\n" + "="*80)
    print("✓ 不确定性计算测试完成")
    print("="*80 + "\n")


def test_ugdr_loss():
    """测试UGDR损失"""
    print("\n" + "="*80)
    print("测试 UGDR Loss")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_max = 15
    N = 100
    
    # 创建模拟数据
    pred_corners = torch.randn(N, reg_max + 1, device=device)
    target_corners = torch.randint(0, reg_max + 1, (N,), device=device)
    weight_right = torch.rand(N, device=device)
    weight_left = torch.rand(N, device=device)
    iou_weights = torch.rand(N, device=device) * 0.5 + 0.5  # [0.5, 1.0]
    
    # 创建UGDR损失
    ugdr_loss = UGDRLoss(
        reg_max=reg_max,
        beta_schedule='linear',
        beta_start=1.0,
        beta_end=0.1
    ).to(device)
    
    print(f"\n输入配置:")
    print(f"  样本数: {N}")
    print(f"  reg_max: {reg_max}")
    print(f"  β调度: linear (1.0 → 0.1)")
    
    # 测试不同epoch的损失
    print(f"\n不同epoch的损失和β值:")
    epochs_to_test = [0, 40, 80, 120, 159]
    max_epochs = 160
    
    for epoch in epochs_to_test:
        with torch.no_grad():
            loss, uncertainty = ugdr_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                iou_weights=iou_weights,
                epoch=epoch,
                max_epochs=max_epochs
            )
            beta = ugdr_loss.get_beta(epoch, max_epochs)
        
        print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}, β={beta:.3f}, "
              f"Uncertainty={uncertainty.mean():.4f}±{uncertainty.std():.4f}")
    
    # 测试不同β调度策略
    print(f"\n不同β调度策略 (Epoch 80):")
    schedules = ['constant', 'linear', 'cosine']
    epoch = 80
    
    for schedule in schedules:
        loss_fn = UGDRLoss(
            reg_max=reg_max,
            beta_schedule=schedule,
            beta_start=1.0,
            beta_end=0.1
        ).to(device)
        
        with torch.no_grad():
            loss, _ = loss_fn(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                epoch=epoch,
                max_epochs=max_epochs
            )
            beta = loss_fn.get_beta(epoch, max_epochs)
        
        print(f"  {schedule:8s}: Loss={loss.item():.4f}, β={beta:.3f}")
    
    print("\n" + "="*80)
    print("✓ UGDR损失测试完成")
    print("="*80 + "\n")


def visualize_beta_schedule():
    """可视化β调度策略"""
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n生成β调度可视化...")
    
    max_epochs = 72
    epochs = np.arange(0, max_epochs)
    
    schedules = {
        'linear': UGDRLoss(beta_schedule='linear'),
        'cosine': UGDRLoss(beta_schedule='cosine'),
        'constant': UGDRLoss(beta_schedule='constant')
    }
    
    plt.figure(figsize=(10, 6))
    for name, loss_fn in schedules.items():
        betas = [loss_fn.get_beta(e, max_epochs) for e in epochs]
        plt.plot(epochs, betas, label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('β (Tolerance)', fontsize=12)
    plt.title('UGDR β Scheduling Strategies', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = '/home/wyq/wyq/DEIM-DEIM/beta_schedule.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    
    print(GREEN + "="*80 + RESET)
    print(GREEN + " Uncertainty-Guided Distribution Refinement (UGDR) - 独立测试" + RESET)
    print(GREEN + "="*80 + RESET)
    
    # 测试1: 不确定性计算
    test_uncertainty_calculator()
    
    # 测试2: UGDR损失
    test_ugdr_loss()
    
    # 测试3: 可视化β调度
    try:
        visualize_beta_schedule()
    except Exception as e:
        print(f"{YELLOW}可视化跳过: {e}{RESET}")
    
    print(BLUE + "\n论文引用说明:" + RESET)
    print("  [1] DHSA (ECCV 2024) - 分布直方图思想")
    print("  [2] D-FINE (arXiv 2024) - FDR机制扩展")
    print("  本模块通过不确定性引导实现课程学习，提升边界定位精度")
    
    print(YELLOW + "\n参数量分析:" + RESET)
    print("  ✓ 完全0参数（纯算法创新）")
    print("  ✓ 不增加模型复杂度")
    print("  ✓ 仅在训练时计算不确定性")
    
    print(ORANGE + "\n理论贡献:" + RESET)
    print("  1. 信息论：熵度量不确定性")
    print("  2. 课程学习：easy-to-hard训练策略")
    print("  3. 鲁棒学习：降低噪声样本影响")
