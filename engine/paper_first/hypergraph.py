"""
超图增强模块 - 基于Hyper-YOLO的高阶关联建模
Hypergraph Enhancement Module for DFINE

核心思想：
1. 语义收集(Semantic Collecting): 统一多尺度特征到同一空间
2. 超图计算(Hypergraph Computation): 基于距离构建超图，进行高阶消息传递
3. 语义散射(Semantic Scattering): 将增强后的特征分发回各尺度

Reference: Hyper-YOLO (TPAMI 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Route(nn.Module):
    """特征路由模块 - 多尺度特征选择与传递"""
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List[Tensor] - 多尺度特征列表
        Returns:
            routed_features: List[Tensor] - 选择后的特征列表
        """
        return features[self.idx]


class MessageAgg(nn.Module):
    """消息聚合模块 - 超图卷积的基础操作"""
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
        Args:
            X: [B, N, C] 节点特征
            path: [B, N, N] 路径矩阵 (col->row)
        Returns:
            聚合后的特征
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            return X
        return X


class HyperGraphConv(nn.Module):
    """超图卷积层 - 两阶段消息传递机制"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.v2e = MessageAgg(agg_method="mean")  # Vertex to Hyperedge
        self.e2v = MessageAgg(agg_method="mean")  # Hyperedge to Vertex

    def forward(self, x, H):
        """
        超图卷积: X' = X + D_v^-1 * H * D_e^-1 * H^T * X * Θ
        
        Args:
            x: [B, N, C] 顶点特征
            H: [B, N, N] 超图关联矩阵
        Returns:
            增强后的特征
        """
        x = self.fc(x)
        # 阶段1: 顶点 -> 超边 (V->E)
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # 阶段2: 超边 -> 顶点 (E->V)
        x_out = self.e2v(E, H)
        
        return x + x_out  # 残差连接


class HyperComputeCore(nn.Module):
    """超图计算核心模块 - 构建超图并进行卷积"""
    def __init__(self, channels, threshold=8):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyperGraphConv(channels, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            超图增强后的特征
        """
        b, c, h, w = x.shape
        
        # 1. 展平为点集: [B, C, H, W] -> [B, N, C]
        x_flat = x.view(b, c, -1).transpose(1, 2).contiguous()
        
        # 2. 构建超图 (基于欧氏距离的ε-ball)
        feature = x_flat.clone()
        distance = torch.cdist(feature, feature)  # [B, N, N]
        hypergraph = (distance < self.threshold).float()  # 距离阈值构建超边
        
        # 3. 超图卷积 (高阶消息传递)
        x_enhanced = self.hgconv(x_flat, hypergraph)
        
        # 4. 恢复形状: [B, N, C] -> [B, C, H, W]
        x_out = x_enhanced.transpose(1, 2).contiguous().view(b, c, h, w)
        
        # 5. 归一化和激活
        x_out = self.act(self.bn(x_out))
        
        return x_out


class HyperGraphEnhance(nn.Module):
    """
    超图多尺度特征增强模块 (HGC-SCS框架实例化)
    
    功能：
    - 接收多尺度特征 [P3, P4, P5]
    - 通过超图计算捕获跨层级、跨位置的高阶关联
    - 输出增强后的多尺度特征
    
    Args:
        hidden_dim: 特征通道数
        threshold: 超图距离阈值 (推荐: N=6, S=8, M=10)
        target_size: 统一的特征尺寸 (默认40, 中间尺度)
        residual_weight: 残差连接权重
    """
    def __init__(self, hidden_dim=256, threshold=8, target_size=40, residual_weight=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.target_size = target_size
        self.residual_weight = residual_weight
        
        # 1. 语义收集: 统一多尺度特征到相同尺寸（使用中间尺度）
        # P3(80x80) -> downsample, P4(40x40) -> keep, P5(20x20) -> upsample
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((target_size, target_size)),  # P3: 下采样
            nn.AdaptiveAvgPool2d((target_size, target_size)),  # P4: 保持/调整
            nn.AdaptiveAvgPool2d((target_size, target_size))   # P5: 上采样
        ])
        
        # 2. 特征融合卷积 (3个尺度concat后降维)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        # 3. 超图计算核心
        self.hyper_compute = HyperComputeCore(hidden_dim, threshold)
        
        # 4. 语义散射: 特征恢复卷积
        self.restore_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ) for _ in range(3)
        ])
    
    def forward(self, features):
        """
        超图增强前向传播
        
        Args:
            features: List[Tensor] - [P3(B,C,H3,W3), P4(B,C,H4,W4), P5(B,C,H5,W5)]
        Returns:
            enhanced_features: List[Tensor] - 增强后的多尺度特征
        """
        # =============== 阶段1: 语义收集 (Semantic Collecting) ===============
        # 统一尺寸到中间尺度: [80x80, 40x40, 20x20] -> [40x40, 40x40, 40x40]
        # P3下采样保留更多细节，P5上采样获得更大感受野
        pooled = [pool(feat) for pool, feat in zip(self.adaptive_pools, features)]
        
        # 拼接多尺度特征: [B, 256, 40, 40] x 3 -> [B, 768, 40, 40]
        x_mixed = torch.cat(pooled, dim=1)
        
        # 降维融合: [B, 768, 40, 40] -> [B, 256, 40, 40]
        x_mixed = self.fusion_conv(x_mixed)
        
        # =============== 阶段2: 超图计算 (Hypergraph Computation) ===============
        # 在中间尺度语义空间进行高阶消息传递（1600个点而非400个点）
        x_hyper = self.hyper_compute(x_mixed)  # [B, 256, 40, 40]
        
        # =============== 阶段3: 语义散射 (Semantic Scattering) ===============
        # 将增强后的特征分发回各尺度
        enhanced_features = []
        for i, (feat, restore_conv) in enumerate(zip(features, self.restore_convs)):
            # 插值回原始尺寸
            target_size = feat.shape[-2:]
            x_i = F.interpolate(x_hyper, size=target_size, mode='bilinear', align_corners=False)
            
            # 恢复卷积
            x_i = restore_conv(x_i)
            
            # 残差连接 (加权融合)
            enhanced = feat + self.residual_weight * x_i
            enhanced_features.append(enhanced)
        
        return enhanced_features


# 为了兼容性，创建别名
HyperGraphModule = HyperGraphEnhance
