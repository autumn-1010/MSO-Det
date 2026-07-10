import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Route(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return features[self.idx]


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.v2e = MessageAgg(agg_method="mean")  # Vertex to Hyperedge
        self.e2v = MessageAgg(agg_method="mean")  # Hyperedge to Vertex

    def forward(self, x, H):
        x = self.fc(x)
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        x_out = self.e2v(E, H)
        
        return x + x_out 


class HyperComputeCore(nn.Module):
    def __init__(self, channels, threshold=8):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyperGraphConv(channels, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
       
        b, c, h, w = x.shape
        
        x_flat = x.view(b, c, -1).transpose(1, 2).contiguous()
        
        feature = x_flat.clone()
        distance = torch.cdist(feature, feature)  # [B, N, N]
        hypergraph = (distance < self.threshold).float() 
        
        x_enhanced = self.hgconv(x_flat, hypergraph)
        
        x_out = x_enhanced.transpose(1, 2).contiguous().view(b, c, h, w)
        
        x_out = self.act(self.bn(x_out))
        
        return x_out


class HyperGraphEnhance(nn.Module):
    
    def __init__(self, hidden_dim=256, threshold=8, target_size=40, residual_weight=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.target_size = target_size
        self.residual_weight = residual_weight
        
        # P3(80x80) -> downsample, P4(40x40) -> keep, P5(20x20) -> upsample
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((target_size, target_size)),  
            nn.AdaptiveAvgPool2d((target_size, target_size)),  
            nn.AdaptiveAvgPool2d((target_size, target_size))   
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        self.hyper_compute = HyperComputeCore(hidden_dim, threshold)
        
        self.restore_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ) for _ in range(3)
        ])
    
    def forward(self, features):
        
        pooled = [pool(feat) for pool, feat in zip(self.adaptive_pools, features)]
        
        x_mixed = torch.cat(pooled, dim=1)
        
        x_mixed = self.fusion_conv(x_mixed)
        
        x_hyper = self.hyper_compute(x_mixed)  # [B, 256, 40, 40]
        
        enhanced_features = []
        for i, (feat, restore_conv) in enumerate(zip(features, self.restore_convs)):
            target_size = feat.shape[-2:]
            x_i = F.interpolate(x_hyper, size=target_size, mode='bilinear', align_corners=False)
            
            x_i = restore_conv(x_i)
            
            enhanced = feat + self.residual_weight * x_i
            enhanced_features.append(enhanced)
        
        return enhanced_features


HyperGraphModule = HyperGraphEnhance
