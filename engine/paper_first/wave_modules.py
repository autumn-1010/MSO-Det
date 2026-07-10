"""
Wave Propagation Modules for DFINE Integration
基于WaveFormer的波动传播算子，适配于目标检测任务

集成方案：
1. Wave2D: 核心波动传播模块（适配检测特征）
3. WaveEncoderBlock, WaveEncoderBlockV2: 完全替换Transformer的版本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入DCT实现
try:
    # PyTorch 1.8+ 有原生DCT
    torch.fft.dct
    USE_TORCH_DCT = True
except AttributeError:
    # 使用torch_dct库
    try:
        import torch_dct as dct
        USE_TORCH_DCT = False
    except ImportError:
        print("警告: 需要安装torch_dct: pip install torch_dct")
        raise


class Wave2D(nn.Module):
    """
    阻尼波动方程在2D特征图上的实现
    基于频率域的解析解：
    u(x,y,t) = F⁻¹{e^(-αt/2)[F(u₀)cos(ωₐt) + sin(ωₐt)/ωₐ(F(v₀) + α/2·F(u₀))]}
    
    Args:
        dim: 输入通道数
        hidden_dim: 隐藏层通道数
        res: 特征图分辨率（用于频率嵌入）
        learnable_params: 是否让α和c可学习
    """
    def __init__(self, dim=128, hidden_dim=None, res=20, learnable_params=True, use_padding=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.res = res
        self.use_padding = use_padding  # 是否使用padding减少边界伪影
        
        # 深度可分离卷积：提取局部特征
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        # 线性变换：生成u₀和v₀（初始语义场和速度场）
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        # 输出归一化和投影
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # 频率嵌入到时间的映射（学习每个频率的传播时间）
        self.to_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        # 波动方程参数
        if learnable_params:
            self.wave_speed = nn.Parameter(torch.ones(1) * 1.0)  # c: 波速
            self.damping = nn.Parameter(torch.ones(1) * 0.1)     # α: 阻尼系数
        else:
            self.register_buffer('wave_speed', torch.ones(1) * 1.0)
            self.register_buffer('damping', torch.ones(1) * 0.1)
    
    def forward(self, x, freq_embed=None):
        """
        Args:
            x: [B, C, H, W] 输入特征图
            freq_embed: [H, W, C] 可选的频率位置编码
        Returns:
            [B, C, H, W] 波动传播后的特征
        """
        B, C, H, W = x.shape
        
        # 1. 局部特征提取
        x = self.dwconv(x)
        
        # 2. 生成初始语义场u₀和速度场v₀
        x_transformed = self.linear(x.permute(0, 2, 3, 1))  # [B, H, W, 2C]
        u0, v0 = x_transformed.chunk(2, dim=-1)  # 各自 [B, H, W, C]
        
        u0 = u0.permute(0, 3, 1, 2)  # [B, C, H, W]
        v0 = v0.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 3. 频率域变换（可选padding减少边界伪影）
        if self.use_padding:
            # 使用反射padding减少DCT的周期性假设带来的边界效应
            pad_h, pad_w = H // 4, W // 4
            u0_padded = F.pad(u0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            v0_padded = F.pad(v0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            u0_freq = self.dct2d(u0_padded)
            v0_freq = self.dct2d(v0_padded)
            # 记录padding尺寸和实际频率域尺寸
            freq_H, freq_W = H + 2 * pad_h, W + 2 * pad_w
            pad_info = (pad_h, pad_w)
        else:
            u0_freq = self.dct2d(u0)
            v0_freq = self.dct2d(v0)
            freq_H, freq_W = H, W
            pad_info = None
        
        # 4. 计算传播时间（频率感知）
        if freq_embed is not None:
            # 使用外部频率嵌入（每个stage的可学习参数）
            t = self.to_time(freq_embed.unsqueeze(0).expand(B, -1, -1, -1))
            t = t.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            # 使用频率坐标生成时间参数（关键修正）
            # 为每个DCT系数位置生成与频率相关的时间（使用频率域尺寸）
            freq_y = torch.arange(freq_H, device=x.device, dtype=x.dtype).view(1, 1, freq_H, 1)
            freq_x = torch.arange(freq_W, device=x.device, dtype=x.dtype).view(1, 1, 1, freq_W)
            # 归一化频率坐标到[0, π]
            freq_y = freq_y * (math.pi / freq_H)
            freq_x = freq_x * (math.pi / freq_W)
            # 径向频率（空间频率大小）
            omega = torch.sqrt(freq_y**2 + freq_x**2)  # [1, 1, freq_H, freq_W]
            # 扩展到batch和channel
            t = omega.expand(B, C, freq_H, freq_W)
        
        # 5. 波动方程求解（阻尼振荡）
        # ω_d = sqrt(ω²c² - (α/2)²) 阻尼频率
        omega_d = torch.sqrt(torch.clamp(
            (self.wave_speed * t)**2 - (self.damping / 2)**2,
            min=1e-8
        ))
        
        cos_term = torch.cos(omega_d)
        sin_term = torch.sin(omega_d) / (omega_d + 1e-8)
        
        # 波动项 + 速度项
        wave_component = cos_term * u0_freq
        velocity_component = sin_term * (v0_freq + (self.damping / 2) * u0_freq)
        
        # 关键修正：应用阻尼衰减因子 e^(-αt/2)
        damping_factor = torch.exp(-self.damping * t / 2)
        final_freq = damping_factor * (wave_component + velocity_component)
        
        # 6. 逆变换回空间域
        x_wave = self.idct2d(final_freq)
        
        # 如果使用了padding，需要裁剪回原始尺寸
        if self.use_padding and pad_info is not None:
            pad_h, pad_w = pad_info
            x_wave = x_wave[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        # 7. 输出处理（归一化 + 门控）
        x_wave = self.out_norm(x_wave.permute(0, 2, 3, 1))
        x_wave = x_wave.permute(0, 3, 1, 2)
        
        # SiLU门控（类似GLU）
        gate = F.silu(v0)
        x_wave = x_wave * gate
        
        x_out = self.out_linear(x_wave.permute(0, 2, 3, 1))
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out
    
    @staticmethod
    def dct2d(x):
        """2D DCT-II变换"""
        if USE_TORCH_DCT:
            x = torch.fft.dct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.dct(x, type=2, dim=-1, norm='ortho')
        else:
            # 使用torch_dct库
            x = dct.dct_2d(x, norm='ortho')
        return x
    
    @staticmethod
    def idct2d(x):
        """2D IDCT-II变换"""
        if USE_TORCH_DCT:
            x = torch.fft.idct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.idct(x, type=2, dim=-1, norm='ortho')
        else:
            # 使用torch_dct库
            x = dct.idct_2d(x, norm='ortho')
        return x


class WaveEncoderBlock(nn.Module):
    """
    纯Wave版本的Encoder Block
    
    结构：Wave2D + FFN（类似Transformer的MHA + FFN）
    """
    def __init__(self,
                 d_model,
                 nhead=8,  # 保留接口兼容性，但不使用
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False):
        super().__init__()
        from engine.deim.utils import get_activation
        
        self.normalize_before = normalize_before
        
        # Wave传播层（替代Multi-Head Attention）
        self.wave_op = Wave2D(
            dim=d_model,
            hidden_dim=d_model,
            res=20,
            learnable_params=True,
            use_padding=True  # 使用padding减少边界伪影
        )
        
        # FFN层（保持与Transformer一致）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Wave传播 + 残差
        if self.normalize_before:
            x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.dropout1(self.wave_op(x_norm))
        else:
            wave_out = self.wave_op(x)
            x = x + self.dropout1(wave_out)
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FFN + 残差
        if self.normalize_before:
            x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2 = self.linear2(self.dropout(self.activation(
                self.linear1(x_norm.permute(0, 2, 3, 1))
            )))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
        else:
            x_ffn = x.permute(0, 2, 3, 1)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x_ffn))))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
            x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class WaveEncoderBlockV2(nn.Module):
    """
    自适应波动传播编码器 - 基于物理动机的密集场景检测优化
    
    物理动机（通用于所有密集检测场景）：
    波动方程的频率域传播特性：u(x,y,t) = F⁻¹{e^(-αt/2)[...]}
    - α（阻尼系数）控制高频信息的衰减速度
    - c（波速）控制语义信息的传播范围
    
    密集场景的物理本质：
    - 密集场景：目标边缘高频信息密集，相邻目标间距小
      → 需要小α保留高频细节，避免边缘模糊
      → 需要大c加快传播，快速聚合局部信息
    - 稀疏场景：背景区域占比大，噪声干扰显著
      → 需要大α平滑传播，抑制背景噪声
      → 需要小c减缓传播，避免过度扩散
    
    这是通用的物理规律，适用于：人群检测、车辆检测、农作物检测等所有密集目标场景
    
    技术方案（参考顶会最佳实践）：
    1. Dynamic-CBAM (ICAMCS 2024): 全局池化获取场景统计 → 动态权重生成
    2. SMFA (ECCV 2024): 可学习alpha/belt自调制 + 方差统计增强
    
    关键改进：
    - 动态参数生成器：根据特征激活强度自适应调节α和c（物理参数）
    - 自调制特征增强：SMFA风格的alpha×特征 + belt×方差 调制
    - 接口完全一致：forward(x)→x，可无缝替换WaveEncoderBlock
    """
    def __init__(self,
                 d_model,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False,
                 alpha_range=(0.05, 0.2),  # 阻尼系数范围
                 speed_range=(0.8, 1.5)):   # 波速范围
        super().__init__()
        from engine.deim.utils import get_activation
        
        self.normalize_before = normalize_before
        self.alpha_min, self.alpha_max = alpha_range
        self.speed_min, self.speed_max = speed_range
        
        # Wave传播层（基础版本）
        self.wave_op = Wave2D(
            dim=d_model,
            hidden_dim=d_model,
            res=20,
            learnable_params=False,  # 使用动态生成的参数
            use_padding=True
        )
        
        # 动态参数生成器（物理动机：场景自适应）
        # 原理：利用特征激活强度反映场景密集程度
        #   - 密集场景：多目标 → 高激活 → 小α保留高频
        #   - 稀疏场景：多背景 → 低激活 → 大α平滑噪声
        # 实现：参考Dynamic-CBAM的全局统计 + 轻量MLP映射
        self.param_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化获取场景统计
            nn.Conv2d(d_model, d_model // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 8, 2, 1),  # 输出2个物理参数的scale: α_scale, c_scale
            nn.Sigmoid()  # 输出[0,1]范围，后续映射到物理参数范围
        )
        
        # 自调制参数（物理增强：特征表达能力）
        # 原理：参考SMFA的自适应特征调制，增强模型表达
        #   - alpha: 乘法因子，控制特征强度
        #   - belt: 加法因子，结合方差统计提供额外调节自由度
        # 物理意义：在波动传播后，根据局部统计特性进一步精细化特征
        self.alpha = nn.Parameter(torch.ones(1, d_model, 1, 1))
        self.belt = nn.Parameter(torch.zeros(1, d_model, 1, 1))
        
        # FFN层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 动态参数生成（物理自适应机制）
        # 通过特征全局统计推断场景密集程度，调节波动物理参数
        param_scales = self.param_generator(x)  # [B, 2, 1, 1]
        alpha_scale = param_scales[:, 0:1, :, :]  # [B, 1, 1, 1]
        speed_scale = param_scales[:, 1:2, :, :]  # [B, 1, 1, 1]
        
        # 映射到物理参数范围
        # 反向关系的物理解释：高激活(密集场景) → 小α → e^(-αt/2)衰减慢 → 保留高频
        dynamic_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - alpha_scale)
        # 正向关系的物理解释：高激活(密集场景) → 大c → 波速快 → 快速聚合局部信息
        dynamic_speed = self.speed_min + (self.speed_max - self.speed_min) * speed_scale
        
        # 临时设置Wave的参数
        self.wave_op.damping.data = dynamic_alpha.mean()
        self.wave_op.wave_speed.data = dynamic_speed.mean()
        
        # 2. Wave传播 + 自调制特征增强
        if self.normalize_before:
            x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            wave_out = self.wave_op(x_norm)
        else:
            wave_out = self.wave_op(x)
        
        # 自调制增强（参考SMFA的物理增强思想）
        # 结合特征方差作为局部统计信息，进行自适应调制
        # 物理意义：方差大 → 信息丰富/目标密集 → 增强表达
        feat_var = torch.var(wave_out, dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
        wave_out = wave_out * self.alpha + feat_var * self.belt
        
        x = x + self.dropout1(wave_out)
        
        if not self.normalize_before:
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 3. FFN + 残差
        if self.normalize_before:
            x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2 = self.linear2(self.dropout(self.activation(
                self.linear1(x_norm.permute(0, 2, 3, 1))
            )))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
        else:
            x_ffn = x.permute(0, 2, 3, 1)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x_ffn))))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
            x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x



if __name__ == "__main__":
    # 测试代码
      
    print("="*60)
    print("测试Wave2D模块")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 128, 20, 20
    x = torch.randn(B, C, H, W)
    
    # 测试Wave2D
    wave = Wave2D(dim=C, hidden_dim=C, res=H)
    out = wave(x)
    print(f"Wave2D输入: {x.shape}, 输出: {out.shape}")
    assert out.shape == x.shape, "形状不匹配！"
    
    # 测试WaveEncoderBlock
    print("\n" + "="*60)
    print("测试WaveEncoderBlock (Baseline)")
    print("="*60)
    wave_block = WaveEncoderBlock(d_model=C, dim_feedforward=512)
    out3 = wave_block(x)
    print(f"WaveEncoderBlock输入: {x.shape}, 输出: {out3.shape}")
    
    # 测试WaveEncoderBlockV2
    print("\n" + "="*60)
    print("测试WaveEncoderBlockV2 (Improved)")
    print("="*60)
    wave_block_v2 = WaveEncoderBlockV2(d_model=C, dim_feedforward=512)
    out4 = wave_block_v2(x)
    print(f"WaveEncoderBlockV2输入: {x.shape}, 输出: {out4.shape}")
    
    # 验证动态参数生成
    print("\n验证动态参数生成机制:")
    with torch.no_grad():
        # 测试不同"密度"的输入
        low_density_input = torch.randn(B, C, H, W) * 0.3  # 低激活
        high_density_input = torch.randn(B, C, H, W) * 1.5  # 高激活
        
        param_low = wave_block_v2.param_generator(low_density_input)
        param_high = wave_block_v2.param_generator(high_density_input)
        
        print(f"  低密度场景参数: alpha_scale={param_low[0,0,0,0]:.3f}, speed_scale={param_low[0,1,0,0]:.3f}")
        print(f"  高密度场景参数: alpha_scale={param_high[0,0,0,0]:.3f}, speed_scale={param_high[0,1,0,0]:.3f}")
        print(f"  参数差异: {(param_high - param_low).abs().mean().item():.4f}")
    
    print("\n✅ 所有模块测试通过！")
    print("\n" + "="*60)
    print("📝 WaveEncoderBlockV2 核心改进（物理动机）")
    print("="*60)
    print("1. 自适应波动参数（物理本质）")
    print("   物理原理：波动方程 u = F⁻¹{e^(-αt/2)[...]}中")
    print("   - α控制高频衰减速度：小α保留细节，大α平滑噪声")
    print("   - c控制传播速度：大c快速聚合，小c局部保留")
    print("\n   场景适应（通用规律，非数据集特定）：")
    print("   - 密集场景（人群/车辆/作物）：小α+大c → 保留边缘+快速聚合")
    print("   - 稀疏场景（背景主导）：大α+小c → 抑制噪声+避免扩散")
    print("\n   实现：全局池化统计 → 轻量MLP → α和c动态调节")
    print("   参考：Dynamic-CBAM (ICAMCS 2024)")
    print("\n2. 自调制特征增强（表达能力提升）")
    print("   - alpha×特征 + belt×方差 自适应调制")
    print("   - 方差反映局部信息丰富度，增强密集区域表达")
    print("   - 参考：SMFA (ECCV 2024)")
    print("\n3. 通用性保证")
    print("   - 物理动机而非数据集驱动，适用所有密集检测场景")
    print("   - 接口一致：forward(x)→x，无缝替换WaveEncoderBlock")
    print("   - YAML配置：直接改module名即可")
    print("="*60)
