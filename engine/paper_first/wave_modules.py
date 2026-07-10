import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.fft.dct
    USE_TORCH_DCT = True
except AttributeError:
    try:
        import torch_dct as dct
        USE_TORCH_DCT = False
    except ImportError:
        raise


class Wave2D(nn.Module):
    
    def __init__(self, dim=128, hidden_dim=None, res=20, learnable_params=True, use_padding=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.res = res
        self.use_padding = use_padding 
        
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.to_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        if learnable_params:
            self.wave_speed = nn.Parameter(torch.ones(1) * 1.0) 
            self.damping = nn.Parameter(torch.ones(1) * 0.1)     
        else:
            self.register_buffer('wave_speed', torch.ones(1) * 1.0)
            self.register_buffer('damping', torch.ones(1) * 0.1)
    
    def forward(self, x, freq_embed=None):
        B, C, H, W = x.shape
        
        x = self.dwconv(x)
        
        x_transformed = self.linear(x.permute(0, 2, 3, 1))  # [B, H, W, 2C]
        u0, v0 = x_transformed.chunk(2, dim=-1)  # 各自 [B, H, W, C]
        
        u0 = u0.permute(0, 3, 1, 2)  # [B, C, H, W]
        v0 = v0.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        if self.use_padding:
            pad_h, pad_w = H // 4, W // 4
            u0_padded = F.pad(u0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            v0_padded = F.pad(v0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            u0_freq = self.dct2d(u0_padded)
            v0_freq = self.dct2d(v0_padded)
            freq_H, freq_W = H + 2 * pad_h, W + 2 * pad_w
            pad_info = (pad_h, pad_w)
        else:
            u0_freq = self.dct2d(u0)
            v0_freq = self.dct2d(v0)
            freq_H, freq_W = H, W
            pad_info = None
        
        if freq_embed is not None:
            t = self.to_time(freq_embed.unsqueeze(0).expand(B, -1, -1, -1))
            t = t.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            freq_y = torch.arange(freq_H, device=x.device, dtype=x.dtype).view(1, 1, freq_H, 1)
            freq_x = torch.arange(freq_W, device=x.device, dtype=x.dtype).view(1, 1, 1, freq_W)
            freq_y = freq_y * (math.pi / freq_H)
            freq_x = freq_x * (math.pi / freq_W)
            omega = torch.sqrt(freq_y**2 + freq_x**2)  # [1, 1, freq_H, freq_W]
            t = omega.expand(B, C, freq_H, freq_W)
        
        omega_d = torch.sqrt(torch.clamp(
            (self.wave_speed * t)**2 - (self.damping / 2)**2,
            min=1e-8
        ))
        
        cos_term = torch.cos(omega_d)
        sin_term = torch.sin(omega_d) / (omega_d + 1e-8)
        
        wave_component = cos_term * u0_freq
        velocity_component = sin_term * (v0_freq + (self.damping / 2) * u0_freq)
        
        damping_factor = torch.exp(-self.damping * t / 2)
        final_freq = damping_factor * (wave_component + velocity_component)
        
        x_wave = self.idct2d(final_freq)
        
        if self.use_padding and pad_info is not None:
            pad_h, pad_w = pad_info
            x_wave = x_wave[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        x_wave = self.out_norm(x_wave.permute(0, 2, 3, 1))
        x_wave = x_wave.permute(0, 3, 1, 2)
        
        gate = F.silu(v0)
        x_wave = x_wave * gate
        
        x_out = self.out_linear(x_wave.permute(0, 2, 3, 1))
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out
    
    @staticmethod
    def dct2d(x):
        if USE_TORCH_DCT:
            x = torch.fft.dct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.dct(x, type=2, dim=-1, norm='ortho')
        else:
            x = dct.dct_2d(x, norm='ortho')
        return x
    
    @staticmethod
    def idct2d(x):
        if USE_TORCH_DCT:
            x = torch.fft.idct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.idct(x, type=2, dim=-1, norm='ortho')
        else:
            x = dct.idct_2d(x, norm='ortho')
        return x

class Wave2D4Ablation(nn.Module):
    def __init__(self, dim=128, hidden_dim=None, res=20, learnable_params=True, use_padding=True, damping=0.1, wave_speed=1.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.res = res
        self.use_padding = use_padding 
        
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.to_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        if learnable_params:
            self.wave_speed = nn.Parameter(torch.ones(1) * wave_speed)  # c:
            self.damping = nn.Parameter(torch.ones(1) * damping)     # α: 
        else:
            self.register_buffer('wave_speed', torch.ones(1) * wave_speed)
            self.register_buffer('damping', torch.ones(1) * damping)
    
    def forward(self, x, freq_embed=None):

        B, C, H, W = x.shape
        
        x = self.dwconv(x)
        
        x_transformed = self.linear(x.permute(0, 2, 3, 1))  # [B, H, W, 2C]
        u0, v0 = x_transformed.chunk(2, dim=-1)  # 各自 [B, H, W, C]
        
        u0 = u0.permute(0, 3, 1, 2)  # [B, C, H, W]
        v0 = v0.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        if self.use_padding:
            pad_h, pad_w = H // 4, W // 4
            u0_padded = F.pad(u0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            v0_padded = F.pad(v0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            u0_freq = self.dct2d(u0_padded)
            v0_freq = self.dct2d(v0_padded)
            freq_H, freq_W = H + 2 * pad_h, W + 2 * pad_w
            pad_info = (pad_h, pad_w)
        else:
            u0_freq = self.dct2d(u0)
            v0_freq = self.dct2d(v0)
            freq_H, freq_W = H, W
            pad_info = None
        
        if freq_embed is not None:
            t = self.to_time(freq_embed.unsqueeze(0).expand(B, -1, -1, -1))
            t = t.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            freq_y = torch.arange(freq_H, device=x.device, dtype=x.dtype).view(1, 1, freq_H, 1)
            freq_x = torch.arange(freq_W, device=x.device, dtype=x.dtype).view(1, 1, 1, freq_W)
            freq_y = freq_y * (math.pi / freq_H)
            freq_x = freq_x * (math.pi / freq_W)
            omega = torch.sqrt(freq_y**2 + freq_x**2)  # [1, 1, freq_H, freq_W]
            t = omega.expand(B, C, freq_H, freq_W)
        
        omega_d = torch.sqrt(torch.clamp(
            (self.wave_speed * t)**2 - (self.damping / 2)**2,
            min=1e-8
        ))
        
        cos_term = torch.cos(omega_d)
        sin_term = torch.sin(omega_d) / (omega_d + 1e-8)
        
        wave_component = cos_term * u0_freq
        velocity_component = sin_term * (v0_freq + (self.damping / 2) * u0_freq)
        
        damping_factor = torch.exp(-self.damping * t / 2)
        final_freq = damping_factor * (wave_component + velocity_component)
        
        x_wave = self.idct2d(final_freq)
        
        if self.use_padding and pad_info is not None:
            pad_h, pad_w = pad_info
            x_wave = x_wave[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        x_wave = self.out_norm(x_wave.permute(0, 2, 3, 1))
        x_wave = x_wave.permute(0, 3, 1, 2)
        
        gate = F.silu(v0)
        x_wave = x_wave * gate
        
        x_out = self.out_linear(x_wave.permute(0, 2, 3, 1))
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out
    
    @staticmethod
    def dct2d(x):
        if USE_TORCH_DCT:
            x = torch.fft.dct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.dct(x, type=2, dim=-1, norm='ortho')
        else:
            x = dct.dct_2d(x, norm='ortho')
        return x
    
    @staticmethod
    def idct2d(x):
        if USE_TORCH_DCT:
            x = torch.fft.idct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.idct(x, type=2, dim=-1, norm='ortho')
        else:
            x = dct.idct_2d(x, norm='ortho')
        return x


class WaveEncoderBlock(nn.Module):
    
    def __init__(self,
                 d_model,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False, 
                 damping=0.1, 
                 wave_speed=1.0):
        super().__init__()
        from engine.deim.utils import get_activation
        
        self.normalize_before = normalize_before
        
        self.wave_op = Wave2D4Ablation(
            dim=d_model,
            hidden_dim=d_model,
            res=20,
            learnable_params=False,  
            use_padding=True,  
            damping=damping,
            wave_speed=wave_speed
        )
        
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
        
        if self.normalize_before:
            x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.dropout1(self.wave_op(x_norm))
        else:
            wave_out = self.wave_op(x)
            x = x + self.dropout1(wave_out)
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
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
   
    def __init__(self,
                 d_model,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False,
                 alpha_range=(0.05, 0.2),  
                 speed_range=(0.8, 1.5)):   
        super().__init__()
        from engine.deim.utils import get_activation
        
        self.normalize_before = normalize_before
        self.alpha_min, self.alpha_max = alpha_range
        self.speed_min, self.speed_max = speed_range
        
        self.wave_op = Wave2D(
            dim=d_model,
            hidden_dim=d_model,
            res=20,
            learnable_params=False,  
            use_padding=True
        )
        
        self.param_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(d_model, d_model // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 8, 2, 1),  
            nn.Sigmoid() 
        )
        
        self.alpha = nn.Parameter(torch.ones(1, d_model, 1, 1))
        self.belt = nn.Parameter(torch.zeros(1, d_model, 1, 1))
        
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
        
        param_scales = self.param_generator(x)  # [B, 2, 1, 1]
        alpha_scale = param_scales[:, 0:1, :, :]  # [B, 1, 1, 1]
        speed_scale = param_scales[:, 1:2, :, :]  # [B, 1, 1, 1]
        
        dynamic_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - alpha_scale)
        dynamic_speed = self.speed_min + (self.speed_max - self.speed_min) * speed_scale
        
        self.wave_op.damping.data = dynamic_alpha.mean()
        self.wave_op.wave_speed.data = dynamic_speed.mean()
        
        if self.normalize_before:
            x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            wave_out = self.wave_op(x_norm)
        else:
            wave_out = self.wave_op(x)
        
        feat_var = torch.var(wave_out, dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
        wave_out = wave_out * self.alpha + feat_var * self.belt
        
        x = x + self.dropout1(wave_out)
        
        if not self.normalize_before:
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
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