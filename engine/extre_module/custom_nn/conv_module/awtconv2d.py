"""\
Adaptive Wavelet-Token Conv2d (AWTConv2d)

- Interface-compatible with WTConv2d:
    __init__(in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1')
    forward(x): (B, in_channels, H, W) -> (B, out_channels, H/stride, W/stride)

Design goal
-----------
Keep WTConv2d's wavelet analysis/synthesis scaffold, but make the *processing* inside the
wavelet pyramid and the *fusion* with the spatial branch more adaptive.

Innovation points (inspired by ideas commonly seen in recent CV modules)
------------------------------------------------------------
1) Per-channel 4x4 learnable subband mixing (LL/LH/HL/HH) via grouped 1x1 (groups=in_channels).
2) Subband-wise attention (dynamic re-weighting) conditioned on pooled subband statistics.
3) CoordGate-like coordinate gating (size-agnostic): generate a spatial gate from (x,y) coords.
4) FSA-like strip frequency gating: separable low/high strip decomposition along H and W.
5) CGA-style pixel-aware fusion to merge spatial branch and reconstructed wavelet branch.

No extra runtime dependencies beyond torch + pywt (same as wtconv2d.py).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import pywt

from engine.extre_module.ultralytics_nn.conv import Conv


def create_wavelet_filter(wave: str, in_size: int, out_size: int, dtype=torch.float):
    w = pywt.Wavelet(wave)

    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    y = F.conv2d(x, filters.to(dtype=x.dtype, device=x.device), stride=2, groups=c, padding=pad)
    y = y.reshape(b, c, 4, h // 2, w // 2)
    return y


def inverse_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    y = x.reshape(b, c * 4, h_half, w_half)
    y = F.conv_transpose2d(y, filters.to(dtype=x.dtype, device=x.device), stride=2, groups=c, padding=pad)
    return y


class _WaveletTransformFn(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, filters: torch.Tensor):
        ctx.filters = filters
        with torch.no_grad():
            out = wavelet_transform(input, filters)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = inverse_wavelet_transform(grad_output, ctx.filters)
        return grad, None


class _InverseWaveletTransformFn(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, filters: torch.Tensor):
        ctx.filters = filters
        with torch.no_grad():
            out = inverse_wavelet_transform(input, filters)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = wavelet_transform(grad_output, ctx.filters)
        return grad, None


def wavelet_transform_init(filters: torch.Tensor):
    def apply(input: torch.Tensor):
        return _WaveletTransformFn.apply(input, filters)

    return apply


def inverse_wavelet_transform_init(filters: torch.Tensor):
    def apply(input: torch.Tensor):
        return _InverseWaveletTransformFn.apply(input, filters)

    return apply


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _SubbandAttention(nn.Module):
    """Subband attention with per-channel grouped 1x1 projection.

    Input:  (B, 4C, H, W)
    Output: (B, 4C, 1, 1) in (0, 1)

    The grouped conv (groups=C) learns a 4->4 mapping for each channel group,
    generating dynamic weights for LL/LH/HL/HH.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Conv2d(channels * 4, channels * 4, kernel_size=1, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        attn = torch.sigmoid(self.proj(pooled))
        return attn


class _CoordGate2d(nn.Module):
    """Size-agnostic coordinate gate.

    Generates a spatially-varying gate from normalized (x, y) coordinates.
    This is a lightweight, fully-convolutional adaptation of coordinate gating.
    """

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
        gate = self.net(coords)
        return x * gate


class _StripFreqGating(nn.Module):
    """FSA-like strip low/high gating (B,C,H,W -> B,C,H,W).

    Learns per-channel mixing between low-pass (strip pooled) and high-pass residual
    along horizontal and vertical directions.
    """

    def __init__(self, channels: int, kernel: int = 7):
        super().__init__()
        self.kernel = kernel
        pad = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad, pad))
        self.pad_hori = nn.ReflectionPad2d((pad, pad, 0, 0))
        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)

        self.hori_low = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.vert_low = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (1.0 + self.hori_high) * hori_h

        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (1.0 + self.vert_high) * vert_h

        return x * self.beta + vert_out * self.gamma


class _CGAStyleFusion(nn.Module):
    """Pixel-aware fusion between two same-shape features (x,y) -> fused.

    Computes channel attention + spatial attention to guide a pixel gate.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
        )
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)

        self.pixel_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=7, padding=3, padding_mode='reflect', groups=channels, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        initial = x + y

        cattn = self.ca(initial)
        x_avg = torch.mean(initial, dim=1, keepdim=True)
        x_max, _ = torch.max(initial, dim=1, keepdim=True)
        sattn = self.sa(torch.cat([x_avg, x_max], dim=1))

        pattn1 = cattn + sattn
        gate = self.pixel_gate(torch.cat([initial, pattn1.expand_as(initial)], dim=1))

        out = initial + gate * x + (1.0 - gate) * y
        return self.out_proj(out)


class AWTConv2d(nn.Module):
    """A more adaptive variant of WTConv2d with the same public interface."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
        wt_type: str = 'db1',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        # Spatial branch: depthwise conv (same as WTConv2d style)
        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding='same',
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)

        # Wavelet branch per level:
        # 1) depthwise conv over each subband channel
        # 2) per-channel 4x4 mixing across subbands (grouped 1x1, groups=C)
        # 3) coord gate + strip frequency gate
        # 4) subband attention (dynamic weights)
        self.dw_subband_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding='same',
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(wt_levels)
            ]
        )
        self.mix_subbands = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size=1,
                    groups=in_channels,
                    bias=True,
                )
                for _ in range(wt_levels)
            ]
        )
        self.coord_gate = nn.ModuleList([_CoordGate2d(in_channels * 4) for _ in range(wt_levels)])
        self.strip_gate = nn.ModuleList([_StripFreqGating(in_channels * 4, kernel=7) for _ in range(wt_levels)])
        self.subband_attn = nn.ModuleList([_SubbandAttention(in_channels) for _ in range(wt_levels)])
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(wt_levels)]
        )

        # Fusion: keep a conservative per-channel gate + add pixel-aware fusion
        self.fuse_gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.fusion = _CGAStyleFusion(in_channels, reduction=8)

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)

            def _do_stride(x_in: torch.Tensor):
                return F.conv2d(
                    x_in,
                    self.stride_filter.to(dtype=x_in.dtype, device=x_in.device),
                    bias=None,
                    stride=self.stride,
                    groups=in_channels,
                )

            self.do_stride = _do_stride
        else:
            self.do_stride = None

        if in_channels != out_channels:
            self.conv1x1 = Conv(in_channels, out_channels, 1)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for level in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, pads)

            # (B, C, 4, H/2, W/2)
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            # token-like processing in subband space
            b, c, _, hh, ww = curr_x.shape
            curr_x_tag = curr_x.reshape(b, c * 4, hh, ww)

            curr_x_tag = self.dw_subband_convs[level](curr_x_tag)
            curr_x_tag = self.mix_subbands[level](curr_x_tag)
            curr_x_tag = self.coord_gate[level](curr_x_tag)
            curr_x_tag = self.strip_gate[level](curr_x_tag)
            curr_x_tag = curr_x_tag * self.subband_attn[level](curr_x_tag)
            curr_x_tag = self.wavelet_scale[level](curr_x_tag)

            curr_x_tag = curr_x_tag.reshape(b, c, 4, hh, ww)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_wave = next_x_ll

        x_base = self.base_scale(self.base_conv(x))
        x_wave = torch.sigmoid(self.fuse_gate) * x_wave
        x_fused = self.fusion(x_base, x_wave)

        if self.do_stride is not None:
            x_fused = self.do_stride(x_fused)

        return self.conv1x1(x_fused)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.randn(1, 16, 33, 32, device=device)
    m = AWTConv2d(16, 64, kernel_size=5, stride=1, wt_levels=2, wt_type='db2').to(device)
    y = m(x)
    print('x:', tuple(x.shape), 'y:', tuple(y.shape))
