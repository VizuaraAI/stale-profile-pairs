"""Optimized Triton kernel for depthwise separable convolution."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

DW_BLOCK = 512
PW_BLOCK_P, PW_BLOCK_C = 256, 64
NUM_WARPS = 4


@triton.jit
def depthwise_kernel(x_ptr, w_ptr, y_ptr, C, H, W, OH, OW, stride, padding, dilation, KS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nc = tl.program_id(1)
    c = nc % C
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OH * OW
    oh = offs // OW
    ow = offs % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(KS):
        for kw in tl.static_range(KS):
            ih = oh * stride - padding + kh * dilation
            iw = ow * stride - padding + kw * dilation
            m = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + nc * H * W + ih * W + iw, mask=m, other=0.0)
            wv = tl.load(w_ptr + c * KS * KS + kh * KS + kw)
            acc += xv * wv
    tl.store(y_ptr + nc * OH * OW + offs, acc, mask=mask)


@triton.jit
def pointwise_kernel_optimized(x_ptr, w_ptr, y_ptr, C, OC, P, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.program_id(2)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_p = offs_p < P
    mask_c = offs_c < OC
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Process 4 channels per iteration with better memory access
    for c in range(0, C, 4):
        # Load 4 channels at a time
        xv_0 = tl.load(x_ptr + (n * C + c) * P + offs_p, mask=mask_p, other=0.0)
        wv_0 = tl.load(w_ptr + offs_c * C + c, mask=mask_c, other=0.0)
        acc += xv_0[:, None] * wv_0[None, :]
        
        if c + 1 < C:
            xv_1 = tl.load(x_ptr + (n * C + c + 1) * P + offs_p, mask=mask_p, other=0.0)
            wv_1 = tl.load(w_ptr + offs_c * C + c + 1, mask=mask_c, other=0.0)
            acc += xv_1[:, None] * wv_1[None, :]
            
        if c + 2 < C:
            xv_2 = tl.load(x_ptr + (n * C + c + 2) * P + offs_p, mask=mask_p, other=0.0)
            wv_2 = tl.load(w_ptr + offs_c * C + c + 2, mask=mask_c, other=0.0)
            acc += xv_2[:, None] * wv_2[None, :]
            
        if c + 3 < C:
            xv_3 = tl.load(x_ptr + (n * C + c + 3) * P + offs_p, mask=mask_p, other=0.0)
            wv_3 = tl.load(w_ptr + offs_c * C + c + 3, mask=mask_c, other=0.0)
            acc += xv_3[:, None] * wv_3[None, :]
            
    tl.store(y_ptr + (n * OC + offs_c[None, :]) * P + offs_p[:, None], acc, mask=mask_p[:, None] & mask_c[None, :])


@triton.jit
def pointwise_kernel_improved(x_ptr, w_ptr, y_ptr, C, OC, P, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.program_id(2)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_p = offs_p < P
    mask_c = offs_c < OC
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Process all channels with reduced register pressure
    for c in range(C):
        xv = tl.load(x_ptr + (n * C + c) * P + offs_p, mask=mask_p, other=0.0)
        wv = tl.load(w_ptr + offs_c * C + c, mask=mask_c, other=0.0)
        acc += xv[:, None] * wv[None, :]
        
    tl.store(y_ptr + (n * OC + offs_c[None, :]) * P + offs_p[:, None], acc, mask=mask_p[:, None] & mask_c[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False):
        super().__init__()
        assert not bias, "only the task's settings"
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        NB, C, H, W = x.shape
        dw = self.depthwise
        KS = dw.kernel_size[0]
        stride, padding, dilation = dw.stride[0], dw.padding[0], dw.dilation[0]
        OH = (H + 2 * padding - dilation * (KS - 1) - 1) // stride + 1
        OW = (W + 2 * padding - dilation * (KS - 1) - 1) // stride + 1
        mid = torch.empty((NB, C, OH, OW), device=x.device, dtype=torch.float32)
        depthwise_kernel[(triton.cdiv(OH * OW, DW_BLOCK), NB * C)](x, dw.weight, mid, C, H, W, OH, OW, stride, padding, dilation,
                                                                  KS=KS, BLOCK=DW_BLOCK, num_warps=NUM_WARPS)
        OC = self.pointwise.weight.shape[0]
        P = OH * OW
        y = torch.empty((NB, OC, OH, OW), device=x.device, dtype=torch.float32)
        pointwise_kernel_improved[(triton.cdiv(P, PW_BLOCK_P), triton.cdiv(OC, PW_BLOCK_C), NB)](mid, self.pointwise.weight, y, C, OC, P,
                                                                                               BLOCK_P=PW_BLOCK_P, BLOCK_C=PW_BLOCK_C,
                                                                                               num_warps=NUM_WARPS)
        return y
