"""Hand-written optimized kernel for KernelBench level 2 task 1: a 3 by 3 convolution, ReLU and a bias add, computed
directly in one kernel."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_P, BLOCK_C = 128, 32
NUM_WARPS = 4


@triton.jit
def conv_relu_bias_kernel(x_ptr, w_ptr, cb_ptr, b_ptr, y_ptr, NB, C, H, W, OC, OH, OW,
                          KH: tl.constexpr, KW: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_p = offs_p < NB * OH * OW
    mask_c = offs_c < OC
    
    # Compute n, oh, ow from offs_p
    ow = offs_p % OW
    t = offs_p // OW
    oh = t % OH
    n = t // OH
    
    # Precompute indices for x to avoid repeated computation
    R = C * KH * KW
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Load bias terms once
    cb = tl.load(cb_ptr + offs_c, mask=mask_c, other=0.0)
    b = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    
    # Convolution loop
    for r in range(0, R, 1):
        c = r // (KH * KW)
        rem = r % (KH * KW)
        ih = oh + rem // KW
        iw = ow + rem % KW
        
        # Load input value with proper masking
        x_idx = ((n * C + c) * H + ih) * W + iw
        xv = tl.load(x_ptr + x_idx, mask=(ih < H) & (iw < W), other=0.0)
        
        # Load weight value
        w_idx = offs_c * R + r
        wv = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
        
        # Accumulate
        acc += xv[:, None] * wv[None, :]
    
    # Apply ReLU and bias
    acc = tl.maximum(acc + cb[None, :], 0.0) + b[None, :]
    
    # Store output
    y_off = ((n[:, None] * OC + offs_c[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    tl.store(y_ptr + y_off, acc, mask=mask_p[:, None] & mask_c[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        NB, C, H, W = x.shape
        OC, _, KH, KW = self.conv.weight.shape
        OH, OW = H - KH + 1, W - KW + 1
        y = torch.empty((NB, OC, OH, OW), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(NB * OH * OW, BLOCK_P), triton.cdiv(OC, BLOCK_C))
        conv_relu_bias_kernel[grid](x, self.conv.weight, self.conv.bias, self.bias, y, NB, C, H, W, OC, OH, OW,
                                    KH=KH, KW=KW, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C, num_warps=NUM_WARPS)
        return y
