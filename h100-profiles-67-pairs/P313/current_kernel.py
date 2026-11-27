"""Optimized Triton kernel for KernelBench level 2 task 1."""
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
    
    # Decompose index into batch, output height, output width
    ow = offs_p % OW
    t = offs_p // OW
    oh = t % OH
    n = t // OH
    
    # Precompute offsets for output
    y_off = ((n[:, None] * OC + offs_c[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Convolution loop with better memory access
    for k in range(0, C * KH * KW, 1):
        c = k // (KH * KW)
        rem = k % (KH * KW)
        ih = oh + rem // KW
        iw = ow + rem % KW
        
        # Load input value
        xv = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=mask_p, other=0.0)
        
        # Load weight value
        wv = tl.load(w_ptr + (offs_c * (C * KH * KW) + k), mask=mask_c, other=0.0)
        
        # Accumulate
        acc += xv[:, None] * wv[None, :]
    
    # Load bias and apply ReLU
    cb = tl.load(cb_ptr + offs_c, mask=mask_c, other=0.0)
    b = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    acc = tl.maximum(acc + cb[None, :], 0.0) + b[None, :]
    
    # Store result
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
