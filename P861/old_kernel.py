"""Optimized Triton kernel for 2D convolution."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_P, BLOCK_C = 128, 32
NUM_WARPS = 4


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, NB, C, H, W, OC, OH, OW, stride, padding,
                  KH: tl.constexpr, KW: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_p = offs_p < NB * OH * OW
    mask_c = offs_c < OC
    
    # Compute output coordinates
    ow = offs_p % OW
    t = offs_p // OW
    oh = t % OH
    n = t // OH
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for c in range(0, C):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh * stride - padding + kh
                iw = ow * stride - padding + kw
                mask_h = (ih >= 0) & (ih < H)
                mask_w = (iw >= 0) & (iw < W)
                mask = mask_p & mask_h & mask_w
                
                # Load input data
                x = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=mask, other=0.0)
                
                # Load weights
                w = tl.load(w_ptr + (offs_c * C * KH * KW + c * KH * KW + kh * KW + kw), mask=mask_c, other=0.0)
                
                # Accumulate
                acc += x[:, None] * w[None, :]
    
    # Add bias
    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    acc += bias[None, :]
    
    # Store output
    y_off = ((n[:, None] * OC + offs_c[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    tl.store(y_ptr + y_off, acc, mask=mask_p[:, None] & mask_c[None, :])


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        x = x.contiguous()
        NB, C, H, W = x.shape
        conv = self.conv1
        OC, _, KH, KW = conv.weight.shape
        stride, padding = conv.stride[0], conv.padding[0]
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1
        y = torch.empty((NB, OC, OH, OW), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(NB * OH * OW, BLOCK_P), triton.cdiv(OC, BLOCK_C))
        conv2d_kernel[grid](x, conv.weight, conv.bias, y, NB, C, H, W, OC, OH, OW, stride, padding,
                            KH=KH, KW=KW, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C, num_warps=NUM_WARPS)
        return y
