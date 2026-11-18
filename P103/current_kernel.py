"""Optimized Triton kernel for KernelBench level 2 task 1: a 3 by 3 convolution, ReLU and a bias add."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_P, BLOCK_C = 256, 64
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
    ow = offs_p % OW
    t = offs_p // OW
    oh = t % OH
    n = t // OH

    # Precompute indices for better memory access
    R = C * KH * KW
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    
    # Process 8 elements per iteration to reduce loop overhead
    for r in range(0, R, 8):  # Process 8 elements per iteration
        # Load weights in chunks
        wv = tl.load(w_ptr + offs_c * R + r, mask=mask_c, other=0.0)
        wv1 = tl.load(w_ptr + offs_c * R + r + 1, mask=mask_c, other=0.0)
        wv2 = tl.load(w_ptr + offs_c * R + r + 2, mask=mask_c, other=0.0)
        wv3 = tl.load(w_ptr + offs_c * R + r + 3, mask=mask_c, other=0.0)
        wv4 = tl.load(w_ptr + offs_c * R + r + 4, mask=mask_c, other=0.0)
        wv5 = tl.load(w_ptr + offs_c * R + r + 5, mask=mask_c, other=0.0)
        wv6 = tl.load(w_ptr + offs_c * R + r + 6, mask=mask_c, other=0.0)
        wv7 = tl.load(w_ptr + offs_c * R + r + 7, mask=mask_c, other=0.0)

        # Load input values efficiently
        c = r // (KH * KW)
        rem = r % (KH * KW)
        ih = oh + rem // KW
        iw = ow + rem % KW
        xv = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=mask_p, other=0.0)

        c1 = (r + 1) // (KH * KW)
        rem1 = (r + 1) % (KH * KW)
        ih1 = oh + rem1 // KW
        iw1 = ow + rem1 % KW
        xv1 = tl.load(x_ptr + ((n * C + c1) * H + ih1) * W + iw1, mask=mask_p, other=0.0)

        c2 = (r + 2) // (KH * KW)
        rem2 = (r + 2) % (KH * KW)
        ih2 = oh + rem2 // KW
        iw2 = ow + rem2 % KW
        xv2 = tl.load(x_ptr + ((n * C + c2) * H + ih2) * W + iw2, mask=mask_p, other=0.0)

        c3 = (r + 3) // (KH * KW)
        rem3 = (r + 3) % (KH * KW)
        ih3 = oh + rem3 // KW
        iw3 = ow + rem3 % KW
        xv3 = tl.load(x_ptr + ((n * C + c3) * H + ih3) * W + iw3, mask=mask_p, other=0.0)

        c4 = (r + 4) // (KH * KW)
        rem4 = (r + 4) % (KH * KW)
        ih4 = oh + rem4 // KW
        iw4 = ow + rem4 % KW
        xv4 = tl.load(x_ptr + ((n * C + c4) * H + ih4) * W + iw4, mask=mask_p, other=0.0)

        c5 = (r + 5) // (KH * KW)
        rem5 = (r + 5) % (KH * KW)
        ih5 = oh + rem5 // KW
        iw5 = ow + rem5 % KW
        xv5 = tl.load(x_ptr + ((n * C + c5) * H + ih5) * W + iw5, mask=mask_p, other=0.0)

        c6 = (r + 6) // (KH * KW)
        rem6 = (r + 6) % (KH * KW)
        ih6 = oh + rem6 // KW
        iw6 = ow + rem6 % KW
        xv6 = tl.load(x_ptr + ((n * C + c6) * H + ih6) * W + iw6, mask=mask_p, other=0.0)

        c7 = (r + 7) // (KH * KW)
        rem7 = (r + 7) % (KH * KW)
        ih7 = oh + rem7 // KW
        iw7 = ow + rem7 % KW
        xv7 = tl.load(x_ptr + ((n * C + c7) * H + ih7) * W + iw7, mask=mask_p, other=0.0)

        # Accumulate
        acc += xv[:, None] * wv[None, :]
        acc += xv1[:, None] * wv1[None, :]
        acc += xv2[:, None] * wv2[None, :]
        acc += xv3[:, None] * wv3[None, :]
        acc += xv4[:, None] * wv4[None, :]
        acc += xv5[:, None] * wv5[None, :]
        acc += xv6[:, None] * wv6[None, :]
        acc += xv7[:, None] * wv7[None, :]

    cb = tl.load(cb_ptr + offs_c, mask=mask_c, other=0.0)
    b = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    acc = tl.maximum(acc + cb[None, :], 0.0) + b[None, :]
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
