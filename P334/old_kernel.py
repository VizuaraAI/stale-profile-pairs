"""Hand-written starting kernel for KernelBench level 1 task 50: a 2D convolution, 3 to 96 channels, 11 by 11, stride 4,
padding 2, computed directly in one kernel."""
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
    ow = offs_p % OW
    t = offs_p // OW
    oh = t % OH
    n = t // OH
    ih0 = oh * stride - padding
    iw0 = ow * stride - padding
    R = C * KH * KW
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    for r in range(0, R):
        c = r // (KH * KW)
        rem = r % (KH * KW)
        ih = ih0 + rem // KW
        iw = iw0 + rem % KW
        m = mask_p & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        xv = tl.load(x_ptr + ((n * C + c) * H + ih) * W + iw, mask=m, other=0.0)
        wv = tl.load(w_ptr + offs_c * R + r, mask=mask_c, other=0.0)
        acc += xv[:, None] * wv[None, :]
    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    acc += bias[None, :]
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
