"""Hand-written starting kernel for KernelBench level 1 task 40, layer normalization: one kernel for the statistics of
each sample, one to normalize."""
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

STATS_BLOCK, APPLY_BLOCK = 1024, 1024


@triton.jit
def ln_stats_kernel(x_ptr, mean_ptr, rstd_ptr, L, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * L
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(base + offs, mask=offs < L, other=0.0)
        acc1 += v
        acc2 += v * v
    mean = tl.sum(acc1, axis=0) / L
    var = tl.sum(acc2, axis=0) / L - mean * mean
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, 1.0 / tl.sqrt(var + eps))


@triton.jit
def ln_apply_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, L, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x = tl.load(x_ptr + row * L + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * L + offs, (x - mean) * rstd * w + b, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = math.prod(self.ln.normalized_shape)
        rows = x.numel() // L
        x = x.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        ln_stats_kernel[(rows,)](x, mean, rstd, L, self.ln.eps, BLOCK=STATS_BLOCK)
        ln_apply_kernel[(rows, triton.cdiv(L, APPLY_BLOCK))](x, y, self.ln.weight, self.ln.bias, mean, rstd, L, BLOCK=APPLY_BLOCK)
        return y
