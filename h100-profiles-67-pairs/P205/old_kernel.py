"""Hand-written starting kernel for KernelBench level 2 task 29: linear layer, then Mish twice, in one kernel."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
NUM_WARPS, NUM_STAGES = 4, 1


@triton.jit
def _mish(x):
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
    t = 1.0 - 2.0 / (tl.exp(2.0 * sp) + 1.0)
    return x * t


@triton.jit
def linear_mish_kernel(x_ptr, w_ptr, bias_ptr, y_ptr, M, N, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * K + cols[None, :], mask=(offs_m[:, None] < M) & (cols[None, :] < K), other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + cols[:, None], mask=(cols[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = _mish(_mish(acc + bias[None, :]))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.linear.weight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_mish_kernel[grid](x, self.linear.weight, self.linear.bias, y, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                 BLOCK_K=BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return y
