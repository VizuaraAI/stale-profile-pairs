"""Optimized Triton kernel for KernelBench level 2 task 12."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
NUM_WARPS, NUM_STAGES = 8, 2


@triton.jit
def linear_act_kernel(x_ptr, w_ptr, bias_ptr, y_ptr, M, N, K, multiplier, negative_slope,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles
        a = tl.load(x_ptr + offs_m[:, None] * K + (k * BLOCK_K + offs_k)[None, :], 
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + (k * BLOCK_K + offs_k)[:, None], 
                    mask=offs_n[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    
    # Apply bias, multiplier, and activation
    acc = (acc + bias[None, :]) * multiplier
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Store result
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, 
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.gemm.weight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M * N, BLOCK_M * BLOCK_N),)
        linear_act_kernel[grid](x, self.gemm.weight, self.gemm.bias, y, M, N, K, self.multiplier, self.negative_slope,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return y
