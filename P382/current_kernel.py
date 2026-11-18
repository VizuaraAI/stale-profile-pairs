"""Optimized Triton kernel for square matrix multiplication."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
NUM_WARPS, NUM_STAGES = 8, 3


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        N = B.shape[1]
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return C
