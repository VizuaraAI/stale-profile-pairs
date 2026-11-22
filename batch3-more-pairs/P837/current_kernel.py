"""Optimized Triton kernel for batched matrix multiplication."""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
NUM_WARPS, NUM_STAGES = 4, 3


@triton.jit
def bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bid = tl.program_id(2)
    a_ptr += bid * M * K
    b_ptr += bid * K * N
    c_ptr += bid * M * N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        batch, M, K = A.shape
        N = B.shape[2]
        C = torch.empty((batch, M, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch)
        bmm_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return C
