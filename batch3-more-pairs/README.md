# 16 more pairs, blinded (third package)

Each folder is one checkpoint from the third branch test (topic 31, Exp08). Fifteen of the sixteen come from
trajectories that the earlier packages did not use, and one task is new to you (level2__29_Matmul_Mish_Mish). The
other tasks are the matmul, batched matmul, LayerNorm and GEMM tasks you have seen, now with three or four pairs each,
so that the split between pairs where the bottleneck changed and pairs where it did not can be checked on new pairs
within each task. The pair IDs are random, none of them appears in
either earlier package, the folders are shuffled, and nothing here says how any branch did.

10 of the 16 checkpoints are of the same kind as the second package: a kernel at least 10% slower than the
best one found earlier in its trajectory, with the old kernel three edits back. The other 6 are one or two
edits after the hand-written starting kernel, and their old kernel is that starting kernel. `meta.json` gives the
number of edits between the two kernels for every pair.

In each folder:

- `old_kernel.py`: the earlier kernel whose profile is the stale one.
- `current_kernel.py`: the kernel at the checkpoint.
- `diff.patch`: unified diff from old to current.
- `profile_old.txt` and `profile_current.txt`: the profile summaries the model was shown, built from Nsight Compute on an
  NVIDIA L40S, the same GPU as both earlier packages. Each lists grid and block size, duration, compute and memory
  throughput (percent of peak), L1 and L2 hit rates, occupancy with the block limits from registers (registers per
  thread) and shared memory (bytes per block), the largest stall reasons, and the hottest source lines.
- `meta.json`: the KernelBench task name, the number of edits between the two kernels, and the kernels launched per call.

`labels_template.csv` has one row per pair, with the same columns as before, so the same frozen labelling rules apply.
If you also fill in the per-kernel columns you added in batch 2, add them at the end as before. We hold the key and
join it to the outcomes only after the labels are fixed. The comparison we will run on your labels was written down
and hashed before this run started.

The raw Nsight Compute reports are on our side; ask if a pair needs them.
