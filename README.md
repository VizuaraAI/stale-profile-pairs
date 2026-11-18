# 30 stale-profile pairs, blinded

Each folder is one checkpoint from the stale-against-fresh test (topic 31, Exp05). The pair IDs are random, the folders are shuffled, and nothing here says which branch produced the faster next kernel.

In each folder:

- `old_kernel.py`: the kernel three edits back. Its profile is the stale one.
- `current_kernel.py`: the kernel at the checkpoint.
- `diff.patch`: unified diff from old to current.
- `profile_old.txt` and `profile_current.txt`: the profile summaries the model was shown, built from Nsight Compute on an H100. Each lists grid and block size, duration, compute and memory throughput (percent of peak), L1 and L2 hit rates, occupancy with the block limits from registers (registers per thread) and shared memory (bytes per block), the largest stall reasons, and the hottest source lines.
- `meta.json`: the KernelBench task name and the kernels launched per call.

`labels_template.csv` has one row per pair. Please fill it in and send it back. We hold the key and join it to the outcomes only after the labels are fixed.

The raw Nsight Compute reports are on our side; ask if a pair needs them.
