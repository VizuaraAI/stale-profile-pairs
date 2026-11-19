# 21 more pairs, earlier in the trajectories, blinded

Each folder is one checkpoint from the second branch test (topic 31, Exp07). Unlike the 30 pairs you labelled
before, which sat at or near the best kernel of their trajectory, each of these checkpoints is a kernel at least 10%
slower than the best one found earlier in its trajectory (rule C accepts slower edits). The pair IDs are random and none
of them was used in the first package, the folders are shuffled, and nothing here says how any branch did.

In each folder:

- `old_kernel.py`: the kernel three edits back, as before. Its profile is the stale one.
- `current_kernel.py`: the kernel at the checkpoint.
- `diff.patch`: unified diff from old to current.
- `profile_old.txt` and `profile_current.txt`: the profile summaries the model was shown, built from Nsight Compute on an
  NVIDIA L40S (the same GPU as the first package; that package's README said H100, which was a mistake). Each lists grid
  and block size, duration, compute and memory throughput (percent of peak), L1 and L2 hit rates, occupancy with the
  block limits from registers (registers per thread) and shared memory (bytes per block), the largest stall reasons, and
  the hottest source lines.
- `meta.json`: the KernelBench task name, the number of edits between the two kernels, and the kernels launched per call.

`labels_template.csv` has one row per pair, with the same columns as before, so the same labelling rules apply. Please
fill it in and send it back. We hold the key and join it to the outcomes only after the labels are fixed.

The raw Nsight Compute reports are on our side; ask if a pair needs them.
