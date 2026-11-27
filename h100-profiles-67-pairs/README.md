# H100 profiles for the 67 labelled pairs (batches 1 to 3)

Same pair ids and files as the batch 1, 2 and 3 packages, with the Nsight Compute summaries taken on one NVIDIA H100
80GB HBM3 (driver 620.70.06) instead of the L40S: `profile_old_h100.txt` and `profile_current_h100.txt`. The command
(`ncu --set full`, one warm-up call skipped, then the recorded call) and the summary format are those of the earlier
packages. Every kernel ran correctly on the H100 before it was profiled.

No outcomes, labels or branch results are included.
