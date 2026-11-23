# Per-pair outcomes, batches 1 and 2

For your re-profiling rule. One row per pair you have already labelled: 30 from batch 1 and 21 from batch 2, under the same P### ids as your label files. Batch 3 is not here, and none of its ids appear, so you can build and freeze the rule on these 51 pairs and we test it on batch 3 afterwards.

Columns:

- `batch`, `pair`, `task`: the package, your pair id and the KernelBench task (L1 = level 1, L2 = level 2, then the task number).
- `g_fresh_A`, `g_fresh_B`, `g_stale_A`, `g_stale_B`, `g_none_A`: the gain of the next turn's best kernel in each branch, as g = max(0, -log of the timed speed ratio against the current kernel), so 0 means no improvement and 0.1 is about 10% faster. Fresh and stale ran two rounds (A and B); the no-profile branch ran one.
- `fresh_minus_none`, `fresh_minus_stale`, `stale_minus_none`: the per-pair differences used in our results, the mean over rounds of one branch minus the other.

Batch 1 is the Exp05 and Exp06 checkpoints (near the best kernel found so far); batch 2 is Exp07 (at least 1.10 times slower than the best so far). They are different kinds of checkpoint, so it is worth checking a rule on each batch separately as well as pooled.

Please write down the rule, and how you will score it against always, never and every k turns, before you ask for the batch 3 outcomes.
