# Pi0.5 worker throughput: LIBERO 0906

Date: 2026-09-13. Physical GPU 1 hosts the policy; GPU 0 hosts simulation and the concurrently running PointPolicy evaluation. Formal Pi0.5 evaluation was suspended with SIGSTOP and resumed with SIGCONT after all four runs; live step advancement was verified.

Protocol: same loaded 29999 checkpoint, task code ID 0, initial states 0/1/2, seed 42, ES switching, 20 Hz, 300 policy steps per episode, 900 steps per run, videos enabled. Order: 1, 3, 3, 1 workers. All runs exited successfully. These are timing runs, not formal SR/PR results.

| Run | Workers | Actual steps | Total seconds | Active seconds | Mean request RTT (ms) | Mean environment step (ms) |
|---|---|---|---|---|---|---|
| 1 | 1 | 900 | 47.58 | 41.31 | 90.60 | 19.26 |
| 2 | 3 | 900 | 33.51 | 23.32 | 95.04 | 54.49 |
| 3 | 3 | 900 | 39.97 | 29.84 | 108.99 | 72.62 |
| 4 | 1 | 900 | 49.69 | 42.15 | 86.16 | 20.64 |

Mean total runtime: 1 worker 48.64s; 3 workers 36.74s. Aggregate throughput: 18.51 vs 24.49 steps/s. Speedup 1.324x; time reduction 24.45%.

Active interval (first policy-step log to result-save announcement, including intervening resets and video writes): 41.73s vs 26.58s, speedup 1.570x.

The bridge serializes inference under InferenceServer.inference_lock; workers overlap simulation and inference, not batched model execution. Measured environment-step latency rises with concurrency. Two repeats and short rollouts are not sufficient to establish an exact full-suite speedup. PointPolicy remained active, so the measurement represents current shared-machine conditions.
