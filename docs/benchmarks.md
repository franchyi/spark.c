# Spark measurements

These are retained one-shot measurements from the same project Spark. A row is
comparable only when checkpoint, prompt/output tokens, concurrency, draft, and
cache policy match.

| Engine | Workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| Mia/SGLang Qwen3.8-27B DFlash2 | 12,617 prompt; 47 → 256 decode | 826.78 tok/s | 35.30 tok/s |
| Spark.C Qwen3.8-27B, final prefill | c427 prep/recurrence, mixed tail | **926.48 tok/s** | — |
| Spark.C Qwen3.8-27B, true-M8 DFlash2 | matched 47 → 256 decode | — | **37.76 tok/s** |
| Spark.C Flash-Next, before batched QSA | warm 66-token prompt | 43.3-44.5 tok/s | 9.7-10.9 tok/s |
| Spark.C Flash-Next, batched QSA | warm 57-token prompt / 2 output | **74.80 tok/s** | **10.87 tok/s** |
| Spark.C GLM-5.3 Q2 | 2048 prompt / 128 output | 523.02 tok/s | 14.52 tok/s |

Qwen3.8-27B first measured 537.42 prefill and 17.07 DFlash2 decode tok/s. The
fixed c427 preparation/recurrence plus `8192 + 4096 + 512` physical schedule
raised prefill to 926.48 tok/s, 12.1% above the retained Mia row. GDN state
journaling and true-M8 GDN/NVFP4 verification raised decode to 37.76 tok/s,
7.0% above Mia. Both final values are single canaries, not repeated statistical
claims. The prefill response hash matched; the decode hash still differs, so
token-by-token verifier parity remains required before correctness promotion.

The GLM Nsight Systems sample measured 68.57 ms per steady token (14.58 tok/s),
consistent with its 14.52 tok/s baseline. GPU time was led by Q8 aligned dense
(35.7%), Q4 pair (14.9%), BF16 matvec (11.4%), and indexed attention (9.7%).
The paired KDA `f_a/g_a` projection removes eleven launches per token; no long
benchmark was repeated solely to amplify that small launch-level change.

Flash-Next keeps the 63.282-GiB expert sidecar resident with zero expert
loads/copies during requests. Its measured cold sidecar prefault was 129.06 s
and cold first-request BF16 staging was 21.90 s. SGLang-style load-time GDN
QKVZBA merging removes three GEMM launches from each T=1 GDN layer. Batching
QSA input/output projections across the existing T=16 prompt bucket then cut
the retained warm 57-token canary from 1.182 s to 0.762 s while producing the
same two token IDs. These are single canaries, not repeated statistical claims.
