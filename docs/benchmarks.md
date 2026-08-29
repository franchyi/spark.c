# Spark baselines

All values below are measurements already recorded on the project Spark. They
are not interchangeable: checkpoint, prompt, generation length, MTP, and
concurrency must match before using a row as a speed ratio.

| Implementation | Workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| native Qwen3.8-27B MVP | 13-token prompt, one output token | not a valid throughput test | about 9.66 tok/s staged decode |
| ds4 GLM-5.3 Q2 | ds4 `promessi_sposi.txt`, 2048/128 | 523.02 tok/s | 14.52 tok/s |
| ds4 repository GB10 row | upstream workload/model attribution is incomplete | 825.76 tok/s | 18.05 tok/s |

The native Qwen27 result is a functional decode-first baseline. Its prompt is
currently serialized through the one-token ABI, so it must not be compared to
MiaAI/SGLang's chunked prefill. The next performance gate is a true batched
prefill followed by the same decode workload on both servers.

The pinned MiaAI/SGLang Qwen27 recipe reports workload-dependent generation
rates in its own documentation (including MTP and DFlash2 variants), but a
same-command local comparison has not yet been completed. Those upstream rows
remain targets, not measurements of Spark.C.

Flash-Next reference work has demonstrated that the complete checkpoint fits on
one GB10 when PLE stays file-backed, with roughly 23–50 single-stream decode
tok/s depending on content/speculation and about 95 aggregate tok/s at
concurrency four. Spark.C still needs its own end-to-end native row.

`make <model>-bench` must print separate `prefill_tok_s` and
`generation_tok_s` values plus checkpoint revision, prompt/output tokens,
concurrency, and speculation mode. A missing metric is an incomplete benchmark,
not zero and not an estimate.
