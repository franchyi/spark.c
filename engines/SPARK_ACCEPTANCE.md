# First Spark acceptance pass

Run this only on the GB10 host. Keep one heavyweight model active at a time and
save the command output, container/image inspection, `nvidia-smi`,
`/proc/meminfo`, and thermal clocks beside each result.

## 1. Verify locks without loading a model

```bash
./engines/bin/spark-engine list
./engines/bin/spark-engine qwen38-27b provenance
./engines/bin/spark-engine qwen38-flash-next provenance
./engines/bin/spark-engine glm53-q2 provenance
```

## 2. Qwen3.8-27B resident oracle

```bash
./engines/bin/spark-engine qwen38-27b build
QWEN27_PROFILE=resident-mtp ./engines/bin/spark-engine qwen38-27b serve
./engines/bin/spark-engine qwen38-27b smoke
./engines/bin/spark-engine qwen38-27b bench
./engines/bin/spark-engine qwen38-27b stop
```

Record the resolved Linux/arm64 image digest, cold startup, post-load available
memory, MTP acceptance, code/prose decode, and the upstream TTFT prefill probe.
Do not enable DFlash2 until this resident baseline passes.

## 3. Qwen3.8 Flash-Next vLLM oracle

If the checkpoint is already present in the Hugging Face cache, omit download.

```bash
make -C engines/qwen38-flash-next oracle-build
make -C engines/qwen38-flash-next oracle-download
make -C engines/qwen38-flash-next oracle-serve
make -C engines/qwen38-flash-next oracle-smoke
make -C engines/qwen38-flash-next oracle-stop
```

Keep `GPU_MEM=0.85`, BF16 QSA KV, MTP=2, and eight sequence slots for the first
run. Capture cold and warm PLE behavior separately.

## 4. Qwen3.8 Flash-Next native gate

`serve` stays in the foreground; run `smoke`, `parity`, and `bench` from a
second SSH shell.

```bash
./engines/bin/spark-engine qwen38-flash-next build
make -C engines/qwen38-flash-next parity
./engines/bin/spark-engine qwen38-flash-next serve
./engines/bin/spark-engine qwen38-flash-next smoke
./engines/bin/spark-engine qwen38-flash-next bench
./engines/bin/spark-engine qwen38-flash-next stop
```

The parity adapter loads the native model, then the oracle sequentially, and
stops its exact oracle container after comparison. Promotion requires exact
greedy token continuation, not just a successful kernel launch or plausible
text.

## 5. GLM-5.3 Q2 regression

Run this last because its resident model needs roughly 110 GiB available.
Its `serve` target also stays in the foreground, so use a second shell for the
remaining commands.

```bash
./engines/bin/spark-engine glm53-q2 build
GLM53_Q2_VERIFY_SHA=1 ./engines/bin/spark-engine glm53-q2 serve
./engines/bin/spark-engine glm53-q2 smoke
./engines/bin/spark-engine glm53-q2 bench
./engines/bin/spark-engine glm53-q2 stop
```

The regression floor is the already measured 523.02 prefill / 14.52 generation
tok/s for the exact 2048/128 ds4 workload. IQ3 is outside this pass.
