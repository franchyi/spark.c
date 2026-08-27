# SGLang Qwen4 PLE arithmetic oracle

SparkServe's PLE gather contract is frozen against SGLang PR 36497 at commit
`7c66045d71f067c1c5da2b85baad3c47d9a19cb7`:

- repository: `https://github.com/sgl-project/sglang`
- source: `python/sglang/srt/models/qwen4_exp.py`
- source SHA-256:
  `f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8`
- license: Apache-2.0

The running Spark oracle contains the identical source bytes. Its Triton kernel
loads one FP8-E4M3 row, converts each value to BF16, and writes BF16 output; the
model then multiplies by the checkpoint's BF16 `weight_scale` (`0x3951` for the
locked Qwen checkpoint).

SparkServe does not ship Triton or copy the model module. A compact raw CUDA
kernel reproduces that numerical contract over two-fragment row descriptors
emitted by the Rust fixed-slab scheduler. Private fixtures contain selected
checkpoint rows and SGLang outputs; no model payload is committed here.
