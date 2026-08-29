# Q27 batched prefill MLP

This fixed model capsule joins the validated M=128/M=512 NVFP4 primitives into
one allocation-free dense MLP call: merged gate/up projection, BF16-rounded
`SiLU(gate) * up`, then down projection. It accepts only the Qwen3.8-27B
5120/17408 dimensions and requires the load-time contiguous gate/up weights,
scales, and common alpha already validated by the strict sidecar.

The activation arithmetic matches the accepted M=1 capsule. Packed activation
and scale scratch is reused between the two sequential projections. The ABI
contains no scheduler, allocator, JIT, framework tensor, or M=1 fallback.

Promotion requires a Spark build, a real-checkpoint M=128 full-MLP fixture, and
a timing that includes merged gate/up, activation, activation quantization, and
down. Projection-only microbenchmarks are evidence for its components but are
not a full-MLP performance claim.

The 2026-08-29 Spark real layer-0 fixture passed. The complete merged
gate/up-to-SiLU-to-down path was byte-exact to the accepted M=1 chain across
1,310,720 BF16 output bytes and measured 1.26678 ms per M=128 tile
(9.89669 microseconds per physical row).

The bounded M=512 timing measured 4.86579 ms total, or 9.5035 microseconds per
physical row. That small per-row gain does not justify delaying the first
fixed-M128 joined model path.
