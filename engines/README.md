# Model-specific engines

SparkServe is three small products, not one general-purpose model framework.
Each directory owns one model graph, its launch policy, its upstream oracle, and
its acceptance benchmark. The directories intentionally do not share model
traits, tensor registries, or kernel dispatch code.

| Engine | Shipping path | Reference path | Default port |
| --- | --- | --- | --- |
| `qwen38-27b` | pinned SGLang recipe, then native extraction | MiaAI-Lab recipe | `8888` |
| `qwen38-flash-next` | SparkServe Rust/CUDA native path | blazux vLLM mmap recipe | `8020` native, `8000` oracle |
| `glm53-q2` | pinned ds4 C/CUDA server | the same pristine ds4 build | `8010` |

The commonality is deliberately limited to the operator contract in
[`CONTRACT.md`](CONTRACT.md): the same directory shape, Make targets,
environment vocabulary, OpenAI-compatible endpoints, benchmark output, and
source-locking rules. Arithmetic and scheduling remain local to each capsule.

Use the tiny dispatcher from the repository root:

```bash
./engines/bin/spark-engine list
./engines/bin/spark-engine glm53-q2 build
./engines/bin/spark-engine glm53-q2 serve
./engines/bin/spark-engine qwen38-flash-next smoke
```

It only selects a directory and invokes `make`; it is not a runtime framework.
Every operation can also be run directly with `make -C engines/ENGINE TARGET`.
The exact safe run order for the target host is in
[`SPARK_ACCEPTANCE.md`](SPARK_ACCEPTANCE.md).

## First-version rule

Do not move proven code merely to make the tree look uniform. The first version
keeps existing native sources and build scripts in place and wraps them from the
engine capsule. A source is moved into a capsule only after that exact path has
passed its Spark acceptance run. This keeps the current Qwen work and the
validated GLM binary reproducible while the boundary is introduced.
