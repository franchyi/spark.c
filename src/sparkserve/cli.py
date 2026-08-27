from __future__ import annotations

import argparse
import json
import random
import time

from sparkserve.manifest import fetch_manifest
from sparkserve.model_lock import load_model_lock, verify_model_files
from sparkserve.planner import PROFILES, plan_memory
from sparkserve.ple_store import PleIndex, PleReader, build_ple_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparkserve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="check whether a model memory plan fits")
    plan.add_argument("model", choices=sorted(PROFILES))
    plan.add_argument("--system-gib", type=float, default=121.0)
    plan.add_argument("--sparse-cache-gib", type=float, default=2.0)
    plan.add_argument("--kv-cache-gib", type=float, default=8.0)
    plan.add_argument("--runtime-gib", type=float, default=12.0)
    plan.add_argument("--safety-gib", type=float, default=8.0)
    plan.add_argument("--json", action="store_true")

    manifest = subparsers.add_parser(
        "manifest", help="summarize a Hugging Face checkpoint without downloading weights"
    )
    manifest.add_argument("repo")
    manifest.add_argument("--endpoint", default="https://hf-mirror.com")
    manifest.add_argument("--json", action="store_true")

    lock_check = subparsers.add_parser(
        "lock-check", help="validate an immutable model lock and optionally its local files"
    )
    lock_check.add_argument("lock", nargs="?", default="models.lock.json")
    lock_check.add_argument("--model")
    lock_check.add_argument("--model-root")
    lock_check.add_argument("--json", action="store_true")

    ple_index = subparsers.add_parser(
        "ple-index",
        help="index exact FP8 PLE rows inside existing safetensors without copying them",
    )
    ple_index.add_argument("model_root")
    ple_index.add_argument("output")
    ple_index.add_argument("--revision")
    ple_index.add_argument("--json", action="store_true")

    ple_bench = subparsers.add_parser(
        "ple-bench", help="benchmark bounded random PLE row reads from NVMe"
    )
    ple_bench.add_argument("index")
    ple_bench.add_argument("--model-root", required=True)
    ple_bench.add_argument("--cache-mib", type=int, default=512)
    ple_bench.add_argument("--tokens", type=int, default=4096)
    ple_bench.add_argument("--passes", type=int, default=2)
    ple_bench.add_argument("--workers", type=int, default=16)
    ple_bench.add_argument(
        "--batch-tokens",
        type=int,
        default=512,
        help="tokens per I/O submission; zero submits the full prefill together",
    )
    ple_bench.add_argument("--seed", type=int, default=1)
    ple_bench.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "lock-check":
        lock = load_model_lock(args.lock)
        selected = lock.models if args.model is None else (lock.model(args.model),)
        if args.model_root and len(selected) != 1:
            raise SystemExit("--model-root requires --model")
        if args.model_root:
            verify_model_files(selected[0], args.model_root)
        result = {
            "schema_version": lock.schema_version,
            "mirror": lock.mirror,
            "models": [
                {
                    "id": model.id,
                    "revision": model.revision,
                    "format": model.format,
                    "inventory": model.inventory,
                    "files": len(model.files),
                    "checkpoint_bytes": model.checkpoint_bytes,
                    "files_verified": bool(args.model_root),
                }
                for model in selected
            ],
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for model in result["models"]:
                suffix = " (files verified)" if model["files_verified"] else ""
                print(f"{model['id']} @ {model['revision']}{suffix}")
                print(f"  format            {model['format']}")
                print(f"  inventory         {model['inventory']}")
                print(f"  locked files      {model['files']:12d}")
                print(f"  checkpoint bytes  {model['checkpoint_bytes']:12d}")
        return

    if args.command == "manifest":
        summary = fetch_manifest(args.repo, endpoint=args.endpoint)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(f"{summary.repo} @ {summary.revision or 'unknown'}")
            print(f"  files             {summary.file_count:12d}")
            print(f"  routed experts    {summary.expert_bytes / 10**9:12.3f} GB")
            print(f"  sparse PLE        {summary.ple_bytes / 10**9:12.3f} GB")
            print(f"  other/BF16/meta   {summary.other_bytes / 10**9:12.3f} GB")
            print(f"  total             {summary.total_bytes / 10**9:12.3f} GB")
        return

    if args.command == "ple-index":
        index = build_ple_index(
            args.model_root, args.output, revision=args.revision
        )
        summary = {
            "index": args.output,
            "shards": len(index.shards),
            "rows": index.total_rows,
            "row_bytes": index.row_bytes,
            "payload_gib": round(index.total_rows * index.row_bytes / 1024**3, 3),
            "scale_bf16_bits": f"0x{index.scale_bf16_bits:04x}",
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"indexed {summary['shards']} PLE shards without copying weights")
            print(f"  rows              {summary['rows']:12d}")
            print(f"  FP8 payload       {summary['payload_gib']:12.3f} GiB")
            print(f"  row width         {summary['row_bytes']:12d} bytes")
            print(f"  BF16 scale        {summary['scale_bf16_bits']:>12}")
            print(f"  index             {summary['index']}")
        return

    if args.command == "ple-bench":
        if (
            args.cache_mib <= 0
            or args.tokens <= 0
            or args.passes <= 0
            or args.workers <= 0
            or args.batch_tokens < 0
        ):
            raise SystemExit(
                "cache, token, pass, and worker counts must be positive; "
                "batch tokens must be non-negative"
            )
        index = PleIndex.read(args.index)
        cache_pages = args.cache_mib * 1024**2 // index.page_bytes
        generator = random.Random(args.seed)
        # Flash-Next performs sixteen 160-byte PLE lookups for every token.
        rows = [generator.randrange(index.total_rows) for _ in range(args.tokens * 16)]
        batch_tokens = args.batch_tokens or args.tokens
        batch_rows = batch_tokens * 16
        latencies: list[float] = []
        started = time.perf_counter()
        with PleReader(
            index,
            args.model_root,
            cache_pages=max(1, cache_pages),
            workers=args.workers,
        ) as reader:
            for _ in range(args.passes):
                for start in range(0, len(rows), batch_rows):
                    batch_started = time.perf_counter()
                    reader.fetch_rows(rows[start : start + batch_rows])
                    latencies.append(time.perf_counter() - batch_started)
            elapsed = time.perf_counter() - started
            stats = reader.cache.stats.to_dict()
        tokens = args.tokens * args.passes
        ordered_latencies = sorted(latencies)
        p50 = ordered_latencies[len(ordered_latencies) // 2]
        p99 = ordered_latencies[min(len(ordered_latencies) - 1, int(len(ordered_latencies) * 0.99))]
        result = {
            "tokens": tokens,
            "rows": len(rows) * args.passes,
            "elapsed_seconds": round(elapsed, 6),
            "tokens_per_second": round(tokens / elapsed, 2),
            "useful_mib_per_second": round(
                tokens * 16 * index.row_bytes / elapsed / 1024**2, 2
            ),
            "cache_mib": args.cache_mib,
            "workers": args.workers,
            "batch_tokens": batch_tokens,
            "batches": len(latencies),
            "batch_latency_p50_ms": round(p50 * 1000, 3),
            "batch_latency_p99_ms": round(p99 * 1000, 3),
            **stats,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("exact FP8 PLE NVMe benchmark")
            print(f"  throughput        {result['tokens_per_second']:12.2f} tokens/s")
            print(f"  useful bandwidth  {result['useful_mib_per_second']:12.2f} MiB/s")
            print(f"  page reads        {result['page_reads']:12d}")
            print(f"  cache hit rate    {result['hit_rate'] * 100:11.2f}%")
            print(f"  batch p50         {result['batch_latency_p50_ms']:12.3f} ms")
            print(f"  batch p99         {result['batch_latency_p99_ms']:12.3f} ms")
            print(f"  bytes read        {result['bytes_read'] / 1024**2:12.2f} MiB")
        return

    profile = PROFILES[args.model]
    plan = plan_memory(
        profile,
        system_gib=args.system_gib,
        sparse_cache_gib=args.sparse_cache_gib,
        kv_cache_gib=args.kv_cache_gib,
        runtime_gib=args.runtime_gib,
        safety_gib=args.safety_gib,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    state = "FIT" if plan.fits else "DOES NOT FIT"
    print(f"{state}: {plan.model}")
    print(f"  resident weights  {plan.resident_weights_gib:7.2f} GiB")
    print(f"  sparse row cache  {plan.sparse_cache_gib:7.2f} GiB")
    print(f"  KV cache          {plan.kv_cache_gib:7.2f} GiB")
    print(f"  runtime           {plan.runtime_gib:7.2f} GiB")
    print(f"  safety reserve    {plan.safety_gib:7.2f} GiB")
    print(f"  required          {plan.required_gib:7.2f} GiB")
    print(f"  headroom          {plan.headroom_gib:7.2f} GiB")


if __name__ == "__main__":
    main()
