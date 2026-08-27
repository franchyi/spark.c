from __future__ import annotations

import argparse
import json

from sparkserve.manifest import fetch_manifest
from sparkserve.planner import PROFILES, plan_memory


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
