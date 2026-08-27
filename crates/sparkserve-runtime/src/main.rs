use sparkserve_runtime::checkpoint::{CheckpointPlan, load_flash_next_plan};
use sparkserve_runtime::kernel::{DenseNvfp4Spec, DeviceCaps, select_dense_nvfp4_candidate};
use sparkserve_runtime::model::{plan_memory, profile};
use sparkserve_runtime::storage::{ClockPageCache, FilePageSource, PleIndex};
use std::path::Path;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  sparkserve-runtime plan <model> [system-gib]");
    eprintln!("  sparkserve-runtime checkpoint-plan <model-root>");
    eprintln!("  sparkserve-runtime kernel-plan nvfp4-dense <M> <N> <K> [SM]");
    eprintln!("  sparkserve-runtime ple-inspect <index>");
    eprintln!(
        "  sparkserve-runtime ple-bench <index> <model-root> [tokens] [cache-mib] [chunk-tokens] [workers]"
    );
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("plan") => run_memory_plan(args),
        Some("checkpoint-plan") => run_checkpoint_plan(args),
        Some("kernel-plan") => run_kernel_plan(args),
        Some("ple-inspect") => run_ple_inspect(args),
        Some("ple-bench") => run_ple_bench(args),
        _ => usage(),
    }
}

fn run_checkpoint_plan(mut args: impl Iterator<Item = String>) {
    let model_root = args.next().unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }
    let plan = load_flash_next_plan(Path::new(&model_root)).unwrap_or_else(|error| {
        eprintln!("cannot build standalone checkpoint plan: {error}");
        std::process::exit(1);
    });
    print_checkpoint_plan(&plan);
}

fn print_checkpoint_plan(plan: &CheckpointPlan) {
    let config = &plan.config;
    let total = plan.total();
    println!("standalone Qwen3.8 Flash-Next NVFP4 checkpoint");
    println!("  layers             {:12}", config.layers);
    println!("  hidden size        {:12}", config.hidden_size);
    println!(
        "  experts / active   {:8} / {}",
        config.experts, config.experts_per_token
    );
    println!("  checkpoint files   {:12}", plan.files);
    print_tensor_class("resident", plan.resident);
    print_tensor_class("PLE on NVMe", plan.ple_nvme);
    print_tensor_class("MTP deferred", plan.mtp_deferred);
    print_tensor_class("vision ignored", plan.vision_ignored);
    print_tensor_class("indexed total", total);
    println!("  production stack   Rust + C++/CUDA (no SGLang/Python/Torch)");
}

fn print_tensor_class(label: &str, stats: sparkserve_runtime::checkpoint::TensorStats) {
    println!(
        "  {label:<18} {:8} tensors {:12} bytes ({:9.3} GiB)",
        stats.tensors,
        stats.bytes,
        stats.bytes as f64 / 1_073_741_824.0
    );
}

fn run_ple_inspect(mut args: impl Iterator<Item = String>) {
    let path = args.next().unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }
    let payload = std::fs::read(&path).unwrap_or_else(|error| {
        eprintln!("cannot read PLE index {path}: {error}");
        std::process::exit(1);
    });
    let index = PleIndex::decode(&payload).unwrap_or_else(|error| {
        eprintln!("invalid PLE index {path}: {error}");
        std::process::exit(1);
    });
    println!("exact FP8 PLE index");
    println!("  shards       {:12}", index.shards.len());
    println!("  rows         {:12}", index.total_rows);
    println!("  row bytes    {:12}", index.row_bytes);
    println!("  page bytes   {:12}", index.page_bytes);
    println!(
        "  payload GiB  {:12.3}",
        index.total_rows as f64 * index.row_bytes as f64 / 1_073_741_824.0
    );
    println!("  BF16 scale   0x{:04x}", index.scale_bf16_bits);
}

fn run_ple_bench(mut args: impl Iterator<Item = String>) {
    let index_path = args.next().unwrap_or_else(|| usage());
    let model_root = args.next().unwrap_or_else(|| usage());
    let tokens = parse_optional_usize(args.next(), 4096);
    let cache_mib = parse_optional_usize(args.next(), 512);
    let chunk_tokens = parse_optional_usize(args.next(), 512);
    let workers = parse_optional_usize(args.next(), 16);
    if args.next().is_some() || tokens == 0 || cache_mib == 0 || chunk_tokens == 0 || workers == 0 {
        usage();
    }
    let payload = std::fs::read(&index_path).unwrap_or_else(|error| {
        eprintln!("cannot read PLE index {index_path}: {error}");
        std::process::exit(1);
    });
    let index = PleIndex::decode(&payload).unwrap_or_else(|error| {
        eprintln!("invalid PLE index {index_path}: {error}");
        std::process::exit(1);
    });
    let source = FilePageSource::open_with_workers(&index, Path::new(&model_root), workers)
        .unwrap_or_else(|error| {
            eprintln!("cannot open PLE sources: {error}");
            std::process::exit(1);
        });
    let cache_bytes = cache_mib
        .checked_mul(1024 * 1024)
        .unwrap_or_else(|| usage());
    let cache_pages = cache_bytes / index.page_bytes;
    let mut cache = ClockPageCache::new(source, index.page_bytes, cache_pages)
        .expect("validated PLE cache geometry");
    let started = Instant::now();
    let mut generated = 0_usize;
    let mut random = 0x9e37_79b9_7f4a_7c15_u64;
    let mut checksum = 0_u64;
    while generated < tokens {
        let batch_tokens = chunk_tokens.min(tokens - generated);
        let mut rows = Vec::with_capacity(batch_tokens * 16);
        for _ in 0..batch_tokens * 16 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            rows.push(random % index.total_rows);
        }
        let values = cache.fetch_rows(&index, &rows).unwrap_or_else(|error| {
            eprintln!("PLE benchmark failed: {error}");
            std::process::exit(1);
        });
        checksum = values
            .iter()
            .fold(checksum, |sum, value| sum.wrapping_add(u64::from(*value)));
        generated += batch_tokens;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!("exact FP8 PLE Rust benchmark");
    println!(
        "  throughput       {:12.2} tokens/s",
        tokens as f64 / elapsed
    );
    println!("  elapsed          {:12.3} s", elapsed);
    println!("  page reads       {:12}", cache.stats.page_reads);
    println!("  page hits        {:12}", cache.stats.page_hits);
    println!("  evictions        {:12}", cache.stats.evictions);
    println!("  bytes read       {:12}", cache.stats.bytes_read);
    println!("  output checksum  {checksum:012x}");
}

fn run_memory_plan(mut args: impl Iterator<Item = String>) {
    let key = args.next().unwrap_or_else(|| usage());
    let model = profile(&key).unwrap_or_else(|| {
        eprintln!("unknown model: {key}");
        std::process::exit(2);
    });
    let system_gib = args
        .next()
        .map(|value| value.parse::<f64>().unwrap_or_else(|_| usage()))
        .unwrap_or(121.0);
    let plan = plan_memory(model, system_gib, 2.0, 8.0, 12.0, 8.0)
        .expect("static memory plan must be valid");
    println!(
        "{}: {}",
        if plan.fits { "FIT" } else { "DOES NOT FIT" },
        model.key
    );
    println!("  required  {:7.2} GiB", plan.required_gib);
    println!("  headroom  {:7.2} GiB", plan.headroom_gib);
}

fn run_kernel_plan(mut args: impl Iterator<Item = String>) {
    if args.next().as_deref() != Some("nvfp4-dense") {
        usage();
    }
    let m = parse_u64(args.next());
    let n = parse_u64(args.next());
    let k = parse_u64(args.next());
    let sm = args
        .next()
        .map(|value| value.parse::<u32>().unwrap_or_else(|_| usage()))
        .unwrap_or(121);
    if args.next().is_some() {
        usage();
    }

    let spec = DenseNvfp4Spec::native(m, n, k).unwrap_or_else(|error| {
        eprintln!("invalid dense NVFP4 plan: {error}");
        std::process::exit(2);
    });
    let buffers = spec
        .buffer_requirements()
        .expect("validated size calculation");
    let candidate = select_dense_nvfp4_candidate(
        spec,
        DeviceCaps {
            sm,
            supports_fp4_tensor_cores: sm >= 100,
        },
    )
    .unwrap_or_else(|error| {
        eprintln!("no dense NVFP4 plan: {error}");
        std::process::exit(2);
    });

    println!("dense NVFP4: M={m} N={n} K={k} on SM{sm}");
    println!(
        "  native shape      M={} weight-N={} K={} scale-N={}",
        spec.m, spec.padded_n, spec.padded_k, spec.scale_padded_n
    );
    println!("  packed input      {} bytes", buffers.packed_input_bytes);
    println!("  input scales      {} bytes", buffers.input_scale_bytes);
    println!("  packed weight     {} bytes", buffers.packed_weight_bytes);
    println!("  weight scales     {} bytes", buffers.weight_scale_bytes);
    println!("  BF16 output       {} bytes", buffers.output_bytes);
    println!("  candidate         {}", candidate.id);
    println!("  source revision   {}", candidate.source_revision);
    println!(
        "  linked            {}",
        if candidate.linked {
            "yes"
        } else {
            "no (contract-only milestone)"
        }
    );
}

fn parse_u64(value: Option<String>) -> u64 {
    value
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| usage())
}

fn parse_optional_usize(value: Option<String>, default: usize) -> usize {
    value
        .map(|value| value.parse::<usize>().unwrap_or_else(|_| usage()))
        .unwrap_or(default)
}
