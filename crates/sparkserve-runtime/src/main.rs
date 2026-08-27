use sparkserve_runtime::kernel::{
    DenseNvfp4Spec, DeviceCaps, select_dense_nvfp4_candidate,
};
use sparkserve_runtime::model::{plan_memory, profile};

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  sparkserve-runtime plan <model> [system-gib]");
    eprintln!("  sparkserve-runtime kernel-plan nvfp4-dense <M> <N> <K> [SM]");
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("plan") => run_memory_plan(args),
        Some("kernel-plan") => run_kernel_plan(args),
        _ => usage(),
    }
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
    println!("{}: {}", if plan.fits { "FIT" } else { "DOES NOT FIT" }, model.key);
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
    let buffers = spec.buffer_requirements().expect("validated size calculation");
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
        if candidate.linked { "yes" } else { "no (contract-only milestone)" }
    );
}

fn parse_u64(value: Option<String>) -> u64 {
    value
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| usage())
}
