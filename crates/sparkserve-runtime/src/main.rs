use sparkserve_runtime::checkpoint::{CheckpointPlan, load_flash_next_plan};
use sparkserve_runtime::gguf::{GgmlTensorType, GgufSet};
use sparkserve_runtime::gguf_paging::GgufExpertCatalog;
use sparkserve_runtime::glm_dsa::GlmDsaSpec;
use sparkserve_runtime::glm_dsa_quant::{GlmDsaQuantModelPlan, GlmDsaQuantRole};
use sparkserve_runtime::glm_model::Glm53ModelSpec;
use sparkserve_runtime::glm_topology::GlmTopology;
use sparkserve_runtime::glm_weights::GlmResidentPlan;
use sparkserve_runtime::kernel::{DenseNvfp4Spec, DeviceCaps, select_dense_nvfp4_candidate};
use sparkserve_runtime::model::{plan_memory, profile};
use sparkserve_runtime::model_lock::{LockedModel, load_model_lock};
#[cfg(target_os = "linux")]
use sparkserve_runtime::storage::FixedPleCache;
use sparkserve_runtime::storage::{ClockPageCache, FilePageSource, PleIndex};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  sparkserve-runtime plan <model> [system-gib]");
    eprintln!("  sparkserve-runtime checkpoint-plan <model-root>");
    eprintln!("  sparkserve-runtime model-lock <lock-file> [model-id]");
    eprintln!("  sparkserve-runtime kernel-plan nvfp4-dense <M> <N> <K> [SM]");
    eprintln!("  sparkserve-runtime ple-inspect <index>");
    eprintln!("  sparkserve-runtime gguf-index <shard> [shard ...]");
    eprintln!(
        "  sparkserve-runtime gguf-header-index <prefix> <declared-bytes> [prefix declared-bytes ...]"
    );
    eprintln!(
        "  sparkserve-runtime gguf-header-tensor <name> <prefix> <declared-bytes> [prefix declared-bytes ...]"
    );
    eprintln!(
        "  sparkserve-runtime gguf-header-metadata <key> <prefix> <declared-bytes> [prefix declared-bytes ...]"
    );
    eprintln!(
        "  sparkserve-runtime gguf-header-schema <prefix> <declared-bytes> [prefix declared-bytes ...]"
    );
    eprintln!(
        "  sparkserve-runtime gguf-header-metadata-prefix <key-prefix> <prefix> <declared-bytes> [prefix declared-bytes ...]"
    );
    eprintln!(
        "  sparkserve-runtime ple-bench <index> <model-root> [tokens] [cache-mib] [chunk-tokens] [workers]"
    );
    #[cfg(target_os = "linux")]
    eprintln!(
        "  sparkserve-runtime ple-fixed-bench <index> <model-root> [tokens] [cache-mib] [chunk-tokens] [queue-depth] [parallel-threshold-pages] [parallel-workers]"
    );
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("plan") => run_memory_plan(args),
        Some("checkpoint-plan") => run_checkpoint_plan(args),
        Some("model-lock") => run_model_lock(args),
        Some("kernel-plan") => run_kernel_plan(args),
        Some("ple-inspect") => run_ple_inspect(args),
        Some("gguf-index") => run_gguf_index(args),
        Some("gguf-header-index") => run_gguf_header_index(args),
        Some("gguf-header-tensor") => run_gguf_header_tensor(args),
        Some("gguf-header-metadata") => run_gguf_header_metadata(args),
        Some("gguf-header-schema") => run_gguf_header_schema(args),
        Some("gguf-header-metadata-prefix") => run_gguf_header_metadata_prefix(args),
        Some("ple-bench") => run_ple_bench(args),
        #[cfg(target_os = "linux")]
        Some("ple-fixed-bench") => run_ple_fixed_bench(args),
        _ => usage(),
    }
}

fn run_gguf_index(args: impl Iterator<Item = String>) {
    let paths = args.map(std::path::PathBuf::from).collect::<Vec<_>>();
    if paths.is_empty() {
        usage();
    }
    let index = GgufSet::open(&paths).unwrap_or_else(|error| {
        eprintln!("invalid GGUF set: {error}");
        std::process::exit(1);
    });
    print_gguf_index(&index, false);
}

fn run_gguf_header_index(mut args: impl Iterator<Item = String>) {
    let mut headers = Vec::new();
    while let Some(path) = args.next() {
        let declared_bytes = args
            .next()
            .unwrap_or_else(|| usage())
            .parse::<u64>()
            .unwrap_or_else(|_| usage());
        headers.push((std::path::PathBuf::from(path), declared_bytes));
    }
    if headers.is_empty() {
        usage();
    }
    let index = GgufSet::open_headers(&headers).unwrap_or_else(|error| {
        eprintln!("invalid GGUF header set: {error}");
        std::process::exit(1);
    });
    print_gguf_index(&index, true);
}

fn run_gguf_header_tensor(mut args: impl Iterator<Item = String>) {
    let name = args.next().unwrap_or_else(|| usage());
    let mut headers = Vec::new();
    while let Some(path) = args.next() {
        let declared_bytes = args
            .next()
            .unwrap_or_else(|| usage())
            .parse::<u64>()
            .unwrap_or_else(|_| usage());
        headers.push((std::path::PathBuf::from(path), declared_bytes));
    }
    if headers.is_empty() {
        usage();
    }
    let index = GgufSet::open_headers(&headers).unwrap_or_else(|error| {
        eprintln!("invalid GGUF header set: {error}");
        std::process::exit(1);
    });
    let tensor = index.tensors.get(&name).unwrap_or_else(|| {
        eprintln!("GGUF tensor not found: {name}");
        std::process::exit(1);
    });
    println!("strict GGUF tensor location");
    println!("  name              {name}");
    println!("  shard             {}", tensor.shard);
    println!(
        "  path              {}",
        index.shards[tensor.shard].path.display()
    );
    println!("  absolute offset   {}", tensor.absolute_offset);
    println!("  data bytes        {}", tensor.data_bytes);
    println!("  dimensions        {:?}", tensor.dimensions);
    println!("  type              {:?}", tensor.tensor_type);
}

fn run_gguf_header_metadata(mut args: impl Iterator<Item = String>) {
    let key = args.next().unwrap_or_else(|| usage());
    let mut headers = Vec::new();
    while let Some(path) = args.next() {
        let declared_bytes = args
            .next()
            .unwrap_or_else(|| usage())
            .parse::<u64>()
            .unwrap_or_else(|_| usage());
        headers.push((std::path::PathBuf::from(path), declared_bytes));
    }
    if headers.is_empty() {
        usage();
    }
    let index = GgufSet::open_headers(&headers).unwrap_or_else(|error| {
        eprintln!("invalid GGUF header set: {error}");
        std::process::exit(1);
    });
    let entry = index.shards[0].metadata.get(&key).unwrap_or_else(|| {
        eprintln!("GGUF metadata key not found: {key}");
        std::process::exit(1);
    });
    println!("strict GGUF metadata");
    println!("  key               {key}");
    println!("  type              {:?}", entry.value_type);
    println!("  value             {:?}", entry.value);
    if let sparkserve_runtime::gguf::GgufMetadataValue::Array { element_type, .. } = &entry.value {
        match element_type {
            sparkserve_runtime::gguf::GgufValueType::Int32 => println!(
                "  decoded           {:?}",
                index.shards[0]
                    .metadata_i32_array(&key)
                    .unwrap_or_else(|error| {
                        eprintln!("cannot decode GGUF metadata array {key}: {error}");
                        std::process::exit(1);
                    })
            ),
            sparkserve_runtime::gguf::GgufValueType::Float32 => println!(
                "  decoded           {:?}",
                index.shards[0]
                    .metadata_f32_array(&key)
                    .unwrap_or_else(|error| {
                        eprintln!("cannot decode GGUF metadata array {key}: {error}");
                        std::process::exit(1);
                    })
            ),
            _ => {}
        }
    }
}

fn run_gguf_header_schema(mut args: impl Iterator<Item = String>) {
    let mut headers = Vec::new();
    while let Some(path) = args.next() {
        let declared_bytes = args
            .next()
            .unwrap_or_else(|| usage())
            .parse::<u64>()
            .unwrap_or_else(|_| usage());
        headers.push((std::path::PathBuf::from(path), declared_bytes));
    }
    if headers.is_empty() {
        usage();
    }
    let index = GgufSet::open_headers(&headers).unwrap_or_else(|error| {
        eprintln!("invalid GGUF header set: {error}");
        std::process::exit(1);
    });

    let mut schema = BTreeMap::<
        (String, Vec<u64>, GgmlTensorType),
        (usize, u64, BTreeSet<u16>),
    >::new();
    for (name, tensor) in &index.tensors {
        let (role, layer) = canonical_gguf_tensor_role(name);
        let entry = schema
            .entry((role, tensor.dimensions.clone(), tensor.tensor_type))
            .or_default();
        entry.0 += 1;
        entry.1 = entry
            .1
            .checked_add(tensor.data_bytes)
            .expect("validated GGUF schema byte sum");
        if let Some(layer) = layer {
            entry.2.insert(layer);
        }
    }

    println!("strict GGUF tensor schema");
    println!("  architecture      {}", index.architecture);
    println!("  roles             {}", schema.len());
    for ((role, dimensions, tensor_type), (count, bytes, layers)) in schema {
        if layers.is_empty() {
            println!(
                "  {role:<44} {dimensions:?} {tensor_type:?} x{count} {bytes} bytes"
            );
        } else {
            println!(
                "  {role:<44} {dimensions:?} {tensor_type:?} x{count} {bytes} bytes layers={layers:?}"
            );
        }
    }
}

fn run_gguf_header_metadata_prefix(mut args: impl Iterator<Item = String>) {
    let key_prefix = args.next().unwrap_or_else(|| usage());
    let mut headers = Vec::new();
    while let Some(path) = args.next() {
        let declared_bytes = args
            .next()
            .unwrap_or_else(|| usage())
            .parse::<u64>()
            .unwrap_or_else(|_| usage());
        headers.push((std::path::PathBuf::from(path), declared_bytes));
    }
    if headers.is_empty() {
        usage();
    }
    let index = GgufSet::open_headers(&headers).unwrap_or_else(|error| {
        eprintln!("invalid GGUF header set: {error}");
        std::process::exit(1);
    });

    let entries = index.shards[0]
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with(&key_prefix))
        .collect::<Vec<_>>();
    if entries.is_empty() {
        eprintln!("GGUF metadata prefix not found: {key_prefix}");
        std::process::exit(1);
    }
    println!("strict GGUF metadata prefix");
    println!("  prefix            {key_prefix}");
    for (key, entry) in entries {
        println!("  {key} {:?} {:?}", entry.value_type, entry.value);
    }
}

fn canonical_gguf_tensor_role(name: &str) -> (String, Option<u16>) {
    let Some(rest) = name.strip_prefix("blk.") else {
        return (name.to_owned(), None);
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return (name.to_owned(), None);
    };
    let Ok(layer) = layer.parse::<u16>() else {
        return (name.to_owned(), None);
    };
    (format!("blk.*.{suffix}"), Some(layer))
}

fn print_gguf_index(index: &GgufSet, headers_only: bool) {
    let tensor_bytes = index
        .tensors
        .values()
        .try_fold(0_u64, |total, tensor| total.checked_add(tensor.data_bytes))
        .expect("validated GGUF tensor byte sum");
    let mut type_distribution = BTreeMap::<GgmlTensorType, (usize, u64)>::new();
    let mut expert_layouts = BTreeMap::<(String, Vec<u64>, GgmlTensorType), (usize, u64)>::new();
    let mut glm_attention_layouts = BTreeMap::<(String, Vec<u64>, GgmlTensorType), usize>::new();
    for (name, tensor) in &index.tensors {
        let entry = type_distribution.entry(tensor.tensor_type).or_default();
        entry.0 += 1;
        entry.1 = entry
            .1
            .checked_add(tensor.data_bytes)
            .expect("validated GGUF type byte sum");
        let canonical_name = name
            .strip_prefix("blk.")
            .and_then(|suffix| suffix.split_once('.').map(|(_, rest)| rest))
            .unwrap_or(name)
            .to_owned();
        if name.contains("exp") {
            let entry = expert_layouts
                .entry((
                    canonical_name.clone(),
                    tensor.dimensions.clone(),
                    tensor.tensor_type,
                ))
                .or_default();
            entry.0 += 1;
            entry.1 = entry
                .1
                .checked_add(tensor.data_bytes)
                .expect("validated GGUF expert byte sum");
        }
        if canonical_name.starts_with("attn_")
            || canonical_name.starts_with("ssm_")
            || canonical_name.starts_with("indexer")
        {
            *glm_attention_layouts
                .entry((
                    canonical_name,
                    tensor.dimensions.clone(),
                    tensor.tensor_type,
                ))
                .or_default() += 1;
        }
    }
    println!(
        "strict GGUF v3 {}set",
        if headers_only { "header " } else { "" }
    );
    println!("  architecture      {}", index.architecture);
    println!("  shards            {:12}", index.shards.len());
    println!("  tensors           {:12}", index.tensors.len());
    println!("  tensor bytes      {tensor_bytes:12}");
    println!("  tensor type distribution");
    for (tensor_type, (count, bytes)) in type_distribution {
        println!("    {tensor_type:?} {count:8} tensors {bytes:12} bytes");
    }
    println!("  expert tensor layouts");
    for ((name, dimensions, tensor_type), (count, bytes)) in expert_layouts {
        println!("    {name:<36} {dimensions:?} {tensor_type:?} x{count} {bytes} bytes");
    }
    if index.architecture == "glm5next" {
        let model = Glm53ModelSpec::from_gguf(index).unwrap_or_else(|error| {
            eprintln!("invalid locked GLM-5.3 model contract: {error}");
            std::process::exit(1);
        });
        println!("  locked GLM-5.3 execution contract");
        println!("    trunk blocks      {:12}", model.topology.trunk_layer_count());
        println!("    MTP blocks        {:12}", model.topology.mtp_layer_count());
        println!("    hidden            {:12}", model.hidden_size);
        println!("    vocabulary        {:12}", model.vocabulary);
        println!("    context           {:12}", model.context_length);
        println!("    hyper connections {:12}", model.hyper_connections);
        println!("    routed experts    {:12}", model.routed_experts);
        let resident = GlmResidentPlan::from_gguf(index, false).unwrap_or_else(|error| {
            eprintln!("invalid GLM base resident mapping plan: {error}");
            std::process::exit(1);
        });
        println!("  GLM base resident mapping plan");
        println!("    tensors           {:12}", resident.tensors().len());
        println!("    file ranges       {:12}", resident.ranges().len());
        println!("    tensor bytes      {:12}", resident.tensor_bytes());
        println!("    mapped payload    {:12}", resident.mapped_payload_bytes());
        println!(
            "    mapped 4K pages   {:12}",
            resident.mapped_page_bytes(4096).expect("validated 4K plan")
        );
        println!("    paged experts     {:12}", resident.excluded_expert_bytes());
        println!("    deferred MTP      {:12}", resident.excluded_mtp_bytes());
        println!("  GLM attention tensor layouts");
        for ((name, dimensions, tensor_type), count) in glm_attention_layouts {
            println!("    {name:<36} {dimensions:?} {tensor_type:?} x{count}");
        }
        let topology = GlmTopology::from_gguf(index).unwrap_or_else(|error| {
            eprintln!("invalid GLM layer topology: {error}");
            std::process::exit(1);
        });
        let dsa_layers = topology.dsa_layers().collect::<Vec<_>>();
        println!("  GLM layer topology");
        println!("    blocks            {:12}", topology.layers().len());
        println!("    trunk blocks      {:12}", topology.trunk_layer_count());
        println!("    MTP blocks        {:12}", topology.mtp_layer_count());
        println!(
            "    leading dense     {:12}",
            topology.leading_dense_layers()
        );
        println!("    KDA layers         {:12}", topology.kda_layers());
        println!("    DSA/MLA layers     {dsa_layers:?}");
        println!(
            "    trunk DSA/MLA      {:?}",
            topology.trunk_dsa_layers().collect::<Vec<_>>()
        );
        let dsa = GlmDsaSpec::from_gguf_trunk(index).unwrap_or_else(|error| {
            eprintln!("invalid GLM DSA/KPool contract: {error}");
            std::process::exit(1);
        });
        let dsa_plan = dsa.plan(1, 32 * 1024).expect("locked 32K DSA plan");
        println!("  GLM DSA/KPool batch-one 32K plan");
        println!("    pooled entries    {:12}", dsa_plan.pooled_entries);
        println!(
            "    persistent bytes  {:12}",
            dsa_plan
                .persistent_bytes()
                .expect("validated persistent DSA bytes")
        );
        println!(
            "    decode workspace  {:12}",
            dsa_plan
                .decode_workspace_bytes()
                .expect("validated DSA workspace bytes")
        );
        let quant_plan = GlmDsaQuantModelPlan::from_gguf_trunk(index).unwrap_or_else(|error| {
            eprintln!("invalid GLM DSA quant projection plan: {error}");
            std::process::exit(1);
        });
        let first_quant_layer = quant_plan.layers.first().expect("validated DSA layer set");
        let absorb = first_quant_layer
            .operation(GlmDsaQuantRole::AbsorbKey)
            .expect("validated MLA K absorption");
        let absorb_stride = match &absorb.dispatch {
            sparkserve_runtime::glm_dsa_quant::GlmDsaQuantDispatch::HeadBatched {
                spec, ..
            } => spec.weight_slot_stride_bytes,
            _ => unreachable!("MLA K absorption is head-batched"),
        };
        println!("  GLM DSA direct-GGUF projection plan");
        println!("    layers            {:12}", quant_plan.layers.len());
        println!(
            "    MMVQ arena bytes  {:12}",
            quant_plan.workspace.total_bytes
        );
        println!("    head launches     {:12}", 8);
        println!("    Q8 head stride    {:12}", absorb_stride);
        println!(
            "    BF16 copies saved {:12}",
            quant_plan.avoided_bf16_kv_projection_bytes
        );
        let catalog = GgufExpertCatalog::build_glm53_trunk(index, 16).unwrap_or_else(|error| {
            eprintln!("invalid GLM expert paging catalog: {error}");
            std::process::exit(1);
        });
        println!("  GLM base fixed expert cache (16 slots)");
        println!(
            "    expert payload    {:12} bytes",
            catalog.expert_source_bytes()
        );
        println!(
            "    resident payload  {:12} bytes",
            resident.tensor_bytes()
        );
        println!(
            "    max useful/slot  {:12} bytes",
            catalog.useful_expert_bytes()
        );
        println!("    allocated arena  {:12} bytes", catalog.arena_bytes());
        for component in catalog.components() {
            println!(
                "    {:<8} K {:>4} rows {:>4} max {:>8} stride {:>8} {:?}",
                component.name,
                component.k,
                component.rows,
                component.max_slice_bytes,
                component.slot_stride_bytes,
                component.quant_types
            );
        }
    }
    for (number, shard) in index.shards.iter().enumerate() {
        println!(
            "  shard {:>3}         {:12} bytes  {}",
            number,
            shard.file_bytes,
            shard.path.display()
        );
    }
}

fn run_model_lock(mut args: impl Iterator<Item = String>) {
    let path = args.next().unwrap_or_else(|| usage());
    let model_id = args.next();
    if args.next().is_some() {
        usage();
    }
    let lock = load_model_lock(Path::new(&path)).unwrap_or_else(|error| {
        eprintln!("invalid model lock: {error}");
        std::process::exit(1);
    });
    println!("SparkServe immutable model lock v{}", lock.schema_version);
    println!("  mirror             {}", lock.mirror);
    if let Some(model_id) = model_id {
        let model = lock.model(&model_id).unwrap_or_else(|error| {
            eprintln!("invalid model selection: {error}");
            std::process::exit(1);
        });
        print_locked_model(model);
    } else {
        for model in &lock.models {
            print_locked_model(model);
        }
    }
}

fn print_locked_model(model: &LockedModel) {
    println!("{} @ {}", model.id, model.revision);
    println!("  repository         {}", model.repo);
    println!("  architecture       {}", model.architecture);
    println!(
        "  format             {} / {}",
        model.format, model.quantization
    );
    println!("  inventory          {}", model.inventory);
    println!("  locked files       {:12}", model.files.len());
    println!("  checkpoint bytes   {:12}", model.checkpoint_bytes);
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

#[cfg(target_os = "linux")]
fn run_ple_fixed_bench(mut args: impl Iterator<Item = String>) {
    let index_path = args.next().unwrap_or_else(|| usage());
    let model_root = args.next().unwrap_or_else(|| usage());
    let tokens = parse_optional_usize(args.next(), 4096);
    let cache_mib = parse_optional_usize(args.next(), 4);
    let chunk_tokens = parse_optional_usize(args.next(), 16);
    let queue_depth = parse_optional_usize(args.next(), 16);
    let parallel_threshold_pages = parse_optional_usize(args.next(), 64);
    let parallel_workers = parse_optional_usize(args.next(), 16);
    if args.next().is_some()
        || tokens == 0
        || cache_mib == 0
        || chunk_tokens == 0
        || queue_depth == 0
        || parallel_threshold_pages == 0
        || parallel_workers == 0
    {
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
    let cache_bytes = cache_mib
        .checked_mul(1024 * 1024)
        .unwrap_or_else(|| usage());
    let mut slab = vec![0_u8; cache_bytes];
    let mut cache = FixedPleCache::open_hybrid(
        &index,
        Path::new(&model_root),
        &mut slab,
        queue_depth,
        parallel_threshold_pages,
        parallel_workers,
    )
    .unwrap_or_else(|error| {
        eprintln!("cannot open fixed-buffer PLE cache: {error}");
        std::process::exit(1);
    });
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
        let batch = cache.fetch_rows(&index, &rows).unwrap_or_else(|error| {
            eprintln!("fixed-buffer PLE benchmark failed: {error}");
            std::process::exit(1);
        });
        checksum = checksum.wrapping_add(batch.wrapping_checksum());
        generated += batch_tokens;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!("exact FP8 PLE fixed-slab hybrid benchmark");
    println!(
        "  throughput       {:12.2} tokens/s",
        tokens as f64 / elapsed
    );
    println!("  elapsed          {:12.3} s", elapsed);
    println!("  queue depth      {:12}", queue_depth);
    println!("  pread threshold  {:12} pages", parallel_threshold_pages);
    println!("  page reads       {:12}", cache.stats.page_reads);
    println!("  io_uring reads   {:12}", cache.stats.uring_page_reads);
    println!(
        "  parallel preads {:12}",
        cache.stats.parallel_pread_page_reads
    );
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
