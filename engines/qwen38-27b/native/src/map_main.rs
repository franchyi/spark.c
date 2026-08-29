mod checkpoint;
mod mapping;

use mapping::MappedCheckpoint;
use std::env;
use std::path::Path;

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(root) = arguments.next() else {
        eprintln!("usage: {} CHECKPOINT", Path::new(&program).display());
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("q27-map-inspect accepts exactly one checkpoint path");
        std::process::exit(2);
    }
    let mapped = MappedCheckpoint::open(Path::new(&root)).unwrap_or_else(|error| {
        eprintln!("q27 checkpoint mapping failed: {error}");
        std::process::exit(1);
    });
    let checkpoint = mapped.checkpoint();
    let embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        .expect("strict checkpoint omitted embedding");
    let layer0 = checkpoint.tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .expect("strict checkpoint omitted layer-0 projection");
    let lm_head = checkpoint.tensor("lm_head.weight")
        .expect("strict checkpoint omitted LM head");
    println!("q27_mapping=valid");
    println!("shards={}", checkpoint.plan().files);
    println!("mapped_gib={:.3}", mapped.mapped_bytes() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("embedding_device=0x{:x}", mapped.device_address(embedding).unwrap());
    println!("layer0_qkv_device=0x{:x}", mapped.device_address(layer0).unwrap());
    println!("lm_head_device=0x{:x}", mapped.device_address(lm_head).unwrap());
}
