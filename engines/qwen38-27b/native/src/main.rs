mod checkpoint;

use checkpoint::Q27Checkpoint;
use std::env;
use std::path::Path;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(root) = arguments.next() else {
        eprintln!("usage: {} CHECKPOINT", Path::new(&program).display());
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("q27-inspect accepts exactly one checkpoint path");
        std::process::exit(2);
    }

    match Q27Checkpoint::open(Path::new(&root)) {
        Ok(checkpoint) => {
            let plan = checkpoint.plan();
            println!("q27_checkpoint=valid");
            println!("revision={}", plan.revision);
            println!("architecture={}", plan.config.architecture);
            println!("layers={}", plan.config.layers);
            println!("gdn_layers={}", plan.config.gdn_layers);
            println!("attention_layers={}", plan.config.attention_layers);
            println!("hidden_size={}", plan.config.hidden_size);
            println!("intermediate_size={}", plan.config.intermediate_size);
            println!("vocab_size={}", plan.config.vocab_size);
            println!("text_tensors={}", plan.body.tensors + plan.roots.tensors);
            println!("mtp_tensors={}", plan.mtp.tensors);
            println!("vision_ignored_tensors={}", plan.vision_ignored.tensors);
            println!("text_body_gib={:.3}", gib(plan.body.bytes));
            println!("text_roots_gib={:.3}", gib(plan.roots.bytes));
            println!("mtp_gib={:.3}", gib(plan.mtp.bytes));
            println!("vision_ignored_gib={:.3}", gib(plan.vision_ignored.bytes));
            println!("text_runtime_gib={:.3}", gib(plan.runtime_bytes()));
            println!("checkpoint_gib={:.3}", gib(plan.total.bytes));
        }
        Err(error) => {
            eprintln!("q27 checkpoint rejected: {error}");
            std::process::exit(1);
        }
    }
}
