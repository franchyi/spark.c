mod checkpoint;
mod dflash2_checkpoint;
mod mapping;

use dflash2_checkpoint::DFlash2WeightPlan;
use std::env;
use std::path::Path;

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(checkpoint) = arguments.next() else {
        eprintln!(
            "usage: {} SNAPSHOT_OR_MODEL_SAFETENSORS",
            Path::new(&program).display()
        );
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("q27-dflash2-inspect accepts exactly one checkpoint path");
        std::process::exit(2);
    }
    let plan = DFlash2WeightPlan::open(Path::new(&checkpoint)).unwrap_or_else(|error| {
        eprintln!("DFlash2 checkpoint rejected: {error}");
        std::process::exit(1);
    });
    let evidence = plan.evidence();
    println!("q27_dflash2_checkpoint=valid");
    println!("repository={}", evidence.repository);
    println!("revision={}", evidence.revision);
    println!("model_sha256={}", evidence.model_sha256);
    println!("config_sha256={}", evidence.config_sha256);
    println!("config_contract={}", evidence.config_contract);
    println!("tensors={}", evidence.tensor_count);
    println!("file_bytes={}", evidence.file_bytes);
    println!("payload_bytes={}", evidence.payload_bytes);
    println!("mapped_bytes={}", plan.mapped_bytes());
    println!("checkpoint={}", plan.checkpoint_root().display());
    println!("ffi_weights=0x{:x}", plan.weights_ptr() as usize);
}
