use std::path::Path;

use spark_flash_next::checkpoint::load_flash_next_checkpoint;
use spark_flash_next::qwen_expert_sidecar::ExpertSidecarProgress;
use spark_flash_next::qwen_expert_sidecar_v2::{
    build_expert_sidecar_v2, verify_expert_sidecar_v2,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().unwrap_or_else(|| usage());
    let output = arguments.next().unwrap_or_else(|| usage());
    let mode = arguments.next();
    if arguments.next().is_some() || mode.as_deref().is_some_and(|mode| mode != "--verify") {
        usage();
    }
    let checkpoint = load_flash_next_checkpoint(Path::new(&model))?;
    let report = if mode.as_deref() == Some("--verify") {
        verify_expert_sidecar_v2(&checkpoint, Path::new(&output), print_progress)?
    } else {
        build_expert_sidecar_v2(&checkpoint, Path::new(&output), print_progress)?
    };
    println!(
        "Qwen SoA-v2 expert sidecar ready: layers={} bytes={} resumed={} existing={} fingerprint={}",
        report.records,
        report.bytes,
        report.resumed_records,
        report.already_complete,
        hex(&report.source_fingerprint),
    );
    Ok(())
}

fn print_progress(progress: ExpertSidecarProgress) {
    let percent = progress.completed_records as f64 * 100.0 / progress.total_records as f64;
    eprintln!(
        "Qwen SoA-v2 expert sidecar: {}/{} layers ({percent:.1}%, resumed {})",
        progress.completed_records, progress.total_records, progress.resumed_records
    );
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn usage() -> ! {
    eprintln!(
        "usage: qwen_expert_sidecar_v2 <model-root> <output.ssx> [--verify]\n\
         build resumes <output.ssx>.partial and publishes <output.ssx> atomically"
    );
    std::process::exit(2)
}
