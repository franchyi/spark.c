mod checkpoint;
mod mapping;
mod scale_sidecar;

use scale_sidecar::ScaleSidecar;
use std::env;
use std::path::Path;

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(path) = arguments.next() else {
        eprintln!("usage: {} SIDECAR", Path::new(&program).display());
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("q27-inspect-scales accepts exactly one sidecar path");
        std::process::exit(2);
    }
    let sidecar = ScaleSidecar::open(Path::new(&path)).unwrap_or_else(|error| {
        eprintln!("q27 scale sidecar rejected: {error}");
        std::process::exit(1);
    });
    let first = sidecar.entry(0, 0).unwrap();
    let last = sidecar.entry(63, 2).unwrap();
    println!("q27_scale_sidecar=valid");
    println!("entries=192");
    println!("mapped_gib={:.3}", sidecar.bytes() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("first_scale_device=0x{:x}", sidecar.scale_device_address(0, 0).unwrap());
    println!("first_input_scale_inv_device=0x{:x}", sidecar.input_scale_inv_device_address(0, 0).unwrap());
    println!("first_alpha_device=0x{:x}", sidecar.alpha_device_address(0, 0).unwrap());
    println!("first_alpha={:.9e}", first.alpha);
    println!("last_end={}", last.offset + last.bytes);
}
