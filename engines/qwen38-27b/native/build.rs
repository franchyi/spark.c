use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let library = manifest.join("../../../build/q27");
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/sbsa-linux/lib");
    println!("cargo:rustc-link-lib=dylib=q27-mapping");
    println!("cargo:rustc-link-lib=dylib=q27-kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-changed=build.rs");
}
