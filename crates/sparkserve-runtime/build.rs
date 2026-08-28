use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=SPARKSERVE_DS4_STATIC_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if env::var_os("CARGO_FEATURE_NATIVE_DS4_GLM53").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repository = manifest.join("../..");
    let native_dir = env::var_os("SPARKSERVE_DS4_STATIC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repository.join("build-spark"));
    let archive = native_dir.join("libsparkserve-ds4-glm53.a");
    if !archive.is_file() {
        panic!(
            "missing {}; build it on Spark with `make glm53-ds4-static`",
            archive.display()
        );
    }

    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", native_dir.display());
    println!("cargo:rustc-link-lib=static=sparkserve-ds4-glm53");

    let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()));
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("targets/sbsa-linux/lib").display()
    );
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}
