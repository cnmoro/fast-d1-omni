// Compiles cuda/kernels.cu with nvcc into a static library and links the CUDA runtime and cuBLAS.
// Env: D1_CUDA_ARCH (default "native", e.g. "sm_75" or "sm_86"), CUDA_PATH / D1_CUDA_LIB for library lookup.
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("D1_CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    println!("cargo:rerun-if-env-changed=D1_CUDA_ARCH");
    let mut objs = Vec::new();
    for src in ["kernels", "gemm"] {
        println!("cargo:rerun-if-changed=cuda/{src}.cu");
        let obj = out.join(format!("{src}.o"));
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-c"]).arg(format!("cuda/{src}.cu")).arg("-o").arg(&obj);
        if arch == "native" {
            cmd.arg("-arch=native");
        } else {
            for a in arch.split(',') {
                let n = a.trim().trim_start_matches("sm_");
                cmd.arg(format!("-gencode=arch=compute_{n},code=sm_{n}"));
            }
        }
        let st = cmd.status().expect("failed to run nvcc (set NVCC=/path/to/nvcc)");
        assert!(st.success(), "nvcc failed on {src}.cu");
        objs.push(obj);
    }
    let lib = out.join("libd1kernels.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").args(["crs"]).arg(&lib).args(&objs).status().expect("ar");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=d1kernels");
    for dir in [env::var("D1_CUDA_LIB").ok(), env::var("CUDA_PATH").ok().map(|p| format!("{p}/lib64"))]
        .into_iter()
        .flatten()
    {
        println!("cargo:rustc-link-search=native={dir}");
    }
    for dir in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
