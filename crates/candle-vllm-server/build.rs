use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NCCL");
    println!("cargo:rerun-if-env-changed=NCCL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
    println!("cargo:rerun-if-env-changed=NCCL_ROOT");

    if env::var_os("CARGO_FEATURE_NCCL").is_none() {
        return;
    }

    let mut candidate_dirs: Vec<PathBuf> = Vec::new();

    if let Some(dir) = env::var_os("NCCL_LIB_DIR") {
        candidate_dirs.push(PathBuf::from(dir));
    }

    if let Some(home) = env::var_os("NCCL_HOME").or_else(|| env::var_os("NCCL_ROOT")) {
        let home = PathBuf::from(home);
        candidate_dirs.push(home.join("lib"));
        candidate_dirs.push(home.join("lib64"));
    }

    candidate_dirs.push(PathBuf::from("/usr/local/nccl/lib"));
    candidate_dirs.push(PathBuf::from("/usr/local/nccl/lib64"));
    candidate_dirs.push(PathBuf::from("/opt/nccl/lib"));
    candidate_dirs.push(PathBuf::from("/opt/nccl/lib64"));
    candidate_dirs.push(PathBuf::from("/lib/x86_64-linux-gnu"));
    candidate_dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    candidate_dirs.push(PathBuf::from("/lib"));
    candidate_dirs.push(PathBuf::from("/usr/lib"));

    let mut added_any = false;
    for dir in candidate_dirs {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            added_any = true;
        }
    }

    if !added_any {
        println!(
            "cargo:warning=NCCL feature enabled but no NCCL library search path was added; install NCCL (libnccl) or set NCCL_LIB_DIR/NCCL_HOME."
        );
    }
}
