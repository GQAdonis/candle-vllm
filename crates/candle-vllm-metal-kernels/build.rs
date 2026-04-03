use std::path::PathBuf;
use std::process::Command;
use std::{env, str};
const METAL_SOURCES: [&str; 9] = [
    "copy_blocks",
    "pagedattention",
    "reshape_and_cache",
    "prefill_paged_attn",
    "mask",
    "update_scales",
    "fused_rope",
    "fp8_matmul",
    "gdn",
];

enum Platform {
    MacOS,
    Ios,
}

impl Platform {
    fn sdk(&self) -> &str {
        match self {
            Platform::MacOS => "macosx",
            Platform::Ios => "iphoneos",
        }
    }
}

fn compile(platform: Platform) -> Result<(), String> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
    let working_directory = out_dir.to_string_lossy().to_string();
    let sources = current_dir.join("src");
    let module_cache = out_dir.join("clang-module-cache");
    std::fs::create_dir_all(&module_cache)
        .map_err(|e| format!("failed to create clang module cache dir: {e}"))?;
    let module_cache_flag = format!("-fmodules-cache-path={}", module_cache.display());

    // Compile metal to air
    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-working-directory={working_directory}"))
        .arg(&module_cache_flag)
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-O3")
        .arg("-c")
        .arg("-w");
    for metal_file in METAL_SOURCES {
        compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
    }
    // compile_air_cmd.arg(sources.join("metal_dtype.metal"));
    let mut child = compile_air_cmd.spawn().expect("Failed to compile air");

    match child.try_wait() {
        Ok(Some(status)) => {
            if !status.success() {
                panic!("Compiling metal -> air failed. Exit with status: {status}")
            }
        }
        Ok(None) => {
            let status = child
                .wait()
                .expect("Compiling metal -> air failed while waiting for result");
            if !status.success() {
                panic!("Compiling metal -> air failed. Exit with status: {status}")
            }
        }
        Err(e) => panic!("Compiling metal -> air failed: {e:?}"),
    }

    // Compile air to metallib
    let lib_name = match platform {
        Platform::MacOS => "paged_attention.metallib",
        Platform::Ios => "paged_attention_ios.metallib",
    };
    let metallib = out_dir.join(lib_name);
    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

    for metal_file in METAL_SOURCES {
        compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
    }
    // compile_metallib_cmd.arg(out_dir.join("metal_dtype.air"));

    let mut child = compile_metallib_cmd
        .spawn()
        .expect("Failed to compile air -> metallib");

    match child.try_wait() {
        Ok(Some(status)) => {
            if !status.success() {
                panic!("Compiling air -> metallib failed. Exit with status: {status}")
            }
        }
        Ok(None) => {
            let status = child
                .wait()
                .expect("Compiling air -> metallib failed while waiting for result");
            if !status.success() {
                panic!("Compiling air -> metallib failed. Exit with status: {status}")
            }
        }
        Err(e) => panic!("Compiling air -> metallib failed: {e:?}"),
    }

    Ok(())
}

fn main() -> Result<(), String> {
    // Metal compilation is only available on macOS/iOS (Apple platforms).
    // On other platforms this build script is a no-op.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" && target_os != "ios" {
        return Ok(());
    }

    for src in METAL_SOURCES {
        println!("cargo::rerun-if-changed=src/{src}.metal");
    }
    println!("cargo::rerun-if-changed=src/metal_dtype.metal");
    println!("cargo::rerun-if-changed=build.rs");

    compile(Platform::MacOS)?;
    compile(Platform::Ios)?;

    Ok(())
}
