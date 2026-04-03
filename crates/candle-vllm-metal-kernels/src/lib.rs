// Metal kernels are only available on Apple platforms.
// On non-Apple targets this crate is an empty stub.
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod utils;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod inner;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use inner::*;
