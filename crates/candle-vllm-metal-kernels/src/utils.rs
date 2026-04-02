use candle_core::MetalStorage;
use candle_metal_kernels::metal::{Buffer, Commands, ComputeCommandEncoder};
use std::ffi::c_void;

pub fn set_param<P: EncoderParam, E: std::ops::Deref<Target = ComputeCommandEncoder>>(encoder: &E, position: usize, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self);
}

#[macro_export]
macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            $crate::utils::set_param(&$encoder, _index, $param);
            _index += 1;
        )+
    );
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
                encoder.set_bytes(position, &data);
            }
        }
    };
}
primitive!(bool);
primitive!(usize);
primitive!(i32);
primitive!(i64);
primitive!(u32);
primitive!(u64);
primitive!(f32);

pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes_directly(
            position,
            core::mem::size_of_val(data),
            data.as_ptr() as *const c_void,
        );
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.buffer), data.offset_in_bytes);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for Option<MetalStorage> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        if let Some(s) = data {
            encoder.set_buffer(position, Some(s.buffer()), 0);
        }
    }
}

pub trait EncoderProvider: Sized {
    type Encoder<'a>: std::ops::Deref<Target = ComputeCommandEncoder>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a> {
    pub(crate) inner: ComputeCommandEncoder,
    #[allow(dead_code)]
    pub(crate) buffer: &'a Commands,
}

impl<'a> std::ops::Deref for WrappedEncoder<'a> {
    type Target = ComputeCommandEncoder;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for WrappedEncoder<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl EncoderProvider for &Commands {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        let (_is_new, encoder) = self.command_encoder().expect("Failed to create encoder");
        WrappedEncoder {
            inner: encoder,
            buffer: self,
        }
    }
}
