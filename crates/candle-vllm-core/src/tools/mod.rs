//! Tool calling support for candle-vllm-core.
//!
//! This module re-exports tool-related types from `openai::requests` and provides
//! the streaming tool parser for detecting tool calls in LLM output.

pub mod stream_parser;

// Re-export core tool types from requests for convenience
pub use crate::openai::requests::{FunctionCall, ToolCall};
