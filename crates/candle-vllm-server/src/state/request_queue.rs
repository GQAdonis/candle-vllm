//! Request queue for handling requests during model switching.
//!
//! This module provides queuing functionality to handle requests when
//! the requested model is not currently active.
//!
//! **DEPRECATED**: This module is being replaced by the parking-lot scheduler's
//! built-in queueing. The `LLMEngineV2` uses `prometheus_parking_lot::TaskQueue`
//! for request queueing with configurable backends.
//!
//! This module is retained for backward compatibility during the migration period.

use candle_vllm_core::openai::requests::ChatCompletionRequest;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{info, warn};

/// A request queued for processing after model switch.
pub struct QueuedRequest {
    /// Correlation ID for diagnostics and mailbox storage.
    pub request_id: String,
    /// The model this request is for
    pub model: String,
    /// The chat completion request
    pub request: ChatCompletionRequest,
    /// When the request was queued
    pub queued_at: Instant,
    /// Sender for the response
    pub response_tx: Option<oneshot::Sender<candle_vllm_core::openai::responses::ChatResponder>>,
    /// Whether this is a streaming request
    pub is_streaming: bool,
}

impl QueuedRequest {
    /// Create a new queued request.
    pub fn new(
        request_id: String,
        model: String,
        request: ChatCompletionRequest,
        response_tx: Option<oneshot::Sender<candle_vllm_core::openai::responses::ChatResponder>>,
    ) -> Self {
        Self {
            request_id,
            model,
            is_streaming: request.stream.is_some_and(|s| s),
            request,
            queued_at: Instant::now(),
            response_tx,
        }
    }

    /// Check if this request has timed out.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.queued_at.elapsed() > timeout
    }
}

/// Queue for managing requests during model switches.
pub struct RequestQueue {
    inner: Mutex<VecDeque<QueuedRequest>>,
    max_size: usize,
    timeout: Duration,
}

impl RequestQueue {
    /// Create a new request queue.
    pub fn new(max_size: usize, timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            max_size,
            timeout,
        }
    }

    /// Enqueue a request. Returns Ok(()) if queued, Err if queue is full.
    pub fn enqueue(&self, req: QueuedRequest) -> Result<(), QueueError> {
        let mut guard = self.inner.lock();
        if guard.len() >= self.max_size {
            warn!(
                event = "request_queue_full",
                request_id = %req.request_id,
                model = %req.model,
                queued = guard.len(),
                max_size = self.max_size,
                "Request queue is full"
            );
            return Err(QueueError::QueueFull);
        }
        let new_len = guard.len() + 1;
        info!(
            event = "request_queue_enqueue",
            request_id = %req.request_id,
            model = %req.model,
            is_streaming = req.is_streaming,
            queued = new_len,
            max_size = self.max_size,
            "Request enqueued waiting for model availability"
        );
        guard.push_back(req);
        Ok(())
    }

    /// Dequeue the next request.
    pub fn dequeue(&self) -> Option<QueuedRequest> {
        let mut guard = self.inner.lock();
        let req = guard.pop_front();
        if let Some(ref r) = req {
            info!(
                event = "request_queue_dequeue",
                request_id = %r.request_id,
                model = %r.model,
                is_streaming = r.is_streaming,
                queued_remaining = guard.len(),
                "Dequeued request for processing"
            );
        }
        req
    }

    /// Drain all requests from the queue.
    pub fn drain(&self) -> Vec<QueuedRequest> {
        let mut guard = self.inner.lock();
        let drained: Vec<_> = guard.drain(..).collect();
        if !drained.is_empty() {
            info!(
                event = "request_queue_drain",
                drained = drained.len(),
                "Drained queued requests"
            );
        }
        drained
    }

    /// Remove timed-out requests and return them.
    pub fn remove_timed_out(&self) -> Vec<QueuedRequest> {
        let mut guard = self.inner.lock();
        let mut timed_out = Vec::new();
        let mut i = 0;
        while i < guard.len() {
            if guard[i].is_timed_out(self.timeout) {
                if let Some(req) = guard.remove(i) {
                    timed_out.push(req);
                }
            } else {
                i += 1;
            }
        }
        if !timed_out.is_empty() {
            warn!(
                event = "request_queue_timeout_sweep",
                timed_out = timed_out.len(),
                queued_remaining = guard.len(),
                timeout_ms = self.timeout.as_millis() as u64,
                "Removed timed-out queued requests"
            );
        }
        timed_out
    }

    /// Get the current queue length.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Get the maximum queue size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

/// Errors that can occur when queuing requests.
#[derive(Debug, Clone)]
pub enum QueueError {
    QueueFull,
    Timeout,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::QueueFull => write!(f, "Request queue is full"),
            QueueError::Timeout => write!(f, "Request timed out in queue"),
        }
    }
}

impl std::error::Error for QueueError {}
