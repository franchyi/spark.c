//! Allocation-free Qwen layer composition over borrowed attention and MoE kernels.
//!
//! Each layer uses two fixed coherent hidden-state slabs. Attention reads the
//! published slab and writes the scratch slab; the MLP reads scratch and writes
//! back to the published slab. No hidden-state copy or address change occurs.
//! Child GDN/QSA/MoE schedulers still own their detailed stages; this scheduler
//! owns the cross-block handoff, layer pattern, and failure boundary.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

pub const QWEN_HIDDEN_SIZE: u64 = 2_560;
pub const QWEN_HYPER_STREAMS: u64 = 4;
pub const QWEN_BF16_BYTES: u64 = 2;
pub const QWEN_HYPER_BYTES: u64 = QWEN_HIDDEN_SIZE * QWEN_HYPER_STREAMS * QWEN_BF16_BYTES;
pub const QWEN_LAYERS: u16 = 48;
pub const QWEN_FULL_ATTENTION_INTERVAL: u16 = 4;
const COHERENT_ALIGNMENT: u64 = 256;

static NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenAttentionKind {
    Gdn,
    Qsa,
}

pub fn qwen_attention_kind(layer: u16) -> Result<QwenAttentionKind, QwenLayerPipelineError> {
    if layer >= QWEN_LAYERS {
        return Err(QwenLayerPipelineError::InvalidLayer);
    }
    if (layer + 1).is_multiple_of(QWEN_FULL_ATTENTION_INTERVAL) {
        Ok(QwenAttentionKind::Qsa)
    } else {
        Ok(QwenAttentionKind::Gdn)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoherentHiddenRange {
    pub address: u64,
    pub bytes: u64,
}

impl CoherentHiddenRange {
    fn end(self) -> Result<u64, QwenLayerPipelineError> {
        self.address
            .checked_add(self.bytes)
            .ok_or(QwenLayerPipelineError::AddressOverflow)
    }

    fn validate(self) -> Result<(), QwenLayerPipelineError> {
        if self.address == 0
            || !self.address.is_multiple_of(COHERENT_ALIGNMENT)
            || self.bytes != QWEN_HYPER_BYTES
        {
            return Err(QwenLayerPipelineError::InvalidHiddenRange);
        }
        self.end().map(|_| ())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLayerBuffers {
    /// Published four-stream hidden state. MLP writes the next layer here.
    pub published: CoherentHiddenRange,
    /// Attention output and MLP input. Never copied into a shadow tensor.
    pub scratch: CoherentHiddenRange,
}

impl QwenLayerBuffers {
    pub fn new(
        published: CoherentHiddenRange,
        scratch: CoherentHiddenRange,
    ) -> Result<Self, QwenLayerPipelineError> {
        published.validate()?;
        scratch.validate()?;
        if published.address < scratch.end()? && scratch.address < published.end()? {
            return Err(QwenLayerPipelineError::OverlappingHiddenRanges);
        }
        Ok(Self { published, scratch })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerStage {
    Attention(QwenAttentionKind),
    Mlp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerPhase {
    Idle,
    Running(QwenLayerStage),
    ReadyForMlp,
    Poisoned(QwenAttentionKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Operation {
    pipeline: u64,
    epoch: u64,
    token: u64,
    generation: u64,
    layer: u16,
    attention: QwenAttentionKind,
    buffers: QwenLayerBuffers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLayerLease {
    operation: Operation,
    stage: QwenLayerStage,
    input: CoherentHiddenRange,
    output: CoherentHiddenRange,
}

impl QwenLayerLease {
    pub fn layer(self) -> u16 {
        self.operation.layer
    }

    pub fn token(self) -> u64 {
        self.operation.token
    }

    pub fn stage(self) -> QwenLayerStage {
        self.stage
    }

    pub fn input(self) -> CoherentHiddenRange {
        self.input
    }

    pub fn output(self) -> CoherentHiddenRange {
        self.output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenAttentionReady {
    operation: Operation,
}

impl QwenAttentionReady {
    pub fn mlp_input(self) -> CoherentHiddenRange {
        self.operation.buffers.scratch
    }

    pub fn mlp_output(self) -> CoherentHiddenRange {
        self.operation.buffers.published
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLayerPublication {
    pub token: u64,
    pub generation: u64,
    pub layer: u16,
    pub hidden: CoherentHiddenRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerAbort {
    RestoreAttentionState {
        token: u64,
        generation: u64,
        layer: u16,
        kind: QwenAttentionKind,
    },
    RetryMlp(QwenAttentionReady),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Idle,
    AttentionRunning(Operation),
    AttentionReady(Operation),
    MlpRunning(Operation),
    Poisoned(Operation),
}

/// Cross-block owner for one in-flight layer operation.
pub struct QwenLayerPipeline {
    id: u64,
    next_epoch: u64,
    state: State,
}

impl Default for QwenLayerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenLayerPipeline {
    pub fn new() -> Self {
        Self {
            id: NEXT_PIPELINE_ID.fetch_add(1, Ordering::Relaxed),
            next_epoch: 0,
            state: State::Idle,
        }
    }

    pub fn phase(&self) -> QwenLayerPhase {
        match self.state {
            State::Idle => QwenLayerPhase::Idle,
            State::AttentionRunning(operation) => {
                QwenLayerPhase::Running(QwenLayerStage::Attention(operation.attention))
            }
            State::AttentionReady(_) => QwenLayerPhase::ReadyForMlp,
            State::MlpRunning(_) => QwenLayerPhase::Running(QwenLayerStage::Mlp),
            State::Poisoned(operation) => QwenLayerPhase::Poisoned(operation.attention),
        }
    }

    pub fn begin_layer(
        &mut self,
        token: u64,
        generation: u64,
        layer: u16,
        buffers: QwenLayerBuffers,
    ) -> Result<QwenLayerLease, QwenLayerPipelineError> {
        if !matches!(self.state, State::Idle) {
            return Err(QwenLayerPipelineError::Busy);
        }
        if token == 0 || generation == 0 {
            return Err(QwenLayerPipelineError::InvalidOperation);
        }
        let attention = qwen_attention_kind(layer)?;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(QwenLayerPipelineError::EpochOverflow)?;
        let operation = Operation {
            pipeline: self.id,
            epoch: self.next_epoch,
            token,
            generation,
            layer,
            attention,
            buffers,
        };
        self.state = State::AttentionRunning(operation);
        Ok(attention_lease(operation))
    }

    /// Call only after the selected GDN/QSA child scheduler publishes success.
    pub fn complete_attention(
        &mut self,
        lease: QwenLayerLease,
    ) -> Result<QwenAttentionReady, QwenLayerPipelineError> {
        let operation = match self.state {
            State::AttentionRunning(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if lease != attention_lease(operation) {
            return Err(QwenLayerPipelineError::ForeignLease);
        }
        self.state = State::AttentionReady(operation);
        Ok(QwenAttentionReady { operation })
    }

    pub fn begin_mlp(
        &mut self,
        ready: QwenAttentionReady,
    ) -> Result<QwenLayerLease, QwenLayerPipelineError> {
        let operation = match self.state {
            State::AttentionReady(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if ready.operation != operation {
            return Err(QwenLayerPipelineError::ForeignLease);
        }
        self.state = State::MlpRunning(operation);
        Ok(mlp_lease(operation))
    }

    /// Publish only after the MoE child scheduler has completed its final join.
    pub fn complete_mlp(
        &mut self,
        lease: QwenLayerLease,
    ) -> Result<QwenLayerPublication, QwenLayerPipelineError> {
        let operation = match self.state {
            State::MlpRunning(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if lease != mlp_lease(operation) {
            return Err(QwenLayerPipelineError::ForeignLease);
        }
        self.state = State::Idle;
        Ok(QwenLayerPublication {
            token: operation.token,
            generation: operation.generation,
            layer: operation.layer,
            hidden: operation.buffers.published,
        })
    }

    pub fn abort_attention(
        &mut self,
        lease: QwenLayerLease,
    ) -> Result<QwenLayerAbort, QwenLayerPipelineError> {
        let operation = match self.state {
            State::AttentionRunning(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if lease != attention_lease(operation) {
            return Err(QwenLayerPipelineError::ForeignLease);
        }
        self.state = State::Poisoned(operation);
        Ok(QwenLayerAbort::RestoreAttentionState {
            token: operation.token,
            generation: operation.generation,
            layer: operation.layer,
            kind: operation.attention,
        })
    }

    /// The child attention scheduler must restore its recurrent/KV checkpoint
    /// before this releases the two hidden slabs for another operation.
    pub fn restore_attention(
        &mut self,
        token: u64,
        generation: u64,
        layer: u16,
    ) -> Result<(), QwenLayerPipelineError> {
        let operation = match self.state {
            State::Poisoned(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if operation.token != token
            || operation.generation != generation
            || operation.layer != layer
        {
            return Err(QwenLayerPipelineError::ForeignRestore);
        }
        self.state = State::Idle;
        Ok(())
    }

    /// MLP does not mutate attention state. Its output aliases the old layer
    /// input, so a partial write is safely overwritten from intact scratch.
    pub fn abort_mlp(
        &mut self,
        lease: QwenLayerLease,
    ) -> Result<QwenLayerAbort, QwenLayerPipelineError> {
        let operation = match self.state {
            State::MlpRunning(operation) => operation,
            _ => return Err(QwenLayerPipelineError::WrongPhase),
        };
        if lease != mlp_lease(operation) {
            return Err(QwenLayerPipelineError::ForeignLease);
        }
        self.state = State::AttentionReady(operation);
        Ok(QwenLayerAbort::RetryMlp(QwenAttentionReady { operation }))
    }
}

fn attention_lease(operation: Operation) -> QwenLayerLease {
    QwenLayerLease {
        operation,
        stage: QwenLayerStage::Attention(operation.attention),
        input: operation.buffers.published,
        output: operation.buffers.scratch,
    }
}

fn mlp_lease(operation: Operation) -> QwenLayerLease {
    QwenLayerLease {
        operation,
        stage: QwenLayerStage::Mlp,
        input: operation.buffers.scratch,
        output: operation.buffers.published,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerPipelineError {
    Busy,
    InvalidOperation,
    InvalidLayer,
    InvalidHiddenRange,
    OverlappingHiddenRanges,
    AddressOverflow,
    EpochOverflow,
    WrongPhase,
    ForeignLease,
    ForeignRestore,
}

impl Display for QwenLayerPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Busy => "Qwen layer pipeline is busy or poisoned",
            Self::InvalidOperation => "Qwen layer token or generation is invalid",
            Self::InvalidLayer => "Qwen layer index is outside the 48-layer checkpoint",
            Self::InvalidHiddenRange => "Qwen coherent hidden range has the wrong address or size",
            Self::OverlappingHiddenRanges => "Qwen coherent hidden ranges overlap",
            Self::AddressOverflow => "Qwen coherent hidden range overflows the address space",
            Self::EpochOverflow => "Qwen layer pipeline epoch overflow",
            Self::WrongPhase => "Qwen layer pipeline is in the wrong phase",
            Self::ForeignLease => "Qwen layer lease is stale or foreign",
            Self::ForeignRestore => "Qwen attention restore belongs to another operation",
        })
    }
}

impl std::error::Error for QwenLayerPipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffers() -> QwenLayerBuffers {
        QwenLayerBuffers::new(
            CoherentHiddenRange {
                address: 0x1000,
                bytes: QWEN_HYPER_BYTES,
            },
            CoherentHiddenRange {
                address: 0x10000,
                bytes: QWEN_HYPER_BYTES,
            },
        )
        .expect("buffers")
    }

    #[test]
    fn checkpoint_pattern_is_three_gdn_then_one_qsa() {
        for layer in 0..QWEN_LAYERS {
            let expected = if (layer + 1) % QWEN_FULL_ATTENTION_INTERVAL == 0 {
                QwenAttentionKind::Qsa
            } else {
                QwenAttentionKind::Gdn
            };
            assert_eq!(qwen_attention_kind(layer), Ok(expected));
        }
        assert_eq!(
            qwen_attention_kind(48),
            Err(QwenLayerPipelineError::InvalidLayer)
        );
    }

    #[test]
    fn full_layer_ping_pongs_two_fixed_hidden_slabs_without_copy() {
        let buffers = buffers();
        let mut pipeline = QwenLayerPipeline::new();
        let attention = pipeline.begin_layer(7, 3, 0, buffers).expect("attention");
        assert_eq!(
            attention.stage(),
            QwenLayerStage::Attention(QwenAttentionKind::Gdn)
        );
        assert_eq!(attention.input(), buffers.published);
        assert_eq!(attention.output(), buffers.scratch);
        let ready = pipeline
            .complete_attention(attention)
            .expect("attention done");
        assert_eq!(ready.mlp_input(), buffers.scratch);
        assert_eq!(ready.mlp_output(), buffers.published);
        let mlp = pipeline.begin_mlp(ready).expect("mlp");
        assert_eq!(mlp.input(), attention.output());
        assert_eq!(mlp.output(), attention.input());
        let published = pipeline.complete_mlp(mlp).expect("publish");
        assert_eq!(published.hidden, buffers.published);
        assert_eq!(pipeline.phase(), QwenLayerPhase::Idle);
    }

    #[test]
    fn attention_failure_quarantines_both_slabs_until_child_state_restore() {
        let mut pipeline = QwenLayerPipeline::new();
        let attention = pipeline.begin_layer(9, 4, 3, buffers()).expect("attention");
        assert_eq!(
            attention.stage(),
            QwenLayerStage::Attention(QwenAttentionKind::Qsa)
        );
        assert_eq!(
            pipeline.abort_attention(attention).expect("abort"),
            QwenLayerAbort::RestoreAttentionState {
                token: 9,
                generation: 4,
                layer: 3,
                kind: QwenAttentionKind::Qsa,
            }
        );
        assert_eq!(
            pipeline.begin_layer(10, 4, 4, buffers()),
            Err(QwenLayerPipelineError::Busy)
        );
        assert_eq!(
            pipeline.restore_attention(10, 4, 3),
            Err(QwenLayerPipelineError::ForeignRestore)
        );
        pipeline.restore_attention(9, 4, 3).expect("restored");
        assert_eq!(pipeline.phase(), QwenLayerPhase::Idle);
    }

    #[test]
    fn mlp_failure_retries_from_intact_attention_scratch() {
        let buffers = buffers();
        let mut pipeline = QwenLayerPipeline::new();
        let attention = pipeline.begin_layer(11, 5, 1, buffers).expect("attention");
        let ready = pipeline.complete_attention(attention).expect("ready");
        let mlp = pipeline.begin_mlp(ready).expect("mlp");
        let retry = match pipeline.abort_mlp(mlp).expect("abort") {
            QwenLayerAbort::RetryMlp(ready) => ready,
            other => panic!("unexpected abort {other:?}"),
        };
        let retried = pipeline.begin_mlp(retry).expect("retry");
        assert_eq!(retried.input(), buffers.scratch);
        assert_eq!(retried.output(), buffers.published);
        pipeline.complete_mlp(retried).expect("publish");
    }

    #[test]
    fn hidden_ranges_are_exact_aligned_and_disjoint() {
        assert_eq!(
            QwenLayerBuffers::new(
                CoherentHiddenRange {
                    address: 1,
                    bytes: QWEN_HYPER_BYTES,
                },
                buffers().scratch,
            ),
            Err(QwenLayerPipelineError::InvalidHiddenRange)
        );
        assert_eq!(
            QwenLayerBuffers::new(
                buffers().published,
                CoherentHiddenRange {
                    address: 0x1100,
                    bytes: QWEN_HYPER_BYTES,
                },
            ),
            Err(QwenLayerPipelineError::OverlappingHiddenRanges)
        );
    }
}
