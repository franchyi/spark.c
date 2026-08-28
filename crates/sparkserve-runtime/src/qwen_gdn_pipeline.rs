//! Rust ownership for the Qwen GDN attention half-layer.
//!
//! Borrowed CUDA kernels own arithmetic only. This state machine owns the
//! stage order and requires paired convolution/temporal-state snapshots before
//! either persistent pool can be changed. A failed stateful token is poisoned
//! until its exact checkpoint is restored; no half-updated token is published.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

const STATE_ALIGNMENT: u64 = 256;
static NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGdnStage {
    MhcMix,
    Prepare,
    Recurrent,
    Finish,
    MhcCombine,
}

impl QwenGdnStage {
    fn next(self) -> Option<Self> {
        match self {
            Self::MhcMix => Some(Self::Prepare),
            Self::Prepare => Some(Self::Recurrent),
            Self::Recurrent => Some(Self::Finish),
            Self::Finish => Some(Self::MhcCombine),
            Self::MhcCombine => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnStateRange {
    pub offset: u64,
    pub bytes: u64,
}

impl GdnStateRange {
    fn end(self) -> Result<u64, QwenGdnPipelineError> {
        self.offset
            .checked_add(self.bytes)
            .ok_or(QwenGdnPipelineError::InvalidCheckpoint)
    }

    fn validate(self) -> Result<(), QwenGdnPipelineError> {
        if self.bytes == 0
            || !self.offset.is_multiple_of(STATE_ALIGNMENT)
            || !self.bytes.is_multiple_of(STATE_ALIGNMENT)
        {
            return Err(QwenGdnPipelineError::InvalidCheckpoint);
        }
        self.end().map(|_| ())
    }
}

/// Snapshot locations in the caller-owned coherent slab. Both ranges must be
/// filled on the same CUDA stream before `Prepare` starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnStateCheckpoint {
    pub operation: u64,
    pub sequence_slot: u32,
    pub generation: u64,
    pub convolution: GdnStateRange,
    pub temporal: GdnStateRange,
}

impl GdnStateCheckpoint {
    pub fn new(
        operation: u64,
        sequence_slot: u32,
        generation: u64,
        convolution: GdnStateRange,
        temporal: GdnStateRange,
    ) -> Result<Self, QwenGdnPipelineError> {
        if operation == 0 || generation == 0 {
            return Err(QwenGdnPipelineError::InvalidCheckpoint);
        }
        convolution.validate()?;
        temporal.validate()?;
        if convolution.offset < temporal.end()? && temporal.offset < convolution.end()? {
            return Err(QwenGdnPipelineError::OverlappingCheckpointRanges);
        }
        Ok(Self {
            operation,
            sequence_slot,
            generation,
            convolution,
            temporal,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenGdnLease {
    pipeline: u64,
    operation: u64,
    sequence_slot: u32,
    generation: u64,
    stage: QwenGdnStage,
}

impl QwenGdnLease {
    pub fn stage(self) -> QwenGdnStage {
        self.stage
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenGdnReady {
    pipeline: u64,
    operation: u64,
    sequence_slot: u32,
    generation: u64,
    stage: QwenGdnStage,
}

impl QwenGdnReady {
    pub fn stage(self) -> QwenGdnStage {
        self.stage
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGdnCompletion {
    StageReady(QwenGdnReady),
    Published {
        operation: u64,
        checkpoint: GdnStateCheckpoint,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGdnAbort {
    RetrySafe,
    RestorePairedState(GdnStateCheckpoint),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGdnPhase {
    Idle,
    Running(QwenGdnStage),
    Ready(QwenGdnStage),
    Poisoned(QwenGdnStage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Active {
    operation: u64,
    sequence_slot: u32,
    generation: u64,
    stage: QwenGdnStage,
    checkpoint: Option<GdnStateCheckpoint>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Idle,
    Running(Active),
    Ready(Active),
    Poisoned(Active),
}

pub struct QwenGdnPipeline {
    id: u64,
    state: State,
}

impl Default for QwenGdnPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenGdnPipeline {
    pub fn new() -> Self {
        Self {
            id: NEXT_PIPELINE_ID.fetch_add(1, Ordering::Relaxed),
            state: State::Idle,
        }
    }

    pub fn phase(&self) -> QwenGdnPhase {
        match self.state {
            State::Idle => QwenGdnPhase::Idle,
            State::Running(active) => QwenGdnPhase::Running(active.stage),
            State::Ready(active) => QwenGdnPhase::Ready(active.stage),
            State::Poisoned(active) => QwenGdnPhase::Poisoned(active.stage),
        }
    }

    pub fn begin(
        &mut self,
        operation: u64,
        sequence_slot: u32,
        generation: u64,
    ) -> Result<QwenGdnLease, QwenGdnPipelineError> {
        if !matches!(self.state, State::Idle) {
            return Err(QwenGdnPipelineError::Busy);
        }
        if operation == 0 || generation == 0 {
            return Err(QwenGdnPipelineError::InvalidOperation);
        }
        let active = Active {
            operation,
            sequence_slot,
            generation,
            stage: QwenGdnStage::MhcMix,
            checkpoint: None,
        };
        self.state = State::Running(active);
        Ok(self.lease(active))
    }

    pub fn complete(
        &mut self,
        lease: QwenGdnLease,
    ) -> Result<QwenGdnCompletion, QwenGdnPipelineError> {
        let active = match self.state {
            State::Running(active) => active,
            _ => return Err(QwenGdnPipelineError::WrongPhase),
        };
        self.validate_lease(lease, active)?;
        if active.stage == QwenGdnStage::MhcCombine {
            let checkpoint = active
                .checkpoint
                .ok_or(QwenGdnPipelineError::MissingCheckpoint)?;
            self.state = State::Idle;
            return Ok(QwenGdnCompletion::Published {
                operation: active.operation,
                checkpoint,
            });
        }
        self.state = State::Ready(active);
        Ok(QwenGdnCompletion::StageReady(self.ready(active)))
    }

    /// Install the two-state snapshot and start the first mutating stage.
    pub fn checkpoint_and_begin_prepare(
        &mut self,
        ready: QwenGdnReady,
        checkpoint: GdnStateCheckpoint,
    ) -> Result<QwenGdnLease, QwenGdnPipelineError> {
        let mut active = self.ready_active(ready)?;
        if active.stage != QwenGdnStage::MhcMix {
            return Err(QwenGdnPipelineError::WrongStage);
        }
        if checkpoint.operation != active.operation
            || checkpoint.sequence_slot != active.sequence_slot
            || checkpoint.generation != active.generation
        {
            return Err(QwenGdnPipelineError::ForeignCheckpoint);
        }
        active.stage = QwenGdnStage::Prepare;
        active.checkpoint = Some(checkpoint);
        self.state = State::Running(active);
        Ok(self.lease(active))
    }

    pub fn begin_next(
        &mut self,
        ready: QwenGdnReady,
    ) -> Result<QwenGdnLease, QwenGdnPipelineError> {
        let mut active = self.ready_active(ready)?;
        if active.stage == QwenGdnStage::MhcMix {
            return Err(QwenGdnPipelineError::MissingCheckpoint);
        }
        active.stage = active
            .stage
            .next()
            .ok_or(QwenGdnPipelineError::NoNextStage)?;
        if active.checkpoint.is_none() {
            return Err(QwenGdnPipelineError::MissingCheckpoint);
        }
        self.state = State::Running(active);
        Ok(self.lease(active))
    }

    pub fn abort_running(
        &mut self,
        lease: QwenGdnLease,
    ) -> Result<QwenGdnAbort, QwenGdnPipelineError> {
        let active = match self.state {
            State::Running(active) => active,
            _ => return Err(QwenGdnPipelineError::WrongPhase),
        };
        self.validate_lease(lease, active)?;
        self.abort(active)
    }

    pub fn abort_ready(
        &mut self,
        ready: QwenGdnReady,
    ) -> Result<QwenGdnAbort, QwenGdnPipelineError> {
        let active = self.ready_active(ready)?;
        self.abort(active)
    }

    pub fn restore(&mut self, checkpoint: GdnStateCheckpoint) -> Result<(), QwenGdnPipelineError> {
        let active = match self.state {
            State::Poisoned(active) => active,
            _ => return Err(QwenGdnPipelineError::WrongPhase),
        };
        if active.checkpoint != Some(checkpoint) {
            return Err(QwenGdnPipelineError::ForeignCheckpoint);
        }
        self.state = State::Idle;
        Ok(())
    }

    fn abort(&mut self, active: Active) -> Result<QwenGdnAbort, QwenGdnPipelineError> {
        match active.checkpoint {
            None => {
                self.state = State::Idle;
                Ok(QwenGdnAbort::RetrySafe)
            }
            Some(checkpoint) => {
                self.state = State::Poisoned(active);
                Ok(QwenGdnAbort::RestorePairedState(checkpoint))
            }
        }
    }

    fn ready_active(&self, ready: QwenGdnReady) -> Result<Active, QwenGdnPipelineError> {
        let active = match self.state {
            State::Ready(active) => active,
            _ => return Err(QwenGdnPipelineError::WrongPhase),
        };
        if ready != self.ready(active) {
            return Err(QwenGdnPipelineError::ForeignLease);
        }
        Ok(active)
    }

    fn validate_lease(
        &self,
        lease: QwenGdnLease,
        active: Active,
    ) -> Result<(), QwenGdnPipelineError> {
        if lease != self.lease(active) {
            return Err(QwenGdnPipelineError::ForeignLease);
        }
        Ok(())
    }

    fn lease(&self, active: Active) -> QwenGdnLease {
        QwenGdnLease {
            pipeline: self.id,
            operation: active.operation,
            sequence_slot: active.sequence_slot,
            generation: active.generation,
            stage: active.stage,
        }
    }

    fn ready(&self, active: Active) -> QwenGdnReady {
        QwenGdnReady {
            pipeline: self.id,
            operation: active.operation,
            sequence_slot: active.sequence_slot,
            generation: active.generation,
            stage: active.stage,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGdnPipelineError {
    Busy,
    InvalidOperation,
    InvalidCheckpoint,
    OverlappingCheckpointRanges,
    ForeignCheckpoint,
    ForeignLease,
    WrongPhase,
    WrongStage,
    MissingCheckpoint,
    NoNextStage,
}

impl Display for QwenGdnPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Busy => "Qwen GDN pipeline is busy or poisoned",
            Self::InvalidOperation => "Qwen GDN operation or generation is invalid",
            Self::InvalidCheckpoint => "Qwen GDN checkpoint range is invalid",
            Self::OverlappingCheckpointRanges => "Qwen GDN checkpoint ranges overlap",
            Self::ForeignCheckpoint => "Qwen GDN checkpoint belongs to another token",
            Self::ForeignLease => "Qwen GDN lease is stale or foreign",
            Self::WrongPhase => "Qwen GDN pipeline is in the wrong phase",
            Self::WrongStage => "Qwen GDN pipeline is in the wrong stage",
            Self::MissingCheckpoint => "Qwen GDN paired-state checkpoint is missing",
            Self::NoNextStage => "Qwen GDN pipeline has no next stage",
        })
    }
}

impl std::error::Error for QwenGdnPipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(operation: u64) -> GdnStateCheckpoint {
        GdnStateCheckpoint::new(
            operation,
            3,
            9,
            GdnStateRange {
                offset: 0,
                bytes: 61_440,
            },
            GdnStateRange {
                offset: 61_440,
                bytes: 1_572_864,
            },
        )
        .expect("valid checkpoint")
    }

    fn ready(completion: QwenGdnCompletion) -> QwenGdnReady {
        match completion {
            QwenGdnCompletion::StageReady(ready) => ready,
            QwenGdnCompletion::Published { .. } => panic!("expected ready stage"),
        }
    }

    #[test]
    fn exact_stage_order_requires_checkpoint_before_mutation() {
        let mut pipeline = QwenGdnPipeline::new();
        let mix = pipeline.begin(7, 3, 9).expect("begin");
        assert_eq!(mix.stage(), QwenGdnStage::MhcMix);
        let mix_ready = ready(pipeline.complete(mix).expect("complete mix"));
        assert_eq!(
            pipeline.begin_next(mix_ready),
            Err(QwenGdnPipelineError::MissingCheckpoint)
        );
        let prepare = pipeline
            .checkpoint_and_begin_prepare(mix_ready, checkpoint(7))
            .expect("checkpoint and prepare");
        assert_eq!(prepare.stage(), QwenGdnStage::Prepare);
        let prepare_ready = ready(pipeline.complete(prepare).expect("prepare"));
        let recurrent = pipeline.begin_next(prepare_ready).expect("recurrent");
        assert_eq!(recurrent.stage(), QwenGdnStage::Recurrent);
        let recurrent_ready = ready(pipeline.complete(recurrent).expect("recurrent"));
        let finish = pipeline.begin_next(recurrent_ready).expect("finish");
        let finish_ready = ready(pipeline.complete(finish).expect("finish"));
        let combine = pipeline.begin_next(finish_ready).expect("combine");
        let published = pipeline.complete(combine).expect("publish");
        assert_eq!(
            published,
            QwenGdnCompletion::Published {
                operation: 7,
                checkpoint: checkpoint(7)
            }
        );
        assert_eq!(pipeline.phase(), QwenGdnPhase::Idle);
    }

    #[test]
    fn failed_stateful_stage_requires_both_ranges_to_be_restored() {
        let mut pipeline = QwenGdnPipeline::new();
        let mix = pipeline.begin(11, 3, 9).expect("begin");
        let mix_ready = ready(pipeline.complete(mix).expect("mix"));
        let prepare = pipeline
            .checkpoint_and_begin_prepare(mix_ready, checkpoint(11))
            .expect("prepare");
        assert_eq!(
            pipeline.abort_running(prepare).expect("abort"),
            QwenGdnAbort::RestorePairedState(checkpoint(11))
        );
        assert_eq!(
            pipeline.phase(),
            QwenGdnPhase::Poisoned(QwenGdnStage::Prepare)
        );
        assert_eq!(pipeline.begin(12, 3, 9), Err(QwenGdnPipelineError::Busy));
        pipeline.restore(checkpoint(11)).expect("restore");
        assert_eq!(pipeline.phase(), QwenGdnPhase::Idle);
    }

    #[test]
    fn pre_checkpoint_failure_is_retry_safe() {
        let mut pipeline = QwenGdnPipeline::new();
        let mix = pipeline.begin(13, 3, 9).expect("begin");
        assert_eq!(
            pipeline.abort_running(mix).expect("abort"),
            QwenGdnAbort::RetrySafe
        );
        assert_eq!(pipeline.phase(), QwenGdnPhase::Idle);
    }

    #[test]
    fn checkpoint_ranges_are_aligned_disjoint_and_token_scoped() {
        assert_eq!(
            GdnStateCheckpoint::new(
                1,
                0,
                1,
                GdnStateRange {
                    offset: 1,
                    bytes: 256
                },
                GdnStateRange {
                    offset: 512,
                    bytes: 256
                }
            ),
            Err(QwenGdnPipelineError::InvalidCheckpoint)
        );
        assert_eq!(
            GdnStateCheckpoint::new(
                1,
                0,
                1,
                GdnStateRange {
                    offset: 0,
                    bytes: 512
                },
                GdnStateRange {
                    offset: 256,
                    bytes: 512
                }
            ),
            Err(QwenGdnPipelineError::OverlappingCheckpointRanges)
        );
    }
}
