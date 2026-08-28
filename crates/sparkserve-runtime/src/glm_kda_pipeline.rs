//! Transactional GLM KDA scheduling. Arithmetic stays in the pinned raw CUDA
//! adapters; this module owns fixed activation addresses and fail-closed state
//! publication across convolution and recurrence mutations.

use std::fmt::{Display, Formatter};

use crate::glm_kda::{FP32_BYTES, GLM_KDA_HEAD_DIM, GLM_KDA_HEADS};

pub const GLM_KDA_ARENA_ALIGNMENT: u64 = 256;
pub const GLM_KDA_BOTTLENECK: u64 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmKdaRange {
    pub offset: u64,
    pub bytes: u64,
}

impl GlmKdaRange {
    pub fn end(self) -> Result<u64, GlmKdaPipelineError> {
        self.offset
            .checked_add(self.bytes)
            .ok_or(GlmKdaPipelineError::Overflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmKdaArenaPlan {
    pub q: GlmKdaRange,
    pub k: GlmKdaRange,
    pub v: GlmKdaRange,
    pub log_decay: GlmKdaRange,
    pub gate: GlmKdaRange,
    pub attention_output: GlmKdaRange,
    pub beta: GlmKdaRange,
    pub bottleneck: GlmKdaRange,
    pub arena_bytes: u64,
}

impl GlmKdaArenaPlan {
    pub fn new(sequences: u64, tokens: u64) -> Result<Self, GlmKdaPipelineError> {
        if sequences == 0 || tokens == 0 {
            return Err(GlmKdaPipelineError::InvalidGeometry);
        }
        let vector_bytes = product(&[
            sequences,
            tokens,
            GLM_KDA_HEADS,
            GLM_KDA_HEAD_DIM,
            FP32_BYTES,
        ])?;
        let beta_bytes = product(&[sequences, tokens, GLM_KDA_HEADS, FP32_BYTES])?;
        let bottleneck_bytes = product(&[sequences, tokens, GLM_KDA_BOTTLENECK, FP32_BYTES])?;
        let mut cursor = 0;
        let q = allocate(&mut cursor, vector_bytes)?;
        let k = allocate(&mut cursor, vector_bytes)?;
        let v = allocate(&mut cursor, vector_bytes)?;
        let log_decay = allocate(&mut cursor, vector_bytes)?;
        let gate = allocate(&mut cursor, vector_bytes)?;
        let attention_output = allocate(&mut cursor, vector_bytes)?;
        let beta = allocate(&mut cursor, beta_bytes)?;
        let bottleneck = allocate(&mut cursor, bottleneck_bytes)?;
        let arena_bytes = align_up(cursor, GLM_KDA_ARENA_ALIGNMENT)?;
        Ok(Self {
            q,
            k,
            v,
            log_decay,
            gate,
            attention_output,
            beta,
            bottleneck,
            arena_bytes,
        })
    }

    pub fn ranges(self) -> [GlmKdaRange; 8] {
        [
            self.q,
            self.k,
            self.v,
            self.log_decay,
            self.gate,
            self.attention_output,
            self.beta,
            self.bottleneck,
        ]
    }
}

fn allocate(cursor: &mut u64, bytes: u64) -> Result<GlmKdaRange, GlmKdaPipelineError> {
    let offset = align_up(*cursor, GLM_KDA_ARENA_ALIGNMENT)?;
    *cursor = offset
        .checked_add(bytes)
        .ok_or(GlmKdaPipelineError::Overflow)?;
    Ok(GlmKdaRange { offset, bytes })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GlmKdaPipelineError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GlmKdaPipelineError::Overflow)
}

fn product(values: &[u64]) -> Result<u64, GlmKdaPipelineError> {
    values.iter().try_fold(1_u64, |accumulator, value| {
        accumulator
            .checked_mul(*value)
            .ok_or(GlmKdaPipelineError::Overflow)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmKdaStage {
    Idle,
    Checkpointed,
    Normalized,
    ProjectionsReady,
    ConvInFlight,
    Convolved,
    Prepared,
    RecurrenceInFlight,
    Recurrent,
    Gated,
    OutputProjected,
    Published,
    NeedsRestore { convolution: bool, recurrence: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmKdaStep {
    epoch: u64,
    stage: GlmKdaStage,
}

impl Default for GlmKdaStep {
    fn default() -> Self {
        Self {
            epoch: 0,
            stage: GlmKdaStage::Idle,
        }
    }
}

impl GlmKdaStep {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn stage(&self) -> GlmKdaStage {
        self.stage
    }

    pub fn checkpoint(&mut self) -> Result<u64, GlmKdaPipelineError> {
        self.require(GlmKdaStage::Idle)?;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(GlmKdaPipelineError::Overflow)?;
        self.stage = GlmKdaStage::Checkpointed;
        Ok(self.epoch)
    }

    pub fn normalized(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(epoch, GlmKdaStage::Checkpointed, GlmKdaStage::Normalized)
    }

    pub fn projections_ready(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(
            epoch,
            GlmKdaStage::Normalized,
            GlmKdaStage::ProjectionsReady,
        )
    }

    pub fn begin_convolution(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(
            epoch,
            GlmKdaStage::ProjectionsReady,
            GlmKdaStage::ConvInFlight,
        )
    }

    pub fn finish_convolution(
        &mut self,
        epoch: u64,
        success: bool,
    ) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        self.require(GlmKdaStage::ConvInFlight)?;
        self.stage = if success {
            GlmKdaStage::Convolved
        } else {
            GlmKdaStage::NeedsRestore {
                convolution: true,
                recurrence: false,
            }
        };
        Ok(())
    }

    pub fn prepared(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(epoch, GlmKdaStage::Convolved, GlmKdaStage::Prepared)
    }

    pub fn begin_recurrence(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(
            epoch,
            GlmKdaStage::Prepared,
            GlmKdaStage::RecurrenceInFlight,
        )
    }

    pub fn finish_recurrence(
        &mut self,
        epoch: u64,
        success: bool,
    ) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        self.require(GlmKdaStage::RecurrenceInFlight)?;
        self.stage = if success {
            GlmKdaStage::Recurrent
        } else {
            GlmKdaStage::NeedsRestore {
                convolution: true,
                recurrence: true,
            }
        };
        Ok(())
    }

    pub fn gated(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(epoch, GlmKdaStage::Recurrent, GlmKdaStage::Gated)
    }

    pub fn output_projected(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(epoch, GlmKdaStage::Gated, GlmKdaStage::OutputProjected)
    }

    pub fn publish(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.advance(epoch, GlmKdaStage::OutputProjected, GlmKdaStage::Published)
    }

    pub fn finish_published(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        self.require(GlmKdaStage::Published)?;
        self.stage = GlmKdaStage::Idle;
        Ok(())
    }

    /// A failure after convolution completion must restore convolution state;
    /// after recurrence completion it must restore both state families.
    pub fn fail_compute(&mut self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        self.stage = match self.stage {
            GlmKdaStage::Idle | GlmKdaStage::Published | GlmKdaStage::NeedsRestore { .. } => {
                return Err(GlmKdaPipelineError::InvalidStage);
            }
            GlmKdaStage::Checkpointed | GlmKdaStage::Normalized | GlmKdaStage::ProjectionsReady => {
                GlmKdaStage::Idle
            }
            GlmKdaStage::ConvInFlight | GlmKdaStage::Convolved | GlmKdaStage::Prepared => {
                GlmKdaStage::NeedsRestore {
                    convolution: true,
                    recurrence: false,
                }
            }
            GlmKdaStage::RecurrenceInFlight
            | GlmKdaStage::Recurrent
            | GlmKdaStage::Gated
            | GlmKdaStage::OutputProjected => GlmKdaStage::NeedsRestore {
                convolution: true,
                recurrence: true,
            },
        };
        Ok(())
    }

    pub fn restore(
        &mut self,
        epoch: u64,
        convolution_restored: bool,
        recurrence_restored: bool,
    ) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        let GlmKdaStage::NeedsRestore {
            convolution,
            recurrence,
        } = self.stage
        else {
            return Err(GlmKdaPipelineError::InvalidStage);
        };
        if convolution != convolution_restored || recurrence != recurrence_restored {
            return Err(GlmKdaPipelineError::IncompleteRestore);
        }
        self.stage = GlmKdaStage::Idle;
        Ok(())
    }

    fn advance(
        &mut self,
        epoch: u64,
        expected: GlmKdaStage,
        next: GlmKdaStage,
    ) -> Result<(), GlmKdaPipelineError> {
        self.check_epoch(epoch)?;
        self.require(expected)?;
        self.stage = next;
        Ok(())
    }

    fn check_epoch(&self, epoch: u64) -> Result<(), GlmKdaPipelineError> {
        if self.epoch != epoch {
            return Err(GlmKdaPipelineError::StaleEpoch);
        }
        Ok(())
    }

    fn require(&self, expected: GlmKdaStage) -> Result<(), GlmKdaPipelineError> {
        if self.stage != expected {
            return Err(GlmKdaPipelineError::InvalidStage);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmKdaPipelineError {
    InvalidGeometry,
    InvalidStage,
    StaleEpoch,
    IncompleteRestore,
    Overflow,
}

impl Display for GlmKdaPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGeometry => formatter.write_str("invalid GLM KDA arena geometry"),
            Self::InvalidStage => formatter.write_str("invalid GLM KDA pipeline stage"),
            Self::StaleEpoch => formatter.write_str("stale GLM KDA pipeline epoch"),
            Self::IncompleteRestore => {
                formatter.write_str("GLM KDA state restore set is incomplete")
            }
            Self::Overflow => formatter.write_str("GLM KDA pipeline geometry overflows u64"),
        }
    }
}

impl std::error::Error for GlmKdaPipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_arena_is_fixed_aligned_and_disjoint() {
        let plan = GlmKdaArenaPlan::new(1, 1).expect("arena");
        assert_eq!(plan.arena_bytes, 197_376);
        let ranges = plan.ranges();
        for (index, range) in ranges.iter().enumerate() {
            assert_eq!(range.offset % GLM_KDA_ARENA_ALIGNMENT, 0);
            for other in &ranges[index + 1..] {
                assert!(range.end().expect("end") <= other.offset);
            }
        }
    }

    #[test]
    fn successful_step_cannot_publish_before_every_stage() {
        let mut step = GlmKdaStep::default();
        let epoch = step.checkpoint().expect("checkpoint");
        assert_eq!(step.publish(epoch), Err(GlmKdaPipelineError::InvalidStage));
        step.normalized(epoch).expect("norm");
        step.projections_ready(epoch).expect("projections");
        step.begin_convolution(epoch).expect("conv begin");
        step.finish_convolution(epoch, true).expect("conv finish");
        step.prepared(epoch).expect("prepare");
        step.begin_recurrence(epoch).expect("recurrence begin");
        step.finish_recurrence(epoch, true)
            .expect("recurrence finish");
        step.gated(epoch).expect("gate");
        step.output_projected(epoch).expect("output projection");
        step.publish(epoch).expect("publish");
        step.finish_published(epoch).expect("finish");
        assert_eq!(step.stage(), GlmKdaStage::Idle);
    }

    #[test]
    fn recurrence_failure_requires_both_state_families_to_restore() {
        let mut step = GlmKdaStep::default();
        let epoch = step.checkpoint().expect("checkpoint");
        step.normalized(epoch).expect("norm");
        step.projections_ready(epoch).expect("projections");
        step.begin_convolution(epoch).expect("conv begin");
        step.finish_convolution(epoch, true).expect("conv finish");
        step.prepared(epoch).expect("prepare");
        step.begin_recurrence(epoch).expect("recurrence begin");
        step.finish_recurrence(epoch, false).expect("failure");
        assert_eq!(
            step.stage(),
            GlmKdaStage::NeedsRestore {
                convolution: true,
                recurrence: true
            }
        );
        assert_eq!(
            step.restore(epoch, true, false),
            Err(GlmKdaPipelineError::IncompleteRestore)
        );
        step.restore(epoch, true, true).expect("complete restore");
        assert_eq!(step.stage(), GlmKdaStage::Idle);
    }

    #[test]
    fn pre_mutation_failure_retries_without_restore_and_rejects_stale_epoch() {
        let mut step = GlmKdaStep::default();
        let first = step.checkpoint().expect("checkpoint");
        step.normalized(first).expect("norm");
        step.fail_compute(first).expect("fail before mutation");
        let second = step.checkpoint().expect("retry checkpoint");
        assert!(second > first);
        assert_eq!(step.normalized(first), Err(GlmKdaPipelineError::StaleEpoch));
    }
}
