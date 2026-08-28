//! Qwen MoE scheduling around borrowed CUDA arithmetic.
//!
//! The gate, routed NVFP4 chain, shared BF16 branch, and final join remain raw
//! kernel calls. Rust owns the useful work between them: coherent top-k handoff,
//! transactional expert fills, fixed-slot publication, overlap, and failure
//! recovery. A CUDA event completion is the only valid input to `complete_stage`.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fabric::{ExpertCacheStats, ExpertLoad};
use crate::scheduler::{
    MoeScheduleError, MoeSchedulerConfig, PendingMoeStep, ReadyMoeStep, RoutedMoeScheduler,
};

static NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenMoeStage {
    Gate,
    Shared,
    Routed,
    Join,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchStatus {
    Waiting,
    Running,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenMoePhase {
    Idle,
    GateRunning,
    Branches {
        expert_io_complete: bool,
        shared: BranchStatus,
        routed: BranchStatus,
        join: BranchStatus,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenMoeOperation {
    pipeline_id: u64,
    epoch: u64,
    layer: u16,
    num_tokens: u32,
}

impl QwenMoeOperation {
    pub fn layer(self) -> u16 {
        self.layer
    }

    pub fn num_tokens(self) -> u32 {
        self.num_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenMoeLease {
    operation: QwenMoeOperation,
    stage: QwenMoeStage,
}

impl QwenMoeLease {
    pub fn operation(self) -> QwenMoeOperation {
        self.operation
    }

    pub fn stage(self) -> QwenMoeStage {
        self.stage
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QwenMoePipelineStats {
    pub tokens_started: u64,
    pub tokens_completed: u64,
    pub expert_fills_completed: u64,
    pub shared_stages_completed: u64,
    pub routed_stages_completed: u64,
    pub joins_completed: u64,
    pub retries: u64,
}

struct ActiveOperation {
    operation: QwenMoeOperation,
    pending: Option<PendingMoeStep>,
    ready: Option<ReadyMoeStep>,
    shared: BranchStatus,
    routed: BranchStatus,
    join: BranchStatus,
}

// Keep the fixed route inline: boxing would allocate once per token exactly on
// the latency-sensitive gate handoff this scheduler is meant to control.
#[allow(clippy::large_enum_variant)]
enum PipelineState {
    Idle,
    GateRunning(QwenMoeOperation),
    Branches(ActiveOperation),
}

/// Allocation-bounded policy scheduler for one Qwen MoE layer at a time.
///
/// `begin_shared` is legal immediately after gate completion, while NVMe fills
/// are still outstanding. `begin_routed` becomes legal only after every fixed
/// expert slot is published. `begin_join` requires both CUDA branches to have
/// completed their events.
pub struct QwenMoePipeline {
    pipeline_id: u64,
    next_epoch: u64,
    routes: RoutedMoeScheduler,
    state: PipelineState,
    stats: QwenMoePipelineStats,
}

impl QwenMoePipeline {
    pub fn new(config: MoeSchedulerConfig) -> Result<Self, QwenMoePipelineError> {
        let pipeline_id = NEXT_PIPELINE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| QwenMoePipelineError::IdOverflow)?;
        Ok(Self {
            pipeline_id,
            next_epoch: 0,
            routes: RoutedMoeScheduler::new(config)?,
            state: PipelineState::Idle,
            stats: QwenMoePipelineStats::default(),
        })
    }

    pub fn phase(&self) -> QwenMoePhase {
        match &self.state {
            PipelineState::Idle => QwenMoePhase::Idle,
            PipelineState::GateRunning(_) => QwenMoePhase::GateRunning,
            PipelineState::Branches(active) => QwenMoePhase::Branches {
                expert_io_complete: active.ready.is_some(),
                shared: active.shared,
                routed: active.routed,
                join: active.join,
            },
        }
    }

    pub fn stats(&self) -> QwenMoePipelineStats {
        self.stats
    }

    pub fn cache_stats(&self) -> ExpertCacheStats {
        self.routes.cache_stats()
    }

    pub fn begin_gate(
        &mut self,
        layer: u16,
        num_tokens: u32,
    ) -> Result<QwenMoeLease, QwenMoePipelineError> {
        if !matches!(self.state, PipelineState::Idle) {
            return Err(QwenMoePipelineError::Busy);
        }
        let epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(QwenMoePipelineError::EpochOverflow)?;
        self.next_epoch = epoch;
        let operation = QwenMoeOperation {
            pipeline_id: self.pipeline_id,
            epoch,
            layer,
            num_tokens,
        };
        self.state = PipelineState::GateRunning(operation);
        self.stats.tokens_started = self.stats.tokens_started.saturating_add(1);
        Ok(QwenMoeLease {
            operation,
            stage: QwenMoeStage::Gate,
        })
    }

    /// Consume ids only after the gate stream's CUDA event has completed.
    pub fn complete_gate(
        &mut self,
        lease: QwenMoeLease,
        coherent_expert_ids: &[i32],
    ) -> Result<QwenMoeOperation, QwenMoePipelineError> {
        self.validate_gate(lease)?;
        let pending = match self.routes.prepare_gate_step(
            lease.operation.layer,
            lease.operation.num_tokens,
            coherent_expert_ids,
        ) {
            Ok(pending) => pending,
            Err(error) => {
                self.state = PipelineState::Idle;
                return Err(error.into());
            }
        };
        self.state = PipelineState::Branches(ActiveOperation {
            operation: lease.operation,
            pending: Some(pending),
            ready: None,
            shared: BranchStatus::Waiting,
            routed: BranchStatus::Waiting,
            join: BranchStatus::Waiting,
        });
        Ok(lease.operation)
    }

    pub fn abort_gate(&mut self, lease: QwenMoeLease) -> Result<(), QwenMoePipelineError> {
        self.validate_gate(lease)?;
        self.state = PipelineState::Idle;
        self.stats.retries = self.stats.retries.saturating_add(1);
        Ok(())
    }

    pub fn expert_loads(
        &self,
        operation: QwenMoeOperation,
    ) -> Result<&[ExpertLoad], QwenMoePipelineError> {
        let active = self.validate_operation(operation)?;
        active
            .pending
            .as_ref()
            .map(PendingMoeStep::loads)
            .ok_or(QwenMoePipelineError::ExpertIoAlreadyComplete)
    }

    pub fn complete_expert_io(
        &mut self,
        operation: QwenMoeOperation,
    ) -> Result<(), QwenMoePipelineError> {
        self.validate_operation(operation)?;
        let pending = match &mut self.state {
            PipelineState::Branches(active) => active
                .pending
                .take()
                .ok_or(QwenMoePipelineError::ExpertIoAlreadyComplete)?,
            _ => return Err(QwenMoePipelineError::WrongState),
        };
        let ready = match self.routes.commit_step(pending) {
            Ok(ready) => ready,
            Err(error) => {
                self.state = PipelineState::Idle;
                self.stats.retries = self.stats.retries.saturating_add(1);
                return Err(error.into());
            }
        };
        match &mut self.state {
            PipelineState::Branches(active) => active.ready = Some(ready),
            _ => return Err(QwenMoePipelineError::WrongState),
        }
        self.stats.expert_fills_completed = self.stats.expert_fills_completed.saturating_add(1);
        Ok(())
    }

    pub fn abort_expert_io(
        &mut self,
        operation: QwenMoeOperation,
    ) -> Result<(), QwenMoePipelineError> {
        self.validate_operation(operation)?;
        self.state = PipelineState::Idle;
        self.stats.retries = self.stats.retries.saturating_add(1);
        Ok(())
    }

    pub fn ready_step(
        &self,
        operation: QwenMoeOperation,
    ) -> Result<&ReadyMoeStep, QwenMoePipelineError> {
        self.validate_operation(operation)?
            .ready
            .as_ref()
            .ok_or(QwenMoePipelineError::ExpertIoPending)
    }

    pub fn begin_shared(
        &mut self,
        operation: QwenMoeOperation,
    ) -> Result<QwenMoeLease, QwenMoePipelineError> {
        self.begin_branch(operation, QwenMoeStage::Shared)
    }

    pub fn begin_routed(
        &mut self,
        operation: QwenMoeOperation,
    ) -> Result<QwenMoeLease, QwenMoePipelineError> {
        if self.validate_operation(operation)?.ready.is_none() {
            return Err(QwenMoePipelineError::ExpertIoPending);
        }
        self.begin_branch(operation, QwenMoeStage::Routed)
    }

    pub fn begin_join(
        &mut self,
        operation: QwenMoeOperation,
    ) -> Result<QwenMoeLease, QwenMoePipelineError> {
        let active = self.validate_operation(operation)?;
        if active.shared != BranchStatus::Complete || active.routed != BranchStatus::Complete {
            return Err(QwenMoePipelineError::BranchesPending);
        }
        self.begin_branch(operation, QwenMoeStage::Join)
    }

    /// Publish a CUDA stage only after its completion event succeeds.
    pub fn complete_stage(
        &mut self,
        lease: QwenMoeLease,
    ) -> Result<Option<ReadyMoeStep>, QwenMoePipelineError> {
        self.validate_lease(lease, BranchStatus::Running)?;
        if lease.stage == QwenMoeStage::Join {
            let state = std::mem::replace(&mut self.state, PipelineState::Idle);
            let mut active = match state {
                PipelineState::Branches(active) => active,
                other => {
                    self.state = other;
                    return Err(QwenMoePipelineError::WrongState);
                }
            };
            self.stats.joins_completed = self.stats.joins_completed.saturating_add(1);
            self.stats.tokens_completed = self.stats.tokens_completed.saturating_add(1);
            return active
                .ready
                .take()
                .map(Some)
                .ok_or(QwenMoePipelineError::ExpertIoPending);
        }
        let active = match &mut self.state {
            PipelineState::Branches(active) => active,
            _ => return Err(QwenMoePipelineError::WrongState),
        };
        match lease.stage {
            QwenMoeStage::Shared => {
                active.shared = BranchStatus::Complete;
                self.stats.shared_stages_completed =
                    self.stats.shared_stages_completed.saturating_add(1);
            }
            QwenMoeStage::Routed => {
                active.routed = BranchStatus::Complete;
                self.stats.routed_stages_completed =
                    self.stats.routed_stages_completed.saturating_add(1);
            }
            QwenMoeStage::Gate | QwenMoeStage::Join => {
                return Err(QwenMoePipelineError::WrongStage);
            }
        }
        Ok(None)
    }

    pub fn abort_stage(&mut self, lease: QwenMoeLease) -> Result<(), QwenMoePipelineError> {
        self.validate_lease(lease, BranchStatus::Running)?;
        let active = match &mut self.state {
            PipelineState::Branches(active) => active,
            _ => return Err(QwenMoePipelineError::WrongState),
        };
        *branch_status_mut(active, lease.stage)? = BranchStatus::Waiting;
        self.stats.retries = self.stats.retries.saturating_add(1);
        Ok(())
    }

    fn begin_branch(
        &mut self,
        operation: QwenMoeOperation,
        stage: QwenMoeStage,
    ) -> Result<QwenMoeLease, QwenMoePipelineError> {
        let active = match &mut self.state {
            PipelineState::Branches(active) if active.operation == operation => active,
            PipelineState::Branches(_) => return Err(QwenMoePipelineError::StaleOperation),
            _ => return Err(QwenMoePipelineError::WrongState),
        };
        let status = branch_status_mut(active, stage)?;
        if *status != BranchStatus::Waiting {
            return Err(QwenMoePipelineError::StageBusyOrComplete);
        }
        *status = BranchStatus::Running;
        Ok(QwenMoeLease { operation, stage })
    }

    fn validate_gate(&self, lease: QwenMoeLease) -> Result<(), QwenMoePipelineError> {
        if lease.operation.pipeline_id != self.pipeline_id {
            return Err(QwenMoePipelineError::ForeignLease);
        }
        if lease.stage != QwenMoeStage::Gate {
            return Err(QwenMoePipelineError::WrongStage);
        }
        match self.state {
            PipelineState::GateRunning(operation) if operation == lease.operation => Ok(()),
            PipelineState::GateRunning(_) => Err(QwenMoePipelineError::StaleOperation),
            _ => Err(QwenMoePipelineError::WrongState),
        }
    }

    fn validate_operation(
        &self,
        operation: QwenMoeOperation,
    ) -> Result<&ActiveOperation, QwenMoePipelineError> {
        if operation.pipeline_id != self.pipeline_id {
            return Err(QwenMoePipelineError::ForeignLease);
        }
        match &self.state {
            PipelineState::Branches(active) if active.operation == operation => Ok(active),
            PipelineState::Branches(_) => Err(QwenMoePipelineError::StaleOperation),
            _ => Err(QwenMoePipelineError::WrongState),
        }
    }

    fn validate_lease(
        &self,
        lease: QwenMoeLease,
        expected: BranchStatus,
    ) -> Result<(), QwenMoePipelineError> {
        let active = self.validate_operation(lease.operation)?;
        if *branch_status(active, lease.stage)? != expected {
            return Err(QwenMoePipelineError::StageNotRunning);
        }
        Ok(())
    }
}

fn branch_status(
    active: &ActiveOperation,
    stage: QwenMoeStage,
) -> Result<&BranchStatus, QwenMoePipelineError> {
    match stage {
        QwenMoeStage::Shared => Ok(&active.shared),
        QwenMoeStage::Routed => Ok(&active.routed),
        QwenMoeStage::Join => Ok(&active.join),
        QwenMoeStage::Gate => Err(QwenMoePipelineError::WrongStage),
    }
}

fn branch_status_mut(
    active: &mut ActiveOperation,
    stage: QwenMoeStage,
) -> Result<&mut BranchStatus, QwenMoePipelineError> {
    match stage {
        QwenMoeStage::Shared => Ok(&mut active.shared),
        QwenMoeStage::Routed => Ok(&mut active.routed),
        QwenMoeStage::Join => Ok(&mut active.join),
        QwenMoeStage::Gate => Err(QwenMoePipelineError::WrongStage),
    }
}

#[derive(Debug)]
pub enum QwenMoePipelineError {
    Busy,
    WrongState,
    WrongStage,
    ForeignLease,
    StaleOperation,
    StageBusyOrComplete,
    StageNotRunning,
    ExpertIoPending,
    ExpertIoAlreadyComplete,
    BranchesPending,
    EpochOverflow,
    IdOverflow,
    Moe(MoeScheduleError),
}

impl From<MoeScheduleError> for QwenMoePipelineError {
    fn from(error: MoeScheduleError) -> Self {
        Self::Moe(error)
    }
}

impl Display for QwenMoePipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => formatter.write_str("Qwen MoE pipeline is busy"),
            Self::WrongState => formatter.write_str("Qwen MoE operation is in the wrong state"),
            Self::WrongStage => formatter.write_str("Qwen MoE lease names the wrong stage"),
            Self::ForeignLease => formatter.write_str("Qwen MoE lease belongs to another pipeline"),
            Self::StaleOperation => formatter.write_str("Qwen MoE operation is stale"),
            Self::StageBusyOrComplete => formatter.write_str("Qwen MoE stage already started"),
            Self::StageNotRunning => formatter.write_str("Qwen MoE stage is not running"),
            Self::ExpertIoPending => formatter.write_str("expert I/O has not been published"),
            Self::ExpertIoAlreadyComplete => formatter.write_str("expert I/O is already complete"),
            Self::BranchesPending => {
                formatter.write_str("shared and routed branches are not complete")
            }
            Self::EpochOverflow => formatter.write_str("Qwen MoE epoch overflow"),
            Self::IdOverflow => formatter.write_str("Qwen MoE pipeline id overflow"),
            Self::Moe(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for QwenMoePipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MoeSchedulerConfig {
        MoeSchedulerConfig {
            layers: 48,
            num_experts: 512,
            top_k: 10,
            hidden_size: 2560,
            expert_slots: 16,
            expert_slot_bytes: 4096,
        }
    }

    #[test]
    fn overlaps_shared_with_expert_fill_and_joins_only_after_both_branches() {
        let mut pipeline = QwenMoePipeline::new(config()).expect("pipeline");
        let gate = pipeline.begin_gate(0, 1).expect("gate");
        let operation = pipeline
            .complete_gate(gate, &[399, 27, 334, 37, 139, 35, 322, 67, 225, 151])
            .expect("coherent gate handoff");
        assert_eq!(pipeline.expert_loads(operation).expect("loads").len(), 10);

        let shared = pipeline.begin_shared(operation).expect("shared overlap");
        assert!(matches!(
            pipeline.begin_routed(operation),
            Err(QwenMoePipelineError::ExpertIoPending)
        ));
        pipeline
            .complete_expert_io(operation)
            .expect("publish slots");
        let ready = pipeline.ready_step(operation).expect("ready route");
        assert_eq!(ready.execution_route.num_experts, 16);
        assert_eq!(ready.execution_route.active_experts(), 10);
        assert_eq!(ready.execution_route.grouped.total_rows, 40);

        let routed = pipeline.begin_routed(operation).expect("routed");
        pipeline.complete_stage(shared).expect("shared event");
        assert!(matches!(
            pipeline.begin_join(operation),
            Err(QwenMoePipelineError::BranchesPending)
        ));
        pipeline.complete_stage(routed).expect("routed event");
        let join = pipeline.begin_join(operation).expect("join");
        let finished = pipeline
            .complete_stage(join)
            .expect("join event")
            .expect("ready telemetry");
        assert_eq!(finished.route.route_experts.len(), 10);
        assert_eq!(pipeline.phase(), QwenMoePhase::Idle);
        assert_eq!(pipeline.stats().tokens_completed, 1);
        assert_eq!(pipeline.cache_stats().misses, 10);
    }

    #[test]
    fn failed_io_and_cuda_stages_retry_without_publishing_partial_work() {
        let mut pipeline = QwenMoePipeline::new(config()).expect("pipeline");
        let gate = pipeline.begin_gate(1, 1).expect("gate");
        let operation = pipeline
            .complete_gate(gate, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .expect("gate handoff");
        let shared = pipeline.begin_shared(operation).expect("shared");
        pipeline.abort_stage(shared).expect("retry shared");
        assert!(pipeline.begin_shared(operation).is_ok());
        pipeline.abort_expert_io(operation).expect("failed storage");
        assert_eq!(pipeline.cache_stats(), ExpertCacheStats::default());
        assert_eq!(pipeline.phase(), QwenMoePhase::Idle);

        let next_gate = pipeline.begin_gate(1, 1).expect("next gate");
        assert!(matches!(
            pipeline.complete_gate(gate, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            Err(QwenMoePipelineError::StaleOperation)
        ));
        pipeline.abort_gate(next_gate).expect("abort current gate");
    }
}
