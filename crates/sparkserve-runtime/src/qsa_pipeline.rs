//! End-to-end QSA token ordering owned by Rust.
//!
//! Arithmetic donors operate on raw pointers and CUDA streams. This module
//! makes the semantic gaps between those donors explicit, prevents stages from
//! being skipped, and publishes a transition only after its CUDA event has
//! completed. Persistent index state is quarantined after a partial index-prep
//! or decode failure until the caller restores the token checkpoint.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::qsa::{
    QsaArenaPhase, QsaArenaScheduler, QsaArenaStats, QsaArenaView, QsaDecodeLease, QsaPackLease,
    QsaPlanError, QsaReadyLease, QsaScheduleError, QsaSparseDecodePlan, QsaWorkspaceZeroLease,
};

static NEXT_QSA_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaPipelineStage {
    IndexPrep,
    Score,
    BlockTopk,
    SelectionExpand,
    KvPack,
    Decode,
}

impl QsaPipelineStage {
    fn next(self) -> Option<Self> {
        match self {
            Self::IndexPrep => Some(Self::Score),
            Self::Score => Some(Self::BlockTopk),
            Self::BlockTopk => Some(Self::SelectionExpand),
            Self::SelectionExpand => Some(Self::KvPack),
            Self::KvPack => Some(Self::Decode),
            Self::Decode => None,
        }
    }

    fn previous(self) -> Option<Self> {
        match self {
            Self::IndexPrep => None,
            Self::Score => Some(Self::IndexPrep),
            Self::BlockTopk => Some(Self::Score),
            Self::SelectionExpand => Some(Self::BlockTopk),
            Self::KvPack => Some(Self::SelectionExpand),
            Self::Decode => Some(Self::KvPack),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaPipelinePhase {
    WorkspaceNeedsZero,
    ZeroingWorkspace,
    Idle,
    Running(QsaPipelineStage),
    Ready(QsaPipelineStage),
    Poisoned(QsaPipelineStage),
    ArenaInconsistent(QsaArenaPhase),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PipelineOperation {
    epoch: u64,
    active_batch: usize,
    graph_batch: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunningInner {
    Pack(QsaPackLease),
    Decode(QsaDecodeLease),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadyInner {
    Pack(QsaReadyLease),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PipelineState {
    Idle,
    Running {
        stage: QsaPipelineStage,
        operation: PipelineOperation,
        inner: Option<RunningInner>,
    },
    Ready {
        stage: QsaPipelineStage,
        operation: PipelineOperation,
        inner: Option<ReadyInner>,
    },
    Poisoned {
        stage: QsaPipelineStage,
        operation: PipelineOperation,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaPipelineLease {
    scheduler_id: u64,
    operation: PipelineOperation,
    stage: QsaPipelineStage,
    inner: Option<RunningInner>,
}

impl QsaPipelineLease {
    pub fn stage(self) -> QsaPipelineStage {
        self.stage
    }

    pub fn active_batch(self) -> usize {
        self.operation.active_batch
    }

    pub fn graph_batch(self) -> usize {
        self.operation.graph_batch
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaPipelineReady {
    scheduler_id: u64,
    operation: PipelineOperation,
    stage: QsaPipelineStage,
    inner: Option<ReadyInner>,
}

impl QsaPipelineReady {
    pub fn stage(self) -> QsaPipelineStage {
        self.stage
    }

    pub fn active_batch(self) -> usize {
        self.operation.active_batch
    }

    pub fn graph_batch(self) -> usize {
        self.operation.graph_batch
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// The ready lease embeds the fixed arena view. Keeping it inline avoids a heap
// allocation on every decode stage transition.
#[allow(clippy::large_enum_variant)]
pub enum QsaPipelineCompletion {
    StageReady(QsaPipelineReady),
    TokenComplete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// Retry is an error path, but retaining the inline lease keeps this scheduler
// allocation-free and preserves the exact arena identity.
#[allow(clippy::large_enum_variant)]
pub enum QsaPipelineAbort {
    RetryFrom(QsaPipelineReady),
    RestoreTokenState,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QsaPipelineStats {
    pub tokens_started: u64,
    pub tokens_completed: u64,
    pub stages_submitted: u64,
    pub stages_completed: u64,
    pub stages_failed: u64,
    pub state_quarantines: u64,
    pub stage_stalls: u64,
    pub peak_graph_batch: usize,
}

/// Allocation-free semantic scheduler around the fixed QSA arena.
///
/// `Score` and `SelectionExpand` are first-class stages even though their
/// production kernels are not connected yet. This prevents the already
/// validated index-prep/top-k/pack/XQA donors from being presented as a joined
/// layer before the missing transformations exist.
pub struct QsaPipelineScheduler {
    scheduler_id: u64,
    arena: QsaArenaScheduler,
    state: PipelineState,
    next_epoch: u64,
    stats: QsaPipelineStats,
}

impl QsaPipelineScheduler {
    pub fn new(
        plan: QsaSparseDecodePlan,
        device_base: u64,
        device_bytes: usize,
        graph_buckets: Vec<usize>,
    ) -> Result<Self, QsaPipelineError> {
        let arena = QsaArenaScheduler::new(plan, device_base, device_bytes, graph_buckets)?;
        let scheduler_id = NEXT_QSA_PIPELINE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| QsaPipelineError::SchedulerIdOverflow)?;
        Ok(Self {
            scheduler_id,
            arena,
            state: PipelineState::Idle,
            next_epoch: 0,
            stats: QsaPipelineStats::default(),
        })
    }

    pub fn phase(&self) -> QsaPipelinePhase {
        match self.state {
            PipelineState::Idle => match self.arena.phase() {
                QsaArenaPhase::WorkspaceNeedsZero => QsaPipelinePhase::WorkspaceNeedsZero,
                QsaArenaPhase::ZeroingWorkspace => QsaPipelinePhase::ZeroingWorkspace,
                QsaArenaPhase::Idle => QsaPipelinePhase::Idle,
                phase => QsaPipelinePhase::ArenaInconsistent(phase),
            },
            PipelineState::Running { stage, .. } => QsaPipelinePhase::Running(stage),
            PipelineState::Ready { stage, .. } => QsaPipelinePhase::Ready(stage),
            PipelineState::Poisoned { stage, .. } => QsaPipelinePhase::Poisoned(stage),
        }
    }

    pub fn arena(&self) -> QsaArenaView {
        self.arena.arena()
    }

    pub fn graph_buckets(&self) -> &[usize] {
        self.arena.graph_buckets()
    }

    pub fn arena_stats(&self) -> QsaArenaStats {
        self.arena.stats()
    }

    pub fn stats(&self) -> QsaPipelineStats {
        self.stats
    }

    pub fn begin_workspace_zero(&mut self) -> Result<QsaWorkspaceZeroLease, QsaPipelineError> {
        self.require_idle_pipeline()?;
        self.arena.begin_workspace_zero().map_err(Into::into)
    }

    pub fn complete_workspace_zero(
        &mut self,
        lease: QsaWorkspaceZeroLease,
    ) -> Result<(), QsaPipelineError> {
        self.require_idle_pipeline()?;
        self.arena
            .complete_workspace_zero(lease)
            .map_err(Into::into)
    }

    pub fn abort_workspace_zero(
        &mut self,
        lease: QsaWorkspaceZeroLease,
    ) -> Result<(), QsaPipelineError> {
        self.require_idle_pipeline()?;
        self.arena.abort_workspace_zero(lease).map_err(Into::into)
    }

    pub fn begin_token(
        &mut self,
        active_batch: usize,
    ) -> Result<QsaPipelineLease, QsaPipelineError> {
        self.require_idle_pipeline()?;
        if self.arena.phase() != QsaArenaPhase::Idle {
            return Err(QsaPipelineError::WorkspaceNotInitialized);
        }
        let capacity = self.arena.graph_buckets().last().copied().unwrap_or(0);
        if active_batch == 0 {
            return Err(QsaPlanError::ZeroBatch.into());
        }
        if active_batch > capacity {
            return Err(QsaPlanError::BatchExceedsCapacity {
                active: active_batch,
                capacity,
            }
            .into());
        }
        let graph_batch = self
            .arena
            .graph_buckets()
            .iter()
            .copied()
            .find(|bucket| *bucket >= active_batch)
            .ok_or(QsaPipelineError::InvalidGraphBuckets)?;
        let epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(QsaPipelineError::EpochOverflow)?;
        self.next_epoch = epoch;
        let operation = PipelineOperation {
            epoch,
            active_batch,
            graph_batch,
        };
        let lease = QsaPipelineLease {
            scheduler_id: self.scheduler_id,
            operation,
            stage: QsaPipelineStage::IndexPrep,
            inner: None,
        };
        self.state = running_state(lease);
        self.stats.tokens_started = self.stats.tokens_started.saturating_add(1);
        self.note_stage_submission(operation);
        Ok(lease)
    }

    /// Publish one stage only after its CUDA event completed.
    pub fn complete_stage(
        &mut self,
        lease: QsaPipelineLease,
    ) -> Result<QsaPipelineCompletion, QsaPipelineError> {
        self.validate_running(lease)?;
        let completion = match (lease.stage, lease.inner) {
            (QsaPipelineStage::KvPack, Some(RunningInner::Pack(inner))) => {
                let ready = self.arena.complete_pack(inner)?;
                let stage_ready = QsaPipelineReady {
                    scheduler_id: self.scheduler_id,
                    operation: lease.operation,
                    stage: lease.stage,
                    inner: Some(ReadyInner::Pack(ready)),
                };
                self.state = ready_state(stage_ready);
                QsaPipelineCompletion::StageReady(stage_ready)
            }
            (QsaPipelineStage::Decode, Some(RunningInner::Decode(inner))) => {
                self.arena.complete_decode(inner)?;
                self.state = PipelineState::Idle;
                self.stats.tokens_completed = self.stats.tokens_completed.saturating_add(1);
                QsaPipelineCompletion::TokenComplete
            }
            (
                QsaPipelineStage::IndexPrep
                | QsaPipelineStage::Score
                | QsaPipelineStage::BlockTopk
                | QsaPipelineStage::SelectionExpand,
                None,
            ) => {
                let stage_ready = QsaPipelineReady {
                    scheduler_id: self.scheduler_id,
                    operation: lease.operation,
                    stage: lease.stage,
                    inner: None,
                };
                self.state = ready_state(stage_ready);
                QsaPipelineCompletion::StageReady(stage_ready)
            }
            _ => return Err(QsaPipelineError::InternalStageInvariant),
        };
        self.stats.stages_completed = self.stats.stages_completed.saturating_add(1);
        Ok(completion)
    }

    pub fn begin_next(
        &mut self,
        ready: QsaPipelineReady,
    ) -> Result<QsaPipelineLease, QsaPipelineError> {
        self.validate_ready(ready)?;
        let next = ready.stage.next().ok_or(QsaPipelineError::NoNextStage)?;
        let inner = match (next, ready.inner) {
            (QsaPipelineStage::KvPack, None) => Some(RunningInner::Pack(
                self.arena.begin_pack(ready.operation.active_batch)?,
            )),
            (QsaPipelineStage::Decode, Some(ReadyInner::Pack(pack))) => {
                Some(RunningInner::Decode(self.arena.begin_decode(pack)?))
            }
            (
                QsaPipelineStage::Score
                | QsaPipelineStage::BlockTopk
                | QsaPipelineStage::SelectionExpand,
                None,
            ) => None,
            _ => return Err(QsaPipelineError::InternalStageInvariant),
        };
        let lease = QsaPipelineLease {
            scheduler_id: self.scheduler_id,
            operation: ready.operation,
            stage: next,
            inner,
        };
        self.state = running_state(lease);
        self.note_stage_submission(ready.operation);
        Ok(lease)
    }

    /// Release a failed stage only after its stream can no longer touch memory.
    ///
    /// Scratch-only failures return the previous completed stage for a retry.
    /// Index prep and decode can leave persistent token state partially updated,
    /// so they poison the pipeline until the caller restores its checkpoint.
    pub fn abort_stage(
        &mut self,
        lease: QsaPipelineLease,
    ) -> Result<QsaPipelineAbort, QsaPipelineError> {
        self.validate_running(lease)?;
        self.stats.stages_failed = self.stats.stages_failed.saturating_add(1);
        match (lease.stage, lease.inner) {
            (QsaPipelineStage::IndexPrep, None) => {
                self.poison(lease);
                Ok(QsaPipelineAbort::RestoreTokenState)
            }
            (QsaPipelineStage::Decode, Some(RunningInner::Decode(inner))) => {
                self.arena.abort_decode(inner)?;
                self.poison(lease);
                Ok(QsaPipelineAbort::RestoreTokenState)
            }
            (QsaPipelineStage::KvPack, Some(RunningInner::Pack(inner))) => {
                self.arena.abort_pack(inner)?;
                self.retry_from_previous(lease)
            }
            (
                QsaPipelineStage::Score
                | QsaPipelineStage::BlockTopk
                | QsaPipelineStage::SelectionExpand,
                None,
            ) => self.retry_from_previous(lease),
            _ => Err(QsaPipelineError::InternalStageInvariant),
        }
    }

    /// Clear a quarantine only after external persistent state was restored.
    pub fn recover_after_state_restore(&mut self) -> Result<(), QsaPipelineError> {
        if !matches!(self.state, PipelineState::Poisoned { .. }) {
            return Err(QsaPipelineError::PipelineNotPoisoned);
        }
        if !matches!(
            self.arena.phase(),
            QsaArenaPhase::Idle | QsaArenaPhase::WorkspaceNeedsZero
        ) {
            return Err(QsaPipelineError::InternalStageInvariant);
        }
        self.state = PipelineState::Idle;
        Ok(())
    }

    fn retry_from_previous(
        &mut self,
        lease: QsaPipelineLease,
    ) -> Result<QsaPipelineAbort, QsaPipelineError> {
        let previous = lease
            .stage
            .previous()
            .ok_or(QsaPipelineError::InternalStageInvariant)?;
        let ready = QsaPipelineReady {
            scheduler_id: self.scheduler_id,
            operation: lease.operation,
            stage: previous,
            inner: None,
        };
        self.state = ready_state(ready);
        Ok(QsaPipelineAbort::RetryFrom(ready))
    }

    fn poison(&mut self, lease: QsaPipelineLease) {
        self.state = PipelineState::Poisoned {
            stage: lease.stage,
            operation: lease.operation,
        };
        self.stats.state_quarantines = self.stats.state_quarantines.saturating_add(1);
    }

    fn validate_running(&self, lease: QsaPipelineLease) -> Result<(), QsaPipelineError> {
        if lease.scheduler_id != self.scheduler_id || self.state != running_state(lease) {
            return Err(QsaPipelineError::StaleStageLease);
        }
        Ok(())
    }

    fn validate_ready(&self, ready: QsaPipelineReady) -> Result<(), QsaPipelineError> {
        if ready.scheduler_id != self.scheduler_id || self.state != ready_state(ready) {
            return Err(QsaPipelineError::StaleStageReady);
        }
        Ok(())
    }

    fn require_idle_pipeline(&mut self) -> Result<(), QsaPipelineError> {
        if self.state != PipelineState::Idle {
            self.stats.stage_stalls = self.stats.stage_stalls.saturating_add(1);
            return Err(QsaPipelineError::PipelineBusy(self.phase()));
        }
        Ok(())
    }

    fn note_stage_submission(&mut self, operation: PipelineOperation) {
        self.stats.stages_submitted = self.stats.stages_submitted.saturating_add(1);
        self.stats.peak_graph_batch = self.stats.peak_graph_batch.max(operation.graph_batch);
    }
}

/// QSA pipeline plus the coherent allocation behind every pack/XQA pointer.
///
/// The model's persistent index state has separate lifetime ownership, while
/// this object makes it impossible to release the fixed decode arena without
/// also dropping the Rust stage scheduler that controls it.
#[cfg(feature = "native-fabric")]
pub struct QsaCoherentPipeline {
    scheduler: QsaPipelineScheduler,
    region: Option<crate::coherent::CoherentRegionOwner>,
}

#[cfg(feature = "native-fabric")]
impl QsaCoherentPipeline {
    pub fn allocate(
        plan: QsaSparseDecodePlan,
        graph_buckets: Vec<usize>,
        coherent_flags: u32,
    ) -> Result<Self, QsaCoherentPipelineError> {
        let bytes = plan.scratch_layout()?.total_bytes;
        let payload_bytes = u64::try_from(bytes).map_err(|_| QsaPlanError::SizeOverflow)?;
        let region =
            crate::coherent::CoherentRegionOwner::slab(payload_bytes, 4096, coherent_flags)?;
        let scheduler = QsaPipelineScheduler::new(
            plan,
            region.device_address(),
            region.payload_bytes()?,
            graph_buckets,
        )?;
        Ok(Self {
            scheduler,
            region: Some(region),
        })
    }

    pub fn scheduler(&self) -> &QsaPipelineScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut QsaPipelineScheduler {
        &mut self.scheduler
    }

    pub fn region(&self) -> &crate::coherent::CoherentRegionOwner {
        self.region
            .as_ref()
            .expect("QSA pipeline coherent region is owned")
    }

    /// Borrow the CPU alias only when the decode arena has no CUDA owner.
    ///
    /// # Safety
    ///
    /// The caller must also ensure a front-end donor is not using bytes that
    /// it placed in this mapping outside the standard arena layout.
    pub unsafe fn host_payload_mut(&mut self) -> Result<&mut [u8], QsaCoherentPipelineError> {
        let phase = self.scheduler.arena.phase();
        if matches!(
            phase,
            QsaArenaPhase::ZeroingWorkspace
                | QsaArenaPhase::Packing
                | QsaArenaPhase::Ready
                | QsaArenaPhase::Decoding
        ) {
            return Err(QsaCoherentPipelineError::ArenaBusy(phase));
        }
        // SAFETY: the caller contract and arena phase exclude CUDA access.
        Ok(unsafe {
            self.region
                .as_mut()
                .expect("QSA pipeline coherent region is owned")
                .host_payload_mut()?
        })
    }
}

#[cfg(feature = "native-fabric")]
impl Drop for QsaCoherentPipeline {
    fn drop(&mut self) {
        if matches!(
            self.scheduler.arena.phase(),
            QsaArenaPhase::ZeroingWorkspace | QsaArenaPhase::Packing | QsaArenaPhase::Decoding
        ) {
            // A live asynchronous writer is safer leaked at process teardown
            // than unregistered while CUDA still owns its device pointers.
            if let Some(region) = self.region.take() {
                std::mem::forget(region);
            }
        }
    }
}

#[cfg(feature = "native-fabric")]
#[derive(Debug)]
pub enum QsaCoherentPipelineError {
    Plan(QsaPlanError),
    Coherent(crate::coherent::CoherentRegionError),
    Pipeline(QsaPipelineError),
    ArenaBusy(QsaArenaPhase),
}

#[cfg(feature = "native-fabric")]
impl From<QsaPlanError> for QsaCoherentPipelineError {
    fn from(error: QsaPlanError) -> Self {
        Self::Plan(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<crate::coherent::CoherentRegionError> for QsaCoherentPipelineError {
    fn from(error: crate::coherent::CoherentRegionError) -> Self {
        Self::Coherent(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<QsaPipelineError> for QsaCoherentPipelineError {
    fn from(error: QsaPipelineError) -> Self {
        Self::Pipeline(error)
    }
}

#[cfg(feature = "native-fabric")]
impl fmt::Display for QsaCoherentPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(formatter),
            Self::Coherent(error) => error.fmt(formatter),
            Self::Pipeline(error) => error.fmt(formatter),
            Self::ArenaBusy(phase) => write!(formatter, "QSA pipeline arena is busy in {phase:?}"),
        }
    }
}

#[cfg(feature = "native-fabric")]
impl std::error::Error for QsaCoherentPipelineError {}

fn running_state(lease: QsaPipelineLease) -> PipelineState {
    PipelineState::Running {
        stage: lease.stage,
        operation: lease.operation,
        inner: lease.inner,
    }
}

fn ready_state(ready: QsaPipelineReady) -> PipelineState {
    PipelineState::Ready {
        stage: ready.stage,
        operation: ready.operation,
        inner: ready.inner,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaPipelineError {
    Arena(QsaScheduleError),
    Plan(QsaPlanError),
    InvalidGraphBuckets,
    SchedulerIdOverflow,
    EpochOverflow,
    WorkspaceNotInitialized,
    PipelineBusy(QsaPipelinePhase),
    StaleStageLease,
    StaleStageReady,
    NoNextStage,
    PipelineNotPoisoned,
    InternalStageInvariant,
}

impl From<QsaScheduleError> for QsaPipelineError {
    fn from(error: QsaScheduleError) -> Self {
        Self::Arena(error)
    }
}

impl From<QsaPlanError> for QsaPipelineError {
    fn from(error: QsaPlanError) -> Self {
        Self::Plan(error)
    }
}

impl fmt::Display for QsaPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arena(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::InvalidGraphBuckets => formatter.write_str("QSA pipeline graph buckets invalid"),
            Self::SchedulerIdOverflow => formatter.write_str("QSA pipeline scheduler ID overflow"),
            Self::EpochOverflow => formatter.write_str("QSA pipeline epoch overflow"),
            Self::WorkspaceNotInitialized => {
                formatter.write_str("QSA pipeline workspace zero has not completed")
            }
            Self::PipelineBusy(phase) => write!(formatter, "QSA pipeline is busy in {phase:?}"),
            Self::StaleStageLease => formatter.write_str("stale or foreign QSA stage lease"),
            Self::StaleStageReady => formatter.write_str("stale or foreign QSA stage-ready lease"),
            Self::NoNextStage => formatter.write_str("completed QSA stage has no successor"),
            Self::PipelineNotPoisoned => formatter.write_str("QSA pipeline is not quarantined"),
            Self::InternalStageInvariant => {
                formatter.write_str("QSA internal stage invariant failed")
            }
        }
    }
}

impl std::error::Error for QsaPipelineError {}

#[cfg(feature = "native-fabric")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QsaPipelineCudaPending {
    WorkspaceZero(QsaWorkspaceZeroLease),
    Stage(QsaPipelineLease),
}

#[cfg(feature = "native-fabric")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// CUDA completion carries the same inline fixed-address lease; boxing it would
// add an allocation to the event publication path.
#[allow(clippy::large_enum_variant)]
pub enum QsaPipelineCudaCompletion {
    WorkspaceReady,
    StageReady(QsaPipelineReady),
    TokenComplete,
}

/// One reusable CUDA event that owns exactly one pending pipeline lease.
#[cfg(feature = "native-fabric")]
pub struct QsaPipelineCudaFence {
    event: crate::cuda::CudaEventOwner,
    pending: Option<QsaPipelineCudaPending>,
}

#[cfg(feature = "native-fabric")]
impl QsaPipelineCudaFence {
    pub fn create() -> Result<Self, QsaPipelineCudaFenceError> {
        Ok(Self {
            event: crate::cuda::CudaEventOwner::create()?,
            pending: None,
        })
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn record_workspace_zero(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        lease: QsaWorkspaceZeroLease,
    ) -> Result<(), QsaPipelineCudaFenceError> {
        self.record(stream, QsaPipelineCudaPending::WorkspaceZero(lease))
    }

    pub fn record_stage(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        lease: QsaPipelineLease,
    ) -> Result<(), QsaPipelineCudaFenceError> {
        self.record(stream, QsaPipelineCudaPending::Stage(lease))
    }

    pub fn poll(
        &mut self,
        scheduler: &mut QsaPipelineScheduler,
    ) -> Result<Option<QsaPipelineCudaCompletion>, QsaPipelineCudaFenceError> {
        if self.pending.is_none() {
            return Err(QsaPipelineCudaFenceError::NoPendingEvent);
        }
        if !self.event.query()? {
            return Ok(None);
        }
        self.publish(scheduler).map(Some)
    }

    pub fn wait(
        &mut self,
        scheduler: &mut QsaPipelineScheduler,
    ) -> Result<QsaPipelineCudaCompletion, QsaPipelineCudaFenceError> {
        if self.pending.is_none() {
            return Err(QsaPipelineCudaFenceError::NoPendingEvent);
        }
        self.event.synchronize()?;
        self.publish(scheduler)
    }

    fn record(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        pending: QsaPipelineCudaPending,
    ) -> Result<(), QsaPipelineCudaFenceError> {
        if self.pending.is_some() {
            return Err(QsaPipelineCudaFenceError::EventBusy);
        }
        self.event.record(stream)?;
        self.pending = Some(pending);
        Ok(())
    }

    fn publish(
        &mut self,
        scheduler: &mut QsaPipelineScheduler,
    ) -> Result<QsaPipelineCudaCompletion, QsaPipelineCudaFenceError> {
        let pending = self
            .pending
            .take()
            .ok_or(QsaPipelineCudaFenceError::NoPendingEvent)?;
        match pending {
            QsaPipelineCudaPending::WorkspaceZero(lease) => {
                scheduler.complete_workspace_zero(lease)?;
                Ok(QsaPipelineCudaCompletion::WorkspaceReady)
            }
            QsaPipelineCudaPending::Stage(lease) => match scheduler.complete_stage(lease)? {
                QsaPipelineCompletion::StageReady(ready) => {
                    Ok(QsaPipelineCudaCompletion::StageReady(ready))
                }
                QsaPipelineCompletion::TokenComplete => {
                    Ok(QsaPipelineCudaCompletion::TokenComplete)
                }
            },
        }
    }
}

#[cfg(feature = "native-fabric")]
#[derive(Debug)]
pub enum QsaPipelineCudaFenceError {
    Runtime(crate::cuda::CudaRuntimeError),
    Pipeline(QsaPipelineError),
    EventBusy,
    NoPendingEvent,
}

#[cfg(feature = "native-fabric")]
impl From<crate::cuda::CudaRuntimeError> for QsaPipelineCudaFenceError {
    fn from(error: crate::cuda::CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<QsaPipelineError> for QsaPipelineCudaFenceError {
    fn from(error: QsaPipelineError) -> Self {
        Self::Pipeline(error)
    }
}

#[cfg(feature = "native-fabric")]
impl fmt::Display for QsaPipelineCudaFenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Pipeline(error) => error.fmt(formatter),
            Self::EventBusy => formatter.write_str("QSA pipeline CUDA fence already owns a lease"),
            Self::NoPendingEvent => {
                formatter.write_str("QSA pipeline CUDA fence has no pending lease")
            }
        }
    }
}

#[cfg(feature = "native-fabric")]
impl std::error::Error for QsaPipelineCudaFenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE_BASE: u64 = 0x2_0000_0000;

    fn scheduler() -> QsaPipelineScheduler {
        let plan = QsaSparseDecodePlan::qwen38_flash(8).expect("plan");
        let bytes = plan.scratch_layout().expect("layout").total_bytes;
        QsaPipelineScheduler::new(plan, DEVICE_BASE, bytes, vec![1, 2, 4, 8]).expect("scheduler")
    }

    fn initialize(scheduler: &mut QsaPipelineScheduler) {
        let zero = scheduler.begin_workspace_zero().expect("begin zero");
        scheduler
            .complete_workspace_zero(zero)
            .expect("complete zero");
    }

    fn complete_ready(
        scheduler: &mut QsaPipelineScheduler,
        lease: QsaPipelineLease,
    ) -> QsaPipelineReady {
        let QsaPipelineCompletion::StageReady(ready) =
            scheduler.complete_stage(lease).expect("complete stage")
        else {
            panic!("stage completed token early");
        };
        ready
    }

    #[test]
    fn explicit_pipeline_cannot_skip_missing_glue_stages() {
        let mut scheduler = scheduler();
        assert_eq!(scheduler.phase(), QsaPipelinePhase::WorkspaceNeedsZero);
        assert_eq!(
            scheduler.begin_token(1),
            Err(QsaPipelineError::WorkspaceNotInitialized)
        );
        initialize(&mut scheduler);

        let prep = scheduler.begin_token(3).expect("prep");
        assert_eq!(prep.stage(), QsaPipelineStage::IndexPrep);
        assert_eq!(prep.graph_batch(), 4);
        let prep_ready = complete_ready(&mut scheduler, prep);
        assert_eq!(prep_ready.stage(), QsaPipelineStage::IndexPrep);

        let score = scheduler.begin_next(prep_ready).expect("score");
        assert_eq!(score.stage(), QsaPipelineStage::Score);
        let score_ready = complete_ready(&mut scheduler, score);

        let topk = scheduler.begin_next(score_ready).expect("topk");
        assert_eq!(topk.stage(), QsaPipelineStage::BlockTopk);
        let topk_ready = complete_ready(&mut scheduler, topk);

        let expand = scheduler.begin_next(topk_ready).expect("expand");
        assert_eq!(expand.stage(), QsaPipelineStage::SelectionExpand);
        let selection_ready = complete_ready(&mut scheduler, expand);

        let pack = scheduler.begin_next(selection_ready).expect("pack");
        assert_eq!(pack.stage(), QsaPipelineStage::KvPack);
        let pack_ready = complete_ready(&mut scheduler, pack);

        let decode = scheduler.begin_next(pack_ready).expect("decode");
        assert_eq!(decode.stage(), QsaPipelineStage::Decode);
        assert_eq!(
            scheduler.complete_stage(decode).expect("decode completion"),
            QsaPipelineCompletion::TokenComplete
        );
        assert_eq!(scheduler.phase(), QsaPipelinePhase::Idle);
        assert_eq!(scheduler.stats().tokens_started, 1);
        assert_eq!(scheduler.stats().tokens_completed, 1);
        assert_eq!(scheduler.stats().stages_submitted, 6);
        assert_eq!(scheduler.stats().stages_completed, 6);
        assert_eq!(scheduler.arena_stats().packs_completed, 1);
        assert_eq!(scheduler.arena_stats().decodes_completed, 1);
    }

    #[test]
    fn stale_and_foreign_stage_handoffs_fail_closed() {
        let mut left = scheduler();
        let mut right = scheduler();
        initialize(&mut left);
        initialize(&mut right);
        let left_prep = left.begin_token(1).expect("left prep");
        let right_prep = right.begin_token(1).expect("right prep");
        assert_eq!(
            left.complete_stage(right_prep),
            Err(QsaPipelineError::StaleStageLease)
        );
        let left_ready = complete_ready(&mut left, left_prep);
        assert_eq!(
            left.complete_stage(left_prep),
            Err(QsaPipelineError::StaleStageLease)
        );
        assert_eq!(
            right.begin_next(left_ready),
            Err(QsaPipelineError::StaleStageReady)
        );
    }

    #[test]
    fn scratch_failure_retries_but_persistent_failure_quarantines() {
        let mut scheduler = scheduler();
        initialize(&mut scheduler);
        let prep = scheduler.begin_token(1).expect("prep");
        assert_eq!(
            scheduler.abort_stage(prep).expect("abort prep"),
            QsaPipelineAbort::RestoreTokenState
        );
        assert_eq!(
            scheduler.phase(),
            QsaPipelinePhase::Poisoned(QsaPipelineStage::IndexPrep)
        );
        assert_eq!(
            scheduler.begin_token(1),
            Err(QsaPipelineError::PipelineBusy(QsaPipelinePhase::Poisoned(
                QsaPipelineStage::IndexPrep
            )))
        );
        scheduler
            .recover_after_state_restore()
            .expect("restore state");

        let prep = scheduler.begin_token(1).expect("retry prep");
        let prep_ready = complete_ready(&mut scheduler, prep);
        let score = scheduler.begin_next(prep_ready).expect("score");
        let QsaPipelineAbort::RetryFrom(retry_ready) =
            scheduler.abort_stage(score).expect("abort score")
        else {
            panic!("score failure should be retryable");
        };
        assert_eq!(retry_ready.stage(), QsaPipelineStage::IndexPrep);
        assert_eq!(scheduler.begin_next(retry_ready).expect("retry"), score);
        assert_eq!(scheduler.stats().stages_failed, 2);
        assert_eq!(scheduler.stats().state_quarantines, 1);
    }
}
