//! Fixed-address, transactional scheduling for GLM-5.3 DSA decode.
//!
//! The pinned SGLang arithmetic is exposed as allocation-free CUDA leaves.
//! Rust owns their addresses and ordering. In particular, it snapshots the
//! complete four-slot KPool key/score ring before mutation and does not publish
//! MLA or pooled-cache lengths until the final output event has completed.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::glm_dsa::{GlmDsaError, GlmDsaMemoryPlan, GlmDsaSpec};

pub const GLM_DSA_ARENA_ALIGNMENT: u64 = 256;

static NEXT_GLM_DSA_SCHEDULER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaRange {
    pub address: u64,
    pub bytes: u64,
}

impl GlmDsaRange {
    pub fn end(self) -> Result<u64, GlmDsaPipelineError> {
        self.address
            .checked_add(self.bytes)
            .ok_or(GlmDsaPipelineError::Overflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaArenaView {
    pub mla_cache: GlmDsaRange,
    pub pooled_index_cache: GlmDsaRange,
    pub tail_key: GlmDsaRange,
    pub tail_score: GlmDsaRange,
    pub query: GlmDsaRange,
    pub topk: GlmDsaRange,
    pub history_indices: GlmDsaRange,
    pub tail_indices: GlmDsaRange,
    pub history_lengths: GlmDsaRange,
    pub tail_lengths: GlmDsaRange,
    pub history_mid_out: GlmDsaRange,
    pub history_mid_lse: GlmDsaRange,
    /// Final sparse-MLA output after the tail LSE merge. The donor first
    /// writes its history result here and the merge updates it in place.
    pub attention_output: GlmDsaRange,
    pub attention_lse: GlmDsaRange,
    pub tail_mid_out: GlmDsaRange,
    pub tail_mid_lse: GlmDsaRange,
    pub tail_output: GlmDsaRange,
    pub tail_lse: GlmDsaRange,
    pub mqa_schedule: GlmDsaRange,
    pub checkpoint_key: GlmDsaRange,
    pub checkpoint_score: GlmDsaRange,
    pub persistent_required_bytes: u64,
    pub workspace_required_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaLayerView {
    pub mla_cache: GlmDsaRange,
    pub pooled_index_cache: GlmDsaRange,
    pub tail_key: GlmDsaRange,
    pub tail_score: GlmDsaRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaTailCheckpoint {
    pub key_source: GlmDsaRange,
    pub key_snapshot: GlmDsaRange,
    pub score_source: GlmDsaRange,
    pub score_snapshot: GlmDsaRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaFixedArena {
    view: GlmDsaArenaView,
    memory: GlmDsaMemoryPlan,
    layers: Vec<u16>,
}

impl GlmDsaFixedArena {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: &GlmDsaSpec,
        sequences: u64,
        context_tokens: u64,
        persistent_base: u64,
        persistent_capacity: u64,
        workspace_base: u64,
        workspace_capacity: u64,
    ) -> Result<Self, GlmDsaPipelineError> {
        validate_base(persistent_base)?;
        validate_base(workspace_base)?;
        let memory = spec.plan(sequences, context_tokens)?;
        if memory.layers == 0 || memory.tail_bytes % (memory.layers * 2) != 0 {
            return Err(GlmDsaPipelineError::InvalidMemoryPlan);
        }

        let tail_half = memory.tail_bytes / 2;
        let checkpoint_half = tail_half / memory.layers;

        let mut persistent_cursor = persistent_base;
        let mla_cache = allocate(&mut persistent_cursor, memory.mla_cache_bytes)?;
        let pooled_index_cache = allocate(&mut persistent_cursor, memory.pooled_index_cache_bytes)?;
        let tail_key = allocate(&mut persistent_cursor, tail_half)?;
        let tail_score = allocate(&mut persistent_cursor, tail_half)?;
        let persistent_end = align_up(persistent_cursor, GLM_DSA_ARENA_ALIGNMENT)?;
        let persistent_required_bytes = persistent_end
            .checked_sub(persistent_base)
            .ok_or(GlmDsaPipelineError::Overflow)?;
        if persistent_required_bytes > persistent_capacity {
            return Err(GlmDsaPipelineError::InsufficientPersistentCapacity {
                required: persistent_required_bytes,
                available: persistent_capacity,
            });
        }

        let mut workspace_cursor = workspace_base;
        let query = allocate(&mut workspace_cursor, memory.query_workspace_bytes)?;
        let topk = allocate(&mut workspace_cursor, memory.topk_bytes)?;
        let history_indices = allocate(&mut workspace_cursor, memory.history_indices_bytes)?;
        let tail_indices = allocate(&mut workspace_cursor, memory.tail_indices_bytes)?;
        let history_lengths = allocate(&mut workspace_cursor, memory.selection_lengths_bytes)?;
        let tail_lengths = allocate(&mut workspace_cursor, memory.selection_lengths_bytes)?;
        let history_mid_out = allocate(&mut workspace_cursor, memory.history_mid_out_bytes)?;
        let history_mid_lse = allocate(&mut workspace_cursor, memory.history_mid_lse_bytes)?;
        let attention_output = allocate(&mut workspace_cursor, memory.attention_output_bytes)?;
        let attention_lse = allocate(&mut workspace_cursor, memory.attention_lse_bytes)?;
        let tail_mid_out = allocate(&mut workspace_cursor, memory.tail_mid_out_bytes)?;
        let tail_mid_lse = allocate(&mut workspace_cursor, memory.tail_mid_lse_bytes)?;
        let tail_output = allocate(&mut workspace_cursor, memory.attention_output_bytes)?;
        let tail_lse = allocate(&mut workspace_cursor, memory.attention_lse_bytes)?;
        let mqa_schedule = allocate(&mut workspace_cursor, memory.mqa_schedule_bytes)?;
        let checkpoint_key = allocate(&mut workspace_cursor, checkpoint_half)?;
        let checkpoint_score = allocate(&mut workspace_cursor, checkpoint_half)?;
        let workspace_end = align_up(workspace_cursor, GLM_DSA_ARENA_ALIGNMENT)?;
        let workspace_required_bytes = workspace_end
            .checked_sub(workspace_base)
            .ok_or(GlmDsaPipelineError::Overflow)?;
        if workspace_required_bytes > workspace_capacity {
            return Err(GlmDsaPipelineError::InsufficientWorkspaceCapacity {
                required: workspace_required_bytes,
                available: workspace_capacity,
            });
        }

        if ranges_overlap(
            GlmDsaRange {
                address: persistent_base,
                bytes: persistent_required_bytes,
            },
            GlmDsaRange {
                address: workspace_base,
                bytes: workspace_required_bytes,
            },
        )? {
            return Err(GlmDsaPipelineError::OverlappingArenas);
        }

        Ok(Self {
            view: GlmDsaArenaView {
                mla_cache,
                pooled_index_cache,
                tail_key,
                tail_score,
                query,
                topk,
                history_indices,
                tail_indices,
                history_lengths,
                tail_lengths,
                history_mid_out,
                history_mid_lse,
                attention_output,
                attention_lse,
                tail_mid_out,
                tail_mid_lse,
                tail_output,
                tail_lse,
                mqa_schedule,
                checkpoint_key,
                checkpoint_score,
                persistent_required_bytes,
                workspace_required_bytes,
            },
            memory,
            layers: spec.dsa_layers.clone(),
        })
    }

    pub fn view(&self) -> GlmDsaArenaView {
        self.view
    }

    pub fn memory(&self) -> GlmDsaMemoryPlan {
        self.memory
    }

    pub fn layers(&self) -> &[u16] {
        &self.layers
    }

    pub fn layer_view(&self, layer: u16) -> Result<GlmDsaLayerView, GlmDsaPipelineError> {
        let ordinal = self.layer_ordinal(layer)?;
        Ok(GlmDsaLayerView {
            mla_cache: layer_subrange(self.view.mla_cache, ordinal, self.memory.layers)?,
            pooled_index_cache: layer_subrange(
                self.view.pooled_index_cache,
                ordinal,
                self.memory.layers,
            )?,
            tail_key: layer_subrange(self.view.tail_key, ordinal, self.memory.layers)?,
            tail_score: layer_subrange(self.view.tail_score, ordinal, self.memory.layers)?,
        })
    }

    pub fn tail_checkpoint(&self, layer: u16) -> Result<GlmDsaTailCheckpoint, GlmDsaPipelineError> {
        let layer = self.layer_view(layer)?;
        Ok(GlmDsaTailCheckpoint {
            key_source: layer.tail_key,
            key_snapshot: self.view.checkpoint_key,
            score_source: layer.tail_score,
            score_snapshot: self.view.checkpoint_score,
        })
    }

    fn layer_ordinal(&self, layer: u16) -> Result<u64, GlmDsaPipelineError> {
        self.layers
            .iter()
            .position(|candidate| *candidate == layer)
            .and_then(|ordinal| u64::try_from(ordinal).ok())
            .ok_or(GlmDsaPipelineError::UnknownDsaLayer(layer))
    }
}

fn validate_base(base: u64) -> Result<(), GlmDsaPipelineError> {
    if base == 0 || !base.is_multiple_of(GLM_DSA_ARENA_ALIGNMENT) {
        return Err(GlmDsaPipelineError::InvalidArenaBase(base));
    }
    Ok(())
}

fn allocate(cursor: &mut u64, bytes: u64) -> Result<GlmDsaRange, GlmDsaPipelineError> {
    if bytes == 0 {
        return Err(GlmDsaPipelineError::InvalidMemoryPlan);
    }
    let address = align_up(*cursor, GLM_DSA_ARENA_ALIGNMENT)?;
    *cursor = address
        .checked_add(bytes)
        .ok_or(GlmDsaPipelineError::Overflow)?;
    Ok(GlmDsaRange { address, bytes })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GlmDsaPipelineError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GlmDsaPipelineError::Overflow)
}

fn layer_subrange(
    range: GlmDsaRange,
    ordinal: u64,
    layers: u64,
) -> Result<GlmDsaRange, GlmDsaPipelineError> {
    if layers == 0 || range.bytes % layers != 0 || ordinal >= layers {
        return Err(GlmDsaPipelineError::InvalidMemoryPlan);
    }
    let bytes = range.bytes / layers;
    let address = ordinal
        .checked_mul(bytes)
        .and_then(|offset| range.address.checked_add(offset))
        .ok_or(GlmDsaPipelineError::Overflow)?;
    Ok(GlmDsaRange { address, bytes })
}

fn ranges_overlap(left: GlmDsaRange, right: GlmDsaRange) -> Result<bool, GlmDsaPipelineError> {
    Ok(left.address < right.end()? && right.address < left.end()?)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GlmDsaStage {
    TailCheckpoint,
    Projections,
    KPoolUpdate,
    IndexerScore,
    PooledTopk,
    SelectionExpand,
    MlaCacheWrite,
    SparseMlaSegment,
    MlaDecode,
    OutputProjection,
}

impl GlmDsaStage {
    fn next(self) -> Option<Self> {
        match self {
            Self::TailCheckpoint => Some(Self::Projections),
            Self::Projections => Some(Self::KPoolUpdate),
            Self::KPoolUpdate => Some(Self::IndexerScore),
            Self::IndexerScore => Some(Self::PooledTopk),
            Self::PooledTopk => Some(Self::SelectionExpand),
            Self::SelectionExpand => Some(Self::MlaCacheWrite),
            Self::MlaCacheWrite => Some(Self::SparseMlaSegment),
            Self::SparseMlaSegment => Some(Self::MlaDecode),
            Self::MlaDecode => Some(Self::OutputProjection),
            Self::OutputProjection => None,
        }
    }

    fn previous(self) -> Option<Self> {
        match self {
            Self::TailCheckpoint => None,
            Self::Projections => Some(Self::TailCheckpoint),
            Self::KPoolUpdate => Some(Self::Projections),
            Self::IndexerScore => Some(Self::KPoolUpdate),
            Self::PooledTopk => Some(Self::IndexerScore),
            Self::SelectionExpand => Some(Self::PooledTopk),
            Self::MlaCacheWrite => Some(Self::SelectionExpand),
            Self::SparseMlaSegment => Some(Self::MlaCacheWrite),
            Self::MlaDecode => Some(Self::SparseMlaSegment),
            Self::OutputProjection => Some(Self::MlaDecode),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmDsaPhase {
    Idle,
    Running(GlmDsaStage),
    Ready(GlmDsaStage),
    Poisoned(GlmDsaStage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlmDsaOperation {
    epoch: u64,
    layer: u16,
    position: u64,
    active_rows: u64,
    graph_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GlmDsaState {
    Idle,
    Running {
        stage: GlmDsaStage,
        operation: GlmDsaOperation,
    },
    Ready {
        stage: GlmDsaStage,
        operation: GlmDsaOperation,
    },
    Poisoned {
        stage: GlmDsaStage,
        operation: GlmDsaOperation,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaLease {
    scheduler_id: u64,
    operation: GlmDsaOperation,
    stage: GlmDsaStage,
}

impl GlmDsaLease {
    pub fn stage(self) -> GlmDsaStage {
        self.stage
    }

    pub fn layer(self) -> u16 {
        self.operation.layer
    }

    pub fn position(self) -> u64 {
        self.operation.position
    }

    pub fn active_rows(self) -> u64 {
        self.operation.active_rows
    }

    pub fn graph_rows(self) -> u64 {
        self.operation.graph_rows
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaReady {
    scheduler_id: u64,
    operation: GlmDsaOperation,
    stage: GlmDsaStage,
}

impl GlmDsaReady {
    pub fn stage(self) -> GlmDsaStage {
        self.stage
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaPublication {
    pub layer: u16,
    pub position: u64,
    pub sequence_length: u64,
    pub pooled_entries: u64,
    pub unpooled_tail_tokens: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaRollback {
    pub tail: GlmDsaTailCheckpoint,
    pub discard_mla_position: u64,
    pub discard_pooled_entry: Option<u64>,
    pub restore_sequence_length: u64,
    pub restore_pooled_entries: u64,
    pub restore_tail_tokens: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmDsaCompletion {
    StageReady(GlmDsaReady),
    Published(GlmDsaPublication),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmDsaAbort {
    RetryToken,
    RetryFrom(GlmDsaReady),
    RestoreState(GlmDsaRollback),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GlmDsaPipelineStats {
    pub tokens_started: u64,
    pub tokens_published: u64,
    pub stages_submitted: u64,
    pub stages_completed: u64,
    pub stages_failed: u64,
    pub state_rollbacks: u64,
    pub stalls: u64,
    pub peak_graph_rows: u64,
}

pub struct GlmDsaPipelineScheduler {
    scheduler_id: u64,
    arena: GlmDsaFixedArena,
    graph_buckets: Vec<u64>,
    next_epoch: u64,
    state: GlmDsaState,
    stats: GlmDsaPipelineStats,
}

impl GlmDsaPipelineScheduler {
    pub fn new(
        arena: GlmDsaFixedArena,
        graph_buckets: Vec<u64>,
    ) -> Result<Self, GlmDsaPipelineError> {
        if graph_buckets.is_empty()
            || graph_buckets[0] == 0
            || !graph_buckets.windows(2).all(|pair| pair[0] < pair[1])
            || graph_buckets.last().copied().unwrap_or(0) > arena.memory.sequences
        {
            return Err(GlmDsaPipelineError::InvalidGraphBuckets);
        }
        let scheduler_id = NEXT_GLM_DSA_SCHEDULER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| GlmDsaPipelineError::Overflow)?;
        Ok(Self {
            scheduler_id,
            arena,
            graph_buckets,
            next_epoch: 0,
            state: GlmDsaState::Idle,
            stats: GlmDsaPipelineStats::default(),
        })
    }

    pub fn arena(&self) -> &GlmDsaFixedArena {
        &self.arena
    }

    pub fn graph_buckets(&self) -> &[u64] {
        &self.graph_buckets
    }

    pub fn phase(&self) -> GlmDsaPhase {
        match self.state {
            GlmDsaState::Idle => GlmDsaPhase::Idle,
            GlmDsaState::Running { stage, .. } => GlmDsaPhase::Running(stage),
            GlmDsaState::Ready { stage, .. } => GlmDsaPhase::Ready(stage),
            GlmDsaState::Poisoned { stage, .. } => GlmDsaPhase::Poisoned(stage),
        }
    }

    pub fn stats(&self) -> GlmDsaPipelineStats {
        self.stats
    }

    pub fn begin_token(
        &mut self,
        layer: u16,
        position: u64,
        active_rows: u64,
    ) -> Result<GlmDsaLease, GlmDsaPipelineError> {
        if self.state != GlmDsaState::Idle {
            self.stats.stalls = self.stats.stalls.saturating_add(1);
            return Err(GlmDsaPipelineError::Busy(self.phase()));
        }
        self.arena.layer_view(layer)?;
        if active_rows == 0
            || active_rows > self.arena.memory.sequences
            || position >= self.arena.memory.context_tokens
        {
            return Err(GlmDsaPipelineError::InvalidOperation);
        }
        let graph_rows = self
            .graph_buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= active_rows)
            .ok_or(GlmDsaPipelineError::InvalidOperation)?;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(GlmDsaPipelineError::Overflow)?;
        let operation = GlmDsaOperation {
            epoch: self.next_epoch,
            layer,
            position,
            active_rows,
            graph_rows,
        };
        let lease = self.lease(operation, GlmDsaStage::TailCheckpoint);
        self.state = running_state(lease);
        self.stats.tokens_started = self.stats.tokens_started.saturating_add(1);
        self.note_submission(operation);
        Ok(lease)
    }

    /// Advance only after the stage's CUDA event reports completion.
    pub fn complete_stage(
        &mut self,
        lease: GlmDsaLease,
    ) -> Result<GlmDsaCompletion, GlmDsaPipelineError> {
        self.validate_running(lease)?;
        self.stats.stages_completed = self.stats.stages_completed.saturating_add(1);
        if lease.stage == GlmDsaStage::OutputProjection {
            let publication = publication(lease.operation)?;
            self.state = GlmDsaState::Idle;
            self.stats.tokens_published = self.stats.tokens_published.saturating_add(1);
            return Ok(GlmDsaCompletion::Published(publication));
        }
        let ready = self.ready(lease.operation, lease.stage);
        self.state = ready_state(ready);
        Ok(GlmDsaCompletion::StageReady(ready))
    }

    pub fn begin_next(&mut self, ready: GlmDsaReady) -> Result<GlmDsaLease, GlmDsaPipelineError> {
        self.validate_ready(ready)?;
        let stage = ready.stage.next().ok_or(GlmDsaPipelineError::NoNextStage)?;
        let lease = self.lease(ready.operation, stage);
        self.state = running_state(lease);
        self.note_submission(ready.operation);
        Ok(lease)
    }

    /// Release a failed stage only after its stream is quiescent. A partial
    /// KPool update requires tail restoration. Later scratch/cache leaves can
    /// retry because their cache writes are append-only and still unpublished.
    pub fn abort_stage(&mut self, lease: GlmDsaLease) -> Result<GlmDsaAbort, GlmDsaPipelineError> {
        self.validate_running(lease)?;
        self.stats.stages_failed = self.stats.stages_failed.saturating_add(1);
        match lease.stage {
            GlmDsaStage::TailCheckpoint => {
                self.state = GlmDsaState::Idle;
                Ok(GlmDsaAbort::RetryToken)
            }
            GlmDsaStage::KPoolUpdate => self.poison(lease.stage, lease.operation),
            stage => {
                let previous = stage
                    .previous()
                    .ok_or(GlmDsaPipelineError::InternalInvariant)?;
                let ready = self.ready(lease.operation, previous);
                self.state = ready_state(ready);
                Ok(GlmDsaAbort::RetryFrom(ready))
            }
        }
    }

    /// Abandoning an in-flight token after KPool mutation restores the ring.
    /// MLA and pooled entries need no copy-back because their published lengths
    /// never advanced; the retry deterministically overwrites those slots.
    pub fn cancel_ready(&mut self, ready: GlmDsaReady) -> Result<GlmDsaAbort, GlmDsaPipelineError> {
        self.validate_ready(ready)?;
        if ready.stage < GlmDsaStage::KPoolUpdate {
            self.state = GlmDsaState::Idle;
            return Ok(GlmDsaAbort::RetryToken);
        }
        self.poison(ready.stage, ready.operation)
    }

    /// The caller invokes this only after both BF16 snapshot halves have been
    /// copied back and the unpublished cache lengths remain unchanged.
    pub fn recover_after_restore(&mut self) -> Result<(), GlmDsaPipelineError> {
        if !matches!(self.state, GlmDsaState::Poisoned { .. }) {
            return Err(GlmDsaPipelineError::PipelineNotPoisoned);
        }
        self.state = GlmDsaState::Idle;
        Ok(())
    }

    fn poison(
        &mut self,
        stage: GlmDsaStage,
        operation: GlmDsaOperation,
    ) -> Result<GlmDsaAbort, GlmDsaPipelineError> {
        let rollback = rollback(&self.arena, operation)?;
        self.state = GlmDsaState::Poisoned { stage, operation };
        self.stats.state_rollbacks = self.stats.state_rollbacks.saturating_add(1);
        Ok(GlmDsaAbort::RestoreState(rollback))
    }

    fn lease(&self, operation: GlmDsaOperation, stage: GlmDsaStage) -> GlmDsaLease {
        GlmDsaLease {
            scheduler_id: self.scheduler_id,
            operation,
            stage,
        }
    }

    fn ready(&self, operation: GlmDsaOperation, stage: GlmDsaStage) -> GlmDsaReady {
        GlmDsaReady {
            scheduler_id: self.scheduler_id,
            operation,
            stage,
        }
    }

    fn validate_running(&self, lease: GlmDsaLease) -> Result<(), GlmDsaPipelineError> {
        if lease.scheduler_id != self.scheduler_id || self.state != running_state(lease) {
            return Err(GlmDsaPipelineError::StaleLease);
        }
        Ok(())
    }

    fn validate_ready(&self, ready: GlmDsaReady) -> Result<(), GlmDsaPipelineError> {
        if ready.scheduler_id != self.scheduler_id || self.state != ready_state(ready) {
            return Err(GlmDsaPipelineError::StaleReady);
        }
        Ok(())
    }

    fn note_submission(&mut self, operation: GlmDsaOperation) {
        self.stats.stages_submitted = self.stats.stages_submitted.saturating_add(1);
        self.stats.peak_graph_rows = self.stats.peak_graph_rows.max(operation.graph_rows);
    }
}

fn running_state(lease: GlmDsaLease) -> GlmDsaState {
    GlmDsaState::Running {
        stage: lease.stage,
        operation: lease.operation,
    }
}

fn ready_state(ready: GlmDsaReady) -> GlmDsaState {
    GlmDsaState::Ready {
        stage: ready.stage,
        operation: ready.operation,
    }
}

fn publication(operation: GlmDsaOperation) -> Result<GlmDsaPublication, GlmDsaPipelineError> {
    let sequence_length = operation
        .position
        .checked_add(1)
        .ok_or(GlmDsaPipelineError::Overflow)?;
    Ok(GlmDsaPublication {
        layer: operation.layer,
        position: operation.position,
        sequence_length,
        pooled_entries: sequence_length / 4,
        unpooled_tail_tokens: sequence_length % 4,
    })
}

fn rollback(
    arena: &GlmDsaFixedArena,
    operation: GlmDsaOperation,
) -> Result<GlmDsaRollback, GlmDsaPipelineError> {
    let prior_length = operation.position;
    Ok(GlmDsaRollback {
        tail: arena.tail_checkpoint(operation.layer)?,
        discard_mla_position: operation.position,
        discard_pooled_entry: if operation.position % 4 == 3 {
            Some(operation.position / 4)
        } else {
            None
        },
        restore_sequence_length: prior_length,
        restore_pooled_entries: prior_length / 4,
        restore_tail_tokens: prior_length % 4,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum GlmDsaPipelineError {
    Dsa(GlmDsaError),
    InvalidArenaBase(u64),
    InvalidMemoryPlan,
    InsufficientPersistentCapacity { required: u64, available: u64 },
    InsufficientWorkspaceCapacity { required: u64, available: u64 },
    OverlappingArenas,
    UnknownDsaLayer(u16),
    InvalidGraphBuckets,
    InvalidOperation,
    Busy(GlmDsaPhase),
    StaleLease,
    StaleReady,
    NoNextStage,
    PipelineNotPoisoned,
    InternalInvariant,
    Overflow,
}

impl From<GlmDsaError> for GlmDsaPipelineError {
    fn from(error: GlmDsaError) -> Self {
        Self::Dsa(error)
    }
}

impl Display for GlmDsaPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dsa(error) => write!(formatter, "GLM DSA plan failed: {error}"),
            Self::InvalidArenaBase(base) => {
                write!(
                    formatter,
                    "GLM DSA arena base {base:#x} is not 256-byte aligned"
                )
            }
            Self::InvalidMemoryPlan => formatter.write_str("invalid GLM DSA memory plan"),
            Self::InsufficientPersistentCapacity {
                required,
                available,
            } => write!(
                formatter,
                "GLM DSA persistent arena needs {required} bytes, has {available}"
            ),
            Self::InsufficientWorkspaceCapacity {
                required,
                available,
            } => write!(
                formatter,
                "GLM DSA workspace arena needs {required} bytes, has {available}"
            ),
            Self::OverlappingArenas => {
                formatter.write_str("GLM DSA persistent and workspace arenas overlap")
            }
            Self::UnknownDsaLayer(layer) => write!(formatter, "layer {layer} is not GLM DSA"),
            Self::InvalidGraphBuckets => formatter.write_str("invalid GLM DSA graph buckets"),
            Self::InvalidOperation => formatter.write_str("invalid GLM DSA token operation"),
            Self::Busy(phase) => write!(formatter, "GLM DSA pipeline is busy: {phase:?}"),
            Self::StaleLease => formatter.write_str("stale or foreign GLM DSA stage lease"),
            Self::StaleReady => formatter.write_str("stale or foreign GLM DSA ready lease"),
            Self::NoNextStage => formatter.write_str("GLM DSA stage has no successor"),
            Self::PipelineNotPoisoned => formatter.write_str("GLM DSA pipeline is not poisoned"),
            Self::InternalInvariant => formatter.write_str("GLM DSA scheduler invariant failed"),
            Self::Overflow => formatter.write_str("GLM DSA address or epoch overflow"),
        }
    }
}

impl std::error::Error for GlmDsaPipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> GlmDsaSpec {
        GlmDsaSpec {
            dsa_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45],
            heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            qk_rope_head_dim: 0,
            v_head_dim: 256,
            index_heads: 32,
            index_head_dim: 128,
            index_topk: 2048,
            index_kpool: 4,
            layer_norm_epsilon: 1.0e-6,
        }
    }

    fn arena() -> GlmDsaFixedArena {
        GlmDsaFixedArena::new(
            &spec(),
            1,
            32 * 1024,
            0x1000_0000,
            500_000_000,
            0x8000_0000,
            8_000_000,
        )
        .expect("fixed arena")
    }

    fn scheduler() -> GlmDsaPipelineScheduler {
        GlmDsaPipelineScheduler::new(arena(), vec![1]).expect("scheduler")
    }

    fn complete_ready(scheduler: &mut GlmDsaPipelineScheduler, lease: GlmDsaLease) -> GlmDsaReady {
        let GlmDsaCompletion::StageReady(ready) =
            scheduler.complete_stage(lease).expect("complete stage")
        else {
            panic!("stage published too early")
        };
        ready
    }

    #[test]
    fn fixed_arena_charges_full_ring_and_checkpoint() {
        let arena = arena();
        let view = arena.view();
        assert_eq!(view.persistent_required_bytes, 270_950_400);
        assert_eq!(view.workspace_required_bytes, 2_572_032);
        assert_eq!(view.tail_key.bytes, 12_288);
        assert_eq!(view.tail_score.bytes, 12_288);
        assert_eq!(view.checkpoint_key.bytes, 1_024);
        assert_eq!(view.checkpoint_score.bytes, 1_024);
        for range in [
            view.mla_cache,
            view.pooled_index_cache,
            view.tail_key,
            view.tail_score,
            view.query,
            view.topk,
            view.history_indices,
            view.tail_indices,
            view.history_lengths,
            view.tail_lengths,
            view.history_mid_out,
            view.history_mid_lse,
            view.attention_output,
            view.attention_lse,
            view.tail_mid_out,
            view.tail_mid_lse,
            view.tail_output,
            view.tail_lse,
            view.mqa_schedule,
            view.checkpoint_key,
            view.checkpoint_score,
        ] {
            assert_eq!(range.address % GLM_DSA_ARENA_ALIGNMENT, 0);
        }
        let layer = arena.layer_view(3).expect("layer");
        assert_eq!(layer.mla_cache.bytes, 21_495_808);
        assert_eq!(layer.pooled_index_cache.bytes, 1_081_344);
        assert_eq!(layer.tail_key.bytes, 1_024);
        assert_eq!(layer.tail_score.bytes, 1_024);
    }

    #[test]
    fn token_publishes_only_after_all_ten_stages() {
        let mut scheduler = scheduler();
        let mut lease = scheduler.begin_token(3, 3, 1).expect("begin token");
        assert_eq!(lease.stage(), GlmDsaStage::TailCheckpoint);
        loop {
            match scheduler.complete_stage(lease).expect("complete") {
                GlmDsaCompletion::StageReady(ready) => {
                    lease = scheduler.begin_next(ready).expect("next");
                }
                GlmDsaCompletion::Published(publication) => {
                    assert_eq!(publication.sequence_length, 4);
                    assert_eq!(publication.pooled_entries, 1);
                    assert_eq!(publication.unpooled_tail_tokens, 0);
                    break;
                }
            }
        }
        assert_eq!(scheduler.phase(), GlmDsaPhase::Idle);
        assert_eq!(scheduler.stats().stages_completed, 10);
        assert_eq!(scheduler.stats().tokens_published, 1);
    }

    #[test]
    fn partial_kpool_update_poison_requires_exact_tail_restore() {
        let mut scheduler = scheduler();
        let checkpoint = scheduler.begin_token(3, 3, 1).expect("begin");
        let checkpointed = complete_ready(&mut scheduler, checkpoint);
        let projections = scheduler.begin_next(checkpointed).expect("projections");
        let projected = complete_ready(&mut scheduler, projections);
        let kpool = scheduler.begin_next(projected).expect("kpool");
        let GlmDsaAbort::RestoreState(rollback) = scheduler.abort_stage(kpool).expect("abort")
        else {
            panic!("KPool failure did not request restore")
        };
        assert_eq!(rollback.discard_mla_position, 3);
        assert_eq!(rollback.discard_pooled_entry, Some(0));
        assert_eq!(rollback.restore_sequence_length, 3);
        assert_eq!(rollback.restore_tail_tokens, 3);
        assert_eq!(rollback.tail.key_source.bytes, 1_024);
        assert_eq!(rollback.tail.key_snapshot.bytes, 1_024);
        assert_eq!(
            scheduler.phase(),
            GlmDsaPhase::Poisoned(GlmDsaStage::KPoolUpdate)
        );
        assert_eq!(
            scheduler.begin_token(3, 3, 1),
            Err(GlmDsaPipelineError::Busy(GlmDsaPhase::Poisoned(
                GlmDsaStage::KPoolUpdate
            )))
        );
        scheduler.recover_after_restore().expect("restored");
        assert_eq!(scheduler.phase(), GlmDsaPhase::Idle);
    }

    #[test]
    fn scratch_failure_after_mutation_retries_without_publishing_lengths() {
        let mut scheduler = scheduler();
        let checkpoint = scheduler.begin_token(7, 4, 1).expect("begin");
        let checkpointed = complete_ready(&mut scheduler, checkpoint);
        let projections = scheduler.begin_next(checkpointed).expect("projections");
        let projected = complete_ready(&mut scheduler, projections);
        let kpool = scheduler.begin_next(projected).expect("kpool");
        let updated = complete_ready(&mut scheduler, kpool);
        let score = scheduler.begin_next(updated).expect("score");
        let GlmDsaAbort::RetryFrom(retry) = scheduler.abort_stage(score).expect("retry") else {
            panic!("scratch failure should be retryable")
        };
        assert_eq!(retry.stage(), GlmDsaStage::KPoolUpdate);
        assert_eq!(scheduler.stats().tokens_published, 0);
        let score_retry = scheduler.begin_next(retry).expect("retry score");
        assert_eq!(score_retry.stage(), GlmDsaStage::IndexerScore);
    }

    #[test]
    fn cancelling_mutated_ready_state_discards_only_unpublished_slots() {
        let mut scheduler = scheduler();
        let checkpoint = scheduler.begin_token(11, 4, 1).expect("begin");
        let checkpointed = complete_ready(&mut scheduler, checkpoint);
        let projections = scheduler.begin_next(checkpointed).expect("projections");
        let projected = complete_ready(&mut scheduler, projections);
        let kpool = scheduler.begin_next(projected).expect("kpool");
        let updated = complete_ready(&mut scheduler, kpool);
        let GlmDsaAbort::RestoreState(rollback) = scheduler.cancel_ready(updated).expect("cancel")
        else {
            panic!("mutated token cancellation must restore")
        };
        assert_eq!(rollback.discard_mla_position, 4);
        assert_eq!(rollback.discard_pooled_entry, None);
        assert_eq!(rollback.restore_pooled_entries, 1);
        assert_eq!(rollback.restore_tail_tokens, 0);
    }

    #[test]
    fn rejects_wrong_layers_overlapping_arenas_and_small_capacity() {
        let mut scheduler = scheduler();
        assert_eq!(
            scheduler.begin_token(4, 0, 1),
            Err(GlmDsaPipelineError::UnknownDsaLayer(4))
        );
        assert!(matches!(
            GlmDsaFixedArena::new(
                &spec(),
                1,
                32 * 1024,
                0x1000_0000,
                270_950_399,
                0x8000_0000,
                8_000_000,
            ),
            Err(GlmDsaPipelineError::InsufficientPersistentCapacity { .. })
        ));
        assert_eq!(
            GlmDsaFixedArena::new(
                &spec(),
                1,
                32 * 1024,
                0x1000_0000,
                500_000_000,
                0x1000_1000,
                8_000_000,
            ),
            Err(GlmDsaPipelineError::OverlappingArenas)
        );
    }
}
