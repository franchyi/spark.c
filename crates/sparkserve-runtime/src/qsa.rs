//! Fixed-address QSA sparse-decode scratch owned by the Rust scheduler.
//!
//! CUDA donors receive raw pointers into this layout. They do not allocate,
//! resize, page, or decide residency. On DGX Spark a single device allocation
//! is backed by unified system memory, but remains resident and address-stable
//! for CUDA graph replay.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

pub const QWEN_QSA_TOPK: usize = 2051;
pub const QWEN_QSA_PAGE_TOKENS: usize = 64;
pub const QWEN_QSA_PAGES_PER_ROW: usize = 33;
pub const QWEN_QSA_PACKED_ROW_TOKENS: usize = 2112;
pub const QWEN_QSA_KV_HEADS: usize = 2;
pub const QWEN_QSA_HEAD_DIM: usize = 256;
pub const QWEN_QSA_QUERY_HEADS: usize = 24;
pub const ATTENTION_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;
pub const XQA_SEMAPHORE_BYTES: usize = 8 * 1024 * 1024;

const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;
const SCRATCH_ALIGNMENT: usize = 256;
static NEXT_QSA_SCHEDULER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaSparseDecodePlan {
    max_batch: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaScratchLayout {
    pub packed_key_offset: usize,
    pub packed_key_bytes: usize,
    pub packed_value_offset: usize,
    pub packed_value_bytes: usize,
    pub valid_counts_offset: usize,
    pub valid_counts_bytes: usize,
    pub block_tables_offset: usize,
    pub block_tables_bytes: usize,
    pub attention_output_offset: usize,
    pub attention_output_bytes: usize,
    pub attention_workspace_offset: usize,
    pub attention_workspace_bytes: usize,
    pub xqa_semaphore_offset: usize,
    pub xqa_semaphore_bytes: usize,
    pub xqa_scratch_offset: usize,
    pub xqa_scratch_bytes: usize,
    pub total_bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaPlanError {
    ZeroBatch,
    BatchExceedsCapacity { active: usize, capacity: usize },
    BufferTooSmall { needed: usize, available: usize },
    SizeOverflow,
}

impl fmt::Display for QsaPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatch => write!(formatter, "QSA graph batch must be non-zero"),
            Self::BatchExceedsCapacity { active, capacity } => write!(
                formatter,
                "QSA active batch {active} exceeds graph capacity {capacity}"
            ),
            Self::BufferTooSmall { needed, available } => write!(
                formatter,
                "QSA metadata buffer needs {needed} entries but has {available}"
            ),
            Self::SizeOverflow => write!(formatter, "QSA scratch size overflow"),
        }
    }
}

impl std::error::Error for QsaPlanError {}

impl QsaSparseDecodePlan {
    pub fn qwen38_flash(max_batch: usize) -> Result<Self, QsaPlanError> {
        if max_batch == 0 {
            return Err(QsaPlanError::ZeroBatch);
        }
        let plan = Self { max_batch };
        plan.scratch_layout()?;
        Ok(plan)
    }

    pub fn max_batch(self) -> usize {
        self.max_batch
    }

    pub fn packed_row_bytes(self) -> usize {
        QWEN_QSA_PACKED_ROW_TOKENS * QWEN_QSA_KV_HEADS * QWEN_QSA_HEAD_DIM * BF16_BYTES
    }

    pub fn scratch_layout(self) -> Result<QsaScratchLayout, QsaPlanError> {
        let packed_bytes = self
            .packed_row_bytes()
            .checked_mul(self.max_batch)
            .ok_or(QsaPlanError::SizeOverflow)?;
        let valid_counts_bytes = self
            .max_batch
            .checked_mul(I32_BYTES)
            .ok_or(QsaPlanError::SizeOverflow)?;
        let block_tables_bytes = self
            .max_batch
            .checked_mul(QWEN_QSA_PAGES_PER_ROW)
            .and_then(|entries| entries.checked_mul(I32_BYTES))
            .ok_or(QsaPlanError::SizeOverflow)?;
        let attention_output_bytes = self
            .max_batch
            .checked_mul(QWEN_QSA_QUERY_HEADS)
            .and_then(|elements| elements.checked_mul(QWEN_QSA_HEAD_DIM))
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(QsaPlanError::SizeOverflow)?;

        let packed_key_offset: usize = 0;
        let packed_value_offset = align_up(
            packed_key_offset
                .checked_add(packed_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let valid_counts_offset = align_up(
            packed_value_offset
                .checked_add(packed_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let block_tables_offset = align_up(
            valid_counts_offset
                .checked_add(valid_counts_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let attention_output_offset = align_up(
            block_tables_offset
                .checked_add(block_tables_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let attention_workspace_offset = align_up(
            attention_output_offset
                .checked_add(attention_output_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let total_bytes = attention_workspace_offset
            .checked_add(ATTENTION_WORKSPACE_BYTES)
            .ok_or(QsaPlanError::SizeOverflow)?;
        let xqa_semaphore_offset = attention_workspace_offset;
        let xqa_scratch_offset = xqa_semaphore_offset
            .checked_add(XQA_SEMAPHORE_BYTES)
            .ok_or(QsaPlanError::SizeOverflow)?;
        let xqa_scratch_bytes = ATTENTION_WORKSPACE_BYTES
            .checked_sub(XQA_SEMAPHORE_BYTES)
            .ok_or(QsaPlanError::SizeOverflow)?;

        Ok(QsaScratchLayout {
            packed_key_offset,
            packed_key_bytes: packed_bytes,
            packed_value_offset,
            packed_value_bytes: packed_bytes,
            valid_counts_offset,
            valid_counts_bytes,
            block_tables_offset,
            block_tables_bytes,
            attention_output_offset,
            attention_output_bytes,
            attention_workspace_offset,
            attention_workspace_bytes: ATTENTION_WORKSPACE_BYTES,
            xqa_semaphore_offset,
            xqa_semaphore_bytes: XQA_SEMAPHORE_BYTES,
            xqa_scratch_offset,
            xqa_scratch_bytes,
            total_bytes,
            alignment: SCRATCH_ALIGNMENT,
        })
    }

    /// Fill the immutable row-major XQA block table once, before graph
    /// capture. Each request owns 33 consecutive 64-token pages.
    pub fn fill_block_tables(self, output: &mut [i32]) -> Result<usize, QsaPlanError> {
        let needed = self
            .max_batch
            .checked_mul(QWEN_QSA_PAGES_PER_ROW)
            .ok_or(QsaPlanError::SizeOverflow)?;
        if output.len() < needed {
            return Err(QsaPlanError::BufferTooSmall {
                needed,
                available: output.len(),
            });
        }
        for row in 0..self.max_batch {
            for page in 0..QWEN_QSA_PAGES_PER_ROW {
                output[row * QWEN_QSA_PAGES_PER_ROW + page] =
                    i32::try_from(row * QWEN_QSA_PAGES_PER_ROW + page)
                        .map_err(|_| QsaPlanError::SizeOverflow)?;
            }
        }
        Ok(needed)
    }

    pub fn validate_active_batch(self, active_batch: usize) -> Result<(), QsaPlanError> {
        if active_batch == 0 {
            return Err(QsaPlanError::ZeroBatch);
        }
        if active_batch > self.max_batch {
            return Err(QsaPlanError::BatchExceedsCapacity {
                active: active_batch,
                capacity: self.max_batch,
            });
        }
        Ok(())
    }
}

/// Stable device addresses carved from one max-batch allocation.
///
/// The allocation may be CUDA device memory or a mapped coherent slab on
/// GB10. The scheduler only requires that the device address remains valid for
/// the lifetime of all captured graphs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaScratchAddresses {
    pub packed_key: u64,
    pub packed_value: u64,
    pub valid_counts: u64,
    pub block_tables: u64,
    pub attention_output: u64,
    pub attention_workspace: u64,
    pub xqa_semaphores: u64,
    pub xqa_scratch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaArenaView {
    device_base: u64,
    device_bytes: usize,
    layout: QsaScratchLayout,
}

impl QsaArenaView {
    pub fn new(
        plan: QsaSparseDecodePlan,
        device_base: u64,
        device_bytes: usize,
    ) -> Result<Self, QsaScheduleError> {
        let layout = plan.scratch_layout()?;
        if device_base == 0 || device_base % layout.alignment as u64 != 0 {
            return Err(QsaScheduleError::InvalidArenaBase {
                base: device_base,
                alignment: layout.alignment,
            });
        }
        if device_bytes < layout.total_bytes {
            return Err(QsaScheduleError::ArenaTooSmall {
                needed: layout.total_bytes,
                available: device_bytes,
            });
        }
        device_base
            .checked_add(u64::try_from(layout.total_bytes).map_err(|_| QsaPlanError::SizeOverflow)?)
            .ok_or(QsaScheduleError::AddressOverflow)?;
        Ok(Self {
            device_base,
            device_bytes,
            layout,
        })
    }

    pub fn device_base(self) -> u64 {
        self.device_base
    }

    pub fn device_bytes(self) -> usize {
        self.device_bytes
    }

    pub fn layout(self) -> QsaScratchLayout {
        self.layout
    }

    pub fn addresses(self) -> QsaScratchAddresses {
        let address = |offset: usize| self.device_base + offset as u64;
        QsaScratchAddresses {
            packed_key: address(self.layout.packed_key_offset),
            packed_value: address(self.layout.packed_value_offset),
            valid_counts: address(self.layout.valid_counts_offset),
            block_tables: address(self.layout.block_tables_offset),
            attention_output: address(self.layout.attention_output_offset),
            attention_workspace: address(self.layout.attention_workspace_offset),
            xqa_semaphores: address(self.layout.xqa_semaphore_offset),
            xqa_scratch: address(self.layout.xqa_scratch_offset),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaArenaPhase {
    WorkspaceNeedsZero,
    ZeroingWorkspace,
    Idle,
    Packing,
    Ready,
    Decoding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QsaOperation {
    epoch: u64,
    active_batch: usize,
    graph_batch: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QsaArenaState {
    WorkspaceNeedsZero,
    ZeroingWorkspace { epoch: u64 },
    Idle,
    Packing(QsaOperation),
    Ready(QsaOperation),
    Decoding(QsaOperation),
}

impl QsaArenaState {
    fn phase(self) -> QsaArenaPhase {
        match self {
            Self::WorkspaceNeedsZero => QsaArenaPhase::WorkspaceNeedsZero,
            Self::ZeroingWorkspace { .. } => QsaArenaPhase::ZeroingWorkspace,
            Self::Idle => QsaArenaPhase::Idle,
            Self::Packing(_) => QsaArenaPhase::Packing,
            Self::Ready(_) => QsaArenaPhase::Ready,
            Self::Decoding(_) => QsaArenaPhase::Decoding,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaWorkspaceZeroLease {
    scheduler_id: u64,
    epoch: u64,
    arena: QsaArenaView,
}

impl QsaWorkspaceZeroLease {
    pub fn arena(self) -> QsaArenaView {
        self.arena
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaPackLease {
    scheduler_id: u64,
    operation: QsaOperation,
    arena: QsaArenaView,
}

impl QsaPackLease {
    pub fn active_batch(self) -> usize {
        self.operation.active_batch
    }

    pub fn graph_batch(self) -> usize {
        self.operation.graph_batch
    }

    pub fn arena(self) -> QsaArenaView {
        self.arena
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaReadyLease {
    scheduler_id: u64,
    operation: QsaOperation,
    arena: QsaArenaView,
}

impl QsaReadyLease {
    pub fn active_batch(self) -> usize {
        self.operation.active_batch
    }

    pub fn graph_batch(self) -> usize {
        self.operation.graph_batch
    }

    pub fn arena(self) -> QsaArenaView {
        self.arena
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaDecodeLease {
    scheduler_id: u64,
    operation: QsaOperation,
    arena: QsaArenaView,
}

impl QsaDecodeLease {
    pub fn active_batch(self) -> usize {
        self.operation.active_batch
    }

    pub fn graph_batch(self) -> usize {
        self.operation.graph_batch
    }

    pub fn arena(self) -> QsaArenaView {
        self.arena
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QsaArenaStats {
    pub workspace_zeroes_submitted: u64,
    pub workspace_zeroes_completed: u64,
    pub workspace_zeroes_failed: u64,
    pub packs_submitted: u64,
    pub packs_completed: u64,
    pub packs_failed: u64,
    pub packed_batches_discarded: u64,
    pub decodes_submitted: u64,
    pub decodes_completed: u64,
    pub decodes_failed: u64,
    pub rows_completed: u64,
    pub arena_stalls: u64,
    pub peak_graph_batch: usize,
}

/// Allocation-free owner of the shared QSA pack/XQA arena.
///
/// Graph buckets share one max-batch allocation; a smaller bucket changes only
/// launch metadata, never pointers. Completion methods must be called only
/// after the corresponding CUDA event has completed. Until that acknowledgement
/// arrives, the state machine refuses to repack or replay the arena.
pub struct QsaArenaScheduler {
    scheduler_id: u64,
    plan: QsaSparseDecodePlan,
    arena: QsaArenaView,
    graph_buckets: Box<[usize]>,
    state: QsaArenaState,
    next_epoch: u64,
    stats: QsaArenaStats,
}

impl QsaArenaScheduler {
    pub fn new(
        plan: QsaSparseDecodePlan,
        device_base: u64,
        device_bytes: usize,
        graph_buckets: Vec<usize>,
    ) -> Result<Self, QsaScheduleError> {
        validate_graph_buckets(plan, &graph_buckets)?;
        let arena = QsaArenaView::new(plan, device_base, device_bytes)?;
        let scheduler_id = NEXT_QSA_SCHEDULER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| QsaScheduleError::SchedulerIdOverflow)?;
        Ok(Self {
            scheduler_id,
            plan,
            arena,
            graph_buckets: graph_buckets.into_boxed_slice(),
            state: QsaArenaState::WorkspaceNeedsZero,
            next_epoch: 0,
            stats: QsaArenaStats::default(),
        })
    }

    pub fn arena(&self) -> QsaArenaView {
        self.arena
    }

    pub fn graph_buckets(&self) -> &[usize] {
        &self.graph_buckets
    }

    pub fn phase(&self) -> QsaArenaPhase {
        self.state.phase()
    }

    pub fn stats(&self) -> QsaArenaStats {
        self.stats
    }

    /// Reserve the arena for an asynchronous zero of the full attention
    /// workspace, including XQA's semaphore region.
    pub fn begin_workspace_zero(&mut self) -> Result<QsaWorkspaceZeroLease, QsaScheduleError> {
        if !matches!(self.state, QsaArenaState::WorkspaceNeedsZero) {
            return Err(QsaScheduleError::WorkspaceZeroNotRequired(
                self.state.phase(),
            ));
        }
        let epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(QsaScheduleError::EpochOverflow)?;
        self.next_epoch = epoch;
        self.state = QsaArenaState::ZeroingWorkspace { epoch };
        self.stats.workspace_zeroes_submitted =
            self.stats.workspace_zeroes_submitted.saturating_add(1);
        Ok(QsaWorkspaceZeroLease {
            scheduler_id: self.scheduler_id,
            epoch,
            arena: self.arena,
        })
    }

    /// Publish a clean workspace only after the memset CUDA event completes.
    pub fn complete_workspace_zero(
        &mut self,
        lease: QsaWorkspaceZeroLease,
    ) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != (QsaArenaState::ZeroingWorkspace { epoch: lease.epoch })
        {
            return Err(QsaScheduleError::StaleWorkspaceZeroLease);
        }
        self.state = QsaArenaState::Idle;
        self.stats.workspace_zeroes_completed =
            self.stats.workspace_zeroes_completed.saturating_add(1);
        Ok(())
    }

    /// Keep the arena quarantined after a failed memset.
    pub fn abort_workspace_zero(
        &mut self,
        lease: QsaWorkspaceZeroLease,
    ) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != (QsaArenaState::ZeroingWorkspace { epoch: lease.epoch })
        {
            return Err(QsaScheduleError::StaleWorkspaceZeroLease);
        }
        self.state = QsaArenaState::WorkspaceNeedsZero;
        self.stats.workspace_zeroes_failed = self.stats.workspace_zeroes_failed.saturating_add(1);
        Ok(())
    }

    pub fn begin_pack(&mut self, active_batch: usize) -> Result<QsaPackLease, QsaScheduleError> {
        self.plan.validate_active_batch(active_batch)?;
        match self.state {
            QsaArenaState::WorkspaceNeedsZero => {
                return Err(QsaScheduleError::WorkspaceNotInitialized);
            }
            QsaArenaState::Idle => {}
            _ => {
                self.stats.arena_stalls = self.stats.arena_stalls.saturating_add(1);
                return Err(QsaScheduleError::ArenaBusy(self.state.phase()));
            }
        }
        let graph_batch = self
            .graph_buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= active_batch)
            .ok_or(QsaScheduleError::InvalidGraphBuckets)?;
        let epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(QsaScheduleError::EpochOverflow)?;
        let operation = QsaOperation {
            epoch,
            active_batch,
            graph_batch,
        };
        self.next_epoch = epoch;
        self.state = QsaArenaState::Packing(operation);
        self.stats.packs_submitted = self.stats.packs_submitted.saturating_add(1);
        self.stats.peak_graph_batch = self.stats.peak_graph_batch.max(graph_batch);
        Ok(QsaPackLease {
            scheduler_id: self.scheduler_id,
            operation,
            arena: self.arena,
        })
    }

    /// Publish packed K/V only after the pack stream's CUDA event completes.
    pub fn complete_pack(
        &mut self,
        lease: QsaPackLease,
    ) -> Result<QsaReadyLease, QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Packing(lease.operation)
        {
            return Err(QsaScheduleError::StalePackLease);
        }
        self.state = QsaArenaState::Ready(lease.operation);
        self.stats.packs_completed = self.stats.packs_completed.saturating_add(1);
        Ok(QsaReadyLease {
            scheduler_id: self.scheduler_id,
            operation: lease.operation,
            arena: self.arena,
        })
    }

    /// Release a failed pack only after all writes to the arena have stopped.
    pub fn abort_pack(&mut self, lease: QsaPackLease) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Packing(lease.operation)
        {
            return Err(QsaScheduleError::StalePackLease);
        }
        self.state = QsaArenaState::Idle;
        self.stats.packs_failed = self.stats.packs_failed.saturating_add(1);
        Ok(())
    }

    pub fn begin_decode(
        &mut self,
        lease: QsaReadyLease,
    ) -> Result<QsaDecodeLease, QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Ready(lease.operation)
        {
            return Err(QsaScheduleError::StaleReadyLease);
        }
        self.state = QsaArenaState::Decoding(lease.operation);
        self.stats.decodes_submitted = self.stats.decodes_submitted.saturating_add(1);
        Ok(QsaDecodeLease {
            scheduler_id: self.scheduler_id,
            operation: lease.operation,
            arena: self.arena,
        })
    }

    /// Drop a packed batch that will not be submitted to XQA.
    pub fn discard_ready(&mut self, lease: QsaReadyLease) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Ready(lease.operation)
        {
            return Err(QsaScheduleError::StaleReadyLease);
        }
        self.state = QsaArenaState::Idle;
        self.stats.packed_batches_discarded = self.stats.packed_batches_discarded.saturating_add(1);
        Ok(())
    }

    /// Release the arena only after the XQA stream's CUDA event completes.
    pub fn complete_decode(&mut self, lease: QsaDecodeLease) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Decoding(lease.operation)
        {
            return Err(QsaScheduleError::StaleDecodeLease);
        }
        self.state = QsaArenaState::Idle;
        self.stats.decodes_completed = self.stats.decodes_completed.saturating_add(1);
        self.stats.rows_completed = self
            .stats
            .rows_completed
            .saturating_add(lease.operation.active_batch as u64);
        Ok(())
    }

    /// Release a failed decode only after the failed launch can no longer touch
    /// the shared arena. A partial XQA launch may leave a semaphore non-zero,
    /// so recovery requires another full workspace zero before reuse.
    pub fn abort_decode(&mut self, lease: QsaDecodeLease) -> Result<(), QsaScheduleError> {
        if lease.scheduler_id != self.scheduler_id
            || self.state != QsaArenaState::Decoding(lease.operation)
        {
            return Err(QsaScheduleError::StaleDecodeLease);
        }
        self.state = QsaArenaState::WorkspaceNeedsZero;
        self.stats.decodes_failed = self.stats.decodes_failed.saturating_add(1);
        Ok(())
    }
}

fn validate_graph_buckets(
    plan: QsaSparseDecodePlan,
    graph_buckets: &[usize],
) -> Result<(), QsaScheduleError> {
    if graph_buckets.is_empty()
        || graph_buckets[0] == 0
        || graph_buckets.last().copied() != Some(plan.max_batch())
        || graph_buckets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(QsaScheduleError::InvalidGraphBuckets);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaScheduleError {
    Plan(QsaPlanError),
    InvalidArenaBase { base: u64, alignment: usize },
    ArenaTooSmall { needed: usize, available: usize },
    AddressOverflow,
    InvalidGraphBuckets,
    SchedulerIdOverflow,
    EpochOverflow,
    WorkspaceNotInitialized,
    WorkspaceZeroNotRequired(QsaArenaPhase),
    ArenaBusy(QsaArenaPhase),
    StaleWorkspaceZeroLease,
    StalePackLease,
    StaleReadyLease,
    StaleDecodeLease,
}

impl From<QsaPlanError> for QsaScheduleError {
    fn from(error: QsaPlanError) -> Self {
        Self::Plan(error)
    }
}

impl fmt::Display for QsaScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(formatter),
            Self::InvalidArenaBase { base, alignment } => write!(
                formatter,
                "QSA device base 0x{base:x} is null or not {alignment}-byte aligned"
            ),
            Self::ArenaTooSmall { needed, available } => write!(
                formatter,
                "QSA arena needs {needed} bytes but has {available}"
            ),
            Self::AddressOverflow => formatter.write_str("QSA device address overflow"),
            Self::InvalidGraphBuckets => formatter.write_str(
                "QSA graph buckets must be non-zero, strictly increasing, and end at max batch",
            ),
            Self::SchedulerIdOverflow => formatter.write_str("QSA scheduler ID overflow"),
            Self::EpochOverflow => formatter.write_str("QSA arena epoch overflow"),
            Self::WorkspaceNotInitialized => {
                formatter.write_str("QSA workspace zero has not completed")
            }
            Self::WorkspaceZeroNotRequired(phase) => {
                write!(
                    formatter,
                    "QSA workspace zero is not required in phase {phase:?}"
                )
            }
            Self::ArenaBusy(phase) => write!(formatter, "QSA arena is busy in phase {phase:?}"),
            Self::StaleWorkspaceZeroLease => {
                formatter.write_str("stale or foreign QSA workspace-zero lease")
            }
            Self::StalePackLease => formatter.write_str("stale or foreign QSA pack lease"),
            Self::StaleReadyLease => formatter.write_str("stale or foreign QSA ready lease"),
            Self::StaleDecodeLease => formatter.write_str("stale or foreign QSA decode lease"),
        }
    }
}

impl std::error::Error for QsaScheduleError {}

/// QSA control plane plus the coherent allocation that backs every graph
/// pointer. Keeping both in one owner prevents the registered mapping from
/// being released independently of scheduler state.
#[cfg(feature = "native-fabric")]
pub struct QsaCoherentArena {
    // Drop scheduler metadata before unregistering the mapping.
    scheduler: QsaArenaScheduler,
    region: Option<crate::coherent::CoherentRegionOwner>,
}

#[cfg(feature = "native-fabric")]
impl QsaCoherentArena {
    pub fn allocate(
        plan: QsaSparseDecodePlan,
        graph_buckets: Vec<usize>,
        coherent_flags: u32,
    ) -> Result<Self, QsaCoherentArenaError> {
        let bytes = plan.scratch_layout()?.total_bytes;
        let payload_bytes = u64::try_from(bytes).map_err(|_| QsaPlanError::SizeOverflow)?;
        let region =
            crate::coherent::CoherentRegionOwner::slab(payload_bytes, 4096, coherent_flags)?;
        let scheduler = QsaArenaScheduler::new(
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

    pub fn scheduler(&self) -> &QsaArenaScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut QsaArenaScheduler {
        &mut self.scheduler
    }

    pub fn region(&self) -> &crate::coherent::CoherentRegionOwner {
        self.region.as_ref().expect("QSA coherent region is owned")
    }

    /// Borrow the CPU alias only while the scheduler is not packing or
    /// decoding and no CUDA graph can access the arena.
    ///
    /// # Safety
    ///
    /// The caller must additionally ensure every previously recorded CUDA
    /// event has completed. The phase check prevents ordinary in-flight use,
    /// but it cannot query CUDA by itself.
    pub unsafe fn host_payload_mut(&mut self) -> Result<&mut [u8], QsaCoherentArenaError> {
        if matches!(
            self.scheduler.phase(),
            QsaArenaPhase::ZeroingWorkspace
                | QsaArenaPhase::Packing
                | QsaArenaPhase::Ready
                | QsaArenaPhase::Decoding
        ) {
            return Err(QsaCoherentArenaError::ArenaBusy(self.scheduler.phase()));
        }
        // SAFETY: the caller contract and phase check exclude CUDA access.
        Ok(unsafe {
            self.region
                .as_mut()
                .expect("QSA coherent region is owned")
                .host_payload_mut()?
        })
    }
}

#[cfg(feature = "native-fabric")]
impl Drop for QsaCoherentArena {
    fn drop(&mut self) {
        if matches!(
            self.scheduler.phase(),
            QsaArenaPhase::ZeroingWorkspace | QsaArenaPhase::Packing | QsaArenaPhase::Decoding
        ) {
            // Unregistering a mapping still owned by an asynchronous CUDA
            // operation would create a use-after-free. Leaking on contract
            // violation is bounded to process teardown and is safer than
            // releasing live device pointers.
            if let Some(region) = self.region.take() {
                std::mem::forget(region);
            }
        }
    }
}

#[cfg(feature = "native-fabric")]
#[derive(Debug)]
pub enum QsaCoherentArenaError {
    Plan(QsaPlanError),
    Coherent(crate::coherent::CoherentRegionError),
    Schedule(QsaScheduleError),
    ArenaBusy(QsaArenaPhase),
}

#[cfg(feature = "native-fabric")]
impl From<QsaPlanError> for QsaCoherentArenaError {
    fn from(error: QsaPlanError) -> Self {
        Self::Plan(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<crate::coherent::CoherentRegionError> for QsaCoherentArenaError {
    fn from(error: crate::coherent::CoherentRegionError) -> Self {
        Self::Coherent(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<QsaScheduleError> for QsaCoherentArenaError {
    fn from(error: QsaScheduleError) -> Self {
        Self::Schedule(error)
    }
}

#[cfg(feature = "native-fabric")]
impl fmt::Display for QsaCoherentArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(formatter),
            Self::Coherent(error) => error.fmt(formatter),
            Self::Schedule(error) => error.fmt(formatter),
            Self::ArenaBusy(phase) => {
                write!(formatter, "QSA coherent arena is busy in phase {phase:?}")
            }
        }
    }
}

#[cfg(feature = "native-fabric")]
impl std::error::Error for QsaCoherentArenaError {}

#[cfg(feature = "native-fabric")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QsaCudaPending {
    WorkspaceZero(QsaWorkspaceZeroLease),
    Pack(QsaPackLease),
    Decode(QsaDecodeLease),
}

#[cfg(feature = "native-fabric")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaCudaCompletion {
    WorkspaceReady,
    PackReady(QsaReadyLease),
    DecodeComplete,
}

/// Binds one reusable CUDA event to exactly one QSA scheduler lease.
///
/// The fixed arena permits only one zero/pack/decode writer at a time, so one
/// fence is sufficient. A lease is published only after `query` reports CUDA
/// completion or `wait` synchronizes the recorded event.
#[cfg(feature = "native-fabric")]
pub struct QsaCudaFence {
    event: crate::cuda::CudaEventOwner,
    pending: Option<QsaCudaPending>,
}

#[cfg(feature = "native-fabric")]
impl QsaCudaFence {
    pub fn create() -> Result<Self, QsaCudaFenceError> {
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
    ) -> Result<(), QsaCudaFenceError> {
        self.record(stream, QsaCudaPending::WorkspaceZero(lease))
    }

    pub fn record_pack(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        lease: QsaPackLease,
    ) -> Result<(), QsaCudaFenceError> {
        self.record(stream, QsaCudaPending::Pack(lease))
    }

    pub fn record_decode(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        lease: QsaDecodeLease,
    ) -> Result<(), QsaCudaFenceError> {
        self.record(stream, QsaCudaPending::Decode(lease))
    }

    pub fn poll(
        &mut self,
        scheduler: &mut QsaArenaScheduler,
    ) -> Result<Option<QsaCudaCompletion>, QsaCudaFenceError> {
        if self.pending.is_none() {
            return Err(QsaCudaFenceError::NoPendingEvent);
        }
        if !self.event.query()? {
            return Ok(None);
        }
        self.publish(scheduler).map(Some)
    }

    pub fn wait(
        &mut self,
        scheduler: &mut QsaArenaScheduler,
    ) -> Result<QsaCudaCompletion, QsaCudaFenceError> {
        if self.pending.is_none() {
            return Err(QsaCudaFenceError::NoPendingEvent);
        }
        self.event.synchronize()?;
        self.publish(scheduler)
    }

    fn record(
        &mut self,
        stream: &mut crate::cuda::CudaStreamOwner,
        pending: QsaCudaPending,
    ) -> Result<(), QsaCudaFenceError> {
        if self.pending.is_some() {
            return Err(QsaCudaFenceError::EventBusy);
        }
        self.event.record(stream)?;
        self.pending = Some(pending);
        Ok(())
    }

    fn publish(
        &mut self,
        scheduler: &mut QsaArenaScheduler,
    ) -> Result<QsaCudaCompletion, QsaCudaFenceError> {
        let pending = self
            .pending
            .take()
            .ok_or(QsaCudaFenceError::NoPendingEvent)?;
        match pending {
            QsaCudaPending::WorkspaceZero(lease) => {
                scheduler.complete_workspace_zero(lease)?;
                Ok(QsaCudaCompletion::WorkspaceReady)
            }
            QsaCudaPending::Pack(lease) => Ok(QsaCudaCompletion::PackReady(
                scheduler.complete_pack(lease)?,
            )),
            QsaCudaPending::Decode(lease) => {
                scheduler.complete_decode(lease)?;
                Ok(QsaCudaCompletion::DecodeComplete)
            }
        }
    }
}

#[cfg(feature = "native-fabric")]
#[derive(Debug)]
pub enum QsaCudaFenceError {
    Runtime(crate::cuda::CudaRuntimeError),
    Schedule(QsaScheduleError),
    EventBusy,
    NoPendingEvent,
}

#[cfg(feature = "native-fabric")]
impl From<crate::cuda::CudaRuntimeError> for QsaCudaFenceError {
    fn from(error: crate::cuda::CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

#[cfg(feature = "native-fabric")]
impl From<QsaScheduleError> for QsaCudaFenceError {
    fn from(error: QsaScheduleError) -> Self {
        Self::Schedule(error)
    }
}

#[cfg(feature = "native-fabric")]
impl fmt::Display for QsaCudaFenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Schedule(error) => error.fmt(formatter),
            Self::EventBusy => formatter.write_str("QSA CUDA fence already owns a lease"),
            Self::NoPendingEvent => formatter.write_str("QSA CUDA fence has no pending lease"),
        }
    }
}

#[cfg(feature = "native-fabric")]
impl std::error::Error for QsaCudaFenceError {}

fn align_up(value: usize, alignment: usize) -> Result<usize, QsaPlanError> {
    let add = alignment.checked_sub(1).ok_or(QsaPlanError::SizeOverflow)?;
    value
        .checked_add(add)
        .map(|sum| sum / alignment * alignment)
        .ok_or(QsaPlanError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DEVICE_BASE: u64 = 0x1_0000_0000;

    fn arena_scheduler() -> QsaArenaScheduler {
        let plan = QsaSparseDecodePlan::qwen38_flash(8).expect("plan");
        let bytes = plan.scratch_layout().expect("layout").total_bytes;
        QsaArenaScheduler::new(plan, TEST_DEVICE_BASE, bytes, vec![1, 2, 4, 8]).expect("scheduler")
    }

    fn zero_workspace(scheduler: &mut QsaArenaScheduler) {
        let zero = scheduler.begin_workspace_zero().expect("begin zero");
        assert_eq!(zero.arena(), scheduler.arena());
        scheduler
            .complete_workspace_zero(zero)
            .expect("complete zero");
    }

    #[test]
    fn qwen_geometry_matches_sglang_xqa_path() {
        let plan = QsaSparseDecodePlan::qwen38_flash(8).expect("plan");
        assert_eq!(QWEN_QSA_PAGES_PER_ROW, 33);
        assert_eq!(QWEN_QSA_PACKED_ROW_TOKENS, 2112);
        assert_eq!(plan.packed_row_bytes(), 2_162_688);
    }

    #[test]
    fn scratch_regions_are_fixed_aligned_and_non_overlapping() {
        let layout = QsaSparseDecodePlan::qwen38_flash(4)
            .expect("plan")
            .scratch_layout()
            .expect("layout");
        let regions = [
            (layout.packed_key_offset, layout.packed_key_bytes),
            (layout.packed_value_offset, layout.packed_value_bytes),
            (layout.valid_counts_offset, layout.valid_counts_bytes),
            (layout.block_tables_offset, layout.block_tables_bytes),
            (
                layout.attention_output_offset,
                layout.attention_output_bytes,
            ),
            (
                layout.attention_workspace_offset,
                layout.attention_workspace_bytes,
            ),
        ];
        for (index, (offset, bytes)) in regions.iter().copied().enumerate() {
            assert_eq!(offset % layout.alignment, 0);
            assert!(bytes > 0);
            if let Some((next_offset, _)) = regions.get(index + 1) {
                assert!(offset + bytes <= *next_offset);
            }
        }
        let (last_offset, last_bytes) = regions[regions.len() - 1];
        assert_eq!(layout.total_bytes, last_offset + last_bytes);
        assert_eq!(
            layout.xqa_semaphore_offset,
            layout.attention_workspace_offset
        );
        assert_eq!(layout.xqa_semaphore_bytes, 8 * 1024 * 1024);
        assert_eq!(
            layout.xqa_scratch_offset,
            layout.xqa_semaphore_offset + layout.xqa_semaphore_bytes
        );
        assert_eq!(
            layout.xqa_scratch_bytes,
            layout.attention_workspace_bytes - layout.xqa_semaphore_bytes
        );
    }

    #[test]
    fn block_tables_are_static_consecutive_pages() {
        let plan = QsaSparseDecodePlan::qwen38_flash(3).expect("plan");
        let mut tables = vec![-1; 3 * QWEN_QSA_PAGES_PER_ROW];
        assert_eq!(plan.fill_block_tables(&mut tables).expect("fill"), 99);
        assert_eq!(tables[0], 0);
        assert_eq!(tables[32], 32);
        assert_eq!(tables[33], 33);
        assert_eq!(tables[98], 98);
    }

    #[test]
    fn graph_bucket_rejects_larger_active_batch() {
        let plan = QsaSparseDecodePlan::qwen38_flash(8).expect("plan");
        assert!(plan.validate_active_batch(8).is_ok());
        assert_eq!(
            plan.validate_active_batch(9),
            Err(QsaPlanError::BatchExceedsCapacity {
                active: 9,
                capacity: 8,
            })
        );
    }

    #[test]
    fn workspace_zero_is_required_before_first_pack() {
        let mut scheduler = arena_scheduler();
        assert_eq!(scheduler.phase(), QsaArenaPhase::WorkspaceNeedsZero);
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::WorkspaceNotInitialized)
        );
        let zero = scheduler.begin_workspace_zero().expect("begin zero");
        assert_eq!(scheduler.phase(), QsaArenaPhase::ZeroingWorkspace);
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::ArenaBusy(QsaArenaPhase::ZeroingWorkspace))
        );
        scheduler
            .complete_workspace_zero(zero)
            .expect("complete zero");
        assert_eq!(scheduler.phase(), QsaArenaPhase::Idle);
        assert_eq!(
            scheduler.begin_workspace_zero(),
            Err(QsaScheduleError::WorkspaceZeroNotRequired(
                QsaArenaPhase::Idle
            ))
        );
        assert_eq!(scheduler.stats().workspace_zeroes_submitted, 1);
        assert_eq!(scheduler.stats().workspace_zeroes_completed, 1);
        assert_eq!(scheduler.stats().workspace_zeroes_failed, 0);
    }

    #[test]
    fn failed_workspace_zero_stays_quarantined_for_retry() {
        let mut scheduler = arena_scheduler();
        let failed = scheduler.begin_workspace_zero().expect("begin failed zero");
        scheduler
            .abort_workspace_zero(failed)
            .expect("abort failed zero");
        assert_eq!(scheduler.phase(), QsaArenaPhase::WorkspaceNeedsZero);
        assert_eq!(
            scheduler.complete_workspace_zero(failed),
            Err(QsaScheduleError::StaleWorkspaceZeroLease)
        );
        zero_workspace(&mut scheduler);
        assert_eq!(scheduler.stats().workspace_zeroes_submitted, 2);
        assert_eq!(scheduler.stats().workspace_zeroes_completed, 1);
        assert_eq!(scheduler.stats().workspace_zeroes_failed, 1);
    }

    #[test]
    fn graph_buckets_share_one_fixed_address_arena() {
        let mut scheduler = arena_scheduler();
        zero_workspace(&mut scheduler);
        let bucket_storage = scheduler.graph_buckets().as_ptr();
        let addresses = scheduler.arena().addresses();

        let pack = scheduler.begin_pack(3).expect("pack");
        assert_eq!(pack.active_batch(), 3);
        assert_eq!(pack.graph_batch(), 4);
        assert_eq!(pack.arena().addresses(), addresses);
        let ready = scheduler.complete_pack(pack).expect("ready");
        assert_eq!(ready.graph_batch(), 4);
        let decode = scheduler.begin_decode(ready).expect("decode");
        assert_eq!(decode.arena().addresses(), addresses);
        scheduler.complete_decode(decode).expect("complete");

        let pack = scheduler.begin_pack(1).expect("second pack");
        assert_eq!(pack.graph_batch(), 1);
        assert_eq!(pack.arena().addresses(), addresses);
        assert_eq!(scheduler.graph_buckets().as_ptr(), bucket_storage);
        scheduler.abort_pack(pack).expect("abort second pack");
        assert_eq!(
            scheduler.stats(),
            QsaArenaStats {
                workspace_zeroes_submitted: 1,
                workspace_zeroes_completed: 1,
                workspace_zeroes_failed: 0,
                packs_submitted: 2,
                packs_completed: 1,
                packs_failed: 1,
                packed_batches_discarded: 0,
                decodes_submitted: 1,
                decodes_completed: 1,
                decodes_failed: 0,
                rows_completed: 3,
                arena_stalls: 0,
                peak_graph_batch: 4,
            }
        );
    }

    #[test]
    fn arena_cannot_be_repacked_while_cuda_owns_it() {
        let mut scheduler = arena_scheduler();
        zero_workspace(&mut scheduler);
        let pack = scheduler.begin_pack(2).expect("pack");
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::ArenaBusy(QsaArenaPhase::Packing))
        );
        let ready = scheduler.complete_pack(pack).expect("ready");
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::ArenaBusy(QsaArenaPhase::Ready))
        );
        let decode = scheduler.begin_decode(ready).expect("decode");
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::ArenaBusy(QsaArenaPhase::Decoding))
        );
        scheduler.complete_decode(decode).expect("complete");
        assert_eq!(scheduler.phase(), QsaArenaPhase::Idle);
        assert_eq!(scheduler.stats().arena_stalls, 3);
    }

    #[test]
    fn epochs_and_scheduler_ids_reject_stale_or_foreign_leases() {
        let mut scheduler = arena_scheduler();
        zero_workspace(&mut scheduler);
        let stale_pack = scheduler.begin_pack(1).expect("first pack");
        scheduler.abort_pack(stale_pack).expect("abort");
        let current_pack = scheduler.begin_pack(1).expect("second pack");
        assert_eq!(
            scheduler.complete_pack(stale_pack),
            Err(QsaScheduleError::StalePackLease)
        );
        let ready = scheduler.complete_pack(current_pack).expect("ready");

        let mut foreign = arena_scheduler();
        zero_workspace(&mut foreign);
        let foreign_pack = foreign.begin_pack(1).expect("foreign pack");
        assert_eq!(
            scheduler.complete_pack(foreign_pack),
            Err(QsaScheduleError::StalePackLease)
        );

        let decode = scheduler.begin_decode(ready).expect("decode");
        assert_eq!(
            scheduler.begin_decode(ready),
            Err(QsaScheduleError::StaleReadyLease)
        );
        scheduler.complete_decode(decode).expect("complete");
        assert_eq!(
            scheduler.complete_decode(decode),
            Err(QsaScheduleError::StaleDecodeLease)
        );
    }

    #[test]
    fn failure_and_discard_paths_release_or_reinitialize_safely() {
        let mut scheduler = arena_scheduler();
        zero_workspace(&mut scheduler);

        let failed_pack = scheduler.begin_pack(1).expect("failed pack");
        scheduler.abort_pack(failed_pack).expect("abort pack");
        assert_eq!(scheduler.phase(), QsaArenaPhase::Idle);

        let discarded_pack = scheduler.begin_pack(1).expect("discarded pack");
        let discarded = scheduler
            .complete_pack(discarded_pack)
            .expect("discard ready");
        scheduler.discard_ready(discarded).expect("discard");
        assert_eq!(scheduler.phase(), QsaArenaPhase::Idle);

        let failed_decode_pack = scheduler.begin_pack(1).expect("decode pack");
        let failed_decode_ready = scheduler
            .complete_pack(failed_decode_pack)
            .expect("decode ready");
        let failed_decode = scheduler
            .begin_decode(failed_decode_ready)
            .expect("failed decode");
        scheduler.abort_decode(failed_decode).expect("abort decode");
        assert_eq!(scheduler.phase(), QsaArenaPhase::WorkspaceNeedsZero);
        assert_eq!(
            scheduler.begin_pack(1),
            Err(QsaScheduleError::WorkspaceNotInitialized)
        );
        zero_workspace(&mut scheduler);
        assert!(scheduler.begin_pack(1).is_ok());
        assert_eq!(scheduler.stats().workspace_zeroes_submitted, 2);
        assert_eq!(scheduler.stats().workspace_zeroes_completed, 2);
        assert_eq!(scheduler.stats().packs_failed, 1);
        assert_eq!(scheduler.stats().packed_batches_discarded, 1);
        assert_eq!(scheduler.stats().decodes_failed, 1);
    }

    #[test]
    fn arena_and_bucket_validation_fail_closed() {
        let plan = QsaSparseDecodePlan::qwen38_flash(8).expect("plan");
        let bytes = plan.scratch_layout().expect("layout").total_bytes;
        assert!(matches!(
            QsaArenaScheduler::new(plan, TEST_DEVICE_BASE + 1, bytes, vec![1, 2, 4, 8]),
            Err(QsaScheduleError::InvalidArenaBase { .. })
        ));
        assert!(matches!(
            QsaArenaScheduler::new(plan, TEST_DEVICE_BASE, bytes - 1, vec![1, 2, 4, 8]),
            Err(QsaScheduleError::ArenaTooSmall { .. })
        ));
        assert!(matches!(
            QsaArenaScheduler::new(plan, TEST_DEVICE_BASE, bytes, vec![1, 4, 4, 8]),
            Err(QsaScheduleError::InvalidGraphBuckets)
        ));
        assert!(matches!(
            QsaArenaScheduler::new(plan, TEST_DEVICE_BASE, bytes, vec![1, 2, 4]),
            Err(QsaScheduleError::InvalidGraphBuckets)
        ));
    }
}
