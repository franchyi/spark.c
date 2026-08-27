use std::fmt::{Display, Formatter};

const WINDOW_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleChunk {
    pub ordinal: u64,
    pub token_start: u32,
    pub tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleWindowLayout {
    pub host_slab_base: u64,
    pub device_slab_base: u64,
    pub slab_bytes: u64,
    pub fragment_device_base: u64,
    pub output_device_base: u64,
    pub max_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlePipelineConfig {
    pub page_bytes: u64,
    pub row_bytes: u32,
    pub rows_per_token: u32,
    pub max_chunk_tokens: u32,
    pub windows: [PleWindowLayout; WINDOW_COUNT],
}

impl PlePipelineConfig {
    pub fn qwen_flash(windows: [PleWindowLayout; WINDOW_COUNT]) -> Self {
        Self {
            page_bytes: 4096,
            row_bytes: 160,
            rows_per_token: 16,
            max_chunk_tokens: 16,
            windows,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PleWindowId {
    A,
    B,
}

impl PleWindowId {
    fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    fn for_ordinal(ordinal: u64) -> Self {
        if ordinal & 1 == 0 { Self::A } else { Self::B }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleFillLease {
    lease_id: u64,
    window: PleWindowId,
    chunk: PleChunk,
    layout: PleWindowLayout,
}

impl PleFillLease {
    pub fn window(&self) -> PleWindowId {
        self.window
    }

    pub fn chunk(&self) -> PleChunk {
        self.chunk
    }

    pub fn layout(&self) -> PleWindowLayout {
        self.layout
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleComputeLease {
    lease_id: u64,
    window: PleWindowId,
    chunk: PleChunk,
    layout: PleWindowLayout,
    rows: u32,
}

impl PleComputeLease {
    pub fn window(&self) -> PleWindowId {
        self.window
    }

    pub fn chunk(&self) -> PleChunk {
        self.chunk
    }

    pub fn layout(&self) -> PleWindowLayout {
        self.layout
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlePipelineStats {
    pub fills_submitted: u64,
    pub fills_completed: u64,
    pub fills_failed: u64,
    pub computes_completed: u64,
    pub overlapped_fills: u64,
    pub fill_stalls: u64,
    pub compute_stalls: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WindowState {
    Idle,
    Filling {
        lease_id: u64,
        chunk: PleChunk,
    },
    Ready {
        lease_id: u64,
        chunk: PleChunk,
        rows: u32,
    },
    Computing {
        lease_id: u64,
        chunk: PleChunk,
        rows: u32,
    },
}

/// Allocation-free two-window PLE pipeline.
///
/// Chunk `n` always uses window `n % 2`, so CUDA graph-visible slab,
/// descriptor, and output addresses never change. Rust alone advances the
/// fill/ready/compute state. A window cannot be refilled until its compute
/// lease is completed, while the opposite window may be filled concurrently.
pub struct PlePipelineScheduler {
    config: PlePipelineConfig,
    states: [WindowState; WINDOW_COUNT],
    generations: [u64; WINDOW_COUNT],
    next_fill_ordinal: u64,
    next_fill_token: u32,
    next_compute_ordinal: u64,
    stats: PlePipelineStats,
}

impl PlePipelineScheduler {
    pub fn new(config: PlePipelineConfig) -> Result<Self, PlePipelineError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            states: [WindowState::Idle; WINDOW_COUNT],
            generations: [0; WINDOW_COUNT],
            next_fill_ordinal: 0,
            next_fill_token: 0,
            next_compute_ordinal: 0,
            stats: PlePipelineStats::default(),
        })
    }

    pub fn reserve_fill(&mut self, chunk: PleChunk) -> Result<PleFillLease, PlePipelineError> {
        if self
            .states
            .iter()
            .any(|state| matches!(state, WindowState::Filling { .. }))
        {
            self.stats.fill_stalls = self.stats.fill_stalls.saturating_add(1);
            return Err(PlePipelineError::FillInFlight);
        }
        if chunk.ordinal != self.next_fill_ordinal || chunk.token_start != self.next_fill_token {
            return Err(PlePipelineError::OutOfOrderFill {
                expected_ordinal: self.next_fill_ordinal,
                expected_token: self.next_fill_token,
            });
        }
        if chunk.tokens == 0 || chunk.tokens > self.config.max_chunk_tokens {
            return Err(PlePipelineError::ChunkTokens(chunk.tokens));
        }
        let rows = chunk
            .tokens
            .checked_mul(self.config.rows_per_token)
            .ok_or(PlePipelineError::IntegerOverflow)?;
        let window = PleWindowId::for_ordinal(chunk.ordinal);
        let index = window.index();
        if !matches!(self.states[index], WindowState::Idle) {
            self.stats.fill_stalls = self.stats.fill_stalls.saturating_add(1);
            return Err(PlePipelineError::WindowBusy(window));
        }
        if rows > self.config.windows[index].max_rows {
            return Err(PlePipelineError::TooManyRows {
                rows,
                capacity: self.config.windows[index].max_rows,
            });
        }
        let next_generation = self.generations[index]
            .checked_add(1)
            .ok_or(PlePipelineError::IntegerOverflow)?;
        let next_fill_ordinal = self
            .next_fill_ordinal
            .checked_add(1)
            .ok_or(PlePipelineError::IntegerOverflow)?;
        let next_fill_token = self
            .next_fill_token
            .checked_add(chunk.tokens)
            .ok_or(PlePipelineError::IntegerOverflow)?;
        self.generations[index] = next_generation;
        self.states[index] = WindowState::Filling {
            lease_id: next_generation,
            chunk,
        };
        self.next_fill_ordinal = next_fill_ordinal;
        self.next_fill_token = next_fill_token;
        self.stats.fills_submitted = self.stats.fills_submitted.saturating_add(1);
        if self
            .states
            .iter()
            .any(|state| matches!(state, WindowState::Computing { .. }))
        {
            self.stats.overlapped_fills = self.stats.overlapped_fills.saturating_add(1);
        }
        Ok(PleFillLease {
            lease_id: next_generation,
            window,
            chunk,
            layout: self.config.windows[index],
        })
    }

    pub fn complete_fill(
        &mut self,
        lease: PleFillLease,
        fragment_rows: u32,
    ) -> Result<(), PlePipelineError> {
        let expected_rows = lease
            .chunk
            .tokens
            .checked_mul(self.config.rows_per_token)
            .ok_or(PlePipelineError::IntegerOverflow)?;
        if fragment_rows != expected_rows {
            return Err(PlePipelineError::FragmentRows {
                expected: expected_rows,
                actual: fragment_rows,
            });
        }
        let index = lease.window.index();
        match self.states[index] {
            WindowState::Filling { lease_id, chunk }
                if lease_id == lease.lease_id && chunk == lease.chunk =>
            {
                self.states[index] = WindowState::Ready {
                    lease_id,
                    chunk,
                    rows: fragment_rows,
                };
                self.stats.fills_completed = self.stats.fills_completed.saturating_add(1);
                Ok(())
            }
            _ => Err(PlePipelineError::StaleFillLease),
        }
    }

    pub fn abort_fill(&mut self, lease: PleFillLease) -> Result<(), PlePipelineError> {
        let index = lease.window.index();
        match self.states[index] {
            WindowState::Filling { lease_id, chunk }
                if lease_id == lease.lease_id && chunk == lease.chunk =>
            {
                self.states[index] = WindowState::Idle;
                self.next_fill_ordinal = chunk.ordinal;
                self.next_fill_token = chunk.token_start;
                self.stats.fills_failed = self.stats.fills_failed.saturating_add(1);
                Ok(())
            }
            _ => Err(PlePipelineError::StaleFillLease),
        }
    }

    pub fn begin_compute(&mut self) -> Result<PleComputeLease, PlePipelineError> {
        if self
            .states
            .iter()
            .any(|state| matches!(state, WindowState::Computing { .. }))
        {
            self.stats.compute_stalls = self.stats.compute_stalls.saturating_add(1);
            return Err(PlePipelineError::ComputeInFlight);
        }
        let window = PleWindowId::for_ordinal(self.next_compute_ordinal);
        let index = window.index();
        match self.states[index] {
            WindowState::Ready {
                lease_id,
                chunk,
                rows,
            } if chunk.ordinal == self.next_compute_ordinal => {
                self.states[index] = WindowState::Computing {
                    lease_id,
                    chunk,
                    rows,
                };
                Ok(PleComputeLease {
                    lease_id,
                    window,
                    chunk,
                    layout: self.config.windows[index],
                    rows,
                })
            }
            _ => {
                self.stats.compute_stalls = self.stats.compute_stalls.saturating_add(1);
                Err(PlePipelineError::NextChunkNotReady(
                    self.next_compute_ordinal,
                ))
            }
        }
    }

    pub fn complete_compute(&mut self, lease: PleComputeLease) -> Result<(), PlePipelineError> {
        let index = lease.window.index();
        match self.states[index] {
            WindowState::Computing {
                lease_id,
                chunk,
                rows,
            } if lease_id == lease.lease_id && chunk == lease.chunk && rows == lease.rows => {
                let next_compute_ordinal = self
                    .next_compute_ordinal
                    .checked_add(1)
                    .ok_or(PlePipelineError::IntegerOverflow)?;
                self.states[index] = WindowState::Idle;
                self.next_compute_ordinal = next_compute_ordinal;
                self.stats.computes_completed = self.stats.computes_completed.saturating_add(1);
                Ok(())
            }
            _ => Err(PlePipelineError::StaleComputeLease),
        }
    }

    pub fn retry_compute(&mut self, lease: PleComputeLease) -> Result<(), PlePipelineError> {
        let index = lease.window.index();
        match self.states[index] {
            WindowState::Computing {
                lease_id,
                chunk,
                rows,
            } if lease_id == lease.lease_id && chunk == lease.chunk && rows == lease.rows => {
                self.states[index] = WindowState::Ready {
                    lease_id,
                    chunk,
                    rows,
                };
                Ok(())
            }
            _ => Err(PlePipelineError::StaleComputeLease),
        }
    }

    pub fn stats(&self) -> PlePipelineStats {
        self.stats
    }
}

fn validate_config(config: &PlePipelineConfig) -> Result<(), PlePipelineError> {
    if config.page_bytes == 0
        || !config.page_bytes.is_power_of_two()
        || config.row_bytes == 0
        || config.rows_per_token == 0
        || config.max_chunk_tokens == 0
    {
        return Err(PlePipelineError::InvalidConfig);
    }
    let required_rows = config
        .rows_per_token
        .checked_mul(config.max_chunk_tokens)
        .ok_or(PlePipelineError::IntegerOverflow)?;
    for window in config.windows {
        if window.host_slab_base == 0
            || window.device_slab_base == 0
            || window.fragment_device_base == 0
            || window.output_device_base == 0
            || window.slab_bytes == 0
            || window.slab_bytes % config.page_bytes != 0
            || window.host_slab_base % config.page_bytes != 0
            || window.device_slab_base % config.page_bytes != 0
            || window.max_rows < required_rows
        {
            return Err(PlePipelineError::InvalidConfig);
        }
    }
    if ranges_overlap(
        config.windows[0].host_slab_base,
        config.windows[0].slab_bytes,
        config.windows[1].host_slab_base,
        config.windows[1].slab_bytes,
    )? || ranges_overlap(
        config.windows[0].device_slab_base,
        config.windows[0].slab_bytes,
        config.windows[1].device_slab_base,
        config.windows[1].slab_bytes,
    )? || ranges_overlap(
        config.windows[0].fragment_device_base,
        u64::from(config.windows[0].max_rows)
            .checked_mul(std::mem::size_of::<crate::ffi::PleRowFragment>() as u64)
            .ok_or(PlePipelineError::IntegerOverflow)?,
        config.windows[1].fragment_device_base,
        u64::from(config.windows[1].max_rows)
            .checked_mul(std::mem::size_of::<crate::ffi::PleRowFragment>() as u64)
            .ok_or(PlePipelineError::IntegerOverflow)?,
    )? || ranges_overlap(
        config.windows[0].output_device_base,
        u64::from(config.windows[0].max_rows)
            .checked_mul(u64::from(config.row_bytes))
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(PlePipelineError::IntegerOverflow)?,
        config.windows[1].output_device_base,
        u64::from(config.windows[1].max_rows)
            .checked_mul(u64::from(config.row_bytes))
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(PlePipelineError::IntegerOverflow)?,
    )? {
        return Err(PlePipelineError::OverlappingWindows);
    }
    Ok(())
}

fn ranges_overlap(
    first: u64,
    first_bytes: u64,
    second: u64,
    second_bytes: u64,
) -> Result<bool, PlePipelineError> {
    let first_end = first
        .checked_add(first_bytes)
        .ok_or(PlePipelineError::IntegerOverflow)?;
    let second_end = second
        .checked_add(second_bytes)
        .ok_or(PlePipelineError::IntegerOverflow)?;
    Ok(first < second_end && second < first_end)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlePipelineError {
    InvalidConfig,
    OverlappingWindows,
    IntegerOverflow,
    FillInFlight,
    WindowBusy(PleWindowId),
    OutOfOrderFill {
        expected_ordinal: u64,
        expected_token: u32,
    },
    ChunkTokens(u32),
    TooManyRows {
        rows: u32,
        capacity: u32,
    },
    FragmentRows {
        expected: u32,
        actual: u32,
    },
    StaleFillLease,
    ComputeInFlight,
    NextChunkNotReady(u64),
    StaleComputeLease,
}

impl Display for PlePipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid PLE pipeline configuration"),
            Self::OverlappingWindows => formatter.write_str("PLE windows overlap"),
            Self::IntegerOverflow => formatter.write_str("PLE pipeline integer overflow"),
            Self::FillInFlight => formatter.write_str("a PLE fill is already in flight"),
            Self::WindowBusy(window) => write!(formatter, "PLE window {window:?} is busy"),
            Self::OutOfOrderFill {
                expected_ordinal,
                expected_token,
            } => write!(
                formatter,
                "PLE fill is out of order; expected chunk {expected_ordinal} at token {expected_token}"
            ),
            Self::ChunkTokens(tokens) => write!(formatter, "invalid PLE chunk size {tokens}"),
            Self::TooManyRows { rows, capacity } => write!(
                formatter,
                "PLE chunk needs {rows} rows but the fixed window holds {capacity}"
            ),
            Self::FragmentRows { expected, actual } => write!(
                formatter,
                "PLE fill produced {actual} row descriptors; expected {expected}"
            ),
            Self::StaleFillLease => formatter.write_str("stale PLE fill lease"),
            Self::ComputeInFlight => formatter.write_str("PLE compute is already in flight"),
            Self::NextChunkNotReady(ordinal) => {
                write!(formatter, "PLE chunk {ordinal} is not ready")
            }
            Self::StaleComputeLease => formatter.write_str("stale PLE compute lease"),
        }
    }
}

impl std::error::Error for PlePipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(base: u64) -> PleWindowLayout {
        PleWindowLayout {
            host_slab_base: base,
            device_slab_base: base + 0x1000_0000,
            slab_bytes: 4 * 1024 * 1024,
            fragment_device_base: base + 0x2000_0000,
            output_device_base: base + 0x3000_0000,
            max_rows: 256,
        }
    }

    fn scheduler() -> PlePipelineScheduler {
        PlePipelineScheduler::new(PlePipelineConfig::qwen_flash([
            layout(0x1000_0000),
            layout(0x2000_0000),
        ]))
        .expect("pipeline")
    }

    fn chunk(ordinal: u64, token_start: u32) -> PleChunk {
        PleChunk {
            ordinal,
            token_start,
            tokens: 16,
        }
    }

    #[test]
    fn ping_pongs_fixed_addresses_and_counts_real_overlap() {
        let mut pipeline = scheduler();
        let fill_a = pipeline.reserve_fill(chunk(0, 0)).expect("fill A");
        assert_eq!(fill_a.window(), PleWindowId::A);
        pipeline.complete_fill(fill_a, 256).expect("ready A");
        let compute_a = pipeline.begin_compute().expect("compute A");
        assert_eq!(compute_a.layout(), layout(0x1000_0000));

        let fill_b = pipeline.reserve_fill(chunk(1, 16)).expect("fill B");
        assert_eq!(fill_b.window(), PleWindowId::B);
        pipeline.complete_fill(fill_b, 256).expect("ready B");
        pipeline.complete_compute(compute_a).expect("release A");
        let compute_b = pipeline.begin_compute().expect("compute B");
        assert_eq!(compute_b.layout(), layout(0x2000_0000));
        pipeline.complete_compute(compute_b).expect("release B");

        assert_eq!(
            pipeline.stats(),
            PlePipelineStats {
                fills_submitted: 2,
                fills_completed: 2,
                fills_failed: 0,
                computes_completed: 2,
                overlapped_fills: 1,
                fill_stalls: 0,
                compute_stalls: 0,
            }
        );
    }

    #[test]
    fn never_recycles_a_window_before_compute_finishes() {
        let mut pipeline = scheduler();
        let fill_a = pipeline.reserve_fill(chunk(0, 0)).expect("fill A");
        pipeline.complete_fill(fill_a, 256).expect("ready A");
        let compute_a = pipeline.begin_compute().expect("compute A");
        let fill_b = pipeline.reserve_fill(chunk(1, 16)).expect("fill B");
        pipeline.complete_fill(fill_b, 256).expect("ready B");
        assert_eq!(
            pipeline.reserve_fill(chunk(2, 32)),
            Err(PlePipelineError::WindowBusy(PleWindowId::A))
        );
        pipeline.complete_compute(compute_a).expect("release A");
        assert!(pipeline.reserve_fill(chunk(2, 32)).is_ok());
    }

    #[test]
    fn failed_io_releases_the_same_window_for_retry() {
        let mut pipeline = scheduler();
        let failed = pipeline.reserve_fill(chunk(0, 0)).expect("fill");
        pipeline.abort_fill(failed).expect("abort");
        let retry = pipeline.reserve_fill(chunk(0, 0)).expect("retry");
        assert_eq!(retry.window(), PleWindowId::A);
        assert_ne!(retry.lease_id, failed.lease_id);
        assert_eq!(pipeline.stats().fills_failed, 1);
    }

    #[test]
    fn rejects_out_of_order_and_incomplete_descriptor_sets() {
        let mut pipeline = scheduler();
        assert_eq!(
            pipeline.reserve_fill(chunk(1, 16)),
            Err(PlePipelineError::OutOfOrderFill {
                expected_ordinal: 0,
                expected_token: 0,
            })
        );
        let fill = pipeline.reserve_fill(chunk(0, 0)).expect("fill");
        assert_eq!(
            pipeline.complete_fill(fill, 255),
            Err(PlePipelineError::FragmentRows {
                expected: 256,
                actual: 255,
            })
        );
        pipeline.complete_fill(fill, 256).expect("complete");
    }

    #[test]
    fn compute_failure_can_retry_without_reloading_or_moving() {
        let mut pipeline = scheduler();
        let fill = pipeline.reserve_fill(chunk(0, 0)).expect("fill");
        pipeline.complete_fill(fill, 256).expect("ready");
        let first = pipeline.begin_compute().expect("first compute");
        pipeline.retry_compute(first).expect("retry state");
        let second = pipeline.begin_compute().expect("second compute");
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_overlapping_or_undersized_windows() {
        let first = layout(0x1000_0000);
        let mut overlapping = layout(0x1010_0000);
        assert!(matches!(
            PlePipelineScheduler::new(PlePipelineConfig::qwen_flash([first, overlapping])),
            Err(PlePipelineError::OverlappingWindows)
        ));
        overlapping = layout(0x2000_0000);
        overlapping.max_rows = 255;
        assert!(matches!(
            PlePipelineScheduler::new(PlePipelineConfig::qwen_flash([first, overlapping])),
            Err(PlePipelineError::InvalidConfig)
        ));
    }
}
