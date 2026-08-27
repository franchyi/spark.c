//! Fixed-address QSA sparse-decode scratch owned by the Rust scheduler.
//!
//! CUDA donors receive raw pointers into this layout. They do not allocate,
//! resize, page, or decide residency. On DGX Spark a single device allocation
//! is backed by unified system memory, but remains resident and address-stable
//! for CUDA graph replay.

use std::fmt;

pub const QWEN_QSA_TOPK: usize = 2051;
pub const QWEN_QSA_PAGE_TOKENS: usize = 64;
pub const QWEN_QSA_PAGES_PER_ROW: usize = 33;
pub const QWEN_QSA_PACKED_ROW_TOKENS: usize = 2112;
pub const QWEN_QSA_KV_HEADS: usize = 2;
pub const QWEN_QSA_HEAD_DIM: usize = 256;
pub const QWEN_QSA_QUERY_HEADS: usize = 24;
pub const TRTLLM_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;

const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;
const SCRATCH_ALIGNMENT: usize = 256;

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
    pub trtllm_workspace_offset: usize,
    pub trtllm_workspace_bytes: usize,
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
        let trtllm_workspace_offset = align_up(
            attention_output_offset
                .checked_add(attention_output_bytes)
                .ok_or(QsaPlanError::SizeOverflow)?,
            SCRATCH_ALIGNMENT,
        )?;
        let total_bytes = trtllm_workspace_offset
            .checked_add(TRTLLM_WORKSPACE_BYTES)
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
            trtllm_workspace_offset,
            trtllm_workspace_bytes: TRTLLM_WORKSPACE_BYTES,
            total_bytes,
            alignment: SCRATCH_ALIGNMENT,
        })
    }

    /// Fill the immutable row-major TRT-LLM block table once, before graph
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

    #[test]
    fn qwen_geometry_matches_sglang_trtllm_path() {
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
                layout.trtllm_workspace_offset,
                layout.trtllm_workspace_bytes,
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
}
