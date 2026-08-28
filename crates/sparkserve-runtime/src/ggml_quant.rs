//! Rust-owned shapes, scratch, and paging budgets for the borrowed GGML CUDA
//! MMVQ arithmetic. No donor allocator, graph, cache, or scheduler crosses the
//! ABI. Quant tags remain the exact GGUF v3 values.

use std::fmt::{Display, Formatter};

use crate::gguf::GgmlTensorType;

pub const Q8_1_BLOCK_ELEMENTS: u64 = 32;
pub const Q8_1_BLOCK_BYTES: u64 = 36;
pub const MMVQ_ROW_ALIGNMENT: u64 = 512;
pub const MMVQ_MAX_VECTORS: u64 = 8;

pub fn mmvq_geometry(tensor_type: GgmlTensorType) -> Option<(u64, u64)> {
    match tensor_type {
        GgmlTensorType::Q8_0 => Some((32, 34)),
        GgmlTensorType::Q2K => Some((256, 84)),
        GgmlTensorType::Q3K => Some((256, 110)),
        GgmlTensorType::Q4K => Some((256, 144)),
        GgmlTensorType::Q5K => Some((256, 176)),
        GgmlTensorType::Q6K => Some((256, 210)),
        GgmlTensorType::Iq3Xxs => Some((256, 98)),
        GgmlTensorType::Iq3S => Some((256, 110)),
        GgmlTensorType::Iq2S => Some((256, 82)),
        GgmlTensorType::Iq4Xs => Some((256, 136)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantDenseSpec {
    pub quant_type: GgmlTensorType,
    pub vectors: u64,
    pub rows: u64,
    pub k: u64,
}

impl QuantDenseSpec {
    pub fn plan(self) -> Result<QuantMemoryPlan, QuantPlanError> {
        let (block_elements, block_bytes) =
            validate_common(self.quant_type, self.vectors, self.rows, self.k)?;
        Ok(QuantMemoryPlan {
            q8_scratch_bytes: q8_scratch_bytes(self.vectors, self.k)?,
            output_bytes: product(&[self.vectors, self.rows, 4])?,
            max_selected_weight_bytes: product(&[self.rows, self.k / block_elements, block_bytes])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantRoutedSpec {
    pub quant_type: GgmlTensorType,
    pub tokens: u64,
    pub top_k: u64,
    /// Number of fixed cache slots addressable by the physical expert IDs.
    pub weight_slots: u64,
    pub rows: u64,
    pub k: u64,
    /// Byte distance between fixed cache slots. This may exceed one encoded
    /// expert slice, but must be divisible by the selected quant block size.
    pub weight_slot_stride_bytes: u64,
}

impl QuantRoutedSpec {
    pub fn plan(self) -> Result<QuantMemoryPlan, QuantPlanError> {
        let (block_elements, block_bytes) =
            validate_common(self.quant_type, self.tokens, self.rows, self.k)?;
        if self.weight_slots == 0 || self.weight_slots > i32::MAX as u64 {
            return Err(QuantPlanError::InvalidWeightSlots);
        }
        if self.top_k == 0 || self.top_k > self.weight_slots || self.top_k > i32::MAX as u64 {
            return Err(QuantPlanError::InvalidTopK);
        }
        let slice_bytes = product(&[self.rows, self.k / block_elements, block_bytes])?;
        if self.weight_slot_stride_bytes < slice_bytes
            || !self.weight_slot_stride_bytes.is_multiple_of(block_bytes)
            || self.weight_slot_stride_bytes / block_bytes > i32::MAX as u64
        {
            return Err(QuantPlanError::InvalidWeightStride);
        }
        let selected_slots = self
            .tokens
            .checked_mul(self.top_k)
            .ok_or(QuantPlanError::IntegerOverflow)?
            .min(self.weight_slots);
        Ok(QuantMemoryPlan {
            q8_scratch_bytes: q8_scratch_bytes(self.tokens, self.k)?,
            output_bytes: product(&[self.tokens, self.top_k, self.rows, 4])?,
            // Conservative cold-cache source-read budget. The fixed cache
            // allocation itself is accounted by the GGUF paging catalog.
            max_selected_weight_bytes: product(&[selected_slots, slice_bytes])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantMemoryPlan {
    /// Fixed-address canonical block_q8_1 buffer captured by CUDA graphs.
    pub q8_scratch_bytes: u64,
    /// Fixed output arena for this graph bucket.
    pub output_bytes: u64,
    /// Worst-case encoded GGUF bytes touched on a cold cache miss.
    pub max_selected_weight_bytes: u64,
}

impl QuantMemoryPlan {
    pub fn fixed_workspace_bytes(self) -> Result<u64, QuantPlanError> {
        self.q8_scratch_bytes
            .checked_add(self.output_bytes)
            .ok_or(QuantPlanError::IntegerOverflow)
    }
}

pub fn q8_scratch_bytes(vectors: u64, k: u64) -> Result<u64, QuantPlanError> {
    if vectors == 0 || vectors > MMVQ_MAX_VECTORS {
        return Err(QuantPlanError::InvalidVectorCount);
    }
    if k == 0 || !k.is_multiple_of(Q8_1_BLOCK_ELEMENTS) || k > i32::MAX as u64 {
        return Err(QuantPlanError::InvalidK);
    }
    let padded_k = align_up(k, MMVQ_ROW_ALIGNMENT)?;
    if padded_k > i32::MAX as u64 {
        return Err(QuantPlanError::InvalidK);
    }
    product(&[vectors, padded_k / Q8_1_BLOCK_ELEMENTS, Q8_1_BLOCK_BYTES])
}

fn validate_common(
    quant_type: GgmlTensorType,
    vectors: u64,
    rows: u64,
    k: u64,
) -> Result<(u64, u64), QuantPlanError> {
    q8_scratch_bytes(vectors, k)?;
    if rows == 0 || rows > i32::MAX as u64 {
        return Err(QuantPlanError::InvalidRows);
    }
    let geometry = mmvq_geometry(quant_type).ok_or(QuantPlanError::UnsupportedType)?;
    if !k.is_multiple_of(geometry.0) {
        return Err(QuantPlanError::InvalidK);
    }
    Ok(geometry)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, QuantPlanError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(QuantPlanError::IntegerOverflow)
}

fn product(values: &[u64]) -> Result<u64, QuantPlanError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(QuantPlanError::IntegerOverflow)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantPlanError {
    UnsupportedType,
    InvalidVectorCount,
    InvalidRows,
    InvalidK,
    InvalidWeightSlots,
    InvalidWeightStride,
    InvalidTopK,
    IntegerOverflow,
}

impl Display for QuantPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedType => formatter.write_str("GGUF type has no pinned MMVQ adapter"),
            Self::InvalidVectorCount => write!(formatter, "MMVQ vectors must be in [1, 8]"),
            Self::InvalidRows => write!(formatter, "rows must fit a positive 32-bit dimension"),
            Self::InvalidK => write!(
                formatter,
                "K must fit i32 and satisfy the selected quant block geometry"
            ),
            Self::InvalidWeightSlots => {
                write!(
                    formatter,
                    "weight slots must fit a positive 32-bit dimension"
                )
            }
            Self::InvalidWeightStride => formatter
                .write_str("weight slot stride must cover one slice and be quant-block aligned"),
            Self::InvalidTopK => write!(formatter, "top_k must be in [1, weight slots]"),
            Self::IntegerOverflow => write!(formatter, "GGML quant memory plan overflows u64"),
        }
    }
}

impl std::error::Error for QuantPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_scratch_matches_pinned_ggml_geometry() {
        assert_eq!(q8_scratch_bytes(1, 2560), Ok(2_880));
        assert_eq!(q8_scratch_bytes(8, 2560), Ok(23_040));
    }

    #[test]
    fn dense_plan_keeps_encoded_iq3_weights() {
        let plan = QuantDenseSpec {
            quant_type: GgmlTensorType::Iq3Xxs,
            vectors: 1,
            rows: 4096,
            k: 2560,
        }
        .plan()
        .expect("valid dense plan");
        assert_eq!(plan.max_selected_weight_bytes, 4_014_080);
        assert_eq!(plan.output_bytes, 16_384);
        assert_eq!(plan.fixed_workspace_bytes(), Ok(19_264));
    }

    #[test]
    fn routed_plan_accepts_a_mixed_model_fixed_slot_stride() {
        let contiguous = 256 * 4 * 82;
        let plan = QuantRoutedSpec {
            quant_type: GgmlTensorType::Iq2S,
            tokens: 8,
            top_k: 8,
            weight_slots: 32,
            rows: 256,
            k: 1024,
            weight_slot_stride_bytes: contiguous * 2,
        }
        .plan()
        .expect("valid routed plan");
        assert_eq!(plan.max_selected_weight_bytes, 32 * contiguous);
        assert_eq!(plan.output_bytes, 65_536);
    }

    #[test]
    fn rejects_shapes_that_the_donor_would_abort_on() {
        assert_eq!(
            q8_scratch_bytes(9, 2560),
            Err(QuantPlanError::InvalidVectorCount)
        );
        assert_eq!(q8_scratch_bytes(1, 2559), Err(QuantPlanError::InvalidK));
        assert_eq!(
            QuantRoutedSpec {
                quant_type: GgmlTensorType::Iq2S,
                tokens: 1,
                top_k: 9,
                weight_slots: 8,
                rows: 1,
                k: 256,
                weight_slot_stride_bytes: 82,
            }
            .plan(),
            Err(QuantPlanError::InvalidTopK)
        );
        assert_eq!(
            QuantDenseSpec {
                quant_type: GgmlTensorType::F32,
                vectors: 1,
                rows: 1,
                k: 256,
            }
            .plan(),
            Err(QuantPlanError::UnsupportedType)
        );
    }
}
