//! Fixed-address plan for the pinned DeepGEMM SM120 GLM indexer score kernel.
//! The CUDA leaf owns only metadata construction and FP8 paged-MQA arithmetic;
//! Rust owns every allocation, stride, page table, and publication boundary.

use std::fmt::{Display, Formatter};

use crate::ffi::{GLM_MQA_GB10_SMS, GLM_MQA_SCHEDULE_WORDS};

pub const GLM_MQA_MAX_BATCH: u32 = 32;
pub const GLM_MQA_HEADS: u32 = 32;
pub const GLM_MQA_HEAD_DIM: u32 = 128;
pub const GLM_MQA_PAGE_SIZE: u32 = 64;
pub const GLM_MQA_LOGITS_STRIDE_ALIGNMENT: u32 = 256;
pub const GLM_MQA_KEY_PAGE_BYTES: u64 = 8_192;
pub const GLM_MQA_SCALE_PAGE_BYTES: u64 = 256;
pub const GLM_MQA_SCHEDULE_BYTES: u64 = GLM_MQA_SCHEDULE_WORDS as u64 * 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmPagedMqaPlan {
    pub batch_size: u32,
    pub num_pages: u32,
    pub max_context_len: u32,
    pub logits_stride: u32,
    pub block_table_stride: u32,
    pub num_sms: u32,
    pub key_page_stride_bytes: u64,
    pub scale_page_stride_bytes: u64,
    pub schedule_bytes: u64,
    pub logits_bytes: u64,
    pub block_table_bytes: u64,
}

impl GlmPagedMqaPlan {
    pub fn gb10(
        batch_size: u32,
        max_context_len: u32,
        num_pages: u32,
    ) -> Result<Self, GlmMqaPlanError> {
        if batch_size == 0 || batch_size > GLM_MQA_MAX_BATCH || max_context_len == 0 {
            return Err(GlmMqaPlanError::InvalidGeometry);
        }
        let capacity = num_pages
            .checked_mul(GLM_MQA_PAGE_SIZE)
            .ok_or(GlmMqaPlanError::Overflow)?;
        if num_pages == 0 || max_context_len > capacity {
            return Err(GlmMqaPlanError::InsufficientPages);
        }
        let logits_stride = align_up(max_context_len, GLM_MQA_LOGITS_STRIDE_ALIGNMENT)?;
        let block_table_stride = max_context_len
            .checked_add(GLM_MQA_PAGE_SIZE - 1)
            .ok_or(GlmMqaPlanError::Overflow)?
            / GLM_MQA_PAGE_SIZE;
        let logits_bytes = u64::from(batch_size)
            .checked_mul(u64::from(logits_stride))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or(GlmMqaPlanError::Overflow)?;
        let block_table_bytes = u64::from(batch_size)
            .checked_mul(u64::from(block_table_stride))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or(GlmMqaPlanError::Overflow)?;
        Ok(Self {
            batch_size,
            num_pages,
            max_context_len,
            logits_stride,
            block_table_stride,
            num_sms: GLM_MQA_GB10_SMS,
            key_page_stride_bytes: GLM_MQA_KEY_PAGE_BYTES,
            scale_page_stride_bytes: GLM_MQA_SCALE_PAGE_BYTES,
            schedule_bytes: GLM_MQA_SCHEDULE_BYTES,
            logits_bytes,
            block_table_bytes,
        })
    }
}

fn align_up(value: u32, alignment: u32) -> Result<u32, GlmMqaPlanError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GlmMqaPlanError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmMqaPlanError {
    InvalidGeometry,
    InsufficientPages,
    Overflow,
}

impl Display for GlmMqaPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGeometry => formatter.write_str("invalid GLM paged-MQA geometry"),
            Self::InsufficientPages => {
                formatter.write_str("GLM paged-MQA pages do not cover the context")
            }
            Self::Overflow => formatter.write_str("GLM paged-MQA plan overflow"),
        }
    }
}

impl std::error::Error for GlmMqaPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_one_32k_plan_matches_deepgemm_contract() {
        let plan = GlmPagedMqaPlan::gb10(1, 32 * 1024, 512).expect("plan");
        assert_eq!(plan.logits_stride, 32 * 1024);
        assert_eq!(plan.block_table_stride, 512);
        assert_eq!(plan.schedule_bytes, 392);
        assert_eq!(plan.logits_bytes, 131_072);
        assert_eq!(plan.block_table_bytes, 2_048);
        assert_eq!(plan.num_sms, 48);
        assert_eq!(plan.key_page_stride_bytes, 8_192);
        assert_eq!(plan.scale_page_stride_bytes, 256);
    }

    #[test]
    fn rounds_logits_to_the_donor_1024_byte_stride() {
        let plan = GlmPagedMqaPlan::gb10(2, 193, 8).expect("plan");
        assert_eq!(plan.logits_stride, 256);
        assert_eq!(plan.logits_bytes, 2_048);
        assert_eq!(plan.block_table_stride, 4);
        assert_eq!(plan.block_table_bytes, 32);
    }

    #[test]
    fn rejects_batch_and_page_drift() {
        assert_eq!(
            GlmPagedMqaPlan::gb10(33, 128, 2),
            Err(GlmMqaPlanError::InvalidGeometry)
        );
        assert_eq!(
            GlmPagedMqaPlan::gb10(1, 129, 2),
            Err(GlmMqaPlanError::InsufficientPages)
        );
    }
}
