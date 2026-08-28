//! Fixed-address state and launch geometry for GLM-5.3-Flash KDA. The native
//! kernel preserves the pinned llama.cpp fused recurrence; this module keeps
//! its allocation and lifetime policy in Rust.

use std::fmt::{Display, Formatter};

pub const GLM_KDA_HEAD_DIM: u64 = 128;
pub const GLM_KDA_HEADS: u64 = 64;
pub const GLM_KDA_LAYERS: u64 = 34;
pub const GLM_KDA_CONV_BRANCHES: u64 = 3;
pub const GLM_KDA_CONV_HISTORY: u64 = 3;
pub const FP32_BYTES: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmKdaSpec {
    pub sequences: u64,
    pub tokens: u64,
    pub heads: u64,
    pub head_dim: u64,
}

impl GlmKdaSpec {
    pub fn glm53(sequences: u64, tokens: u64) -> Self {
        Self {
            sequences,
            tokens,
            heads: GLM_KDA_HEADS,
            head_dim: GLM_KDA_HEAD_DIM,
        }
    }

    pub fn plan(self) -> Result<GlmKdaMemoryPlan, GlmKdaPlanError> {
        if self.sequences == 0 || self.sequences > 65_535 {
            return Err(GlmKdaPlanError::InvalidSequences);
        }
        if self.tokens == 0 {
            return Err(GlmKdaPlanError::InvalidTokens);
        }
        if self.heads == 0 || self.heads > u32::MAX as u64 {
            return Err(GlmKdaPlanError::InvalidHeads);
        }
        if self.head_dim != GLM_KDA_HEAD_DIM {
            return Err(GlmKdaPlanError::InvalidHeadDim);
        }
        let vectors = product(&[self.sequences, self.tokens, self.heads, self.head_dim])?;
        let state_elements = product(&[self.sequences, self.heads, self.head_dim, self.head_dim])?;
        Ok(GlmKdaMemoryPlan {
            qkv_log_decay_bytes: product(&[4, vectors, FP32_BYTES])?,
            beta_bytes: product(&[self.sequences, self.tokens, self.heads, FP32_BYTES])?,
            output_bytes: product(&[vectors, FP32_BYTES])?,
            state_bytes: product(&[state_elements, FP32_BYTES])?,
            conv_state_bytes: product(&[
                self.sequences,
                GLM_KDA_CONV_BRANCHES,
                self.heads,
                self.head_dim,
                GLM_KDA_CONV_HISTORY,
                FP32_BYTES,
            ])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmKdaMemoryPlan {
    /// Four FP32 activation vectors: normalized Q, normalized K, V, and log decay.
    pub qkv_log_decay_bytes: u64,
    pub beta_bytes: u64,
    pub output_bytes: u64,
    /// One caller-owned recurrent-state slab. Input/output may alias.
    pub state_bytes: u64,
    /// Three width-4 Q/K/V convolution histories. Input/output may alias.
    pub conv_state_bytes: u64,
}

impl GlmKdaMemoryPlan {
    pub fn activation_bytes(self) -> Result<u64, GlmKdaPlanError> {
        self.qkv_log_decay_bytes
            .checked_add(self.beta_bytes)
            .and_then(|bytes| bytes.checked_add(self.output_bytes))
            .ok_or(GlmKdaPlanError::IntegerOverflow)
    }

    pub fn total_state_bytes(self) -> Result<u64, GlmKdaPlanError> {
        self.state_bytes
            .checked_add(self.conv_state_bytes)
            .ok_or(GlmKdaPlanError::IntegerOverflow)
    }
}

pub fn glm53_all_layer_state_bytes(sequences: u64) -> Result<u64, GlmKdaPlanError> {
    GlmKdaSpec::glm53(sequences, 1)
        .plan()?
        .state_bytes
        .checked_mul(GLM_KDA_LAYERS)
        .ok_or(GlmKdaPlanError::IntegerOverflow)
}

pub fn glm53_all_layer_conv_state_bytes(sequences: u64) -> Result<u64, GlmKdaPlanError> {
    GlmKdaSpec::glm53(sequences, 1)
        .plan()?
        .conv_state_bytes
        .checked_mul(GLM_KDA_LAYERS)
        .ok_or(GlmKdaPlanError::IntegerOverflow)
}

pub fn glm53_all_layer_total_state_bytes(sequences: u64) -> Result<u64, GlmKdaPlanError> {
    GlmKdaSpec::glm53(sequences, 1)
        .plan()?
        .total_state_bytes()?
        .checked_mul(GLM_KDA_LAYERS)
        .ok_or(GlmKdaPlanError::IntegerOverflow)
}

fn product(values: &[u64]) -> Result<u64, GlmKdaPlanError> {
    values.iter().try_fold(1_u64, |accumulator, value| {
        accumulator
            .checked_mul(*value)
            .ok_or(GlmKdaPlanError::IntegerOverflow)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmKdaPlanError {
    InvalidSequences,
    InvalidTokens,
    InvalidHeads,
    InvalidHeadDim,
    IntegerOverflow,
}

impl Display for GlmKdaPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSequences => formatter.write_str("KDA sequences must be in [1, 65535]"),
            Self::InvalidTokens => formatter.write_str("KDA tokens must be positive"),
            Self::InvalidHeads => formatter.write_str("KDA heads must fit a positive u32"),
            Self::InvalidHeadDim => formatter.write_str("GLM KDA requires head_dim=128"),
            Self::IntegerOverflow => formatter.write_str("GLM KDA memory geometry overflows u64"),
        }
    }
}

impl std::error::Error for GlmKdaPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_glm53_decode_state_is_four_mib_per_layer() {
        let plan = GlmKdaSpec::glm53(1, 1).plan().expect("valid GLM KDA");
        assert_eq!(plan.state_bytes, 4 * 1024 * 1024);
        assert_eq!(plan.qkv_log_decay_bytes, 131_072);
        assert_eq!(plan.beta_bytes, 256);
        assert_eq!(plan.output_bytes, 32_768);
        assert_eq!(plan.activation_bytes(), Ok(164_096));
        assert_eq!(plan.conv_state_bytes, 288 * 1024);
        assert_eq!(plan.total_state_bytes(), Ok(4_489_216));
        assert_eq!(glm53_all_layer_state_bytes(1), Ok(136 * 1024 * 1024));
        assert_eq!(glm53_all_layer_conv_state_bytes(1), Ok(10_027_008));
        assert_eq!(glm53_all_layer_total_state_bytes(1), Ok(152_633_344));
    }

    #[test]
    fn prefill_reuses_the_same_fixed_state_slab() {
        let decode = GlmKdaSpec::glm53(2, 1).plan().expect("decode");
        let prefill = GlmKdaSpec::glm53(2, 64).plan().expect("prefill");
        assert_eq!(decode.state_bytes, prefill.state_bytes);
        assert_eq!(decode.conv_state_bytes, prefill.conv_state_bytes);
        assert_eq!(prefill.output_bytes, decode.output_bytes * 64);
    }

    #[test]
    fn rejects_non_glm_head_geometry_and_zero_tokens() {
        assert_eq!(
            GlmKdaSpec {
                sequences: 1,
                tokens: 1,
                heads: 64,
                head_dim: 64,
            }
            .plan(),
            Err(GlmKdaPlanError::InvalidHeadDim)
        );
        assert_eq!(
            GlmKdaSpec::glm53(1, 0).plan(),
            Err(GlmKdaPlanError::InvalidTokens)
        );
    }
}
