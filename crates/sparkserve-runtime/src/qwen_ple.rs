//! Qwen3.8 Flash-Next PLE decode hashing.

pub const QWEN_PLE_HEADS: usize = 16;
pub const QWEN_PLE_HEADS_PER_NGRAM: usize = 8;

pub fn decode_row_ids(
    history: &[i64],
    eos_token_id: i64,
    layer_multipliers: [i64; 3],
    head_vocab_sizes: [i64; QWEN_PLE_HEADS],
    head_offsets: [i64; QWEN_PLE_HEADS],
) -> Result<[u64; QWEN_PLE_HEADS], QwenPleHashError> {
    if history.is_empty() {
        return Err(QwenPleHashError::EmptyHistory);
    }
    for index in 0..QWEN_PLE_HEADS {
        if head_vocab_sizes[index] <= 0 || head_offsets[index] < 0 {
            return Err(QwenPleHashError::InvalidHeadGeometry);
        }
    }
    let current_index = history.len() - 1;
    let segment_start = history[..current_index]
        .iter()
        .rposition(|token| *token == eos_token_id)
        .map_or(0, |index| index + 1);
    let current = history[current_index];
    let previous = if current_index >= segment_start + 1 {
        history[current_index - 1]
    } else {
        eos_token_id
    };
    let previous_previous = if current_index >= segment_start + 2 {
        history[current_index - 2]
    } else {
        eos_token_id
    };

    let bigram = current.wrapping_mul(layer_multipliers[0])
        ^ previous.wrapping_mul(layer_multipliers[1]);
    let trigram = bigram ^ previous_previous.wrapping_mul(layer_multipliers[2]);
    let mut rows = [0_u64; QWEN_PLE_HEADS];
    for head in 0..QWEN_PLE_HEADS {
        let mixed = if head < QWEN_PLE_HEADS_PER_NGRAM {
            bigram
        } else {
            trigram
        };
        let row = mixed
            .rem_euclid(head_vocab_sizes[head])
            .checked_add(head_offsets[head])
            .ok_or(QwenPleHashError::RowOverflow)?;
        rows[head] = u64::try_from(row).map_err(|_| QwenPleHashError::RowOverflow)?;
    }
    Ok(rows)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenPleHashError {
    EmptyHistory,
    InvalidHeadGeometry,
    RowOverflow,
}

impl std::fmt::Display for QwenPleHashError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHistory => formatter.write_str("PLE history is empty"),
            Self::InvalidHeadGeometry => formatter.write_str("PLE head geometry is invalid"),
            Self::RowOverflow => formatter.write_str("PLE row id overflowed"),
        }
    }
}

impl std::error::Error for QwenPleHashError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_hash_resets_history_after_eos() {
        let multipliers = [3, 5, 7];
        let sizes = [101; QWEN_PLE_HEADS];
        let offsets = std::array::from_fn(|head| i64::try_from(head * 101).unwrap());
        let after_eos = decode_row_ids(&[11, 99, 13], 99, multipliers, sizes, offsets).unwrap();
        let fresh = decode_row_ids(&[13], 99, multipliers, sizes, offsets).unwrap();
        assert_eq!(after_eos, fresh);
    }

    #[test]
    fn bigram_and_trigram_heads_use_their_own_mix() {
        let multipliers = [3, 5, 7];
        let sizes = [101; QWEN_PLE_HEADS];
        let offsets = [0; QWEN_PLE_HEADS];
        let rows = decode_row_ids(&[2, 3, 4], 99, multipliers, sizes, offsets).unwrap();
        assert_eq!(rows[0], ((4_i64 * 3) ^ (3_i64 * 5)).rem_euclid(101) as u64);
        assert_eq!(
            rows[8],
            ((4_i64 * 3) ^ (3_i64 * 5) ^ (2_i64 * 7)).rem_euclid(101) as u64
        );
    }
}
