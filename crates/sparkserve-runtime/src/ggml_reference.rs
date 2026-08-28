//! Small CPU correctness references for promotion of borrowed ggml CUDA MMQ.
//!
//! These routines are intentionally scalar and never serve tokens. They freeze
//! block layout and accumulation order before the pinned CUDA donor is enabled.

use std::fmt::{Display, Formatter};

pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Reference matrix-vector multiply for GGML Q8_0 rows.
///
/// The encoded matrix has `rows` consecutive rows, each with `columns / 32`
/// blocks. Every block is little-endian FP16 scale followed by 32 signed bytes.
pub fn q8_0_matvec(
    encoded: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, Q8ReferenceError> {
    if rows == 0 || columns == 0 || columns % Q8_0_BLOCK_ELEMENTS != 0 {
        return Err(Q8ReferenceError::InvalidShape { rows, columns });
    }
    if input.len() != columns {
        return Err(Q8ReferenceError::InputLength {
            actual: input.len(),
            expected: columns,
        });
    }
    let blocks_per_row = columns / Q8_0_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(Q8ReferenceError::IntegerOverflow)?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or(Q8ReferenceError::IntegerOverflow)?;
    if encoded.len() != expected_bytes {
        return Err(Q8ReferenceError::EncodedLength {
            actual: encoded.len(),
            expected: expected_bytes,
        });
    }

    let mut output = vec![0.0_f32; rows];
    for (row, output_value) in output.iter_mut().enumerate() {
        let row_start = row * row_bytes;
        let mut sum = 0.0_f32;
        for block in 0..blocks_per_row {
            let weight_start = row_start + block * Q8_0_BLOCK_BYTES;
            let scale = f16_to_f32(u16::from_le_bytes([
                encoded[weight_start],
                encoded[weight_start + 1],
            ]));
            let input_start = block * Q8_0_BLOCK_ELEMENTS;
            let quantized = &encoded[weight_start + 2..weight_start + Q8_0_BLOCK_BYTES];
            for index in 0..Q8_0_BLOCK_ELEMENTS {
                let weight = quantized[index] as i8 as f32;
                sum += scale * weight * input[input_start + index];
            }
        }
        *output_value = sum;
    }
    Ok(output)
}

/// IEEE-754 binary16 to binary32 conversion used by ggml's Q8_0 block scale.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let output = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let highest_bit = 31 - fraction.leading_zeros();
            let exponent32 = highest_bit + 103;
            let mantissa = (fraction << (23 - highest_bit)) & 0x007f_ffff;
            sign | (exponent32 << 23) | mantissa
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(output)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Q8ReferenceError {
    InvalidShape { rows: usize, columns: usize },
    InputLength { actual: usize, expected: usize },
    EncodedLength { actual: usize, expected: usize },
    IntegerOverflow,
}

impl Display for Q8ReferenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape { rows, columns } => {
                write!(formatter, "invalid Q8_0 matrix shape {rows}x{columns}")
            }
            Self::InputLength { actual, expected } => write!(
                formatter,
                "Q8_0 input has {actual} elements, expected {expected}"
            ),
            Self::EncodedLength { actual, expected } => write!(
                formatter,
                "Q8_0 weights have {actual} bytes, expected {expected}"
            ),
            Self::IntegerOverflow => formatter.write_str("Q8_0 reference size overflow"),
        }
    }
}

impl std::error::Error for Q8ReferenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(scale: u16, values: impl Iterator<Item = i8>) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend(values.map(|value| value as u8));
        assert_eq!(block.len(), Q8_0_BLOCK_BYTES);
        block
    }

    #[test]
    fn q8_reference_preserves_block_layout_and_row_order() {
        let first = block(0x3c00, -16_i8..16);
        let second = block(0x3800, std::iter::repeat_n(2_i8, 32));
        let encoded = [first, second].concat();
        let output = q8_0_matvec(&encoded, 2, 32, &[0.25; 32]).expect("Q8 matvec");
        assert_eq!(output, [-4.0, 8.0]);
    }

    #[test]
    fn q8_reference_accumulates_multiple_blocks_in_ggml_order() {
        let encoded = [
            block(0x3c00, std::iter::repeat_n(1_i8, 32)),
            block(0x4000, std::iter::repeat_n(-1_i8, 32)),
        ]
        .concat();
        let output = q8_0_matvec(&encoded, 1, 64, &[1.0; 64]).expect("Q8 matvec");
        assert_eq!(output, [-32.0]);
    }

    #[test]
    fn binary16_conversion_covers_normal_subnormal_and_special_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn q8_reference_rejects_partial_rows_and_wrong_inputs() {
        assert!(matches!(
            q8_0_matvec(&[], 1, 31, &[0.0; 31]),
            Err(Q8ReferenceError::InvalidShape { .. })
        ));
        assert!(matches!(
            q8_0_matvec(&[0; Q8_0_BLOCK_BYTES], 1, 32, &[0.0; 31]),
            Err(Q8ReferenceError::InputLength { .. })
        ));
    }
}
