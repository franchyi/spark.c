#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowAddress {
    pub shard: usize,
    pub byte_offset: u64,
    pub byte_len: usize,
}

/// Physical layout of one logical embedding table split across safetensors
/// shards. Header offsets are supplied by the validated manifest loader.
#[derive(Debug)]
pub struct RowLayout<'a> {
    pub rows_per_tensor: u64,
    pub row_bytes: usize,
    pub tensors_per_file: usize,
    pub tensor_data_offsets: &'a [u64],
}

impl RowLayout<'_> {
    pub fn address(&self, logical_table: usize, row: u64) -> Result<RowAddress, &'static str> {
        if row >= self.rows_per_tensor || self.tensors_per_file == 0 {
            return Err("PLE row or layout is out of range");
        }
        let shard = logical_table / self.tensors_per_file;
        let base = *self
            .tensor_data_offsets
            .get(logical_table)
            .ok_or("PLE table is out of range")?;
        Ok(RowAddress {
            shard,
            byte_offset: base + row * self.row_bytes as u64,
            byte_len: self.row_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_ple_row_without_touching_weight_data() {
        let offsets = [4096, 400_005_120, 800_006_144];
        let layout = RowLayout {
            rows_per_tensor: 2_500_012,
            row_bytes: 160,
            tensors_per_file: 13,
            tensor_data_offsets: &offsets,
        };
        assert_eq!(
            layout.address(1, 7).expect("valid row"),
            RowAddress {
                shard: 0,
                byte_offset: 400_006_240,
                byte_len: 160,
            }
        );
    }
}
