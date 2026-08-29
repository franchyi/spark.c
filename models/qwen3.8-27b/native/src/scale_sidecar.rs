use crate::mapping::{MappedFile, MappingError};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const HEADER_BYTES: u64 = 64;
const ENTRY_BYTES: u64 = 40;
const ENTRIES: u32 = 192;
const REVISION: &[u8; 40] = b"009632fef96dd349150baa780c984e62e70e91fe";

#[derive(Clone, Copy, Debug)]
pub struct ScaleEntry {
    pub layer: u32,
    pub projection: u32,
    pub n: u32,
    pub k: u32,
    pub offset: u64,
    pub bytes: u64,
    pub input_scale_inv: f32,
    pub alpha: f32,
}

#[derive(Debug)]
pub enum SidecarError {
    Io(std::io::Error),
    Mapping(MappingError),
    Invalid(String),
}

impl Display for SidecarError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Mapping(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SidecarError {}
impl From<std::io::Error> for SidecarError {
    fn from(error: std::io::Error) -> Self { Self::Io(error) }
}
impl From<MappingError> for SidecarError {
    fn from(error: MappingError) -> Self { Self::Mapping(error) }
}

fn invalid(message: impl Into<String>) -> SidecarError { SidecarError::Invalid(message.into()) }

fn u32_le(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap())
}
fn u64_le(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap())
}
fn f32_le(input: &[u8], offset: usize) -> f32 { f32::from_bits(u32_le(input, offset)) }

pub struct ScaleSidecar {
    entries: Vec<ScaleEntry>,
    mapping: MappedFile,
}

impl ScaleSidecar {
    pub fn open(path: &Path) -> Result<Self, SidecarError> {
        let mut file = File::open(path)?;
        let file_bytes = file.metadata()?.len();
        let table_bytes = HEADER_BYTES + ENTRY_BYTES * u64::from(ENTRIES);
        if file_bytes < table_bytes {
            return Err(invalid("q27 scale sidecar is shorter than its table"));
        }
        let mut table = vec![0_u8; table_bytes as usize];
        file.read_exact(&mut table)?;
        if &table[0..8] != b"Q27SFV1\0" || u32_le(&table, 8) != 1 ||
            u32_le(&table, 12) != ENTRIES || &table[16..56] != REVISION ||
            table[56..64].iter().any(|byte| *byte != 0) {
            return Err(invalid("q27 scale sidecar header does not match revision v1"));
        }
        let mut entries = Vec::with_capacity(ENTRIES as usize);
        let mut expected_offset = table_bytes;
        for index in 0..ENTRIES as usize {
            let begin = HEADER_BYTES as usize + index * ENTRY_BYTES as usize;
            let entry = ScaleEntry {
                layer: u32_le(&table, begin),
                projection: u32_le(&table, begin + 4),
                n: u32_le(&table, begin + 8),
                k: u32_le(&table, begin + 12),
                offset: u64_le(&table, begin + 16),
                bytes: u64_le(&table, begin + 24),
                input_scale_inv: f32_le(&table, begin + 32),
                alpha: f32_le(&table, begin + 36),
            };
            let wanted_layer = index as u32 / 3;
            let wanted_projection = index as u32 % 3;
            let (wanted_n, wanted_k) = if wanted_projection < 2 {
                (17_408, 5_120)
            } else {
                (5_120, 17_408)
            };
            let wanted_bytes = u64::from(wanted_n) * u64::from(wanted_k) / 16;
            if entry.layer != wanted_layer || entry.projection != wanted_projection ||
                entry.n != wanted_n || entry.k != wanted_k ||
                entry.offset != expected_offset || entry.bytes != wanted_bytes ||
                !entry.input_scale_inv.is_finite() || entry.input_scale_inv <= 0.0 ||
                !entry.alpha.is_finite() || entry.alpha <= 0.0 {
                return Err(invalid(format!("invalid q27 scale sidecar entry {index}")));
            }
            expected_offset = expected_offset.checked_add(entry.bytes)
                .ok_or_else(|| invalid("q27 scale sidecar offset overflow"))?;
            entries.push(entry);
        }
        if expected_offset != file_bytes {
            return Err(invalid(format!(
                "q27 scale sidecar must contain {expected_offset} bytes, found {file_bytes}"
            )));
        }
        for layer in 0..64_usize {
            let gate = entries[layer * 3];
            let up = entries[layer * 3 + 1];
            if gate.input_scale_inv.to_bits() != up.input_scale_inv.to_bits() ||
                gate.alpha.to_bits() != up.alpha.to_bits() {
                return Err(invalid(format!(
                    "layer {layer} gate/up scales cannot use the fused projection"
                )));
            }
        }
        drop(file);
        let mapping = MappedFile::open(path)?;
        Ok(Self { entries, mapping })
    }

    pub fn bytes(&self) -> u64 { self.mapping.bytes() }

    pub fn entry(&self, layer: u32, projection: u32) -> Result<&ScaleEntry, SidecarError> {
        if layer >= 64 || projection >= 3 {
            return Err(invalid("q27 scale sidecar index is out of range"));
        }
        Ok(&self.entries[(layer * 3 + projection) as usize])
    }

    pub fn scale_device_address(&self, layer: u32, projection: u32) -> Result<usize, SidecarError> {
        let entry = self.entry(layer, projection)?;
        Ok(self.mapping.device_address(entry.offset, entry.bytes)?)
    }

    pub fn input_scale_inv_device_address(&self, layer: u32, projection: u32) -> Result<usize, SidecarError> {
        let entry_index = u64::from(layer * 3 + projection);
        Ok(self.mapping.device_address(HEADER_BYTES + entry_index * ENTRY_BYTES + 32, 4)?)
    }

    pub fn alpha_device_address(&self, layer: u32, projection: u32) -> Result<usize, SidecarError> {
        let entry_index = u64::from(layer * 3 + projection);
        Ok(self.mapping.device_address(HEADER_BYTES + entry_index * ENTRY_BYTES + 36, 4)?)
    }
}
