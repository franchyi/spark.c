//! Strict, metadata-only GGUF v3 index for paged model access.
//!
//! The invariants and quant block geometry follow llama.cpp/ggml revision
//! `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`. Tensor payloads are never copied:
//! the index records immutable `(file, absolute offset, byte length)` slices for
//! the Rust residency controller and the borrowed ggml CUDA MMQ kernels.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const GGML_MAX_DIMS: u32 = 4;
const GGML_MAX_NAME: usize = 64;
const MAX_METADATA: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KEY_BYTES: u64 = 1 << 20;
const MAX_VALUE_STRING_BYTES: u64 = 16 << 20;
const MAX_ARRAY_ELEMENTS: u64 = 16_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::InvalidValueType(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufValueType,
        elements: u64,
        payload_offset: u64,
        payload_bytes: u64,
    },
}

impl GgufMetadataValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            Self::Signed(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufMetadataEntry {
    pub value_type: GgufValueType,
    pub value: GgufMetadataValue,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum GgmlTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
}

impl GgmlTensorType {
    pub fn block_geometry(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 40),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq3S => (256, 110),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 | Self::F64 => (1, 8),
            Self::Iq1M => (256, 56),
            Self::Tq1_0 => (256, 54),
            Self::Tq2_0 => (256, 66),
            Self::Mxfp4 => (32, 17),
            Self::Nvfp4 => (64, 36),
            Self::Q1_0 => (128, 18),
        }
    }
}

impl TryFrom<u32> for GgmlTensorType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            34 => Ok(Self::Tq1_0),
            35 => Ok(Self::Tq2_0),
            39 => Ok(Self::Mxfp4),
            40 => Ok(Self::Nvfp4),
            41 => Ok(Self::Q1_0),
            _ => Err(GgufError::InvalidTensorType(value)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlTensorType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub data_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufShard {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
    pub metadata: BTreeMap<String, GgufMetadataEntry>,
    pub tensors: Vec<GgufTensorInfo>,
}

impl GgufShard {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let mut file = File::open(path).map_err(|error| GgufError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|error| GgufError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?
            .len();
        Self::read(path.to_path_buf(), &mut file, file_bytes)
    }

    /// Index a prefix containing all metadata and tensor descriptors while
    /// validating payload extents against the immutable full-file size. This
    /// is inspection tooling for huge locked shards; serving still opens the
    /// complete files through `open` before any payload read.
    pub fn open_header(path: &Path, declared_file_bytes: u64) -> Result<Self, GgufError> {
        let mut file = File::open(path).map_err(|error| GgufError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let prefix_bytes = file
            .metadata()
            .map_err(|error| GgufError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?
            .len();
        if declared_file_bytes == 0 || prefix_bytes > declared_file_bytes {
            return Err(GgufError::Truncated {
                needed: prefix_bytes,
                available: declared_file_bytes,
            });
        }
        Self::read(path.to_path_buf(), &mut file, declared_file_bytes)
    }

    fn read<R: Read + Seek>(
        path: PathBuf,
        input: &mut R,
        file_bytes: u64,
    ) -> Result<Self, GgufError> {
        let mut reader = CheckedReader::new(input, file_bytes);
        if reader.bytes(4)? != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }
        let version = reader.u32()?;
        if version == GGUF_VERSION.swap_bytes() {
            return Err(GgufError::UnsupportedEndian);
        }
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA {
            return Err(GgufError::CountLimit {
                tensors: tensor_count,
                metadata: metadata_count,
            });
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.string(MAX_KEY_BYTES)?;
            if key.is_empty() {
                return Err(GgufError::EmptyMetadataKey);
            }
            if metadata.contains_key(&key) {
                return Err(GgufError::DuplicateMetadata(key));
            }
            let value_type = GgufValueType::try_from(reader.u32()?)?;
            let value = read_metadata_value(&mut reader, value_type)?;
            metadata.insert(key, GgufMetadataEntry { value_type, value });
        }
        let alignment = match metadata.get("general.alignment") {
            None => GGUF_DEFAULT_ALIGNMENT,
            Some(GgufMetadataEntry {
                value_type: GgufValueType::Uint32,
                value: GgufMetadataValue::Unsigned(value),
            }) => *value,
            Some(_) => return Err(GgufError::InvalidAlignmentType),
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        struct PendingTensor {
            name: String,
            dimensions: Vec<u64>,
            tensor_type: GgmlTensorType,
            relative_offset: u64,
            data_bytes: u64,
        }

        let mut names = BTreeSet::new();
        let mut pending = Vec::with_capacity(
            usize::try_from(tensor_count).map_err(|_| GgufError::IntegerOverflow)?,
        );
        for _ in 0..tensor_count {
            let name = reader.string(GGML_MAX_NAME as u64 - 1)?;
            if name.is_empty() {
                return Err(GgufError::EmptyTensorName);
            }
            if !names.insert(name.clone()) {
                return Err(GgufError::DuplicateTensor(name));
            }
            let dimensions_count = reader.u32()?;
            if !(1..=GGML_MAX_DIMS).contains(&dimensions_count) {
                return Err(GgufError::InvalidDimensions(dimensions_count));
            }
            let mut dimensions = Vec::with_capacity(dimensions_count as usize);
            let mut elements = 1_u64;
            for _ in 0..dimensions_count {
                let dimension = reader.u64()?;
                if dimension == 0 || dimension > i64::MAX as u64 {
                    return Err(GgufError::InvalidDimension(dimension));
                }
                elements = elements
                    .checked_mul(dimension)
                    .ok_or(GgufError::IntegerOverflow)?;
                dimensions.push(dimension);
            }
            if elements > i64::MAX as u64 {
                return Err(GgufError::IntegerOverflow);
            }
            let tensor_type = GgmlTensorType::try_from(reader.u32()?)?;
            let (block_elements, block_bytes) = tensor_type.block_geometry();
            if dimensions[0] % block_elements != 0 {
                return Err(GgufError::InvalidQuantRow {
                    name,
                    row_elements: dimensions[0],
                    block_elements,
                });
            }
            let data_bytes = elements
                .checked_div(block_elements)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or(GgufError::IntegerOverflow)?;
            let relative_offset = reader.u64()?;
            pending.push(PendingTensor {
                name,
                dimensions,
                tensor_type,
                relative_offset,
                data_bytes,
            });
        }

        // Split GGUF writers may emit a metadata-only shard without trailing
        // padding because there is no tensor payload whose start must be
        // aligned. Keep its physical EOF as the data offset; shards with at
        // least one tensor still require the canonical aligned payload start.
        let descriptor_end = reader.position()?;
        let data_offset = if pending.is_empty() {
            descriptor_end
        } else {
            align_up(descriptor_end, alignment)?
        };
        if data_offset > file_bytes {
            return Err(GgufError::Truncated {
                needed: data_offset,
                available: file_bytes,
            });
        }
        let mut expected_relative = 0_u64;
        let mut tensors = Vec::with_capacity(pending.len());
        for tensor in pending {
            if tensor.relative_offset != expected_relative {
                return Err(GgufError::NonSequentialTensor {
                    name: tensor.name,
                    actual: tensor.relative_offset,
                    expected: expected_relative,
                });
            }
            let absolute_offset = data_offset
                .checked_add(tensor.relative_offset)
                .ok_or(GgufError::IntegerOverflow)?;
            let end = absolute_offset
                .checked_add(tensor.data_bytes)
                .ok_or(GgufError::IntegerOverflow)?;
            if end > file_bytes {
                return Err(GgufError::Truncated {
                    needed: end,
                    available: file_bytes,
                });
            }
            expected_relative = expected_relative
                .checked_add(align_up(tensor.data_bytes, alignment)?)
                .ok_or(GgufError::IntegerOverflow)?;
            tensors.push(GgufTensorInfo {
                name: tensor.name,
                dimensions: tensor.dimensions,
                tensor_type: tensor.tensor_type,
                relative_offset: tensor.relative_offset,
                absolute_offset,
                data_bytes: tensor.data_bytes,
            });
        }
        let padded_end = data_offset
            .checked_add(expected_relative)
            .ok_or(GgufError::IntegerOverflow)?;
        if padded_end > file_bytes {
            return Err(GgufError::Truncated {
                needed: padded_end,
                available: file_bytes,
            });
        }

        Ok(Self {
            path,
            version,
            alignment,
            data_offset,
            file_bytes,
            metadata,
            tensors,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.get(key).map(|entry| &entry.value)
    }

    /// Decode a signed 32-bit metadata array directly from the immutable GGUF
    /// header. Arrays remain zero-copy in the index and are materialized only
    /// when the model contract needs their values.
    pub fn metadata_i32_array(&self, key: &str) -> Result<Vec<i32>, GgufError> {
        let (elements, payload_offset) = self.metadata_array(key, GgufValueType::Int32)?;
        let mut file = self.open_metadata_payload(payload_offset)?;
        let file_bytes = file
            .metadata()
            .map_err(|error| GgufError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?
            .len();
        let mut reader = CheckedReader::new(&mut file, file_bytes);
        (0..elements).map(|_| reader.i32()).collect()
    }

    /// Decode a float32 metadata array without retaining a second copy in the
    /// long-lived GGUF tensor index.
    pub fn metadata_f32_array(&self, key: &str) -> Result<Vec<f32>, GgufError> {
        let (elements, payload_offset) = self.metadata_array(key, GgufValueType::Float32)?;
        let mut file = self.open_metadata_payload(payload_offset)?;
        let file_bytes = file
            .metadata()
            .map_err(|error| GgufError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?
            .len();
        let mut reader = CheckedReader::new(&mut file, file_bytes);
        (0..elements).map(|_| reader.f32()).collect()
    }

    /// Decode a GGUF string array. This is the native tokenizer path: token
    /// strings are read from the model itself rather than a Python sidecar.
    pub fn metadata_string_array(&self, key: &str) -> Result<Vec<String>, GgufError> {
        let (elements, payload_offset) = self.metadata_array(key, GgufValueType::String)?;
        let mut file = self.open_metadata_payload(payload_offset)?;
        let file_bytes = file
            .metadata()
            .map_err(|error| GgufError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?
            .len();
        let mut reader = CheckedReader::new(&mut file, file_bytes);
        (0..elements)
            .map(|_| reader.string(MAX_VALUE_STRING_BYTES))
            .collect()
    }

    fn metadata_array(
        &self,
        key: &str,
        expected: GgufValueType,
    ) -> Result<(u64, u64), GgufError> {
        let entry = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        let GgufMetadataValue::Array {
            element_type,
            elements,
            payload_offset,
            ..
        } = &entry.value
        else {
            return Err(GgufError::MetadataIsNotArray(key.to_owned()));
        };
        if *element_type != expected {
            return Err(GgufError::MetadataArrayType {
                key: key.to_owned(),
                expected,
                actual: *element_type,
            });
        }
        Ok((*elements, *payload_offset))
    }

    fn open_metadata_payload(&self, payload_offset: u64) -> Result<File, GgufError> {
        let mut file = File::open(&self.path).map_err(|error| GgufError::Io {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        file.seek(SeekFrom::Start(payload_offset))
            .map_err(|error| GgufError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        Ok(file)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensorLocation {
    pub shard: usize,
    pub absolute_offset: u64,
    pub data_bytes: u64,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlTensorType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufSet {
    pub architecture: String,
    pub shards: Vec<GgufShard>,
    pub tensors: BTreeMap<String, GgufTensorLocation>,
}

impl GgufSet {
    pub fn open(paths: &[PathBuf]) -> Result<Self, GgufError> {
        if paths.is_empty() {
            return Err(GgufError::NoShards);
        }
        let indexed = paths
            .iter()
            .map(|path| GgufShard::open(path))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_shards(indexed)
    }

    pub fn open_headers(headers: &[(PathBuf, u64)]) -> Result<Self, GgufError> {
        if headers.is_empty() {
            return Err(GgufError::NoShards);
        }
        let indexed = headers
            .iter()
            .map(|(path, declared_bytes)| GgufShard::open_header(path, *declared_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_shards(indexed)
    }

    fn from_shards(mut indexed: Vec<GgufShard>) -> Result<Self, GgufError> {
        let expected_shards = metadata_u64(&indexed[0], "split.count")?;
        if expected_shards != indexed.len() as u64 {
            return Err(GgufError::SplitCount {
                metadata: expected_shards,
                files: indexed.len() as u64,
            });
        }
        indexed.sort_by_key(|shard| metadata_u64(shard, "split.no").unwrap_or(u64::MAX));
        for (index, shard) in indexed.iter().enumerate() {
            let split = metadata_u64(shard, "split.no")?;
            if split != index as u64 {
                return Err(GgufError::SplitOrder {
                    actual: split,
                    expected: index as u64,
                });
            }
            if metadata_u64(shard, "split.count")? != expected_shards {
                return Err(GgufError::InconsistentSplitMetadata("split.count"));
            }
        }
        let architecture = metadata_string(&indexed[0], "general.architecture")?.to_owned();
        let tensors_expected = metadata_u64(&indexed[0], "split.tensors.count")?;
        let tensors_actual = indexed.iter().try_fold(0_u64, |total, shard| {
            total
                .checked_add(shard.tensors.len() as u64)
                .ok_or(GgufError::IntegerOverflow)
        })?;
        if tensors_expected != tensors_actual {
            return Err(GgufError::SplitTensorCount {
                metadata: tensors_expected,
                actual: tensors_actual,
            });
        }
        let mut tensors = BTreeMap::new();
        for (shard_index, shard) in indexed.iter().enumerate() {
            if shard
                .metadata("general.architecture")
                .and_then(GgufMetadataValue::as_str)
                .is_some_and(|value| value != architecture)
            {
                return Err(GgufError::InconsistentSplitMetadata("general.architecture"));
            }
            if metadata_u64(shard, "split.tensors.count")? != tensors_expected {
                return Err(GgufError::InconsistentSplitMetadata("split.tensors.count"));
            }
            for tensor in &shard.tensors {
                let location = GgufTensorLocation {
                    shard: shard_index,
                    absolute_offset: tensor.absolute_offset,
                    data_bytes: tensor.data_bytes,
                    dimensions: tensor.dimensions.clone(),
                    tensor_type: tensor.tensor_type,
                };
                if tensors.insert(tensor.name.clone(), location).is_some() {
                    return Err(GgufError::DuplicateTensorAcrossShards(tensor.name.clone()));
                }
            }
        }
        Ok(Self {
            architecture,
            shards: indexed,
            tensors,
        })
    }
}

fn metadata_u64(shard: &GgufShard, key: &'static str) -> Result<u64, GgufError> {
    shard
        .metadata(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or(GgufError::MissingSplitMetadata(key))
}

fn metadata_string<'a>(shard: &'a GgufShard, key: &'static str) -> Result<&'a str, GgufError> {
    shard
        .metadata(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or(GgufError::MissingSplitMetadata(key))
}

fn read_metadata_value<R: Read + Seek>(
    reader: &mut CheckedReader<'_, R>,
    value_type: GgufValueType,
) -> Result<GgufMetadataValue, GgufError> {
    match value_type {
        GgufValueType::Uint8 => Ok(GgufMetadataValue::Unsigned(u64::from(reader.u8()?))),
        GgufValueType::Int8 => Ok(GgufMetadataValue::Signed(i64::from(reader.i8()?))),
        GgufValueType::Uint16 => Ok(GgufMetadataValue::Unsigned(u64::from(reader.u16()?))),
        GgufValueType::Int16 => Ok(GgufMetadataValue::Signed(i64::from(reader.i16()?))),
        GgufValueType::Uint32 => Ok(GgufMetadataValue::Unsigned(u64::from(reader.u32()?))),
        GgufValueType::Int32 => Ok(GgufMetadataValue::Signed(i64::from(reader.i32()?))),
        GgufValueType::Float32 => Ok(GgufMetadataValue::Float(f64::from(reader.f32()?))),
        GgufValueType::Bool => Ok(GgufMetadataValue::Bool(reader.boolean()?)),
        GgufValueType::String => Ok(GgufMetadataValue::String(
            reader.string(MAX_VALUE_STRING_BYTES)?,
        )),
        GgufValueType::Array => {
            let element_type = GgufValueType::try_from(reader.u32()?)?;
            if element_type == GgufValueType::Array {
                return Err(GgufError::NestedArray);
            }
            let elements = reader.u64()?;
            if elements > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::ArrayLimit(elements));
            }
            let payload_offset = reader.position()?;
            skip_array(reader, element_type, elements)?;
            let payload_bytes = reader
                .position()?
                .checked_sub(payload_offset)
                .ok_or(GgufError::IntegerOverflow)?;
            Ok(GgufMetadataValue::Array {
                element_type,
                elements,
                payload_offset,
                payload_bytes,
            })
        }
        GgufValueType::Uint64 => Ok(GgufMetadataValue::Unsigned(reader.u64()?)),
        GgufValueType::Int64 => Ok(GgufMetadataValue::Signed(reader.i64()?)),
        GgufValueType::Float64 => Ok(GgufMetadataValue::Float(reader.f64()?)),
    }
}

fn skip_array<R: Read + Seek>(
    reader: &mut CheckedReader<'_, R>,
    element_type: GgufValueType,
    elements: u64,
) -> Result<(), GgufError> {
    match element_type {
        GgufValueType::String => {
            for _ in 0..elements {
                let bytes = reader.u64()?;
                reader.skip(bytes)?;
            }
            Ok(())
        }
        GgufValueType::Bool => {
            let mut remaining = elements;
            let mut buffer = [0_u8; 4096];
            while remaining != 0 {
                let bytes = remaining.min(buffer.len() as u64) as usize;
                reader.read_exact(&mut buffer[..bytes])?;
                if buffer[..bytes].iter().any(|value| *value > 1) {
                    return Err(GgufError::InvalidBool);
                }
                remaining -= bytes as u64;
            }
            Ok(())
        }
        GgufValueType::Array => Err(GgufError::NestedArray),
        scalar => {
            let width = scalar_width(scalar).ok_or(GgufError::InvalidValueType(scalar as u32))?;
            let bytes = elements
                .checked_mul(width)
                .ok_or(GgufError::IntegerOverflow)?;
            reader.skip(bytes)
        }
    }
}

fn scalar_width(value_type: GgufValueType) -> Option<u64> {
    match value_type {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => Some(1),
        GgufValueType::Uint16 | GgufValueType::Int16 => Some(2),
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => Some(4),
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => Some(8),
        GgufValueType::String | GgufValueType::Array => None,
    }
}

struct CheckedReader<'a, R> {
    input: &'a mut R,
    file_bytes: u64,
}

impl<'a, R: Read + Seek> CheckedReader<'a, R> {
    fn new(input: &'a mut R, file_bytes: u64) -> Self {
        Self { input, file_bytes }
    }

    fn position(&mut self) -> Result<u64, GgufError> {
        self.input
            .stream_position()
            .map_err(|error| GgufError::Read(error.to_string()))
    }

    fn read_exact(&mut self, output: &mut [u8]) -> Result<(), GgufError> {
        let position = self.position()?;
        let end = position
            .checked_add(output.len() as u64)
            .ok_or(GgufError::IntegerOverflow)?;
        if end > self.file_bytes {
            return Err(GgufError::Truncated {
                needed: end,
                available: self.file_bytes,
            });
        }
        self.input
            .read_exact(output)
            .map_err(|error| GgufError::Read(error.to_string()))
    }

    fn bytes<const N: usize>(&mut self, count: usize) -> Result<[u8; N], GgufError> {
        if count != N {
            return Err(GgufError::IntegerOverflow);
        }
        let mut output = [0_u8; N];
        self.read_exact(&mut output)?;
        Ok(output)
    }

    fn skip(&mut self, bytes: u64) -> Result<(), GgufError> {
        let position = self.position()?;
        let end = position
            .checked_add(bytes)
            .ok_or(GgufError::IntegerOverflow)?;
        if end > self.file_bytes {
            return Err(GgufError::Truncated {
                needed: end,
                available: self.file_bytes,
            });
        }
        self.input
            .seek(SeekFrom::Start(end))
            .map_err(|error| GgufError::Read(error.to_string()))?;
        Ok(())
    }

    fn string(&mut self, maximum: u64) -> Result<String, GgufError> {
        let bytes = self.u64()?;
        if bytes > maximum {
            return Err(GgufError::StringLimit { bytes, maximum });
        }
        let length = usize::try_from(bytes).map_err(|_| GgufError::IntegerOverflow)?;
        let mut payload = vec![0_u8; length];
        self.read_exact(&mut payload)?;
        String::from_utf8(payload).map_err(|_| GgufError::InvalidUtf8)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.bytes::<1>(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.bytes::<2>(2)?))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.bytes::<2>(2)?))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes::<4>(4)?))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.bytes::<4>(4)?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes::<8>(8)?))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.bytes::<8>(8)?))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn boolean(&mut self) -> Result<bool, GgufError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(GgufError::InvalidBool),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(GgufError::IntegerOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufError {
    Io {
        path: PathBuf,
        message: String,
    },
    Read(String),
    InvalidMagic,
    UnsupportedEndian,
    UnsupportedVersion(u32),
    CountLimit {
        tensors: u64,
        metadata: u64,
    },
    EmptyMetadataKey,
    DuplicateMetadata(String),
    MissingMetadata(String),
    MetadataIsNotArray(String),
    MetadataArrayType {
        key: String,
        expected: GgufValueType,
        actual: GgufValueType,
    },
    InvalidValueType(u32),
    NestedArray,
    ArrayLimit(u64),
    StringLimit {
        bytes: u64,
        maximum: u64,
    },
    InvalidUtf8,
    InvalidBool,
    InvalidAlignmentType,
    InvalidAlignment(u64),
    EmptyTensorName,
    DuplicateTensor(String),
    InvalidDimensions(u32),
    InvalidDimension(u64),
    InvalidTensorType(u32),
    InvalidQuantRow {
        name: String,
        row_elements: u64,
        block_elements: u64,
    },
    NonSequentialTensor {
        name: String,
        actual: u64,
        expected: u64,
    },
    Truncated {
        needed: u64,
        available: u64,
    },
    IntegerOverflow,
    NoShards,
    MissingSplitMetadata(&'static str),
    SplitCount {
        metadata: u64,
        files: u64,
    },
    SplitOrder {
        actual: u64,
        expected: u64,
    },
    InconsistentSplitMetadata(&'static str),
    SplitTensorCount {
        metadata: u64,
        actual: u64,
    },
    DuplicateTensorAcrossShards(String),
}

impl Display for GgufError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(formatter, "cannot open GGUF {}: {message}", path.display())
            }
            Self::Read(error) => write!(formatter, "cannot read GGUF: {error}"),
            Self::InvalidMagic => formatter.write_str("invalid GGUF magic"),
            Self::UnsupportedEndian => formatter.write_str("big-endian GGUF is not supported"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "GGUF version {version} is not the required v3")
            }
            Self::CountLimit { tensors, metadata } => write!(
                formatter,
                "GGUF counts exceed limits: {tensors} tensors, {metadata} metadata entries"
            ),
            Self::EmptyMetadataKey => formatter.write_str("GGUF metadata key is empty"),
            Self::DuplicateMetadata(key) => write!(formatter, "duplicate GGUF metadata key {key}"),
            Self::MissingMetadata(key) => write!(formatter, "missing GGUF metadata key {key}"),
            Self::MetadataIsNotArray(key) => {
                write!(formatter, "GGUF metadata key {key} is not an array")
            }
            Self::MetadataArrayType {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "GGUF metadata array {key} has element type {actual:?}, expected {expected:?}"
            ),
            Self::InvalidValueType(value) => {
                write!(formatter, "invalid GGUF metadata type {value}")
            }
            Self::NestedArray => formatter.write_str("nested GGUF metadata arrays are invalid"),
            Self::ArrayLimit(elements) => write!(
                formatter,
                "GGUF metadata array has {elements} elements, above the safety limit"
            ),
            Self::StringLimit { bytes, maximum } => write!(
                formatter,
                "GGUF string has {bytes} bytes, maximum is {maximum}"
            ),
            Self::InvalidUtf8 => {
                formatter.write_str("GGUF key, name, or scalar string is not UTF-8")
            }
            Self::InvalidBool => formatter.write_str("GGUF boolean is not zero or one"),
            Self::InvalidAlignmentType => formatter.write_str("general.alignment is not uint32"),
            Self::InvalidAlignment(alignment) => write!(
                formatter,
                "GGUF alignment {alignment} is not a nonzero power of two"
            ),
            Self::EmptyTensorName => formatter.write_str("GGUF tensor name is empty"),
            Self::DuplicateTensor(name) => write!(formatter, "duplicate GGUF tensor {name}"),
            Self::InvalidDimensions(dimensions) => write!(
                formatter,
                "GGUF tensor has invalid dimension count {dimensions}"
            ),
            Self::InvalidDimension(dimension) => {
                write!(formatter, "GGUF tensor has invalid dimension {dimension}")
            }
            Self::InvalidTensorType(value) => {
                write!(formatter, "unsupported or removed ggml tensor type {value}")
            }
            Self::InvalidQuantRow {
                name,
                row_elements,
                block_elements,
            } => write!(
                formatter,
                "GGUF tensor {name} row has {row_elements} elements, not divisible by quant block {block_elements}"
            ),
            Self::NonSequentialTensor {
                name,
                actual,
                expected,
            } => write!(
                formatter,
                "GGUF tensor {name} has relative offset {actual}, expected {expected}"
            ),
            Self::Truncated { needed, available } => write!(
                formatter,
                "GGUF needs {needed} bytes but file has {available}"
            ),
            Self::IntegerOverflow => formatter.write_str("GGUF size arithmetic overflow"),
            Self::NoShards => formatter.write_str("GGUF split has no shards"),
            Self::MissingSplitMetadata(key) => {
                write!(formatter, "GGUF split is missing or mistypes {key}")
            }
            Self::SplitCount { metadata, files } => write!(
                formatter,
                "GGUF split.count is {metadata}, but {files} files were supplied"
            ),
            Self::SplitOrder { actual, expected } => write!(
                formatter,
                "GGUF split number {actual} is not expected shard {expected}"
            ),
            Self::InconsistentSplitMetadata(key) => {
                write!(formatter, "GGUF split metadata {key} differs across shards")
            }
            Self::SplitTensorCount { metadata, actual } => write!(
                formatter,
                "GGUF split.tensors.count is {metadata}, indexed {actual}"
            ),
            Self::DuplicateTensorAcrossShards(name) => {
                write!(formatter, "GGUF tensor {name} appears in multiple shards")
            }
        }
    }
}

impl std::error::Error for GgufError {}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    enum TestValue<'a> {
        U32(u32),
        U64(u64),
        I32(i32),
        I32s(&'a [i32]),
        F32s(&'a [f32]),
        String(&'a str),
        Strings(&'a [&'a str]),
    }

    struct TestTensor<'a> {
        name: &'a str,
        dimensions: &'a [u64],
        tensor_type: GgmlTensorType,
        offset_override: Option<u64>,
    }

    fn put_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn build_gguf(metadata: &[(&str, TestValue<'_>)], tensors: &[TestTensor<'_>]) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        output.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            put_string(&mut output, key);
            match value {
                TestValue::U32(value) => {
                    output.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
                    output.extend_from_slice(&value.to_le_bytes());
                }
                TestValue::U64(value) => {
                    output.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
                    output.extend_from_slice(&value.to_le_bytes());
                }
                TestValue::I32(value) => {
                    output.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
                    output.extend_from_slice(&value.to_le_bytes());
                }
                TestValue::I32s(values) => {
                    output.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    output.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
                    output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in *values {
                        output.extend_from_slice(&value.to_le_bytes());
                    }
                }
                TestValue::F32s(values) => {
                    output.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    output.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
                    output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in *values {
                        output.extend_from_slice(&value.to_le_bytes());
                    }
                }
                TestValue::String(value) => {
                    output.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    put_string(&mut output, value);
                }
                TestValue::Strings(values) => {
                    output.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    output.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in *values {
                        put_string(&mut output, value);
                    }
                }
            }
        }
        let alignment = metadata
            .iter()
            .find_map(|(key, value)| match (key, value) {
                (&"general.alignment", TestValue::U32(value)) => Some(*value as u64),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let mut expected_offset = 0_u64;
        let mut tensor_sizes = Vec::new();
        for tensor in tensors {
            put_string(&mut output, tensor.name);
            output.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
            for dimension in tensor.dimensions {
                output.extend_from_slice(&dimension.to_le_bytes());
            }
            output.extend_from_slice(&(tensor.tensor_type as u32).to_le_bytes());
            output.extend_from_slice(
                &tensor
                    .offset_override
                    .unwrap_or(expected_offset)
                    .to_le_bytes(),
            );
            let elements = tensor.dimensions.iter().product::<u64>();
            let (block, bytes) = tensor.tensor_type.block_geometry();
            let size = elements / block * bytes;
            tensor_sizes.push(size);
            expected_offset += align_up(size, alignment).expect("fixture size");
        }
        let data_offset = align_up(output.len() as u64, alignment).expect("fixture alignment");
        output.resize(data_offset as usize, 0);
        for size in tensor_sizes {
            output.resize(output.len() + size as usize, 0x5a);
            let aligned = align_up(output.len() as u64, alignment).expect("tensor alignment");
            output.resize(aligned as usize, 0);
        }
        output
    }

    fn parse(payload: Vec<u8>) -> Result<GgufShard, GgufError> {
        let bytes = payload.len() as u64;
        GgufShard::read(
            PathBuf::from("fixture.gguf"),
            &mut Cursor::new(payload),
            bytes,
        )
    }

    #[test]
    fn indexes_a_metadata_prefix_against_the_locked_full_file_size() {
        let full = build_gguf(
            &[],
            &[TestTensor {
                name: "large.weight",
                dimensions: &[256, 64],
                tensor_type: GgmlTensorType::Iq3Xxs,
                offset_override: None,
            }],
        );
        let full_bytes = full.len() as u64;
        let complete = parse(full.clone()).expect("complete fixture");
        let mut prefix = full;
        prefix.truncate(complete.data_offset as usize);
        let indexed = GgufShard::read(
            PathBuf::from("prefix.gguf"),
            &mut Cursor::new(prefix),
            full_bytes,
        )
        .expect("header-only index");
        assert_eq!(indexed.file_bytes, full_bytes);
        assert_eq!(indexed.tensors[0].data_bytes, 64 * 98);
    }

    #[test]
    fn materializes_typed_metadata_arrays_on_demand() {
        let payload = build_gguf(
            &[
                ("test.i32s", TestValue::I32s(&[-7, 0, 11])),
                ("test.f32s", TestValue::F32s(&[0.5, -2.25])),
                ("test.strings", TestValue::Strings(&["alpha", "β"])),
            ],
            &[],
        );
        let path = std::env::temp_dir().join(format!(
            "sparkserve-gguf-arrays-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, payload).expect("write array fixture");
        let shard = GgufShard::open(&path).expect("open array fixture");
        assert_eq!(
            shard.metadata_i32_array("test.i32s").expect("i32 array"),
            [-7, 0, 11]
        );
        assert_eq!(
            shard.metadata_f32_array("test.f32s").expect("f32 array"),
            [0.5, -2.25]
        );
        assert_eq!(
            shard
                .metadata_string_array("test.strings")
                .expect("string array"),
            ["alpha", "β"]
        );
        assert!(matches!(
            shard.metadata_f32_array("test.i32s"),
            Err(GgufError::MetadataArrayType { .. })
        ));
        std::fs::remove_file(path).expect("remove array fixture");
    }

    #[test]
    fn accepts_an_unpadded_metadata_only_split_shard() {
        let mut payload = build_gguf(
            &[
                ("general.architecture", TestValue::String("glm5next")),
                ("split.no", TestValue::U64(0)),
                ("split.count", TestValue::U64(2)),
                ("split.tensors.count", TestValue::I32(1)),
                ("zz.unpadded", TestValue::String("x")),
            ],
            &[],
        );
        while payload.last() == Some(&0) {
            payload.pop();
        }
        let shard = parse(payload.clone()).expect("metadata-only shard");
        assert!(shard.tensors.is_empty());
        assert_eq!(shard.data_offset, payload.len() as u64);
        assert_eq!(shard.file_bytes, payload.len() as u64);
    }

    #[test]
    fn indexes_iq3_without_copying_tensor_payload() {
        let payload = build_gguf(
            &[
                ("general.alignment", TestValue::U32(32)),
                ("general.architecture", TestValue::String("glm5next")),
                ("tokenizer.ggml.tokens", TestValue::Strings(&["a", "b"])),
            ],
            &[
                TestTensor {
                    name: "blk.0.ffn_gate_exps.weight",
                    dimensions: &[256, 8],
                    tensor_type: GgmlTensorType::Iq3Xxs,
                    offset_override: None,
                },
                TestTensor {
                    name: "output_norm.weight",
                    dimensions: &[256],
                    tensor_type: GgmlTensorType::F32,
                    offset_override: None,
                },
            ],
        );
        let shard = parse(payload).expect("strict fixture");
        assert_eq!(shard.alignment, 32);
        assert_eq!(
            shard
                .metadata("general.architecture")
                .and_then(GgufMetadataValue::as_str),
            Some("glm5next")
        );
        assert_eq!(shard.tensors[0].data_bytes, 8 * 98);
        assert_eq!(shard.tensors[1].relative_offset, 800);
        assert_eq!(shard.tensors[1].data_bytes, 256 * 4);
        assert_eq!(shard.file_bytes, shard.data_offset + 1824);
    }

    #[test]
    fn rejects_nonsequential_offsets_and_invalid_quant_rows() {
        let nonsequential = build_gguf(
            &[],
            &[TestTensor {
                name: "weight",
                dimensions: &[256],
                tensor_type: GgmlTensorType::Iq3Xxs,
                offset_override: Some(32),
            }],
        );
        assert!(matches!(
            parse(nonsequential),
            Err(GgufError::NonSequentialTensor { .. })
        ));

        let invalid_row = build_gguf(
            &[],
            &[TestTensor {
                name: "weight",
                dimensions: &[255],
                tensor_type: GgmlTensorType::Iq3Xxs,
                offset_override: None,
            }],
        );
        assert!(matches!(
            parse(invalid_row),
            Err(GgufError::InvalidQuantRow { .. })
        ));
    }

    #[test]
    fn rejects_truncation_bad_alignment_and_removed_quant_type() {
        let mut truncated = build_gguf(
            &[],
            &[TestTensor {
                name: "weight",
                dimensions: &[32],
                tensor_type: GgmlTensorType::Q8_0,
                offset_override: None,
            }],
        );
        truncated.pop();
        assert!(matches!(parse(truncated), Err(GgufError::Truncated { .. })));

        let alignment = build_gguf(&[("general.alignment", TestValue::U32(24))], &[]);
        assert_eq!(parse(alignment), Err(GgufError::InvalidAlignment(24)));

        let mut removed_type = build_gguf(
            &[],
            &[TestTensor {
                name: "weight",
                dimensions: &[32],
                tensor_type: GgmlTensorType::Q8_0,
                offset_override: None,
            }],
        );
        let type_position = 24 + 8 + "weight".len() + 4 + 8;
        removed_type[type_position..type_position + 4].copy_from_slice(&4_u32.to_le_bytes());
        assert_eq!(parse(removed_type), Err(GgufError::InvalidTensorType(4)));
    }

    #[test]
    fn validates_split_count_architecture_and_tensor_inventory() {
        let first = build_gguf(
            &[
                ("general.architecture", TestValue::String("glm5next")),
                ("split.no", TestValue::U64(0)),
                ("split.count", TestValue::U64(2)),
                ("split.tensors.count", TestValue::I32(2)),
            ],
            &[TestTensor {
                name: "a",
                dimensions: &[32],
                tensor_type: GgmlTensorType::Q8_0,
                offset_override: None,
            }],
        );
        let second = build_gguf(
            &[
                ("split.no", TestValue::U64(1)),
                ("split.count", TestValue::U64(2)),
                ("split.tensors.count", TestValue::I32(2)),
            ],
            &[TestTensor {
                name: "b",
                dimensions: &[256],
                tensor_type: GgmlTensorType::Iq3Xxs,
                offset_override: None,
            }],
        );
        let root = std::env::temp_dir().join(format!(
            "sparkserve-gguf-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir_all(&root).expect("fixture directory");
        let first_path = root.join("part-1.gguf");
        let second_path = root.join("part-2.gguf");
        std::fs::write(&first_path, first).expect("first fixture");
        std::fs::write(&second_path, second).expect("second fixture");
        let set = GgufSet::open(&[second_path, first_path]).expect("split index");
        assert_eq!(set.architecture, "glm5next");
        assert_eq!(set.tensors["a"].shard, 0);
        assert_eq!(set.tensors["b"].shard, 1);
        std::fs::remove_dir_all(root).expect("remove fixtures");
    }
}
