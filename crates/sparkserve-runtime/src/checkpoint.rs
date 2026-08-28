use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const EXPECTED_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const EXPECTED_PLE_SHARDS: u64 = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlashNextConfig {
    pub hidden_size: u64,
    pub layers: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub experts: u64,
    pub experts_per_token: u64,
    pub moe_intermediate_size: u64,
    pub full_attention_interval: u64,
    pub ngram_size: u64,
    pub ngram_vocab_size: u64,
    pub ple_embedding_dim: u64,
    pub ple_shards: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorStats {
    pub tensors: u64,
    pub bytes: u64,
}

impl TensorStats {
    fn add(&mut self, bytes: u64) -> Result<(), CheckpointError> {
        self.tensors = self
            .tensors
            .checked_add(1)
            .ok_or_else(|| CheckpointError::Invalid("tensor count overflow".into()))?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| CheckpointError::Invalid("tensor byte count overflow".into()))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointPlan {
    pub root: PathBuf,
    pub config: FlashNextConfig,
    pub files: u64,
    pub resident: TensorStats,
    pub ple_nvme: TensorStats,
    pub mtp_deferred: TensorStats,
    pub vision_ignored: TensorStats,
}

/// Exact byte location of one tensor in the original safetensors checkpoint.
///
/// SparkServe keeps these locations instead of materializing an intermediate
/// weight archive. The serving owner can map a shard once and pass
/// `absolute_offset` directly to CUDA, or read only selected expert tensors
/// into its fixed hot cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorLocation {
    pub relative_file: PathBuf,
    pub absolute_offset: u64,
    pub data_bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlashNextCheckpoint {
    pub plan: CheckpointPlan,
    pub tensors: BTreeMap<String, SafetensorLocation>,
}

impl FlashNextCheckpoint {
    pub fn tensor(&self, name: &str) -> Result<&SafetensorLocation, CheckpointError> {
        self.tensors.get(name).ok_or_else(|| {
            CheckpointError::Invalid(format!("checkpoint tensor is missing: {name}"))
        })
    }
}

impl CheckpointPlan {
    pub fn total(&self) -> TensorStats {
        TensorStats {
            tensors: self.resident.tensors
                + self.ple_nvme.tensors
                + self.mtp_deferred.tensors
                + self.vision_ignored.tensors,
            bytes: self.resident.bytes
                + self.ple_nvme.bytes
                + self.mtp_deferred.bytes
                + self.vision_ignored.bytes,
        }
    }
}

#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl Display for CheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorClass {
    Resident,
    PleNvme,
    MtpDeferred,
    VisionIgnored,
}

pub fn load_flash_next_plan(root: &Path) -> Result<CheckpointPlan, CheckpointError> {
    Ok(load_flash_next_checkpoint(root)?.plan)
}

pub fn load_flash_next_checkpoint(
    root: &Path,
) -> Result<FlashNextCheckpoint, CheckpointError> {
    let config_value: Value = serde_json::from_reader(File::open(root.join(CONFIG_FILE))?)?;
    let config = parse_config(&config_value)?;
    let index_value: Value = serde_json::from_reader(File::open(root.join(INDEX_FILE))?)?;
    let weight_map = object_field(object(&index_value, "checkpoint index")?, "weight_map")?;

    let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (tensor, file_value) in weight_map {
        let file = file_value.as_str().ok_or_else(|| {
            CheckpointError::Invalid(format!("tensor {tensor} has no shard file"))
        })?;
        validate_relative_path(file)?;
        by_file.entry(file).or_default().push(tensor);
    }
    if by_file.is_empty() {
        return Err(CheckpointError::Invalid(
            "checkpoint index contains no tensors".into(),
        ));
    }

    let mut plan = CheckpointPlan {
        root: root.to_path_buf(),
        config,
        files: by_file.len() as u64,
        resident: TensorStats::default(),
        ple_nvme: TensorStats::default(),
        mtp_deferred: TensorStats::default(),
        vision_ignored: TensorStats::default(),
    };
    let mut tensor_index = BTreeMap::new();
    for (relative_file, tensors) in by_file {
        scan_safetensors_header(
            root,
            relative_file,
            &tensors,
            &mut plan,
            &mut tensor_index,
        )?;
    }
    if plan.ple_nvme.tensors != EXPECTED_PLE_SHARDS {
        return Err(CheckpointError::Invalid(format!(
            "expected {EXPECTED_PLE_SHARDS} PLE shard tensors, found {}",
            plan.ple_nvme.tensors
        )));
    }
    Ok(FlashNextCheckpoint {
        plan,
        tensors: tensor_index,
    })
}

fn parse_config(value: &Value) -> Result<FlashNextConfig, CheckpointError> {
    let root = object(value, "config")?;
    expect_string(root, "model_type", "qwen4_exp")?;
    let architectures = array_field(root, "architectures")?;
    if !architectures
        .iter()
        .any(|item| item.as_str() == Some(EXPECTED_ARCHITECTURE))
    {
        return Err(CheckpointError::Invalid(format!(
            "checkpoint architecture must include {EXPECTED_ARCHITECTURE}"
        )));
    }
    let text = object_field(root, "text_config")?;
    expect_string(text, "model_type", "qwen4_exp_text")?;
    expect_string(text, "ple_embedding_dtype", "float8_e4m3fn")?;

    let layers = integer_field(text, "num_hidden_layers")?;
    let full_attention_interval = integer_field(text, "full_attention_interval")?;
    let layer_types = array_field(text, "layer_types")?;
    if layer_types.len() as u64 != layers {
        return Err(CheckpointError::Invalid(format!(
            "layer_types has {} entries, expected {layers}",
            layer_types.len()
        )));
    }
    for (index, layer_type) in layer_types.iter().enumerate() {
        let expected = if (index as u64 + 1) % full_attention_interval == 0 {
            "full_attention"
        } else {
            "linear_attention"
        };
        if layer_type.as_str() != Some(expected) {
            return Err(CheckpointError::Invalid(format!(
                "layer {index} must be {expected}"
            )));
        }
    }

    validate_nvfp4(root)?;
    let ple_shards = integer_field(text, "split_ngram_parts")?;
    if ple_shards != EXPECTED_PLE_SHARDS {
        return Err(CheckpointError::Invalid(format!(
            "split_ngram_parts must be {EXPECTED_PLE_SHARDS}, got {ple_shards}"
        )));
    }

    Ok(FlashNextConfig {
        hidden_size: expect_integer(text, "hidden_size", 2560)?,
        layers: expect_integer(text, "num_hidden_layers", 48)?,
        attention_heads: expect_integer(text, "num_attention_heads", 24)?,
        kv_heads: expect_integer(text, "num_key_value_heads", 2)?,
        experts: expect_integer(text, "num_experts", 512)?,
        experts_per_token: expect_integer(text, "num_experts_per_tok", 10)?,
        moe_intermediate_size: expect_integer(text, "moe_intermediate_size", 640)?,
        full_attention_interval: expect_integer(text, "full_attention_interval", 4)?,
        ngram_size: expect_integer(text, "ngram_size", 3)?,
        ngram_vocab_size: expect_integer(text, "ngram_vocab_size_base", 20_000_000)?,
        ple_embedding_dim: expect_integer(text, "ple_embed_dim", 2560)?,
        ple_shards,
    })
}

fn validate_nvfp4(root: &Map<String, Value>) -> Result<(), CheckpointError> {
    let quant = object_field(root, "quantization_config")?;
    expect_string(quant, "quant_method", "modelopt")?;
    expect_string(quant, "quant_algo", "NVFP4")?;
    let producer = object_field(quant, "producer")?;
    expect_string(producer, "name", "modelopt")?;
    let groups = object_field(quant, "config_groups")?;
    let group = object_field(groups, "group_0")?;
    for field in ["input_activations", "weights"] {
        let spec = object_field(group, field)?;
        expect_integer(spec, "num_bits", 4)?;
        expect_integer(spec, "group_size", 16)?;
        expect_string(spec, "type", "float")?;
        if bool_field(spec, "dynamic")? {
            return Err(CheckpointError::Invalid(format!(
                "NVFP4 {field} must use static scales"
            )));
        }
    }
    Ok(())
}

fn scan_safetensors_header(
    root: &Path,
    relative_file: &str,
    tensors: &[&str],
    plan: &mut CheckpointPlan,
    tensor_index: &mut BTreeMap<String, SafetensorLocation>,
) -> Result<(), CheckpointError> {
    let path = root.join(relative_file);
    let mut file = File::open(&path)?;
    let file_bytes = file.metadata()?.len();
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)?;
    let header_bytes = u64::from_le_bytes(length_bytes);
    if header_bytes == 0 || header_bytes > 256 * 1024 * 1024 {
        return Err(CheckpointError::Invalid(format!(
            "implausible safetensors header in {relative_file}: {header_bytes} bytes"
        )));
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| CheckpointError::Invalid("safetensors header is too large".into()))?;
    let mut header_payload = vec![0_u8; header_len];
    file.read_exact(&mut header_payload)?;
    let header_value: Value = serde_json::from_slice(&header_payload)?;
    let header = object(&header_value, relative_file)?;
    let data_start = 8_u64
        .checked_add(header_bytes)
        .ok_or_else(|| CheckpointError::Invalid("safetensors offset overflow".into()))?;

    for tensor in tensors {
        let metadata = header
            .get(*tensor)
            .ok_or_else(|| {
                CheckpointError::Invalid(format!(
                    "tensor {tensor} is absent from {relative_file} header"
                ))
            })?
            .as_object()
            .ok_or_else(|| CheckpointError::Invalid(format!("invalid metadata for {tensor}")))?;
        let offsets = array_field(metadata, "data_offsets")?;
        if offsets.len() != 2 {
            return Err(CheckpointError::Invalid(format!(
                "tensor {tensor} must have two data offsets"
            )));
        }
        let start = json_u64(&offsets[0], "tensor start offset")?;
        let end = json_u64(&offsets[1], "tensor end offset")?;
        let bytes = end.checked_sub(start).ok_or_else(|| {
            CheckpointError::Invalid(format!("tensor {tensor} has reversed offsets"))
        })?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| CheckpointError::Invalid("tensor offset overflow".into()))?;
        if absolute_end > file_bytes {
            return Err(CheckpointError::Invalid(format!(
                "tensor {tensor} exceeds {relative_file}"
            )));
        }
        let absolute_offset = data_start
            .checked_add(start)
            .ok_or_else(|| CheckpointError::Invalid("tensor offset overflow".into()))?;
        let dtype = metadata
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| CheckpointError::Invalid(format!("tensor {tensor} has no dtype")))?
            .to_owned();
        let shape = array_field(metadata, "shape")?
            .iter()
            .map(|dimension| json_u64(dimension, "tensor dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        // Safetensors represents a scalar as rank zero (`shape: []`). Qwen's
        // NVFP4 input_scale and weight_scale_2 tensors use that valid form.
        if shape.contains(&0) || bytes == 0 {
            return Err(CheckpointError::Invalid(format!(
                "tensor {tensor} has a zero dimension or empty payload"
            )));
        }
        let class = classify_tensor(tensor);
        if class == TensorClass::PleNvme {
            expect_string(metadata, "dtype", "F8_E4M3")?;
            validate_ple_shape(metadata, tensor, bytes)?;
        }
        match class {
            TensorClass::Resident => plan.resident.add(bytes)?,
            TensorClass::PleNvme => plan.ple_nvme.add(bytes)?,
            TensorClass::MtpDeferred => plan.mtp_deferred.add(bytes)?,
            TensorClass::VisionIgnored => plan.vision_ignored.add(bytes)?,
        }
        let location = SafetensorLocation {
            relative_file: PathBuf::from(relative_file),
            absolute_offset,
            data_bytes: bytes,
            dtype,
            shape,
        };
        if tensor_index.insert((*tensor).to_owned(), location).is_some() {
            return Err(CheckpointError::Invalid(format!(
                "checkpoint tensor is duplicated: {tensor}"
            )));
        }
    }
    Ok(())
}

fn validate_ple_shape(
    metadata: &Map<String, Value>,
    tensor: &str,
    bytes: u64,
) -> Result<(), CheckpointError> {
    let shape = array_field(metadata, "shape")?;
    if shape.len() != 2 {
        return Err(CheckpointError::Invalid(format!(
            "PLE tensor {tensor} must be rank two"
        )));
    }
    let rows = json_u64(&shape[0], "PLE rows")?;
    let width = json_u64(&shape[1], "PLE width")?;
    if rows == 0 || width != 160 || rows.checked_mul(width) != Some(bytes) {
        return Err(CheckpointError::Invalid(format!(
            "PLE tensor {tensor} has invalid shape [{rows}, {width}] for {bytes} bytes"
        )));
    }
    Ok(())
}

fn classify_tensor(name: &str) -> TensorClass {
    if name.contains(".ngram_embedding.shard_") && name.ends_with(".weight") {
        TensorClass::PleNvme
    } else if name.starts_with("model.visual.") || name.contains(".visual.") {
        TensorClass::VisionIgnored
    } else if name.starts_with("mtp.") || name.contains(".mtp.") {
        TensorClass::MtpDeferred
    } else {
        TensorClass::Resident
    }
}

fn validate_relative_path(value: &str) -> Result<(), CheckpointError> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::RootDir))
    {
        return Err(CheckpointError::Invalid(format!(
            "unsafe checkpoint shard path: {value}"
        )));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, CheckpointError> {
    value
        .as_object()
        .ok_or_else(|| CheckpointError::Invalid(format!("{label} must be a JSON object")))
}

fn object_field<'a>(
    value: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Map<String, Value>, CheckpointError> {
    object(
        value
            .get(field)
            .ok_or_else(|| CheckpointError::Invalid(format!("missing {field}")))?,
        field,
    )
}

fn array_field<'a>(
    value: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Vec<Value>, CheckpointError> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| CheckpointError::Invalid(format!("missing or invalid {field}")))
}

fn integer_field(value: &Map<String, Value>, field: &str) -> Result<u64, CheckpointError> {
    value
        .get(field)
        .ok_or_else(|| CheckpointError::Invalid(format!("missing {field}")))
        .and_then(|item| json_u64(item, field))
}

fn bool_field(value: &Map<String, Value>, field: &str) -> Result<bool, CheckpointError> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| CheckpointError::Invalid(format!("missing or invalid {field}")))
}

fn expect_integer(
    value: &Map<String, Value>,
    field: &str,
    expected: u64,
) -> Result<u64, CheckpointError> {
    let actual = integer_field(value, field)?;
    if actual != expected {
        return Err(CheckpointError::Invalid(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(actual)
}

fn expect_string(
    value: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), CheckpointError> {
    let actual = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| CheckpointError::Invalid(format!("missing or invalid string {field}")))?;
    if actual != expected {
        return Err(CheckpointError::Invalid(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn json_u64(value: &Value, label: &str) -> Result<u64, CheckpointError> {
    value
        .as_u64()
        .ok_or_else(|| CheckpointError::Invalid(format!("{label} must be a non-negative integer")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "sparkserve-checkpoint-test-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn builds_strict_flash_next_load_plan_from_headers_only() {
        let directory = TestDir::new();
        write_valid_checkpoint(&directory.0);
        let checkpoint =
            load_flash_next_checkpoint(&directory.0).expect("valid checkpoint index");
        let plan = &checkpoint.plan;
        assert_eq!(plan.files, 2);
        assert_eq!(
            plan.resident,
            TensorStats {
                tensors: 1,
                bytes: 8
            }
        );
        assert_eq!(plan.ple_nvme.tensors, 128);
        assert_eq!(plan.ple_nvme.bytes, 128 * 160);
        assert_eq!(plan.total().tensors, 129);
        let embedding = checkpoint
            .tensor("model.language_model.embed_tokens.weight")
            .expect("embedding location");
        assert_eq!(embedding.relative_file, Path::new("resident.safetensors"));
        assert_eq!(embedding.dtype, "BF16");
        assert_eq!(embedding.shape, [2, 2]);
        assert_eq!(embedding.data_bytes, 8);
        assert!(embedding.absolute_offset >= 8);
    }

    #[test]
    fn rejects_non_nvfp4_checkpoint() {
        let directory = TestDir::new();
        write_valid_checkpoint(&directory.0);
        let config_path = directory.0.join(CONFIG_FILE);
        let mut config: Value =
            serde_json::from_slice(&fs::read(&config_path).expect("read config"))
                .expect("parse config");
        config["quantization_config"]["quant_algo"] = json!("FP8");
        fs::write(
            config_path,
            serde_json::to_vec(&config).expect("encode config"),
        )
        .expect("write config");
        let error = load_flash_next_plan(&directory.0).expect_err("must reject FP8");
        assert!(error.to_string().contains("quant_algo must be NVFP4"));
    }

    fn write_valid_checkpoint(root: &Path) {
        let mut layer_types = Vec::new();
        for layer in 0..48 {
            layer_types.push(if (layer + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            });
        }
        let config = json!({
            "architectures": [EXPECTED_ARCHITECTURE],
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "full_attention_interval": 4,
                "layer_types": layer_types,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20_000_000,
                "ple_embed_dim": 2560,
                "split_ngram_parts": 128,
                "ple_embedding_dtype": "float8_e4m3fn"
            },
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "producer": {"name": "modelopt"},
                "config_groups": {"group_0": {
                    "input_activations": {
                        "dynamic": false, "group_size": 16,
                        "num_bits": 4, "type": "float"
                    },
                    "weights": {
                        "dynamic": false, "group_size": 16,
                        "num_bits": 4, "type": "float"
                    }
                }}
            }
        });
        fs::write(
            root.join(CONFIG_FILE),
            serde_json::to_vec(&config).expect("encode config"),
        )
        .expect("write config");

        let resident_name = "model.language_model.embed_tokens.weight";
        write_safetensors(
            &root.join("resident.safetensors"),
            &[(resident_name.to_string(), "BF16", vec![2, 2], 8)],
        );
        let mut ple_tensors = Vec::new();
        for shard in 0..128 {
            ple_tensors.push((
                format!(
                    "model.language_model.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                ),
                "F8_E4M3",
                vec![1, 160],
                160,
            ));
        }
        write_safetensors(&root.join("ple.safetensors"), &ple_tensors);

        let mut weight_map = Map::new();
        weight_map.insert(resident_name.into(), json!("resident.safetensors"));
        for (name, _, _, _) in &ple_tensors {
            weight_map.insert(name.clone(), json!("ple.safetensors"));
        }
        let index = json!({"weight_map": weight_map});
        fs::write(
            root.join(INDEX_FILE),
            serde_json::to_vec(&index).expect("encode index"),
        )
        .expect("write index");
    }

    fn write_safetensors(path: &Path, tensors: &[(String, &str, Vec<u64>, u64)]) {
        let mut header = Map::new();
        let mut offset = 0_u64;
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                name.clone(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
        }
        let encoded = serde_json::to_vec(&header).expect("encode safetensors header");
        let mut output = Vec::with_capacity(8 + encoded.len() + offset as usize);
        output.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        output.extend_from_slice(&encoded);
        output.resize(output.len() + offset as usize, 0);
        fs::write(path, output).expect("write safetensors");
    }
}
