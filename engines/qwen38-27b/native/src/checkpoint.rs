//! Strict, payload-free checkpoint validation for one Qwen3.8-27B export.
//!
//! This module intentionally does not implement a general safetensors loader.
//! It accepts the exact text-only graph used by the Spark capsule, records the
//! byte ranges that the native engine will mmap, and rejects every other graph
//! before touching tensor payloads.

use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const EXPECTED_REVISION: &str = "009632fef96dd349150baa780c984e62e70e91fe";
const EXPECTED_TENSORS: u64 = 2_191;
const EXPECTED_TEXT_TENSORS: u64 = 1_843;
const EXPECTED_MTP_TENSORS: u64 = 15;
const EXPECTED_VISION_TENSORS: u64 = 333;
const EXPECTED_TOTAL_BYTES: u64 = 23_749_063_264;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q27Config {
    pub architecture: String,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub layers: u64,
    pub gdn_layers: u64,
    pub attention_layers: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub gdn_key_heads: u64,
    pub gdn_value_heads: u64,
    pub gdn_key_dim: u64,
    pub gdn_value_dim: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub mtp_layers: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorStats {
    pub tensors: u64,
    pub bytes: u64,
}

impl TensorStats {
    fn add(&mut self, bytes: u64) -> Result<(), Q27Error> {
        self.tensors = self.tensors.checked_add(1).ok_or_else(|| invalid("tensor count overflow"))?;
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(|| invalid("tensor byte count overflow"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLocation {
    pub relative_file: PathBuf,
    pub absolute_offset: u64,
    pub data_bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q27Plan {
    pub root: PathBuf,
    pub revision: String,
    pub config: Q27Config,
    pub files: u64,
    pub body: TensorStats,
    pub roots: TensorStats,
    pub mtp: TensorStats,
    pub vision_ignored: TensorStats,
    pub total: TensorStats,
}

impl Q27Plan {
    pub fn runtime_bytes(&self) -> u64 {
        self.body.bytes + self.roots.bytes + self.mtp.bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q27Checkpoint {
    plan: Q27Plan,
    tensors: BTreeMap<String, TensorLocation>,
}

impl Q27Checkpoint {
    pub fn open(root: &Path) -> Result<Self, Q27Error> {
        let root = root.canonicalize()?;
        let revision = root.file_name().and_then(|part| part.to_str()).unwrap_or_default();
        if revision != EXPECTED_REVISION {
            return Err(invalid(format!(
                "checkpoint revision must be {EXPECTED_REVISION}, got {revision}"
            )));
        }
        let config_value: Value = serde_json::from_reader(File::open(root.join(CONFIG_FILE))?)?;
        let config = validate_config(&config_value)?;
        let expected = expected_text_tensors(&config);
        if expected.len() as u64 != EXPECTED_TEXT_TENSORS + EXPECTED_MTP_TENSORS {
            return Err(invalid("internal q27 tensor contract has the wrong size"));
        }

        let index_value: Value = serde_json::from_reader(File::open(root.join(INDEX_FILE))?)?;
        let index = object(&index_value, "checkpoint index")?;
        let metadata = object_field(index, "metadata")?;
        expect_integer(metadata, "total_size", EXPECTED_TOTAL_BYTES)?;
        expect_integer(metadata, "total_parameters", 18_164_649_200)?;
        let weight_map = object_field(index, "weight_map")?;
        if weight_map.len() as u64 != EXPECTED_TENSORS {
            return Err(invalid(format!(
                "checkpoint must contain {EXPECTED_TENSORS} tensors, found {}",
                weight_map.len()
            )));
        }

        let actual_text: BTreeSet<&str> = weight_map
            .keys()
            .map(String::as_str)
            .filter(|name| !name.starts_with("model.visual."))
            .collect();
        let expected_text: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
        if actual_text != expected_text {
            let missing = expected_text.difference(&actual_text).next().copied();
            let extra = actual_text.difference(&expected_text).next().copied();
            return Err(invalid(format!(
                "text tensor set mismatch (first missing={missing:?}, first unexpected={extra:?})"
            )));
        }

        let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (tensor, file_value) in weight_map {
            let relative_file = file_value
                .as_str()
                .ok_or_else(|| invalid(format!("tensor {tensor} has no shard file")))?;
            validate_relative_path(relative_file)?;
            by_file.entry(relative_file).or_default().push(tensor);
        }
        if by_file.len() != 3 {
            return Err(invalid(format!("checkpoint must contain 3 shards, found {}", by_file.len())));
        }

        let mut plan = Q27Plan {
            root: root.clone(),
            revision: revision.to_owned(),
            config,
            files: by_file.len() as u64,
            body: TensorStats::default(),
            roots: TensorStats::default(),
            mtp: TensorStats::default(),
            vision_ignored: TensorStats::default(),
            total: TensorStats::default(),
        };
        let mut tensors = BTreeMap::new();
        for (relative_file, names) in by_file {
            scan_shard(&root, relative_file, &names, &expected, &mut plan, &mut tensors)?;
        }
        validate_stats(&plan)?;
        Ok(Self { plan, tensors })
    }

    pub fn plan(&self) -> &Q27Plan {
        &self.plan
    }

    #[allow(dead_code)]
    pub fn tensor(&self, name: &str) -> Result<&TensorLocation, Q27Error> {
        self.tensors.get(name).ok_or_else(|| invalid(format!("missing tensor {name}")))
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorLocation)> {
        self.tensors.iter().map(|(name, location)| (name.as_str(), location))
    }
}

#[derive(Debug)]
pub enum Q27Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl Display for Q27Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Q27Error {}

impl From<std::io::Error> for Q27Error {
    fn from(error: std::io::Error) -> Self { Self::Io(error) }
}

impl From<serde_json::Error> for Q27Error {
    fn from(error: serde_json::Error) -> Self { Self::Json(error) }
}

fn invalid(message: impl Into<String>) -> Q27Error { Q27Error::Invalid(message.into()) }

fn validate_config(value: &Value) -> Result<Q27Config, Q27Error> {
    let root = object(value, "config")?;
    expect_string(root, "model_type", "qwen3_5")?;
    expect_string(root, "dtype", "bfloat16")?;
    let architectures = array_field(root, "architectures")?;
    let architecture = "Qwen3_5ForConditionalGeneration";
    if !architectures.iter().any(|item| item.as_str() == Some(architecture)) {
        return Err(invalid(format!("architecture must include {architecture}")));
    }
    let text = object_field(root, "text_config")?;
    expect_string(text, "model_type", "qwen3_5_text")?;
    expect_string(text, "dtype", "bfloat16")?;
    expect_bool(text, "attn_output_gate", true)?;
    expect_bool(text, "attention_bias", false)?;
    expect_integer(text, "full_attention_interval", 4)?;
    expect_integer(text, "linear_conv_kernel_dim", 4)?;
    let partial_rotary = number_field(text, "partial_rotary_factor")?;
    if partial_rotary != 0.25 {
        return Err(invalid(format!("partial_rotary_factor must be 0.25, got {partial_rotary}")));
    }
    let rope = object_field(text, "rope_parameters")?;
    expect_string(rope, "rope_type", "default")?;
    expect_integer(rope, "rope_theta", 10_000_000)?;
    expect_bool(rope, "mrope_interleaved", true)?;
    let section = array_field(rope, "mrope_section")?;
    if section != &vec![Value::from(11), Value::from(11), Value::from(10)] {
        return Err(invalid("mrope_section must be [11, 11, 10]"));
    }

    let layers = expect_integer(text, "num_hidden_layers", 64)?;
    let layer_types = array_field(text, "layer_types")?;
    if layer_types.len() as u64 != layers {
        return Err(invalid("layer_types must contain 64 entries"));
    }
    let mut gdn_layers = 0;
    let mut attention_layers = 0;
    for (layer, kind) in layer_types.iter().enumerate() {
        let expected = if (layer + 1) % 4 == 0 { "full_attention" } else { "linear_attention" };
        if kind.as_str() != Some(expected) {
            return Err(invalid(format!("layer {layer} must be {expected}")));
        }
        if expected == "full_attention" { attention_layers += 1; } else { gdn_layers += 1; }
    }
    validate_quantization(root, text)?;

    Ok(Q27Config {
        architecture: architecture.to_owned(),
        hidden_size: expect_integer(text, "hidden_size", 5_120)?,
        intermediate_size: expect_integer(text, "intermediate_size", 17_408)?,
        layers,
        gdn_layers,
        attention_layers,
        attention_heads: expect_integer(text, "num_attention_heads", 24)?,
        kv_heads: expect_integer(text, "num_key_value_heads", 4)?,
        head_dim: expect_integer(text, "head_dim", 256)?,
        gdn_key_heads: expect_integer(text, "linear_num_key_heads", 16)?,
        gdn_value_heads: expect_integer(text, "linear_num_value_heads", 48)?,
        gdn_key_dim: expect_integer(text, "linear_key_head_dim", 128)?,
        gdn_value_dim: expect_integer(text, "linear_value_head_dim", 128)?,
        vocab_size: expect_integer(text, "vocab_size", 248_320)?,
        max_position_embeddings: expect_integer(text, "max_position_embeddings", 262_144)?,
        mtp_layers: expect_integer(text, "mtp_num_hidden_layers", 1)?,
    })
}

fn validate_quantization(root: &Map<String, Value>, text: &Map<String, Value>) -> Result<(), Q27Error> {
    let quant = object_field(root, "quantization_config")?;
    expect_string(quant, "quant_method", "modelopt")?;
    expect_string(quant, "quant_algo", "MIXED_PRECISION")?;
    let producer = object_field(quant, "producer")?;
    expect_string(producer, "name", "modelopt")?;
    let kv = object_field(quant, "kv_cache_scheme")?;
    expect_bool(kv, "dynamic", false)?;
    expect_integer(kv, "num_bits", 8)?;
    expect_string(kv, "type", "float")?;
    let ignored = array_field(quant, "ignore")?;
    if !ignored.iter().any(|value| value.as_str() == Some("mtp*")) {
        return Err(invalid("quantization ignore list must include mtp*"));
    }

    let groups = object_field(quant, "config_groups")?;
    let fp8 = object_field(groups, "group_0")?;
    validate_quant_spec(fp8, 8, None)?;
    let nvfp4 = object_field(groups, "group_1")?;
    validate_quant_spec(nvfp4, 4, Some(16))?;

    let mut expected_fp8 = BTreeSet::new();
    let mut expected_nvfp4 = BTreeSet::new();
    for layer in 0..64 {
        let prefix = format!("model.language_model.layers.{layer}");
        if (layer + 1) % 4 == 0 {
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                expected_fp8.insert(format!("{prefix}.self_attn.{projection}"));
            }
        } else {
            for projection in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                expected_fp8.insert(format!("{prefix}.linear_attn.{projection}"));
            }
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            expected_nvfp4.insert(format!("{prefix}.mlp.{projection}"));
        }
    }
    expect_target_set(fp8, &expected_fp8, "FP8")?;
    expect_target_set(nvfp4, &expected_nvfp4, "NVFP4")?;
    expect_string(text, "mamba_ssm_dtype", "float32")?;
    Ok(())
}

fn validate_quant_spec(group: &Map<String, Value>, bits: u64, group_size: Option<u64>) -> Result<(), Q27Error> {
    for field in ["input_activations", "weights"] {
        let spec = object_field(group, field)?;
        expect_bool(spec, "dynamic", false)?;
        expect_integer(spec, "num_bits", bits)?;
        expect_string(spec, "type", "float")?;
        if let Some(size) = group_size { expect_integer(spec, "group_size", size)?; }
    }
    Ok(())
}

fn expect_target_set(group: &Map<String, Value>, expected: &BTreeSet<String>, label: &str) -> Result<(), Q27Error> {
    let actual: BTreeSet<String> = array_field(group, "targets")?
        .iter()
        .map(|item| item.as_str().map(str::to_owned).ok_or_else(|| invalid(format!("{label} target must be a string"))))
        .collect::<Result<_, _>>()?;
    if &actual != expected {
        return Err(invalid(format!("{label} target set does not match the q27 graph")));
    }
    Ok(())
}

fn expected_text_tensors(config: &Q27Config) -> BTreeMap<String, (&'static str, Vec<u64>)> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, dtype: &'static str, shape: &[u64]| {
        tensors.insert(name, (dtype, shape.to_vec()));
    };
    for layer in 0..config.layers {
        let prefix = format!("model.language_model.layers.{layer}");
        add(format!("{prefix}.input_layernorm.weight"), "BF16", &[5_120]);
        add(format!("{prefix}.post_attention_layernorm.weight"), "BF16", &[5_120]);
        if (layer + 1) % 4 == 0 {
            for (name, shape) in [
                ("q_proj", vec![12_288, 5_120]),
                ("k_proj", vec![1_024, 5_120]),
                ("v_proj", vec![1_024, 5_120]),
                ("o_proj", vec![5_120, 6_144]),
            ] {
                let base = format!("{prefix}.self_attn.{name}");
                add(format!("{base}.weight"), "F8_E4M3", &shape);
                add(format!("{base}.input_scale"), "F32", &[]);
                add(format!("{base}.weight_scale"), "F32", &[]);
            }
            add(format!("{prefix}.self_attn.q_norm.weight"), "BF16", &[256]);
            add(format!("{prefix}.self_attn.k_norm.weight"), "BF16", &[256]);
        } else {
            let base = format!("{prefix}.linear_attn");
            add(format!("{base}.A_log"), "BF16", &[48]);
            add(format!("{base}.dt_bias"), "BF16", &[48]);
            add(format!("{base}.conv1d.weight"), "BF16", &[10_240, 1, 4]);
            add(format!("{base}.in_proj_a.weight"), "BF16", &[48, 5_120]);
            add(format!("{base}.in_proj_b.weight"), "BF16", &[48, 5_120]);
            add(format!("{base}.norm.weight"), "BF16", &[128]);
            for (name, shape) in [
                ("in_proj_qkv", vec![10_240, 5_120]),
                ("in_proj_z", vec![6_144, 5_120]),
                ("out_proj", vec![5_120, 6_144]),
            ] {
                let projection = format!("{base}.{name}");
                add(format!("{projection}.weight"), "F8_E4M3", &shape);
                add(format!("{projection}.input_scale"), "F32", &[]);
                add(format!("{projection}.weight_scale"), "F32", &[]);
            }
        }
        add_nvfp4_mlp(&mut add, &format!("{prefix}.mlp"));
    }
    add("model.language_model.embed_tokens.weight".into(), "BF16", &[248_320, 5_120]);
    add("model.language_model.norm.weight".into(), "BF16", &[5_120]);
    add("lm_head.weight".into(), "BF16", &[248_320, 5_120]);
    for (name, shape) in [
        ("mtp.fc.weight", vec![5_120, 10_240]),
        ("mtp.layers.0.input_layernorm.weight", vec![5_120]),
        ("mtp.layers.0.mlp.down_proj.weight", vec![5_120, 17_408]),
        ("mtp.layers.0.mlp.gate_proj.weight", vec![17_408, 5_120]),
        ("mtp.layers.0.mlp.up_proj.weight", vec![17_408, 5_120]),
        ("mtp.layers.0.post_attention_layernorm.weight", vec![5_120]),
        ("mtp.layers.0.self_attn.k_norm.weight", vec![256]),
        ("mtp.layers.0.self_attn.k_proj.weight", vec![1_024, 5_120]),
        ("mtp.layers.0.self_attn.o_proj.weight", vec![5_120, 6_144]),
        ("mtp.layers.0.self_attn.q_norm.weight", vec![256]),
        ("mtp.layers.0.self_attn.q_proj.weight", vec![12_288, 5_120]),
        ("mtp.layers.0.self_attn.v_proj.weight", vec![1_024, 5_120]),
        ("mtp.norm.weight", vec![5_120]),
        ("mtp.pre_fc_norm_embedding.weight", vec![5_120]),
        ("mtp.pre_fc_norm_hidden.weight", vec![5_120]),
    ] {
        add(name.into(), "BF16", &shape);
    }
    tensors
}

fn add_nvfp4_mlp(add: &mut impl FnMut(String, &'static str, &[u64]), prefix: &str) {
    for (name, weight_shape, scale_shape) in [
        ("gate_proj", [17_408, 2_560], [17_408, 320]),
        ("up_proj", [17_408, 2_560], [17_408, 320]),
        ("down_proj", [5_120, 8_704], [5_120, 1_088]),
    ] {
        let base = format!("{prefix}.{name}");
        add(format!("{base}.weight"), "U8", &weight_shape);
        add(format!("{base}.weight_scale"), "F8_E4M3", &scale_shape);
        add(format!("{base}.weight_scale_2"), "F32", &[]);
        add(format!("{base}.input_scale"), "F32", &[]);
    }
}

fn scan_shard(
    root: &Path,
    relative_file: &str,
    names: &[&str],
    expected: &BTreeMap<String, (&'static str, Vec<u64>)>,
    plan: &mut Q27Plan,
    locations: &mut BTreeMap<String, TensorLocation>,
) -> Result<(), Q27Error> {
    let path = root.join(relative_file);
    let mut file = File::open(&path)?;
    let file_bytes = file.metadata()?.len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)?;
    let header_bytes = u64::from_le_bytes(length);
    if header_bytes == 0 || header_bytes > 256 * 1024 * 1024 {
        return Err(invalid(format!("implausible safetensors header in {relative_file}")));
    }
    let mut payload = vec![0_u8; usize::try_from(header_bytes).map_err(|_| invalid("header too large"))?];
    file.read_exact(&mut payload)?;
    let header_value: Value = serde_json::from_slice(&payload)?;
    let header = object(&header_value, relative_file)?;
    let data_start = 8_u64.checked_add(header_bytes).ok_or_else(|| invalid("offset overflow"))?;
    for name in names {
        let tensor = object(header.get(*name).ok_or_else(|| invalid(format!("{name} absent from {relative_file}")))?, name)?;
        let offsets = array_field(tensor, "data_offsets")?;
        if offsets.len() != 2 { return Err(invalid(format!("{name} must have two offsets"))); }
        let start = json_u64(&offsets[0], "tensor start")?;
        let end = json_u64(&offsets[1], "tensor end")?;
        let data_bytes = end.checked_sub(start).ok_or_else(|| invalid(format!("reversed offsets for {name}")))?;
        let absolute_offset = data_start.checked_add(start).ok_or_else(|| invalid("offset overflow"))?;
        let absolute_end = data_start.checked_add(end).ok_or_else(|| invalid("offset overflow"))?;
        if absolute_end > file_bytes { return Err(invalid(format!("{name} exceeds {relative_file}"))); }
        let dtype = string_field(tensor, "dtype")?.to_owned();
        let shape = array_field(tensor, "shape")?.iter().map(|v| json_u64(v, "shape dimension")).collect::<Result<Vec<_>, _>>()?;
        let calculated = tensor_bytes(&dtype, &shape)?;
        if calculated != data_bytes {
            return Err(invalid(format!("{name} declares {calculated} bytes but stores {data_bytes}")));
        }
        if let Some((wanted_dtype, wanted_shape)) = expected.get(*name) {
            if dtype != *wanted_dtype || shape != *wanted_shape {
                return Err(invalid(format!(
                    "{name} must be {wanted_dtype} {wanted_shape:?}, got {dtype} {shape:?}"
                )));
            }
        }
        let stats = if name.starts_with("model.visual.") {
            &mut plan.vision_ignored
        } else if name.starts_with("mtp.") {
            &mut plan.mtp
        } else if *name == "lm_head.weight" || name.starts_with("model.language_model.embed_tokens") || *name == "model.language_model.norm.weight" {
            &mut plan.roots
        } else {
            &mut plan.body
        };
        stats.add(data_bytes)?;
        plan.total.add(data_bytes)?;
        if locations.insert((*name).to_owned(), TensorLocation {
            relative_file: relative_file.into(), absolute_offset, data_bytes, dtype, shape,
        }).is_some() {
            return Err(invalid(format!("duplicate tensor {name}")));
        }
    }
    Ok(())
}

fn tensor_bytes(dtype: &str, shape: &[u64]) -> Result<u64, Q27Error> {
    let bytes: u64 = match dtype { "U8" | "F8_E4M3" => 1, "BF16" => 2, "F32" => 4, other => return Err(invalid(format!("unsupported dtype {other}"))) };
    shape.iter().try_fold(bytes, |size, dimension| {
        if *dimension == 0 { return Err(invalid("zero tensor dimension")); }
        size.checked_mul(*dimension).ok_or_else(|| invalid("tensor size overflow"))
    })
}

fn validate_stats(plan: &Q27Plan) -> Result<(), Q27Error> {
    for (label, actual, expected) in [
        ("body tensors", plan.body.tensors, EXPECTED_TEXT_TENSORS - 3),
        ("root tensors", plan.roots.tensors, 3),
        ("MTP tensors", plan.mtp.tensors, EXPECTED_MTP_TENSORS),
        ("vision tensors", plan.vision_ignored.tensors, EXPECTED_VISION_TENSORS),
        ("total tensors", plan.total.tensors, EXPECTED_TENSORS),
        ("body bytes", plan.body.bytes, 16_892_600_448),
        ("root bytes", plan.roots.bytes, 5_085_603_840),
        ("MTP bytes", plan.mtp.bytes, 849_398_784),
        ("vision bytes", plan.vision_ignored.bytes, 921_460_192),
        ("total bytes", plan.total.bytes, EXPECTED_TOTAL_BYTES),
    ] {
        if actual != expected { return Err(invalid(format!("{label} must be {expected}, got {actual}"))); }
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), Q27Error> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() || path.components().any(|part| matches!(part, Component::ParentDir | Component::RootDir)) {
        return Err(invalid(format!("unsafe checkpoint shard path: {value}")));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, Q27Error> {
    value.as_object().ok_or_else(|| invalid(format!("{label} must be a JSON object")))
}

fn object_field<'a>(value: &'a Map<String, Value>, field: &str) -> Result<&'a Map<String, Value>, Q27Error> {
    object(value.get(field).ok_or_else(|| invalid(format!("missing {field}")))?, field)
}

fn array_field<'a>(value: &'a Map<String, Value>, field: &str) -> Result<&'a Vec<Value>, Q27Error> {
    value.get(field).and_then(Value::as_array).ok_or_else(|| invalid(format!("missing or invalid {field}")))
}

fn string_field<'a>(value: &'a Map<String, Value>, field: &str) -> Result<&'a str, Q27Error> {
    value.get(field).and_then(Value::as_str).ok_or_else(|| invalid(format!("missing or invalid {field}")))
}

fn number_field(value: &Map<String, Value>, field: &str) -> Result<f64, Q27Error> {
    value.get(field).and_then(Value::as_f64).ok_or_else(|| invalid(format!("missing or invalid {field}")))
}

fn json_u64(value: &Value, label: &str) -> Result<u64, Q27Error> {
    value.as_u64().ok_or_else(|| invalid(format!("{label} must be a non-negative integer")))
}

fn expect_integer(value: &Map<String, Value>, field: &str, expected: u64) -> Result<u64, Q27Error> {
    let actual = value.get(field).ok_or_else(|| invalid(format!("missing {field}"))).and_then(|v| json_u64(v, field))?;
    if actual != expected { return Err(invalid(format!("{field} must be {expected}, got {actual}"))); }
    Ok(actual)
}

fn expect_string(value: &Map<String, Value>, field: &str, expected: &str) -> Result<(), Q27Error> {
    let actual = string_field(value, field)?;
    if actual != expected { return Err(invalid(format!("{field} must be {expected}, got {actual}"))); }
    Ok(())
}

fn expect_bool(value: &Map<String, Value>, field: &str, expected: bool) -> Result<(), Q27Error> {
    let actual = value.get(field).and_then(Value::as_bool).ok_or_else(|| invalid(format!("missing or invalid {field}")))?;
    if actual != expected { return Err(invalid(format!("{field} must be {expected}, got {actual}"))); }
    Ok(())
}
