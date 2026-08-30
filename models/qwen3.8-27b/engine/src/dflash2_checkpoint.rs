//! Revision-locked DFlash2 draft checkpoint validation and attachment table.
//!
//! This is deliberately not a generic safetensors loader. It accepts exactly
//! the one-file z-lab DFlash2 snapshot used by the Q27 Spark capsule, validates
//! all 81 BF16 tensor records before mapping the payload, and owns the mapping
//! for as long as its C-compatible weight table may be borrowed.

use crate::mapping::{MappedFile, MappingError};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const DFLASH2_ABI_VERSION: u32 = 1;
pub const DFLASH2_LAYERS: usize = 5;
pub const DFLASH2_TENSORS: usize = 81;
pub const DFLASH2_REPOSITORY: &str = "z-lab/Qwen3.8-27B-DFlash2";
pub const DFLASH2_REVISION: &str = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4";
pub const DFLASH2_MODEL_SHA256: &str =
    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c";
pub const DFLASH2_CONFIG_SHA256: &str =
    "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980";
const DFLASH2_CONFIG_CACHE_BLOB: &str = "79279cc5665bced6f3cdaa2095a2ffe819497b2e";
pub const DFLASH2_HEADER_BYTES: u64 = 8_928;
pub const DFLASH2_PAYLOAD_BYTES: u64 = 3_848_808_960;
pub const DFLASH2_FILE_BYTES: u64 = 3_848_817_896;

const CONFIG_FILE: &str = "config.json";
const MODEL_FILE: &str = "model.safetensors";
const HIDDEN: u64 = 5_120;
const INTERMEDIATE: u64 = 17_408;
const VOCAB: u64 = 248_320;
const SELECTOR_RANK: u64 = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorSpec {
    shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorLocation {
    absolute_offset: u64,
    bytes: u64,
}

#[derive(Debug)]
pub enum DFlash2CheckpointError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Mapping(MappingError),
    Invalid(String),
}

impl Display for DFlash2CheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Mapping(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for DFlash2CheckpointError {}

impl From<std::io::Error> for DFlash2CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for DFlash2CheckpointError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<MappingError> for DFlash2CheckpointError {
    fn from(error: MappingError) -> Self {
        Self::Mapping(error)
    }
}

fn invalid(message: impl Into<String>) -> DFlash2CheckpointError {
    DFlash2CheckpointError::Invalid(message.into())
}

/// Exact Rust representation of `q27_dflash2_weight_view`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlash2WeightView {
    pub data: *const c_void,
    pub bytes: u64,
}

impl DFlash2WeightView {
    const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            bytes: 0,
        }
    }
}

/// Exact Rust representation of `q27_dflash2_layer_weights`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlash2LayerWeights {
    pub input_norm: DFlash2WeightView,
    pub attention_conv_base: DFlash2WeightView,
    pub attention_conv_projection: DFlash2WeightView,
    pub q_proj: DFlash2WeightView,
    pub k_proj: DFlash2WeightView,
    pub v_proj: DFlash2WeightView,
    pub o_proj: DFlash2WeightView,
    pub q_norm: DFlash2WeightView,
    pub k_norm: DFlash2WeightView,
    pub post_attention_norm: DFlash2WeightView,
    pub mlp_conv_base: DFlash2WeightView,
    pub mlp_conv_projection: DFlash2WeightView,
    pub mlp_gate: DFlash2WeightView,
    pub mlp_up: DFlash2WeightView,
    pub mlp_down: DFlash2WeightView,
}

impl DFlash2LayerWeights {
    const fn empty() -> Self {
        Self {
            input_norm: DFlash2WeightView::empty(),
            attention_conv_base: DFlash2WeightView::empty(),
            attention_conv_projection: DFlash2WeightView::empty(),
            q_proj: DFlash2WeightView::empty(),
            k_proj: DFlash2WeightView::empty(),
            v_proj: DFlash2WeightView::empty(),
            o_proj: DFlash2WeightView::empty(),
            q_norm: DFlash2WeightView::empty(),
            k_norm: DFlash2WeightView::empty(),
            post_attention_norm: DFlash2WeightView::empty(),
            mlp_conv_base: DFlash2WeightView::empty(),
            mlp_conv_projection: DFlash2WeightView::empty(),
            mlp_gate: DFlash2WeightView::empty(),
            mlp_up: DFlash2WeightView::empty(),
            mlp_down: DFlash2WeightView::empty(),
        }
    }
}

/// Exact Rust representation of `q27_dflash2_weights`.
#[repr(C)]
#[derive(Debug)]
pub struct DFlash2Weights {
    pub struct_size: u32,
    pub abi_version: u32,
    pub context_projection: DFlash2WeightView,
    pub context_norm: DFlash2WeightView,
    pub final_norm: DFlash2WeightView,
    pub layers: [DFlash2LayerWeights; DFLASH2_LAYERS],
    pub selector_hidden_projection: DFlash2WeightView,
    pub selector_predecessor_codebook: DFlash2WeightView,
    pub selector_successor_codebook: DFlash2WeightView,
}

/// Owns the device-visible file mapping behind every pointer in `weights`.
pub struct DFlash2WeightPlan {
    root: PathBuf,
    weights: DFlash2Weights,
    mapping: MappedFile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DFlash2CheckpointEvidence {
    pub repository: &'static str,
    pub revision: &'static str,
    pub model_sha256: &'static str,
    pub config_sha256: &'static str,
    pub config_contract: &'static str,
    pub tensor_count: usize,
    pub file_bytes: u64,
    pub payload_bytes: u64,
}

impl DFlash2WeightPlan {
    /// Accept either the pinned snapshot directory or its model.safetensors
    /// symlink. The symlink itself is intentionally not canonicalized: its LFS
    /// blob name is part of the checkpoint identity proof.
    pub fn open(root_or_safetensors: &Path) -> Result<Self, DFlash2CheckpointError> {
        let root = resolve_checkpoint_root(root_or_safetensors)?;
        validate_snapshot_revision(&root)?;
        let config_path = root.join(CONFIG_FILE);
        let model_path = root.join(MODEL_FILE);
        validate_config_identity(&config_path)?;
        validate_lfs_identity(&model_path)?;
        validate_config(&serde_json::from_reader(File::open(&config_path)?)?)?;
        let locations = scan_safetensors(&model_path)?;

        // The file is mapped only after every payload-free contract check has
        // succeeded, so malformed metadata cannot reach q27_mapping_open.
        let mapping = MappedFile::open(&model_path)?;
        if mapping.bytes() != DFLASH2_FILE_BYTES {
            return Err(invalid(format!(
                "mapped DFlash2 file must contain {DFLASH2_FILE_BYTES} bytes, got {}",
                mapping.bytes()
            )));
        }
        let weights = build_weights(&locations, |location| {
            mapping.device_address(location.absolute_offset, location.bytes)
        })?;
        Ok(Self {
            root,
            weights,
            mapping,
        })
    }

    pub fn weights(&self) -> &DFlash2Weights {
        &self.weights
    }

    /// Pointer passed directly to `q27_dflash2_validate_weights` or model args.
    pub fn weights_ptr(&self) -> *const DFlash2Weights {
        &self.weights
    }

    pub fn checkpoint_root(&self) -> &Path {
        &self.root
    }

    pub fn mapped_bytes(&self) -> u64 {
        self.mapping.bytes()
    }

    pub fn evidence(&self) -> DFlash2CheckpointEvidence {
        DFlash2CheckpointEvidence {
            repository: DFLASH2_REPOSITORY,
            revision: DFLASH2_REVISION,
            model_sha256: DFLASH2_MODEL_SHA256,
            config_sha256: DFLASH2_CONFIG_SHA256,
            config_contract: "q27.dflash2.config.exact.v1",
            tensor_count: DFLASH2_TENSORS,
            file_bytes: DFLASH2_FILE_BYTES,
            payload_bytes: DFLASH2_PAYLOAD_BYTES,
        }
    }
}

fn validate_config_identity(config: &Path) -> Result<(), DFlash2CheckpointError> {
    let metadata = fs::symlink_metadata(config).map_err(|error| {
        invalid(format!("cannot inspect DFlash2 config {}: {error}", config.display()))
    })?;
    if !metadata.file_type().is_symlink() {
        return Err(invalid(
            "DFlash2 config identity requires the pinned Hugging Face cache symlink",
        ));
    }
    let target = fs::read_link(config)?;
    let blob = target
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or_default();
    if blob != DFLASH2_CONFIG_CACHE_BLOB {
        return Err(invalid(format!(
            "DFlash2 config cache blob must be {DFLASH2_CONFIG_CACHE_BLOB}, got {blob}"
        )));
    }
    Ok(())
}

fn resolve_checkpoint_root(path: &Path) -> Result<PathBuf, DFlash2CheckpointError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        return Ok(path.canonicalize()?);
    }
    if path.file_name().and_then(|part| part.to_str()) != Some(MODEL_FILE) {
        return Err(invalid(format!(
            "DFlash2 checkpoint input must be a snapshot directory or {MODEL_FILE}"
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| invalid("DFlash2 model path has no snapshot directory"))?;
    Ok(parent.canonicalize()?)
}

fn validate_snapshot_revision(root: &Path) -> Result<(), DFlash2CheckpointError> {
    let revision = root
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or_default();
    if revision != DFLASH2_REVISION {
        return Err(invalid(format!(
            "DFlash2 checkpoint revision must be {DFLASH2_REVISION}, got {revision}"
        )));
    }
    Ok(())
}

fn validate_lfs_identity(model: &Path) -> Result<(), DFlash2CheckpointError> {
    let metadata = fs::symlink_metadata(model).map_err(|error| {
        invalid(format!("cannot inspect DFlash2 model {}: {error}", model.display()))
    })?;
    if !metadata.file_type().is_symlink() {
        return Err(invalid(
            "DFlash2 model identity requires the pinned Hugging Face LFS symlink; copied regular files need a separately audited SHA-256 path",
        ));
    }
    let target = fs::read_link(model)?;
    let blob = target
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or_default();
    if blob != DFLASH2_MODEL_SHA256 {
        return Err(invalid(format!(
            "DFlash2 LFS blob must be {DFLASH2_MODEL_SHA256}, got {blob}"
        )));
    }
    Ok(())
}

fn expected_config() -> Value {
    json!({
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": null,
        "is_causal": false,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "initializer_range": 0.02,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention"
        ],
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 64,
        "pad_token_id": 248044,
        "rms_norm_eps": 0.000001,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048,
        "tie_word_embeddings": false,
        "transformers_version": "5.15.0",
        "use_cache": true,
        "use_sliding_window": true,
        "vocab_size": 248320,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
        }
    })
}

fn validate_config(actual: &Value) -> Result<(), DFlash2CheckpointError> {
    let expected = expected_config();
    if actual != &expected {
        let actual_keys = actual
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>());
        let expected_keys = expected
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>());
        return Err(invalid(format!(
            "DFlash2 config does not match the pinned contract (actual keys={actual_keys:?}, expected keys={expected_keys:?})"
        )));
    }
    Ok(())
}

fn expected_tensors() -> BTreeMap<String, TensorSpec> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        tensors.insert(
            name,
            TensorSpec {
                shape: shape.to_vec(),
            },
        );
    };
    add(
        "candidate_selector.hidden_projection.weight".into(),
        &[SELECTOR_RANK, HIDDEN],
    );
    add(
        "candidate_selector.predecessor_codebook".into(),
        &[VOCAB, SELECTOR_RANK],
    );
    add(
        "candidate_selector.successor_codebook".into(),
        &[VOCAB, SELECTOR_RANK],
    );
    add("fc.weight".into(), &[HIDDEN, 5 * HIDDEN]);
    add("hidden_norm.weight".into(), &[HIDDEN]);
    add("norm.weight".into(), &[HIDDEN]);
    for layer in 0..DFLASH2_LAYERS {
        let prefix = format!("layers.{layer}");
        add(
            format!("{prefix}.attention_conv.base_kernel"),
            &[2, 2, HIDDEN],
        );
        add(
            format!("{prefix}.attention_conv.kernel_projection.weight"),
            &[1_280, HIDDEN],
        );
        add(format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
        add(
            format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        add(
            format!("{prefix}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(
            format!("{prefix}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(
            format!("{prefix}.mlp_conv.base_kernel"),
            &[2, 2, HIDDEN],
        );
        add(
            format!("{prefix}.mlp_conv.kernel_projection.weight"),
            &[1_280, HIDDEN],
        );
        add(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
        );
        add(format!("{prefix}.self_attn.k_norm.weight"), &[128]);
        add(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[1_024, HIDDEN],
        );
        add(
            format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, 4_096],
        );
        add(format!("{prefix}.self_attn.q_norm.weight"), &[128]);
        add(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[4_096, HIDDEN],
        );
        add(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[1_024, HIDDEN],
        );
    }
    tensors
}

fn scan_safetensors(
    path: &Path,
) -> Result<BTreeMap<String, TensorLocation>, DFlash2CheckpointError> {
    let mut file = File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes != DFLASH2_HEADER_BYTES {
        return Err(invalid(format!(
            "DFlash2 header must contain {DFLASH2_HEADER_BYTES} bytes, got {header_bytes}"
        )));
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| invalid("DFlash2 safetensors header does not fit usize"))?;
    let mut raw_header = vec![0_u8; header_len];
    file.read_exact(&mut raw_header)?;
    let header_value: Value = serde_json::from_slice(&raw_header)?;
    let header = header_value
        .as_object()
        .ok_or_else(|| invalid("DFlash2 safetensors header must be a JSON object"))?;
    validate_header_map(header, header_bytes, file_bytes)
}

fn validate_header_map(
    header: &Map<String, Value>,
    header_bytes: u64,
    file_bytes: u64,
) -> Result<BTreeMap<String, TensorLocation>, DFlash2CheckpointError> {
    if header_bytes != DFLASH2_HEADER_BYTES {
        return Err(invalid("DFlash2 safetensors header byte count changed"));
    }
    if file_bytes != DFLASH2_FILE_BYTES {
        return Err(invalid(format!(
            "DFlash2 model must contain {DFLASH2_FILE_BYTES} bytes, got {file_bytes}"
        )));
    }
    if let Some(metadata) = header.get("__metadata__") {
        if !metadata.is_null()
            && !matches!(metadata.as_object(), Some(object) if object.is_empty())
        {
            return Err(invalid("DFlash2 safetensors metadata must be absent or empty"));
        }
    }

    let expected = expected_tensors();
    if expected.len() != DFLASH2_TENSORS {
        return Err(invalid("internal DFlash2 tensor contract is not 81 tensors"));
    }
    let actual_names = header
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<std::collections::BTreeSet<_>>();
    if actual_names != expected_names {
        let missing = expected_names.difference(&actual_names).next();
        let extra = actual_names.difference(&expected_names).next();
        return Err(invalid(format!(
            "DFlash2 tensor set mismatch (first missing={missing:?}, first unexpected={extra:?})"
        )));
    }

    let data_start = 8_u64
        .checked_add(header_bytes)
        .ok_or_else(|| invalid("DFlash2 data-start overflow"))?;
    let mut payload_ranges = Vec::with_capacity(DFLASH2_TENSORS);
    let mut locations = BTreeMap::new();
    for (name, spec) in &expected {
        let entry = header
            .get(name)
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} metadata is not an object")))?;
        let dtype = entry.get("dtype").and_then(Value::as_str);
        if dtype != Some("BF16") {
            return Err(invalid(format!(
                "DFlash2 tensor {name} must be BF16, got {dtype:?}"
            )));
        }
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} has no shape")))?
            .iter()
            .map(|dimension| {
                dimension.as_u64().ok_or_else(|| {
                    invalid(format!("DFlash2 tensor {name} has an invalid shape dimension"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape != spec.shape {
            return Err(invalid(format!(
                "DFlash2 tensor {name} must have shape {:?}, got {shape:?}",
                spec.shape
            )));
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} has no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(invalid(format!(
                "DFlash2 tensor {name} must have two data offsets"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} start is invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} end is invalid")))?;
        let stored_bytes = end
            .checked_sub(start)
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} offsets are reversed")))?;
        let expected_bytes = bf16_bytes(&shape)?;
        if stored_bytes != expected_bytes {
            return Err(invalid(format!(
                "DFlash2 tensor {name} stores {stored_bytes} bytes, expected {expected_bytes}"
            )));
        }
        let absolute_offset = data_start
            .checked_add(start)
            .ok_or_else(|| invalid("DFlash2 absolute tensor offset overflow"))?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| invalid("DFlash2 absolute tensor end overflow"))?;
        if absolute_end > file_bytes {
            return Err(invalid(format!("DFlash2 tensor {name} exceeds the model file")));
        }
        payload_ranges.push((start, end, name.as_str()));
        locations.insert(
            name.clone(),
            TensorLocation {
                absolute_offset,
                bytes: stored_bytes,
            },
        );
    }

    payload_ranges.sort_unstable_by_key(|range| range.0);
    let mut cursor = 0_u64;
    for (start, end, name) in payload_ranges {
        if start != cursor {
            return Err(invalid(format!(
                "DFlash2 payload is not exactly contiguous before {name}: expected {cursor}, got {start}"
            )));
        }
        cursor = end;
    }
    if cursor != DFLASH2_PAYLOAD_BYTES {
        return Err(invalid(format!(
            "DFlash2 payload must contain {DFLASH2_PAYLOAD_BYTES} bytes, got {cursor}"
        )));
    }
    let expected_file_bytes = data_start
        .checked_add(cursor)
        .ok_or_else(|| invalid("DFlash2 file-size overflow"))?;
    if expected_file_bytes != file_bytes {
        return Err(invalid("DFlash2 file has missing or trailing payload bytes"));
    }
    Ok(locations)
}

fn bf16_bytes(shape: &[u64]) -> Result<u64, DFlash2CheckpointError> {
    shape.iter().try_fold(2_u64, |bytes, dimension| {
        if *dimension == 0 {
            return Err(invalid("DFlash2 tensors may not have zero dimensions"));
        }
        bytes
            .checked_mul(*dimension)
            .ok_or_else(|| invalid("DFlash2 tensor size overflow"))
    })
}

fn build_weights<E>(
    locations: &BTreeMap<String, TensorLocation>,
    mut address: impl FnMut(&TensorLocation) -> Result<usize, E>,
) -> Result<DFlash2Weights, DFlash2CheckpointError>
where
    DFlash2CheckpointError: From<E>,
{
    fn view<E>(
        locations: &BTreeMap<String, TensorLocation>,
        name: &str,
        address: &mut impl FnMut(&TensorLocation) -> Result<usize, E>,
    ) -> Result<DFlash2WeightView, DFlash2CheckpointError>
    where
        DFlash2CheckpointError: From<E>,
    {
        let location = locations
            .get(name)
            .ok_or_else(|| invalid(format!("DFlash2 tensor {name} is absent from the validated plan")))?;
        let data = address(location).map_err(DFlash2CheckpointError::from)?;
        if data == 0 {
            return Err(invalid(format!("DFlash2 tensor {name} mapped to null")));
        }
        Ok(DFlash2WeightView {
            data: data as *const c_void,
            bytes: location.bytes,
        })
    }

    let mut layers = [DFlash2LayerWeights::empty(); DFLASH2_LAYERS];
    for (layer, output) in layers.iter_mut().enumerate() {
        let prefix = format!("layers.{layer}");
        output.input_norm = view(
            locations,
            &format!("{prefix}.input_layernorm.weight"),
            &mut address,
        )?;
        output.attention_conv_base = view(
            locations,
            &format!("{prefix}.attention_conv.base_kernel"),
            &mut address,
        )?;
        output.attention_conv_projection = view(
            locations,
            &format!("{prefix}.attention_conv.kernel_projection.weight"),
            &mut address,
        )?;
        output.q_proj = view(
            locations,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &mut address,
        )?;
        output.k_proj = view(
            locations,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &mut address,
        )?;
        output.v_proj = view(
            locations,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &mut address,
        )?;
        output.o_proj = view(
            locations,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &mut address,
        )?;
        output.q_norm = view(
            locations,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &mut address,
        )?;
        output.k_norm = view(
            locations,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &mut address,
        )?;
        output.post_attention_norm = view(
            locations,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &mut address,
        )?;
        output.mlp_conv_base = view(
            locations,
            &format!("{prefix}.mlp_conv.base_kernel"),
            &mut address,
        )?;
        output.mlp_conv_projection = view(
            locations,
            &format!("{prefix}.mlp_conv.kernel_projection.weight"),
            &mut address,
        )?;
        output.mlp_gate = view(
            locations,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &mut address,
        )?;
        output.mlp_up = view(
            locations,
            &format!("{prefix}.mlp.up_proj.weight"),
            &mut address,
        )?;
        output.mlp_down = view(
            locations,
            &format!("{prefix}.mlp.down_proj.weight"),
            &mut address,
        )?;
    }

    Ok(DFlash2Weights {
        struct_size: size_of::<DFlash2Weights>() as u32,
        abi_version: DFLASH2_ABI_VERSION,
        context_projection: view(locations, "fc.weight", &mut address)?,
        context_norm: view(locations, "hidden_norm.weight", &mut address)?,
        final_norm: view(locations, "norm.weight", &mut address)?,
        layers,
        selector_hidden_projection: view(
            locations,
            "candidate_selector.hidden_projection.weight",
            &mut address,
        )?,
        selector_predecessor_codebook: view(
            locations,
            "candidate_selector.predecessor_codebook",
            &mut address,
        )?,
        selector_successor_codebook: view(
            locations,
            "candidate_selector.successor_codebook",
            &mut address,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_header() -> Map<String, Value> {
        let mut header = Map::new();
        let mut cursor = 0_u64;
        for (name, spec) in expected_tensors() {
            let bytes = bf16_bytes(&spec.shape).expect("valid contract shape");
            header.insert(
                name,
                json!({
                    "dtype": "BF16",
                    "shape": spec.shape,
                    "data_offsets": [cursor, cursor + bytes]
                }),
            );
            cursor += bytes;
        }
        assert_eq!(cursor, DFLASH2_PAYLOAD_BYTES);
        header
    }

    fn valid_locations() -> BTreeMap<String, TensorLocation> {
        validate_header_map(
            &valid_header(),
            DFLASH2_HEADER_BYTES,
            DFLASH2_FILE_BYTES,
        )
        .expect("synthetic exact header")
    }

    #[test]
    fn exact_contract_has_81_tensors_and_expected_payload() {
        assert_eq!(expected_tensors().len(), DFLASH2_TENSORS);
        assert_eq!(valid_locations().len(), DFLASH2_TENSORS);
    }

    #[test]
    fn rejects_missing_or_extra_tensor() {
        let mut missing = valid_header();
        let first = missing.keys().next().cloned().unwrap();
        missing.remove(&first);
        assert!(
            validate_header_map(&missing, DFLASH2_HEADER_BYTES, DFLASH2_FILE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("tensor set mismatch")
        );

        let mut extra = valid_header();
        extra.insert(
            "unexpected.weight".into(),
            json!({"dtype":"BF16", "shape":[1], "data_offsets":[0,2]}),
        );
        assert!(
            validate_header_map(&extra, DFLASH2_HEADER_BYTES, DFLASH2_FILE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("tensor set mismatch")
        );
    }

    #[test]
    fn rejects_dtype_shape_and_payload_gap() {
        let name = expected_tensors().keys().next().cloned().unwrap();

        let mut dtype = valid_header();
        dtype.get_mut(&name).unwrap()["dtype"] = Value::String("F16".into());
        assert!(
            validate_header_map(&dtype, DFLASH2_HEADER_BYTES, DFLASH2_FILE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("must be BF16")
        );

        let mut shape = valid_header();
        shape.get_mut(&name).unwrap()["shape"] = json!([1]);
        assert!(
            validate_header_map(&shape, DFLASH2_HEADER_BYTES, DFLASH2_FILE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("must have shape")
        );

        let mut gap = valid_header();
        let offsets = gap
            .get(&name)
            .unwrap()
            .get("data_offsets")
            .unwrap()
            .as_array()
            .unwrap();
        let start = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        gap.get_mut(&name).unwrap()["data_offsets"] = json!([start + 2, end + 2]);
        assert!(
            validate_header_map(&gap, DFLASH2_HEADER_BYTES, DFLASH2_FILE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("not exactly contiguous")
        );
    }

    #[test]
    fn ffi_table_matches_native_layout_and_maps_every_view() {
        assert_eq!(size_of::<DFlash2WeightView>(), 16);
        assert_eq!(size_of::<DFlash2LayerWeights>(), 240);
        assert_eq!(size_of::<DFlash2Weights>(), 1_304);

        let locations = valid_locations();
        let weights = build_weights::<DFlash2CheckpointError>(&locations, |location| {
            Ok(0x1_0000usize + usize::try_from(location.absolute_offset).unwrap())
        })
        .expect("build attachment table");
        assert_eq!(weights.struct_size, 1_304);
        assert_eq!(weights.abi_version, DFLASH2_ABI_VERSION);
        let mut views = vec![
            weights.context_projection,
            weights.context_norm,
            weights.final_norm,
            weights.selector_hidden_projection,
            weights.selector_predecessor_codebook,
            weights.selector_successor_codebook,
        ];
        for layer in weights.layers {
            views.extend([
                layer.input_norm,
                layer.attention_conv_base,
                layer.attention_conv_projection,
                layer.q_proj,
                layer.k_proj,
                layer.v_proj,
                layer.o_proj,
                layer.q_norm,
                layer.k_norm,
                layer.post_attention_norm,
                layer.mlp_conv_base,
                layer.mlp_conv_projection,
                layer.mlp_gate,
                layer.mlp_up,
                layer.mlp_down,
            ]);
        }
        assert_eq!(views.len(), DFLASH2_TENSORS);
        assert!(views.iter().all(|view| !view.data.is_null() && view.bytes > 0));
    }

    #[test]
    fn config_is_exact_not_compatible_looking() {
        let expected = expected_config();
        validate_config(&expected).expect("exact config");
        let mut changed = expected;
        changed["dflash_config"]["selector_top_k"] = Value::from(8);
        assert!(validate_config(&changed).is_err());
    }
}
