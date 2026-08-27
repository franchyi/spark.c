use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::path::{Component, Path};

const REVISION_HEX_DIGITS: usize = 40;
const SHA256_HEX_DIGITS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockedFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub role: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockedModel {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub architecture: String,
    pub format: String,
    pub quantization: String,
    pub inventory: String,
    pub checkpoint_bytes: u64,
    pub files: Vec<LockedFile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLock {
    pub schema_version: u64,
    pub mirror: String,
    pub models: Vec<LockedModel>,
}

impl ModelLock {
    pub fn model(&self, id: &str) -> Result<&LockedModel, ModelLockError> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| ModelLockError::Invalid(format!("model lock has no model {id}")))
    }
}

#[derive(Debug)]
pub enum ModelLockError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl Display for ModelLockError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ModelLockError {}

impl From<std::io::Error> for ModelLockError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ModelLockError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub fn load_model_lock(path: &Path) -> Result<ModelLock, ModelLockError> {
    parse_model_lock(serde_json::from_reader(File::open(path)?)?)
}

pub fn parse_model_lock(value: Value) -> Result<ModelLock, ModelLockError> {
    let root = object(&value, "model lock")?;
    let schema_version = unsigned(root, "schema_version")?;
    if schema_version != 1 {
        return invalid(format!("unsupported model lock schema {schema_version}"));
    }
    let mirror = string(root, "mirror")?;
    if !mirror.starts_with("https://") {
        return invalid("model mirror must use HTTPS");
    }
    let raw_models = array(root, "models")?;
    if raw_models.is_empty() {
        return invalid("model lock contains no models");
    }

    let mut ids = BTreeSet::new();
    let mut models = Vec::with_capacity(raw_models.len());
    for raw_model in raw_models {
        let model = parse_model(object(raw_model, "model entry")?)?;
        if !ids.insert(model.id.clone()) {
            return invalid(format!("duplicate model id {}", model.id));
        }
        models.push(model);
    }
    Ok(ModelLock {
        schema_version,
        mirror: mirror.to_owned(),
        models,
    })
}

fn parse_model(raw: &Map<String, Value>) -> Result<LockedModel, ModelLockError> {
    let id = string(raw, "id")?;
    if id.is_empty() {
        return invalid("model id must not be empty");
    }
    let revision = string(raw, "revision")?;
    if !lower_hex(revision, REVISION_HEX_DIGITS) {
        return invalid(format!("model {id} revision must be a 40-digit commit"));
    }
    let format = string(raw, "format")?;
    if !matches!(format, "modelopt_nvfp4_safetensors" | "gguf") {
        return invalid(format!("model {id} has unsupported format {format}"));
    }
    let inventory = string(raw, "inventory")?;
    if !matches!(inventory, "critical_metadata" | "complete_checkpoint") {
        return invalid(format!("model {id} has invalid inventory {inventory}"));
    }
    let checkpoint_bytes = unsigned(raw, "checkpoint_bytes")?;
    if checkpoint_bytes == 0 {
        return invalid(format!("model {id} checkpoint size must be positive"));
    }
    let raw_files = array(raw, "files")?;
    if raw_files.is_empty() {
        return invalid(format!("model {id} has no locked files"));
    }
    let mut paths = BTreeSet::new();
    let mut files = Vec::with_capacity(raw_files.len());
    for raw_file in raw_files {
        let file = parse_file(id, object(raw_file, "locked file")?)?;
        if !paths.insert(file.path.clone()) {
            return invalid(format!("model {id} has duplicate file {}", file.path));
        }
        files.push(file);
    }
    if inventory == "complete_checkpoint" {
        let inventory_bytes = files.iter().try_fold(0_u64, |total, file| {
            total.checked_add(file.size).ok_or_else(|| {
                ModelLockError::Invalid(format!("model {id} inventory size overflow"))
            })
        })?;
        if inventory_bytes != checkpoint_bytes {
            return invalid(format!(
                "model {id} inventory is {inventory_bytes} bytes, expected {checkpoint_bytes}"
            ));
        }
    }
    Ok(LockedModel {
        id: id.to_owned(),
        repo: string(raw, "repo")?.to_owned(),
        revision: revision.to_owned(),
        architecture: string(raw, "architecture")?.to_owned(),
        format: format.to_owned(),
        quantization: string(raw, "quantization")?.to_owned(),
        inventory: inventory.to_owned(),
        checkpoint_bytes,
        files,
    })
}

fn parse_file(model_id: &str, raw: &Map<String, Value>) -> Result<LockedFile, ModelLockError> {
    let path = string(raw, "path")?;
    if !safe_relative_path(path) {
        return invalid(format!("model {model_id} has unsafe file path {path}"));
    }
    let size = unsigned(raw, "size")?;
    let sha256 = string(raw, "sha256")?;
    if size == 0 || !lower_hex(sha256, SHA256_HEX_DIGITS) {
        return invalid(format!("model {model_id} has invalid integrity for {path}"));
    }
    Ok(LockedFile {
        path: path.to_owned(),
        size,
        sha256: sha256.to_owned(),
        role: string(raw, "role")?.to_owned(),
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, ModelLockError> {
    value
        .as_object()
        .ok_or_else(|| ModelLockError::Invalid(format!("{label} must be an object")))
}

fn string<'a>(value: &'a Map<String, Value>, key: &str) -> Result<&'a str, ModelLockError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ModelLockError::Invalid(format!("{key} must be a string")))
}

fn unsigned(value: &Map<String, Value>, key: &str) -> Result<u64, ModelLockError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelLockError::Invalid(format!("{key} must be an unsigned integer")))
}

fn array<'a>(value: &'a Map<String, Value>, key: &str) -> Result<&'a [Value], ModelLockError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| ModelLockError::Invalid(format!("{key} must be an array")))
}

fn lower_hex(value: &str, digits: usize) -> bool {
    value.len() == digits
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ModelLockError> {
    Err(ModelLockError::Invalid(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!({
            "schema_version": 1,
            "mirror": "https://hf-mirror.com",
            "models": [{
                "id": "glm",
                "repo": "owner/model",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "architecture": "GLM5NextForConditionalGeneration",
                "format": "gguf",
                "quantization": "UD-IQ3_XXS",
                "inventory": "complete_checkpoint",
                "checkpoint_bytes": 7,
                "files": [{
                    "path": "part-1.gguf",
                    "size": 7,
                    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "role": "gguf_shard"
                }]
            }]
        })
    }

    #[test]
    fn parses_complete_inventory() {
        let lock = parse_model_lock(fixture()).expect("valid model lock");
        let model = lock.model("glm").expect("locked model");
        assert_eq!(model.checkpoint_bytes, 7);
        assert_eq!(model.files.len(), 1);
    }

    #[test]
    fn rejects_unsafe_path() {
        let mut value = fixture();
        value["models"][0]["files"][0]["path"] = Value::String("../part.gguf".into());
        let error = parse_model_lock(value).expect_err("unsafe path must fail");
        assert!(error.to_string().contains("unsafe file path"));
    }

    #[test]
    fn rejects_incomplete_inventory() {
        let mut value = fixture();
        value["models"][0]["checkpoint_bytes"] = Value::from(8);
        let error = parse_model_lock(value).expect_err("size mismatch must fail");
        assert!(error.to_string().contains("inventory is 7 bytes"));
    }
}
