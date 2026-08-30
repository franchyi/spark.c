use crate::checkpoint::{Q27Checkpoint, Q27Error, TensorLocation};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt::{Display, Formatter};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

const ABI_VERSION: u32 = 1;

#[repr(C)]
struct NativeStatus {
    code: i32,
    message: *const c_char,
}

#[repr(C)]
struct NativeMapping {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeView {
    struct_size: u32,
    abi_version: u32,
    host_base: *const c_void,
    device_base: *const c_void,
    bytes: u64,
    page_bytes: u64,
    device_id: i32,
    reserved: u32,
}

unsafe extern "C" {
    fn q27_mapping_open(path: *const c_char, output: *mut *mut NativeMapping) -> NativeStatus;
    fn q27_mapping_get_view(mapping: *const NativeMapping, output: *mut NativeView) -> NativeStatus;
    fn q27_mapping_close(mapping: *mut NativeMapping) -> NativeStatus;
}

#[derive(Debug)]
pub enum MappingError {
    Checkpoint(Q27Error),
    Native(String),
    Invalid(String),
}

impl Display for MappingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(error) => write!(formatter, "{error}"),
            Self::Native(message) | Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for MappingError {}

impl From<Q27Error> for MappingError {
    fn from(error: Q27Error) -> Self { Self::Checkpoint(error) }
}

fn status(status: NativeStatus) -> Result<(), MappingError> {
    if status.code == 0 {
        return Ok(());
    }
    let message = if status.message.is_null() {
        format!("q27 native mapping failed with status {}", status.code)
    } else {
        // SAFETY: native status messages are static or thread-local and are
        // copied before the next call on this thread.
        unsafe { CStr::from_ptr(status.message) }.to_string_lossy().into_owned()
    };
    Err(MappingError::Native(message))
}

pub struct MappedFile {
    handle: NonNull<NativeMapping>,
    device_base: usize,
    bytes: u64,
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self, MappingError> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| MappingError::Invalid("q27 shard path contains NUL".into()))?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the C string and output slot are valid for this call.
        status(unsafe { q27_mapping_open(path.as_ptr(), &mut raw) })?;
        let handle = NonNull::new(raw)
            .ok_or_else(|| MappingError::Invalid("q27 mapping returned null".into()))?;
        let mut view = NativeView {
            struct_size: size_of::<NativeView>() as u32,
            abi_version: ABI_VERSION,
            host_base: std::ptr::null(),
            device_base: std::ptr::null(),
            bytes: 0,
            page_bytes: 0,
            device_id: -1,
            reserved: 0,
        };
        // SAFETY: handle ownership remains with this guard.
        if let Err(error) = status(unsafe { q27_mapping_get_view(handle.as_ptr(), &mut view) }) {
            // SAFETY: creation transferred the unique handle to us.
            let _ = unsafe { q27_mapping_close(handle.as_ptr()) };
            return Err(error);
        }
        if view.device_base.is_null() || view.bytes == 0 {
            // SAFETY: creation transferred the unique handle to us.
            let _ = unsafe { q27_mapping_close(handle.as_ptr()) };
            return Err(MappingError::Invalid("q27 mapping returned an invalid view".into()));
        }
        Ok(Self { handle, device_base: view.device_base as usize, bytes: view.bytes })
    }

    pub fn bytes(&self) -> u64 { self.bytes }

    pub fn device_address(&self, offset: u64, bytes: u64) -> Result<usize, MappingError> {
        let end = offset.checked_add(bytes)
            .ok_or_else(|| MappingError::Invalid("q27 mapped-file offset overflow".into()))?;
        if end > self.bytes {
            return Err(MappingError::Invalid("q27 view exceeds mapped file".into()));
        }
        self.device_base.checked_add(offset as usize)
            .ok_or_else(|| MappingError::Invalid("q27 device address overflow".into()))
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: the guard owns this handle exactly once.
        let _ = unsafe { q27_mapping_close(self.handle.as_ptr()) };
    }
}

pub struct MappedCheckpoint {
    checkpoint: Q27Checkpoint,
    shards: BTreeMap<PathBuf, MappedFile>,
}

impl MappedCheckpoint {
    pub fn open(root: &Path) -> Result<Self, MappingError> {
        let checkpoint = Q27Checkpoint::open(root)?;
        let mut shards = BTreeMap::new();
        for (_, tensor) in checkpoint.tensors() {
            if shards.contains_key(&tensor.relative_file) {
                continue;
            }
            let mapping = MappedFile::open(&checkpoint.plan().root.join(&tensor.relative_file))?;
            shards.insert(tensor.relative_file.clone(), mapping);
        }
        Ok(Self { checkpoint, shards })
    }

    pub fn checkpoint(&self) -> &Q27Checkpoint { &self.checkpoint }

    pub fn mapped_bytes(&self) -> u64 {
        self.shards.values().map(|shard| shard.bytes).sum()
    }

    pub fn device_address(&self, tensor: &TensorLocation) -> Result<usize, MappingError> {
        let shard = self.shards.get(&tensor.relative_file)
            .ok_or_else(|| MappingError::Invalid("q27 tensor shard is not mapped".into()))?;
        shard.device_address(tensor.absolute_offset, tensor.data_bytes)
    }
}
