//! Safe ownership around the native GB10 coherent-region handle.
//!
//! The C++ layer performs `mmap`, CUDA host registration, and device-pointer
//! lookup. Rust owns the handle lifetime and exposes the two aliases only at an
//! explicit boundary; donor kernels never allocate or register memory.

use std::ffi::CStr;
#[cfg(unix)]
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ffi::{
    CoherentRegion, CoherentRegionConfig, CoherentRegionView, Status,
    flash_coherent_region_create, flash_coherent_region_destroy,
    flash_coherent_region_view,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoherentRegionError {
    pub code: i32,
    pub message: String,
}

impl CoherentRegionError {
    fn native(status: Status) -> Self {
        let message = if status.message.is_null() {
            "native coherent-region error".to_owned()
        } else {
            // SAFETY: the fabric ABI returns a NUL-terminated static or
            // thread-local message that is valid until the next native call on
            // this thread. Copy it before making another call.
            unsafe { CStr::from_ptr(status.message) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            code: status.code,
            message,
        }
    }

    fn invalid_view(message: &'static str) -> Self {
        Self {
            code: -1,
            message: message.to_owned(),
        }
    }
}

impl Display for CoherentRegionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} (native status {})", self.message, self.code)
    }
}

impl std::error::Error for CoherentRegionError {}

/// Unique owner of an `mmap`-backed, CUDA-addressable coherent region.
///
/// The owner is intentionally neither `Send` nor `Sync`: its CUDA registration
/// belongs to the creating thread's active device/context. Higher-level Rust
/// schedulers lend fixed offsets from it and must drain their CUDA events before
/// this value is dropped.
pub struct CoherentRegionOwner {
    handle: Option<NonNull<CoherentRegion>>,
    view: CoherentRegionView,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl CoherentRegionOwner {
    pub fn create(config: &CoherentRegionConfig) -> Result<Self, CoherentRegionError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `config` and the output slot follow fabric ABI version one;
        // ownership of a successful non-null handle transfers to this guard.
        status_result(unsafe { flash_coherent_region_create(config, &mut raw) })?;
        let handle = NonNull::new(raw).ok_or_else(|| {
            CoherentRegionError::invalid_view("native fabric returned a null region")
        })?;
        let mut view = CoherentRegionView::empty();
        // SAFETY: the newly created handle remains owned and alive here.
        if let Err(error) =
            status_result(unsafe { flash_coherent_region_view(handle.as_ptr(), &mut view) })
        {
            // SAFETY: creation transferred unique ownership to this function.
            let _ = unsafe { flash_coherent_region_destroy(handle.as_ptr()) };
            return Err(error);
        }
        if view.host_pointer.is_null()
            || view.device_pointer.is_null()
            || view.payload_bytes == 0
            || view.payload_bytes > view.mapped_bytes
            || view.required_alignment == 0
            || !view.required_alignment.is_power_of_two()
            || view.page_bytes == 0
            || !view.page_bytes.is_power_of_two()
        {
            // SAFETY: this function still uniquely owns the handle.
            let _ = unsafe { flash_coherent_region_destroy(handle.as_ptr()) };
            return Err(CoherentRegionError::invalid_view(
                "native fabric returned an invalid coherent-region view",
            ));
        }
        Ok(Self {
            handle: Some(handle),
            view,
            not_send_or_sync: PhantomData,
        })
    }

    pub fn slab(
        payload_bytes: u64,
        required_alignment: u64,
        flags: u32,
    ) -> Result<Self, CoherentRegionError> {
        Self::create(&CoherentRegionConfig::slab(
            payload_bytes,
            required_alignment,
            flags,
        ))
    }

    /// Map bytes directly from their original NVMe file and register those
    /// pages with CUDA. The native owner opens the file during `create`, so the
    /// temporary C path does not need to outlive this call. No weight payload is
    /// copied into a second CPU or device allocation.
    #[cfg(unix)]
    pub fn file_read_only(
        path: &Path,
        file_offset: u64,
        payload_bytes: u64,
        required_alignment: u64,
        flags: u32,
    ) -> Result<Self, CoherentRegionError> {
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            CoherentRegionError::invalid_view("coherent file path contains a NUL byte")
        })?;
        Self::create(&CoherentRegionConfig::file_read_only(
            payload_bytes,
            file_offset,
            required_alignment,
            flags,
            path.as_ptr(),
        ))
    }

    pub fn view(&self) -> &CoherentRegionView {
        &self.view
    }

    pub fn payload_bytes(&self) -> Result<usize, CoherentRegionError> {
        usize::try_from(self.view.payload_bytes).map_err(|_| {
            CoherentRegionError::invalid_view("coherent payload does not fit in usize")
        })
    }

    pub fn device_address(&self) -> u64 {
        self.view.device_pointer as usize as u64
    }

    /// Borrow the CPU alias for storage fills or initialization.
    ///
    /// # Safety
    ///
    /// No CUDA operation may read or write the region for the duration of the
    /// returned borrow. The caller must use scheduler leases and CUDA events to
    /// establish that exclusion.
    pub unsafe fn host_payload_mut(&mut self) -> Result<&mut [u8], CoherentRegionError> {
        let bytes = self.payload_bytes()?;
        // SAFETY: the native view is validated at creation; the caller promises
        // that CUDA does not concurrently access the same coherent bytes.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.view.host_pointer.cast::<u8>(), bytes) })
    }

    /// Borrow the CPU alias after the scheduler has established that all CUDA
    /// writers are complete.
    ///
    /// # Safety
    ///
    /// No CUDA operation may write the region for the returned borrow.
    pub unsafe fn host_payload(&self) -> Result<&[u8], CoherentRegionError> {
        let bytes = self.payload_bytes()?;
        Ok(unsafe { std::slice::from_raw_parts(self.view.host_pointer.cast::<u8>(), bytes) })
    }

    pub fn close(mut self) -> Result<(), CoherentRegionError> {
        self.destroy()
    }

    fn destroy(&mut self) -> Result<(), CoherentRegionError> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // SAFETY: taking the option transfers the guard's unique handle to the
        // native destroy call exactly once.
        status_result(unsafe { flash_coherent_region_destroy(handle.as_ptr()) })
    }
}

impl Drop for CoherentRegionOwner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

fn status_result(status: Status) -> Result<(), CoherentRegionError> {
    if status.code == 0 {
        Ok(())
    } else {
        Err(CoherentRegionError::native(status))
    }
}
