//! Narrow Rust ownership layer over CUDA streams and completion events.
//!
//! CUDA performs execution and dependency tracking. Rust decides which lease
//! an event publishes and when a fixed-address arena may be reused.

use std::ffi::{CStr, c_void};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::ffi::{
    CudaBlas, CudaEvent, CudaStream, Status, sparkserve_cuda_blas_create,
    sparkserve_cuda_blas_destroy, sparkserve_cuda_blas_raw, sparkserve_cuda_event_create,
    sparkserve_cuda_event_destroy, sparkserve_cuda_event_query, sparkserve_cuda_event_record,
    sparkserve_cuda_event_synchronize, sparkserve_cuda_stream_create,
    sparkserve_cuda_stream_destroy, sparkserve_cuda_stream_memset_async,
    sparkserve_cuda_stream_memcpy_async, sparkserve_cuda_stream_raw,
    sparkserve_cuda_stream_synchronize,
    sparkserve_cuda_stream_wait_event,
};

/// Unique owner of one long-lived cuBLAS handle. The scheduler reuses it for
/// every router projection on an execution lane; no handle or workspace is
/// allocated on the token path.
pub struct CudaBlasOwner {
    handle: Option<NonNull<CudaBlas>>,
    raw_blas: NonNull<c_void>,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaBlasOwner {
    pub fn create() -> Result<Self, CudaRuntimeError> {
        let mut raw_owner = std::ptr::null_mut();
        // SAFETY: output storage is valid and unique.
        status_result(unsafe { sparkserve_cuda_blas_create(&mut raw_owner) })?;
        let handle =
            NonNull::new(raw_owner).ok_or_else(|| CudaRuntimeError::null_handle("cuBLAS owner"))?;
        let mut raw_blas = std::ptr::null_mut();
        // SAFETY: the newly created owner remains live for this query.
        if let Err(error) =
            status_result(unsafe { sparkserve_cuda_blas_raw(handle.as_ptr(), &mut raw_blas) })
        {
            // SAFETY: this function still uniquely owns the handle.
            let _ = unsafe { sparkserve_cuda_blas_destroy(handle.as_ptr()) };
            return Err(error);
        }
        let Some(raw_blas) = NonNull::new(raw_blas) else {
            // SAFETY: this function still uniquely owns the handle.
            let _ = unsafe { sparkserve_cuda_blas_destroy(handle.as_ptr()) };
            return Err(CudaRuntimeError::null_handle("raw cuBLAS handle"));
        };
        Ok(Self {
            handle: Some(handle),
            raw_blas,
            not_send_or_sync: PhantomData,
        })
    }

    pub fn raw(&self) -> *mut c_void {
        self.raw_blas.as_ptr()
    }

    pub fn close(mut self) -> Result<(), CudaRuntimeError> {
        self.destroy()
    }

    fn destroy(&mut self) -> Result<(), CudaRuntimeError> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // SAFETY: taking the option transfers unique ownership exactly once.
        status_result(unsafe { sparkserve_cuda_blas_destroy(handle.as_ptr()) })
    }
}

impl Drop for CudaBlasOwner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeError {
    pub code: i32,
    pub message: String,
}

impl CudaRuntimeError {
    fn native(status: Status) -> Self {
        let message = if status.message.is_null() {
            "native CUDA runtime error".to_owned()
        } else {
            // SAFETY: the native ABI returns a static or thread-local,
            // NUL-terminated message. Copy it before the next native call.
            unsafe { CStr::from_ptr(status.message) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            code: status.code,
            message,
        }
    }

    fn null_handle(kind: &'static str) -> Self {
        Self {
            code: -1,
            message: format!("native CUDA runtime returned a null {kind}"),
        }
    }
}

impl Display for CudaRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} (native status {})", self.message, self.code)
    }
}

impl std::error::Error for CudaRuntimeError {}

/// Unique owner of one non-blocking CUDA stream.
///
/// Drop drains the stream before destroying it, so native mappings owned by
/// later fields cannot be released while submitted work is still running.
pub struct CudaStreamOwner {
    handle: Option<NonNull<CudaStream>>,
    raw_stream: NonNull<c_void>,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaStreamOwner {
    pub fn create() -> Result<Self, CudaRuntimeError> {
        let mut raw_owner = std::ptr::null_mut();
        // SAFETY: output storage is valid and unique.
        status_result(unsafe { sparkserve_cuda_stream_create(&mut raw_owner) })?;
        let handle =
            NonNull::new(raw_owner).ok_or_else(|| CudaRuntimeError::null_handle("stream owner"))?;
        let mut raw_stream = std::ptr::null_mut();
        // SAFETY: the newly created native owner remains alive.
        if let Err(error) =
            status_result(unsafe { sparkserve_cuda_stream_raw(handle.as_ptr(), &mut raw_stream) })
        {
            // SAFETY: this function still uniquely owns the handle.
            let _ = unsafe { sparkserve_cuda_stream_destroy(handle.as_ptr()) };
            return Err(error);
        }
        let Some(raw_stream) = NonNull::new(raw_stream) else {
            // SAFETY: this function still uniquely owns the handle.
            let _ = unsafe { sparkserve_cuda_stream_destroy(handle.as_ptr()) };
            return Err(CudaRuntimeError::null_handle("raw stream"));
        };
        Ok(Self {
            handle: Some(handle),
            raw_stream,
            not_send_or_sync: PhantomData,
        })
    }

    pub fn raw(&self) -> *mut c_void {
        self.raw_stream.as_ptr()
    }

    /// Submit a byte memset to this stream.
    ///
    /// # Safety
    ///
    /// `device_address..device_address + bytes` must be a live writable CUDA
    /// range until stream completion, and the scheduler must own that range.
    pub unsafe fn memset_async(
        &mut self,
        device_address: u64,
        value: u8,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let bytes = u64::try_from(bytes).map_err(|_| CudaRuntimeError {
            code: -1,
            message: "CUDA memset size does not fit in u64".to_owned(),
        })?;
        let address = usize::try_from(device_address).map_err(|_| CudaRuntimeError {
            code: -1,
            message: "CUDA device address does not fit in usize".to_owned(),
        })?;
        let pointer = address as *mut c_void;
        // SAFETY: the caller owns the live device range; this owner keeps the
        // stream handle valid through completion.
        status_result(unsafe {
            sparkserve_cuda_stream_memset_async(
                self.handle().as_ptr(),
                pointer,
                u32::from(value),
                bytes,
            )
        })
    }

    /// Copy bytes between two CUDA-visible ranges on this stream. This is used
    /// to promote selected mmap-backed expert tensors into the fixed hot cache.
    ///
    /// # Safety
    ///
    /// Both ranges must remain live through stream completion, the destination
    /// must be writable, and the ranges must not overlap.
    pub unsafe fn memcpy_async(
        &mut self,
        destination_device_address: u64,
        source_device_address: u64,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let bytes = u64::try_from(bytes).map_err(|_| CudaRuntimeError {
            code: -1,
            message: "CUDA memcpy size does not fit in u64".to_owned(),
        })?;
        let destination = usize::try_from(destination_device_address).map_err(|_| {
            CudaRuntimeError {
                code: -1,
                message: "CUDA destination address does not fit in usize".to_owned(),
            }
        })?;
        let source =
            usize::try_from(source_device_address).map_err(|_| CudaRuntimeError {
                code: -1,
                message: "CUDA source address does not fit in usize".to_owned(),
            })?;
        status_result(unsafe {
            sparkserve_cuda_stream_memcpy_async(
                self.handle().as_ptr(),
                destination as *mut c_void,
                source as *const c_void,
                bytes,
            )
        })
    }

    pub fn wait_event(&mut self, event: &CudaEventOwner) -> Result<(), CudaRuntimeError> {
        // SAFETY: both owners keep their handles live for the call.
        status_result(unsafe {
            sparkserve_cuda_stream_wait_event(self.handle().as_ptr(), event.handle().as_ptr())
        })
    }

    pub fn synchronize(&mut self) -> Result<(), CudaRuntimeError> {
        // SAFETY: the unique owner keeps its stream handle live.
        status_result(unsafe { sparkserve_cuda_stream_synchronize(self.handle().as_ptr()) })
    }

    pub fn close(mut self) -> Result<(), CudaRuntimeError> {
        self.destroy()
    }

    fn handle(&self) -> NonNull<CudaStream> {
        self.handle.expect("CUDA stream owner is live")
    }

    fn destroy(&mut self) -> Result<(), CudaRuntimeError> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // SAFETY: taking the option transfers unique ownership exactly once.
        status_result(unsafe { sparkserve_cuda_stream_destroy(handle.as_ptr()) })
    }
}

impl Drop for CudaStreamOwner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Reusable, timing-disabled CUDA completion event.
pub struct CudaEventOwner {
    handle: Option<NonNull<CudaEvent>>,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaEventOwner {
    pub fn create() -> Result<Self, CudaRuntimeError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: output storage is valid and unique.
        status_result(unsafe { sparkserve_cuda_event_create(&mut raw) })?;
        let handle =
            NonNull::new(raw).ok_or_else(|| CudaRuntimeError::null_handle("event owner"))?;
        Ok(Self {
            handle: Some(handle),
            not_send_or_sync: PhantomData,
        })
    }

    pub fn record(&mut self, stream: &mut CudaStreamOwner) -> Result<(), CudaRuntimeError> {
        // SAFETY: both unique owners keep their handles live.
        status_result(unsafe {
            sparkserve_cuda_event_record(self.handle().as_ptr(), stream.handle().as_ptr())
        })
    }

    pub fn query(&self) -> Result<bool, CudaRuntimeError> {
        let mut complete = 0_u32;
        // SAFETY: the owner keeps its event handle live and output is valid.
        status_result(unsafe {
            sparkserve_cuda_event_query(self.handle().as_ptr(), &mut complete)
        })?;
        match complete {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CudaRuntimeError {
                code: -1,
                message: format!("native CUDA event returned invalid completion {complete}"),
            }),
        }
    }

    pub fn synchronize(&mut self) -> Result<(), CudaRuntimeError> {
        // SAFETY: the unique owner keeps its event handle live.
        status_result(unsafe { sparkserve_cuda_event_synchronize(self.handle().as_ptr()) })
    }

    pub fn close(mut self) -> Result<(), CudaRuntimeError> {
        self.destroy()
    }

    fn handle(&self) -> NonNull<CudaEvent> {
        self.handle.expect("CUDA event owner is live")
    }

    fn destroy(&mut self) -> Result<(), CudaRuntimeError> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // SAFETY: taking the option transfers unique ownership exactly once.
        status_result(unsafe { sparkserve_cuda_event_destroy(handle.as_ptr()) })
    }
}

impl Drop for CudaEventOwner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

pub fn status_result(status: Status) -> Result<(), CudaRuntimeError> {
    if status.code == 0 {
        Ok(())
    } else {
        Err(CudaRuntimeError::native(status))
    }
}
