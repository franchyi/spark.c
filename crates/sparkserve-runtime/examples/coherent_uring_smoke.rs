use sparkserve_runtime::ffi::{
    CoherentRegion, CoherentRegionConfig, CoherentRegionView, FABRIC_ABI_VERSION,
    COHERENT_REGION_PREFAULT, sparkserve_coherent_region_create,
    sparkserve_coherent_region_destroy, sparkserve_coherent_region_view,
};
use sparkserve_runtime::uring::{FixedBufferReader, FixedRead};
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::ptr;

struct Region(*mut CoherentRegion);

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the native region handle.
        let status = unsafe { sparkserve_coherent_region_destroy(self.0) };
        assert_eq!(status.code, 0, "coherent region destroy failed");
    }
}

fn status_result(status: sparkserve_runtime::ffi::Status) -> io::Result<()> {
    if status.code == 0 {
        return Ok(());
    }
    let message = if status.message.is_null() {
        "native fabric error".to_owned()
    } else {
        // SAFETY: native status messages remain valid until the next native
        // call on this thread and are NUL-terminated.
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(io::Error::other(message))
}

fn scratch_file() -> io::Result<(std::path::PathBuf, File)> {
    let path = std::env::temp_dir().join(format!(
        "sparkserve-coherent-uring-{}.bin",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&path)?;
    file.write_all(&vec![0x31; 4096])?;
    file.write_all(&vec![0x72; 4096])?;
    file.sync_all()?;
    Ok((path, file))
}

fn main() -> io::Result<()> {
    let config = CoherentRegionConfig::slab(4 * 1024 * 1024, 4096, COHERENT_REGION_PREFAULT);
    let mut raw_region = ptr::null_mut();
    // SAFETY: the config and output pointer follow fabric ABI version one.
    status_result(unsafe { sparkserve_coherent_region_create(&config, &mut raw_region) })?;
    if raw_region.is_null() {
        return Err(io::Error::other("native fabric returned a null region"));
    }
    let region = Region(raw_region);
    let mut view = CoherentRegionView {
        struct_size: std::mem::size_of::<CoherentRegionView>() as u32,
        abi_version: FABRIC_ABI_VERSION,
        kind: 0,
        flags: 0,
        host_pointer: ptr::null_mut(),
        device_pointer: ptr::null_mut(),
        mapped_bytes: 0,
        payload_bytes: 0,
        file_offset: 0,
        required_alignment: 0,
        page_bytes: 0,
        device_id: 0,
        reserved: 0,
    };
    // SAFETY: `region` remains alive until after the view and reader are gone.
    status_result(unsafe { sparkserve_coherent_region_view(region.0, &mut view) })?;
    if view.host_pointer.is_null() || view.device_pointer.is_null() {
        return Err(io::Error::other("coherent region returned null pointers"));
    }
    let payload_bytes = usize::try_from(view.payload_bytes)
        .map_err(|_| io::Error::other("coherent payload exceeds usize"))?;
    // SAFETY: the region exclusively lends its writable slab to this reader.
    let slab = unsafe {
        std::slice::from_raw_parts_mut(view.host_pointer.cast::<u8>(), payload_bytes)
    };
    let mut reader = FixedBufferReader::new(slab, 2, 2)?;
    let (path, file) = scratch_file()?;
    let stats = reader.read(&[
        FixedRead {
            file: &file,
            file_offset: 4096,
            buffer_offset: 0,
            bytes: 4096,
        },
        FixedRead {
            file: &file,
            file_offset: 0,
            buffer_offset: 4096,
            bytes: 4096,
        },
    ])?;
    assert_eq!(stats.operations, 2);
    assert_eq!(&reader.buffer()[..4096], &[0x72; 4096]);
    assert_eq!(&reader.buffer()[4096..8192], &[0x31; 4096]);
    drop(reader);
    drop(file);
    fs::remove_file(path)?;
    drop(region);
    println!("GB10 coherent slab accepted simultaneous CUDA and io_uring registration");
    Ok(())
}
