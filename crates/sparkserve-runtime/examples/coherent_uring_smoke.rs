use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::ffi::COHERENT_REGION_PREFAULT;
use sparkserve_runtime::qsa::{QsaArenaPhase, QsaCoherentArena, QsaSparseDecodePlan};
use sparkserve_runtime::uring::{FixedBufferReader, FixedRead};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};

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
    let mut region = CoherentRegionOwner::slab(4 * 1024 * 1024, 4096, COHERENT_REGION_PREFAULT)
        .map_err(io::Error::other)?;
    // SAFETY: no CUDA work is submitted while the fixed reader owns the CPU
    // alias. The native smoke test validates device visibility afterwards.
    let slab = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
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
    region.close().map_err(io::Error::other)?;

    let qsa_plan = QsaSparseDecodePlan::qwen38_flash(1).map_err(io::Error::other)?;
    let qsa_bytes = qsa_plan
        .scratch_layout()
        .map_err(io::Error::other)?
        .total_bytes;
    let qsa_arena = QsaCoherentArena::allocate(qsa_plan, vec![1], COHERENT_REGION_PREFAULT)
        .map_err(io::Error::other)?;
    assert_eq!(qsa_arena.region().payload_bytes().unwrap(), qsa_bytes);
    assert_eq!(
        qsa_arena.scheduler().phase(),
        QsaArenaPhase::WorkspaceNeedsZero
    );
    assert_eq!(
        qsa_arena.scheduler().arena().device_base(),
        qsa_arena.region().device_address()
    );
    drop(qsa_arena);
    println!("GB10 coherent slabs accepted io_uring registration and a fixed-address QSA arena");
    Ok(())
}
