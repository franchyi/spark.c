mod checkpoint;

use checkpoint::{Q27Checkpoint, TensorLocation};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"Q27SFV1\0";
const VERSION: u32 = 1;
const LAYERS: u32 = 64;
const PROJECTIONS: u32 = 3;
const ENTRIES: u32 = LAYERS * PROJECTIONS;
const HEADER_BYTES: u64 = 64;
const ENTRY_BYTES: u64 = 40;
const REVISION: &[u8; 40] = b"009632fef96dd349150baa780c984e62e70e91fe";

#[derive(Clone, Copy)]
struct Entry {
    layer: u32,
    projection: u32,
    n: u32,
    k: u32,
    offset: u64,
    bytes: u64,
    input_scale_inv: f32,
    alpha: f32,
}

fn read_tensor(root: &Path, location: &TensorLocation) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(root.join(&location.relative_file))?;
    let mut bytes = vec![0_u8; usize::try_from(location.data_bytes)?];
    file.read_exact_at(&mut bytes, location.absolute_offset)?;
    Ok(bytes)
}

fn scalar(checkpoint: &Q27Checkpoint, name: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let location = checkpoint.tensor(name)?;
    if location.dtype != "F32" || !location.shape.is_empty() || location.data_bytes != 4 {
        return Err(format!("{name} is not a q27 FP32 scalar").into());
    }
    let bytes = read_tensor(&checkpoint.plan().root, location)?;
    Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
}

fn projection(projection: u32) -> (&'static str, u32, u32) {
    match projection {
        0 => ("gate_proj", 17_408, 5_120),
        1 => ("up_proj", 17_408, 5_120),
        2 => ("down_proj", 5_120, 17_408),
        _ => unreachable!("fixed q27 projection"),
    }
}

fn interleave_128x4(input: &[u8], rows: usize, columns: usize) -> Result<Vec<u8>, String> {
    if rows % 128 != 0 || columns % 4 != 0 || input.len() != rows * columns {
        return Err(format!("invalid q27 scale matrix [{rows},{columns}]"));
    }
    let mut output = vec![0_u8; input.len()];
    let mut destination = 0;
    for row_block in 0..rows / 128 {
        for column_block in 0..columns / 4 {
            for row_minor in 0..32 {
                for row_quadrant in 0..4 {
                    let row = row_block * 128 + row_quadrant * 32 + row_minor;
                    let source = row * columns + column_block * 4;
                    output[destination..destination + 4]
                        .copy_from_slice(&input[source..source + 4]);
                    destination += 4;
                }
            }
        }
    }
    Ok(output)
}

fn write_u32(output: &mut impl Write, value: u32) -> std::io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_u64(output: &mut impl Write, value: u64) -> std::io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_f32(output: &mut impl Write, value: f32) -> std::io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_header(output: &mut impl Write) -> std::io::Result<()> {
    output.write_all(MAGIC)?;
    write_u32(output, VERSION)?;
    write_u32(output, ENTRIES)?;
    output.write_all(REVISION)?;
    output.write_all(&[0_u8; 8])
}

fn write_entry(output: &mut impl Write, entry: Entry) -> std::io::Result<()> {
    write_u32(output, entry.layer)?;
    write_u32(output, entry.projection)?;
    write_u32(output, entry.n)?;
    write_u32(output, entry.k)?;
    write_u64(output, entry.offset)?;
    write_u64(output, entry.bytes)?;
    write_f32(output, entry.input_scale_inv)?;
    write_f32(output, entry.alpha)
}

fn temporary_path(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}", std::process::id()));
    output.with_file_name(name)
}

fn pack(root: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if output.exists() {
        return Err(format!("refusing to overwrite existing sidecar: {}", output.display()).into());
    }
    let checkpoint = Q27Checkpoint::open(root)?;
    let temporary = temporary_path(output);
    let file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
    drop(file);
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut entries = Vec::with_capacity(ENTRIES as usize);
        let mut offset = HEADER_BYTES + ENTRY_BYTES * ENTRIES as u64;
        for layer in 0..LAYERS {
            let prefix = format!("model.language_model.layers.{layer}.mlp");
            let gate_scale = scalar(&checkpoint, &format!("{prefix}.gate_proj.input_scale"))?;
            let up_scale = scalar(&checkpoint, &format!("{prefix}.up_proj.input_scale"))?;
            if gate_scale.to_bits() != up_scale.to_bits() {
                return Err(format!("layer {layer} gate/up activation scales differ").into());
            }
            for projection_id in 0..PROJECTIONS {
                let (name, n, k) = projection(projection_id);
                let base = format!("{prefix}.{name}");
                let scale_location = checkpoint.tensor(&format!("{base}.weight_scale"))?;
                if scale_location.dtype != "F8_E4M3" ||
                    scale_location.shape != [u64::from(n), u64::from(k / 16)] {
                    return Err(format!("{base}.weight_scale changed physical layout").into());
                }
                let input_scale = scalar(&checkpoint, &format!("{base}.input_scale"))?;
                let weight_scale_2 = scalar(&checkpoint, &format!("{base}.weight_scale_2"))?;
                entries.push(Entry {
                    layer,
                    projection: projection_id,
                    n,
                    k,
                    offset,
                    bytes: scale_location.data_bytes,
                    input_scale_inv: input_scale.recip(),
                    alpha: input_scale * weight_scale_2,
                });
                offset += scale_location.data_bytes;
            }
        }
        // The final stream is bounded to one 5.3 MiB scale transform at a time
        // and never retains the approximately 1 GiB sidecar in RAM.
        let file = OpenOptions::new().write(true).truncate(true).open(&temporary)?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        write_header(&mut writer)?;
        for entry in entries.iter().copied() { write_entry(&mut writer, entry)?; }
        for layer in 0..LAYERS {
            let prefix = format!("model.language_model.layers.{layer}.mlp");
            for projection_id in 0..PROJECTIONS {
                let (name, n, k) = projection(projection_id);
                let location = checkpoint.tensor(&format!("{prefix}.{name}.weight_scale"))?;
                let raw = read_tensor(&checkpoint.plan().root, location)?;
                let packed = interleave_128x4(&raw, n as usize, (k / 16) as usize)?;
                writer.write_all(&packed)?;
            }
        }
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        if file.metadata()?.len() != offset {
            return Err(format!("q27 sidecar length mismatch: expected {offset}").into());
        }
        drop(file);
        fs::rename(&temporary, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    println!("q27_scale_sidecar=valid");
    println!("entries={ENTRIES}");
    println!("bytes={}", fs::metadata(output)?.len());
    println!("path={}", output.display());
    Ok(())
}

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(root) = arguments.next() else {
        eprintln!("usage: {} CHECKPOINT OUTPUT", Path::new(&program).display());
        std::process::exit(2);
    };
    let Some(output) = arguments.next() else {
        eprintln!("usage: {} CHECKPOINT OUTPUT", Path::new(&program).display());
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("q27-pack-scales accepts exactly a checkpoint and output path");
        std::process::exit(2);
    }
    if let Err(error) = pack(Path::new(&root), Path::new(&output)) {
        eprintln!("q27 scale pack failed: {error}");
        std::process::exit(1);
    }
}
