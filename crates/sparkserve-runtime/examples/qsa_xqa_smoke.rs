use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::cuda::{CudaEventOwner, CudaStreamOwner, status_result};
use sparkserve_runtime::ffi::{
    COHERENT_REGION_PREFAULT, DeviceCaps, QsaDecodeArgs, QsaDecodePlan, QsaKvPackArgs,
    QsaKvPackPlan, sparkserve_qsa_decode_launch, sparkserve_qsa_kv_pack_launch,
};
use sparkserve_runtime::qsa::{QsaArenaPhase, QsaCoherentArena, QsaSparseDecodePlan};
use std::ffi::c_void;
use std::io;
use std::path::{Path, PathBuf};

const QUERY_BYTES: usize = 24 * 256 * 2;
const STATE_BYTES: usize = 8192 * 2 * 256 * 2;
const REQUEST_MAP_BYTES: usize = 6 * 4096 * 4;
const REQUEST_INDICES_FILE_BYTES: usize = 4 * 4;
const LOGICAL_INDICES_FILE_BYTES: usize = 4 * 2051 * 4;
const SEQUENCE_LENGTHS_FILE_BYTES: usize = 4 * 4;
const REQUEST_INDEX_BYTES: usize = 4;
const LOGICAL_INDEX_BYTES: usize = 2051 * 4;
const SEQUENCE_LENGTH_BYTES: usize = 4;
const INPUT_ALIGNMENT: usize = 256;

#[derive(Clone, Copy)]
struct InputLayout {
    key_state: usize,
    value_state: usize,
    request_map: usize,
    request_index: usize,
    logical_indices: usize,
    sequence_length: usize,
    query: usize,
    total_bytes: usize,
}

impl InputLayout {
    fn qwen() -> io::Result<Self> {
        let key_state = 0;
        let value_state = align(key_state + STATE_BYTES)?;
        let request_map = align(value_state + STATE_BYTES)?;
        let request_index = align(request_map + REQUEST_MAP_BYTES)?;
        let logical_indices = align(request_index + REQUEST_INDEX_BYTES)?;
        let sequence_length = align(logical_indices + LOGICAL_INDEX_BYTES)?;
        let query = align(sequence_length + SEQUENCE_LENGTH_BYTES)?;
        let total_bytes = align(query + QUERY_BYTES)?;
        Ok(Self {
            key_state,
            value_state,
            request_map,
            request_index,
            logical_indices,
            sequence_length,
            query,
            total_bytes,
        })
    }
}

fn align(value: usize) -> io::Result<usize> {
    value
        .checked_add(INPUT_ALIGNMENT - 1)
        .map(|sum| sum / INPUT_ALIGNMENT * INPUT_ALIGNMENT)
        .ok_or_else(|| io::Error::other("input layout overflow"))
}

fn read_fixture(root: &Path, name: &str, bytes: usize) -> io::Result<Vec<u8>> {
    let path = root.join(name);
    let payload = std::fs::read(&path)?;
    if payload.len() != bytes {
        return Err(io::Error::other(format!(
            "{} has {} bytes; expected {bytes}",
            path.display(),
            payload.len()
        )));
    }
    Ok(payload)
}

fn device_pointer(address: u64) -> io::Result<*mut c_void> {
    let address = usize::try_from(address)
        .map_err(|_| io::Error::other("device address does not fit in usize"))?;
    Ok(address as *mut c_void)
}

fn device_offset(base: u64, offset: usize) -> io::Result<u64> {
    base.checked_add(u64::try_from(offset).map_err(|_| io::Error::other("device offset overflow"))?)
        .ok_or_else(|| io::Error::other("device address overflow"))
}

fn bf16_mismatches(left: &[u8], right: &[u8]) -> usize {
    left.chunks_exact(2)
        .zip(right.chunks_exact(2))
        .filter(|(left, right)| left != right)
        .count()
}

fn main() -> io::Result<()> {
    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let xqa_fixture = arguments
        .next()
        .ok_or_else(|| io::Error::other("usage: qsa_xqa_smoke <xqa-fixture> <kv-pack-fixture>"))?;
    let pack_fixture = arguments
        .next()
        .ok_or_else(|| io::Error::other("usage: qsa_xqa_smoke <xqa-fixture> <kv-pack-fixture>"))?;
    if arguments.next().is_some() {
        return Err(io::Error::other(
            "usage: qsa_xqa_smoke <xqa-fixture> <kv-pack-fixture>",
        ));
    }

    let plan = QsaSparseDecodePlan::qwen38_flash(1).map_err(io::Error::other)?;
    let arena_layout = plan.scratch_layout().map_err(io::Error::other)?;
    let input_layout = InputLayout::qwen()?;

    let query = read_fixture(&xqa_fixture, "query_bf16.bin", QUERY_BYTES)?;
    let expected_key = read_fixture(
        &xqa_fixture,
        "packed_key_bf16.bin",
        arena_layout.packed_key_bytes,
    )?;
    let expected_value = read_fixture(
        &xqa_fixture,
        "packed_value_bf16.bin",
        arena_layout.packed_value_bytes,
    )?;
    let block_tables = read_fixture(
        &xqa_fixture,
        "block_tables_i32.bin",
        arena_layout.block_tables_bytes,
    )?;
    let expected_length = read_fixture(
        &xqa_fixture,
        "sequence_lengths_i32.bin",
        arena_layout.valid_counts_bytes,
    )?;
    let expected_output = read_fixture(&xqa_fixture, "output_bf16.bin", QUERY_BYTES)?;

    let key_state = read_fixture(&pack_fixture, "key_state_bf16.bin", STATE_BYTES)?;
    let value_state = read_fixture(&pack_fixture, "value_state_bf16.bin", STATE_BYTES)?;
    let request_map = read_fixture(&pack_fixture, "req_to_token_i32.bin", REQUEST_MAP_BYTES)?;
    let request_indices = read_fixture(
        &pack_fixture,
        "request_indices_i32.bin",
        REQUEST_INDICES_FILE_BYTES,
    )?;
    let logical_indices = read_fixture(
        &pack_fixture,
        "logical_indices_i32.bin",
        LOGICAL_INDICES_FILE_BYTES,
    )?;
    let sequence_lengths = read_fixture(
        &pack_fixture,
        "sequence_lengths_i32.bin",
        SEQUENCE_LENGTHS_FILE_BYTES,
    )?;

    let mut inputs = CoherentRegionOwner::slab(
        u64::try_from(input_layout.total_bytes)
            .map_err(|_| io::Error::other("input size overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: the input mapping has not been submitted to CUDA.
    let input_payload = unsafe { inputs.host_payload_mut() }.map_err(io::Error::other)?;
    input_payload[input_layout.key_state..input_layout.key_state + STATE_BYTES]
        .copy_from_slice(&key_state);
    input_payload[input_layout.value_state..input_layout.value_state + STATE_BYTES]
        .copy_from_slice(&value_state);
    input_payload[input_layout.request_map..input_layout.request_map + REQUEST_MAP_BYTES]
        .copy_from_slice(&request_map);
    input_payload[input_layout.request_index..input_layout.request_index + REQUEST_INDEX_BYTES]
        .copy_from_slice(&request_indices[..REQUEST_INDEX_BYTES]);
    input_payload[input_layout.logical_indices..input_layout.logical_indices + LOGICAL_INDEX_BYTES]
        .copy_from_slice(&logical_indices[..LOGICAL_INDEX_BYTES]);
    input_payload
        [input_layout.sequence_length..input_layout.sequence_length + SEQUENCE_LENGTH_BYTES]
        .copy_from_slice(&sequence_lengths[..SEQUENCE_LENGTH_BYTES]);
    input_payload[input_layout.query..input_layout.query + QUERY_BYTES].copy_from_slice(&query);

    let mut arena = QsaCoherentArena::allocate(plan, vec![1], COHERENT_REGION_PREFAULT)
        .map_err(io::Error::other)?;
    let mut stream = CudaStreamOwner::create().map_err(io::Error::other)?;
    let mut event = CudaEventOwner::create().map_err(io::Error::other)?;

    let zero = arena
        .scheduler_mut()
        .begin_workspace_zero()
        .map_err(io::Error::other)?;
    // SAFETY: the workspace-zero lease owns this fixed device range.
    unsafe {
        stream.memset_async(
            zero.arena().addresses().attention_workspace,
            0,
            arena_layout.attention_workspace_bytes,
        )
    }
    .map_err(io::Error::other)?;
    event.record(&mut stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;
    arena
        .scheduler_mut()
        .complete_workspace_zero(zero)
        .map_err(io::Error::other)?;

    // The immutable block table is scheduler metadata, initialized once while
    // the arena is idle. The borrowed packer fills K/V and valid lengths.
    // SAFETY: the arena is Idle and no CUDA operation owns these bytes.
    let arena_payload = unsafe { arena.host_payload_mut() }.map_err(io::Error::other)?;
    arena_payload[arena_layout.block_tables_offset
        ..arena_layout.block_tables_offset + arena_layout.block_tables_bytes]
        .copy_from_slice(&block_tables);

    let input_base = inputs.device_address();
    let addresses = arena.scheduler().arena().addresses();
    let caps = DeviceCaps::gb10(arena_layout.attention_workspace_bytes as u64);
    let pack = arena
        .scheduler_mut()
        .begin_pack(1)
        .map_err(io::Error::other)?;
    let pack_args = QsaKvPackArgs {
        struct_size: std::mem::size_of::<QsaKvPackArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaKvPackPlan::qwen38_flash(1, 8192, 6, 4096),
        key_state: device_pointer(device_offset(input_base, input_layout.key_state)?)?.cast_const(),
        value_state: device_pointer(device_offset(input_base, input_layout.value_state)?)?
            .cast_const(),
        req_to_token: device_pointer(device_offset(input_base, input_layout.request_map)?)?
            .cast::<i32>()
            .cast_const(),
        request_indices: device_pointer(device_offset(input_base, input_layout.request_index)?)?
            .cast::<i32>()
            .cast_const(),
        logical_indices: device_pointer(device_offset(input_base, input_layout.logical_indices)?)?
            .cast::<i32>()
            .cast_const(),
        sequence_lengths: device_pointer(device_offset(input_base, input_layout.sequence_length)?)?
            .cast::<i32>()
            .cast_const(),
        valid_counts: device_pointer(addresses.valid_counts)?.cast::<i32>(),
        packed_key: device_pointer(addresses.packed_key)?,
        packed_value: device_pointer(addresses.packed_value)?,
        cuda_stream: stream.raw(),
    };
    // SAFETY: all inputs and fixed outputs are live coherent mappings; the
    // pack lease owns the output addresses through event completion.
    let pack_launch = unsafe { sparkserve_qsa_kv_pack_launch(&caps, &pack_args) };
    if let Err(error) = status_result(pack_launch) {
        let _ = stream.synchronize();
        let _ = arena.scheduler_mut().abort_pack(pack);
        return Err(io::Error::other(error));
    }
    event.record(&mut stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;
    let ready = arena
        .scheduler_mut()
        .complete_pack(pack)
        .map_err(io::Error::other)?;

    let decode = arena
        .scheduler_mut()
        .begin_decode(ready)
        .map_err(io::Error::other)?;
    let decode_args = QsaDecodeArgs {
        struct_size: std::mem::size_of::<QsaDecodeArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaDecodePlan::qwen38_flash(decode.graph_batch() as u32, 48),
        query: device_pointer(device_offset(input_base, input_layout.query)?)?.cast_const(),
        packed_key: device_pointer(addresses.packed_key)?.cast_const(),
        packed_value: device_pointer(addresses.packed_value)?.cast_const(),
        block_tables: device_pointer(addresses.block_tables)?
            .cast::<i32>()
            .cast_const(),
        sequence_lengths: device_pointer(addresses.valid_counts)?
            .cast::<i32>()
            .cast_const(),
        output: device_pointer(addresses.attention_output)?,
        workspace: device_pointer(addresses.attention_workspace)?,
        workspace_bytes: arena_layout.attention_workspace_bytes as u64,
        bmm1_scale: 0.0625,
        bmm2_scale: 1.0,
        cuda_stream: stream.raw(),
    };
    // SAFETY: the ready lease transfers the packer's fixed K/V output directly
    // to XQA; the decode lease holds every address until event completion.
    let decode_launch = unsafe { sparkserve_qsa_decode_launch(&caps, &decode_args) };
    if let Err(error) = status_result(decode_launch) {
        let _ = stream.synchronize();
        let _ = arena.scheduler_mut().abort_decode(decode);
        return Err(io::Error::other(error));
    }
    event.record(&mut stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;
    arena
        .scheduler_mut()
        .complete_decode(decode)
        .map_err(io::Error::other)?;
    assert_eq!(arena.scheduler().phase(), QsaArenaPhase::Idle);

    // SAFETY: both CUDA events completed and the scheduler released the arena.
    let arena_payload = unsafe { arena.host_payload_mut() }.map_err(io::Error::other)?;
    let actual_key = &arena_payload[arena_layout.packed_key_offset
        ..arena_layout.packed_key_offset + arena_layout.packed_key_bytes];
    let actual_value = &arena_payload[arena_layout.packed_value_offset
        ..arena_layout.packed_value_offset + arena_layout.packed_value_bytes];
    let actual_length = &arena_payload[arena_layout.valid_counts_offset
        ..arena_layout.valid_counts_offset + arena_layout.valid_counts_bytes];
    let actual_output = &arena_payload[arena_layout.attention_output_offset
        ..arena_layout.attention_output_offset + arena_layout.attention_output_bytes];
    let key_mismatches = bf16_mismatches(actual_key, &expected_key);
    let value_mismatches = bf16_mismatches(actual_value, &expected_value);
    let length_mismatches = usize::from(actual_length != expected_length);
    let output_mismatches = bf16_mismatches(actual_output, &expected_output);
    println!(
        "Rust QSA pack/XQA mismatches key/value/length/output: \
         {key_mismatches}/{value_mismatches}/{length_mismatches}/{output_mismatches}"
    );
    if key_mismatches + value_mismatches + length_mismatches + output_mismatches != 0 {
        return Err(io::Error::other(
            "Rust joined QSA output differs from oracle",
        ));
    }

    event.close().map_err(io::Error::other)?;
    stream.close().map_err(io::Error::other)?;
    inputs.close().map_err(io::Error::other)?;
    drop(arena);
    Ok(())
}
