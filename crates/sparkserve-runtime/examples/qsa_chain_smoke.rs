use sparkserve_runtime::cuda::{CudaStreamOwner, status_result};
use sparkserve_runtime::ffi::{
    COHERENT_REGION_PREFAULT, DeviceCaps, QsaDecodeArgs, QsaDecodePlan, QsaExpandArgs,
    QsaExpandPlan, QsaIndexPrepArgs, QsaIndexPrepPlan, QsaKvPackArgs, QsaKvPackPlan, QsaScoreArgs,
    QsaScorePlan, QsaTopkArgs, QsaTopkPlan, sparkserve_qsa_decode_launch,
    sparkserve_qsa_expand_launch, sparkserve_qsa_index_prep_launch, sparkserve_qsa_kv_pack_launch,
    sparkserve_qsa_score_launch, sparkserve_qsa_topk_launch,
};
use sparkserve_runtime::qsa::QsaSparseDecodePlan;
use sparkserve_runtime::qsa_pipeline::{
    QsaCoherentPipeline, QsaPipelineCudaCompletion, QsaPipelineCudaFence, QsaPipelineReady,
    QsaPipelineStage,
};
use std::ffi::c_void;
use std::io;
use std::path::{Path, PathBuf};

const ALIGNMENT: usize = 256;
const INDEX_HEADS: usize = 4;
const PADDED_INDEX_HEADS: usize = 8;
const INDEX_HEAD_DIM: usize = 128;
const ROTARY_ROWS: usize = 3008;
const INDEX_STATE_SLOTS: usize = 16;
const COMPRESSED_PAGES: usize = 48;
const COMPRESSED_PAGE_SIZE: usize = 16;
const COMPRESSED_MAX_PAGES: usize = 44;
const COMPRESSED_LENGTH: usize = 700;
const SCORE_COLUMNS: usize = COMPRESSED_MAX_PAGES * COMPRESSED_PAGE_SIZE;
const BLOCK_TOPK: usize = 512;
const FINAL_TOPK: usize = 2051;
const REQUEST_STRIDE: usize = 3072;
const KV_STATE_SLOTS: usize = 4096;
const KV_HEADS: usize = 2;
const ATTENTION_HEAD_DIM: usize = 256;
const ATTENTION_QUERY_HEADS: usize = 24;

#[derive(Default)]
struct LayoutBuilder {
    next: usize,
}

impl LayoutBuilder {
    fn field(&mut self, bytes: usize) -> io::Result<usize> {
        let offset = align(self.next)?;
        self.next = offset
            .checked_add(bytes)
            .ok_or_else(|| io::Error::other("QSA chain layout overflow"))?;
        Ok(offset)
    }

    fn finish(self) -> io::Result<usize> {
        align(self.next)
    }
}

#[derive(Clone, Copy)]
struct ChainLayout {
    qk: usize,
    q_weight: usize,
    k_weight: usize,
    cos_sin: usize,
    axis_map: usize,
    positions: usize,
    cache_locs: usize,
    index_query: usize,
    index_key_state: usize,
    rope_positions: usize,
    compressed_key_cache: usize,
    compressed_page_table: usize,
    compressed_lengths: usize,
    logits: usize,
    row_start: usize,
    block_indices: usize,
    query_positions: usize,
    sequence_lengths: usize,
    logical_indices: usize,
    full_key_state: usize,
    full_value_state: usize,
    request_to_token: usize,
    request_indices: usize,
    attention_query: usize,
    total_bytes: usize,
}

impl ChainLayout {
    fn qwen() -> io::Result<Self> {
        let mut layout = LayoutBuilder::default();
        let qk = layout.field((INDEX_HEADS + 1) * INDEX_HEAD_DIM * 2)?;
        let q_weight = layout.field(INDEX_HEAD_DIM * 2)?;
        let k_weight = layout.field(INDEX_HEAD_DIM * 2)?;
        let cos_sin = layout.field(ROTARY_ROWS * INDEX_HEAD_DIM * 4)?;
        let axis_map = layout.field(INDEX_HEAD_DIM / 2 * 4)?;
        let positions = layout.field(8)?;
        let cache_locs = layout.field(8)?;
        let index_query = layout.field(PADDED_INDEX_HEADS * INDEX_HEAD_DIM * 2)?;
        let index_key_state = layout.field(INDEX_STATE_SLOTS * INDEX_HEAD_DIM * 2)?;
        let rope_positions = layout.field(INDEX_STATE_SLOTS * 3 * 8)?;
        let compressed_key_cache =
            layout.field(COMPRESSED_PAGES * COMPRESSED_PAGE_SIZE * INDEX_HEAD_DIM * 2)?;
        let compressed_page_table = layout.field(COMPRESSED_MAX_PAGES * 4)?;
        let compressed_lengths = layout.field(4)?;
        let logits = layout.field(SCORE_COLUMNS * 4)?;
        let row_start = layout.field(4)?;
        let block_indices = layout.field(BLOCK_TOPK * 4)?;
        let query_positions = layout.field(8)?;
        let sequence_lengths = layout.field(4)?;
        let logical_indices = layout.field(FINAL_TOPK * 4)?;
        let full_key_state = layout.field(KV_STATE_SLOTS * KV_HEADS * ATTENTION_HEAD_DIM * 2)?;
        let full_value_state = layout.field(KV_STATE_SLOTS * KV_HEADS * ATTENTION_HEAD_DIM * 2)?;
        let request_to_token = layout.field(REQUEST_STRIDE * 4)?;
        let request_indices = layout.field(4)?;
        let attention_query = layout.field(ATTENTION_QUERY_HEADS * ATTENTION_HEAD_DIM * 2)?;
        let total_bytes = layout.finish()?;
        Ok(Self {
            qk,
            q_weight,
            k_weight,
            cos_sin,
            axis_map,
            positions,
            cache_locs,
            index_query,
            index_key_state,
            rope_positions,
            compressed_key_cache,
            compressed_page_table,
            compressed_lengths,
            logits,
            row_start,
            block_indices,
            query_positions,
            sequence_lengths,
            logical_indices,
            full_key_state,
            full_value_state,
            request_to_token,
            request_indices,
            attention_query,
            total_bytes,
        })
    }
}

fn align(value: usize) -> io::Result<usize> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|sum| sum / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| io::Error::other("QSA chain alignment overflow"))
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

fn copy_at(output: &mut [u8], offset: usize, input: &[u8]) {
    output[offset..offset + input.len()].copy_from_slice(input);
}

fn device_pointer(base: u64, offset: usize) -> io::Result<*mut c_void> {
    let address = base
        .checked_add(u64::try_from(offset).map_err(|_| io::Error::other("offset overflow"))?)
        .ok_or_else(|| io::Error::other("device address overflow"))?;
    Ok(usize::try_from(address)
        .map_err(|_| io::Error::other("device address does not fit usize"))? as *mut c_void)
}

fn raw_pointer(address: u64) -> io::Result<*mut c_void> {
    Ok(usize::try_from(address)
        .map_err(|_| io::Error::other("device address does not fit usize"))? as *mut c_void)
}

fn element_mismatches(left: &[u8], right: &[u8], width: usize) -> usize {
    left.chunks_exact(width)
        .zip(right.chunks_exact(width))
        .filter(|(left, right)| left != right)
        .count()
}

fn sorted_i32(bytes: &[u8]) -> Vec<i32> {
    let mut values: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 bytes")))
        .collect();
    values.sort_unstable();
    values
}

fn ordered_pack_mismatches(
    actual: &[u8],
    state: &[u8],
    request_to_token: &[u8],
    logical_indices: &[u8],
) -> usize {
    let request_map: Vec<i32> = request_to_token
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 bytes")))
        .collect();
    let logical: Vec<i32> = logical_indices
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 bytes")))
        .collect();
    let state_row_bytes = KV_HEADS * ATTENTION_HEAD_DIM * 2;
    let mut mismatches = 0;
    for (column, &position) in logical.iter().enumerate() {
        let slot = usize::try_from(request_map[position as usize]).expect("non-negative slot");
        let expected = &state[slot * state_row_bytes..(slot + 1) * state_row_bytes];
        let packed = &actual[column * state_row_bytes..(column + 1) * state_row_bytes];
        mismatches += element_mismatches(packed, expected, 2);
    }
    mismatches
        + actual[logical.len() * state_row_bytes..]
            .chunks_exact(2)
            .filter(|element| *element != [0, 0])
            .count()
}

fn bf16_difference(left: &[u8], right: &[u8]) -> (usize, f32) {
    let mut exact_mismatches = 0;
    let mut max_abs = 0.0_f32;
    for (left, right) in left.chunks_exact(2).zip(right.chunks_exact(2)) {
        let left = f32::from_bits(u32::from(u16::from_ne_bytes(left.try_into().unwrap())) << 16);
        let right = f32::from_bits(u32::from(u16::from_ne_bytes(right.try_into().unwrap())) << 16);
        exact_mismatches += usize::from(left != right);
        max_abs = max_abs.max((left - right).abs());
    }
    (exact_mismatches, max_abs)
}

fn stage_ready(completion: QsaPipelineCudaCompletion) -> io::Result<QsaPipelineReady> {
    match completion {
        QsaPipelineCudaCompletion::StageReady(ready) => Ok(ready),
        _ => Err(io::Error::other(
            "QSA stage completed with wrong fence state",
        )),
    }
}

fn main() -> io::Result<()> {
    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let fixture = arguments
        .next()
        .ok_or_else(|| io::Error::other("usage: qsa_chain_smoke <chain-fixture>"))?;
    if arguments.next().is_some() {
        return Err(io::Error::other("usage: qsa_chain_smoke <chain-fixture>"));
    }

    let layout = ChainLayout::qwen()?;
    let qk = read_fixture(
        &fixture,
        "qk_bf16.bin",
        (INDEX_HEADS + 1) * INDEX_HEAD_DIM * 2,
    )?;
    let q_weight = read_fixture(&fixture, "q_weight_bf16.bin", INDEX_HEAD_DIM * 2)?;
    let k_weight = read_fixture(&fixture, "k_weight_bf16.bin", INDEX_HEAD_DIM * 2)?;
    let cos_sin = read_fixture(
        &fixture,
        "cos_sin_f32.bin",
        ROTARY_ROWS * INDEX_HEAD_DIM * 4,
    )?;
    let axis_map = read_fixture(&fixture, "axis_map_i32.bin", INDEX_HEAD_DIM / 2 * 4)?;
    let positions = read_fixture(&fixture, "positions_i64.bin", 8)?;
    let cache_locs = read_fixture(&fixture, "cache_locs_i64.bin", 8)?;
    let compressed_key_cache = read_fixture(
        &fixture,
        "compressed_key_cache_bf16.bin",
        COMPRESSED_PAGES * COMPRESSED_PAGE_SIZE * INDEX_HEAD_DIM * 2,
    )?;
    let compressed_page_table = read_fixture(
        &fixture,
        "compressed_page_table_i32.bin",
        COMPRESSED_MAX_PAGES * 4,
    )?;
    let compressed_lengths = read_fixture(&fixture, "compressed_lengths_i32.bin", 4)?;
    let query_positions = read_fixture(&fixture, "query_positions_i64.bin", 8)?;
    let sequence_lengths = read_fixture(&fixture, "sequence_lengths_i32.bin", 4)?;
    let full_key_state = read_fixture(
        &fixture,
        "full_key_state_bf16.bin",
        KV_STATE_SLOTS * KV_HEADS * ATTENTION_HEAD_DIM * 2,
    )?;
    let full_value_state = read_fixture(
        &fixture,
        "full_value_state_bf16.bin",
        KV_STATE_SLOTS * KV_HEADS * ATTENTION_HEAD_DIM * 2,
    )?;
    let request_to_token = read_fixture(&fixture, "request_to_token_i32.bin", REQUEST_STRIDE * 4)?;
    let request_indices = read_fixture(&fixture, "request_indices_i32.bin", 4)?;
    let attention_query = read_fixture(
        &fixture,
        "attention_query_bf16.bin",
        ATTENTION_QUERY_HEADS * ATTENTION_HEAD_DIM * 2,
    )?;
    let block_table = read_fixture(&fixture, "block_table_i32.bin", 33 * 4)?;

    let expected_index_query = read_fixture(
        &fixture,
        "index_query_bf16.bin",
        PADDED_INDEX_HEADS * INDEX_HEAD_DIM * 2,
    )?;
    let expected_index_state = read_fixture(
        &fixture,
        "index_key_state_bf16.bin",
        INDEX_STATE_SLOTS * INDEX_HEAD_DIM * 2,
    )?;
    let expected_rope = read_fixture(
        &fixture,
        "rope_positions_i64.bin",
        INDEX_STATE_SLOTS * 3 * 8,
    )?;
    let expected_logits = read_fixture(&fixture, "logits_f32.bin", SCORE_COLUMNS * 4)?;
    let expected_blocks = read_fixture(&fixture, "block_indices_i32.bin", BLOCK_TOPK * 4)?;
    let expected_logical = read_fixture(&fixture, "logical_indices_i32.bin", FINAL_TOPK * 4)?;

    let plan = QsaSparseDecodePlan::qwen38_flash(1).map_err(io::Error::other)?;
    let arena_layout = plan.scratch_layout().map_err(io::Error::other)?;
    let expected_valid = read_fixture(&fixture, "valid_counts_i32.bin", 4)?;
    let expected_attention = read_fixture(
        &fixture,
        "attention_output_bf16.bin",
        ATTENTION_QUERY_HEADS * ATTENTION_HEAD_DIM * 2,
    )?;

    let mut front = sparkserve_runtime::coherent::CoherentRegionOwner::slab(
        u64::try_from(layout.total_bytes).map_err(|_| io::Error::other("layout overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: the fresh mapping has not been submitted to CUDA.
    let payload = unsafe { front.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.qk, &qk);
    copy_at(payload, layout.q_weight, &q_weight);
    copy_at(payload, layout.k_weight, &k_weight);
    copy_at(payload, layout.cos_sin, &cos_sin);
    copy_at(payload, layout.axis_map, &axis_map);
    copy_at(payload, layout.positions, &positions);
    copy_at(payload, layout.cache_locs, &cache_locs);
    copy_at(payload, layout.compressed_key_cache, &compressed_key_cache);
    copy_at(
        payload,
        layout.compressed_page_table,
        &compressed_page_table,
    );
    copy_at(payload, layout.compressed_lengths, &compressed_lengths);
    copy_at(payload, layout.query_positions, &query_positions);
    copy_at(payload, layout.sequence_lengths, &sequence_lengths);
    copy_at(payload, layout.full_key_state, &full_key_state);
    copy_at(payload, layout.full_value_state, &full_value_state);
    copy_at(payload, layout.request_to_token, &request_to_token);
    copy_at(payload, layout.request_indices, &request_indices);
    copy_at(payload, layout.attention_query, &attention_query);

    let mut pipeline = QsaCoherentPipeline::allocate(plan, vec![1], COHERENT_REGION_PREFAULT)
        .map_err(io::Error::other)?;
    // SAFETY: no arena lease or CUDA operation exists yet.
    let arena_payload = unsafe { pipeline.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(
        arena_payload,
        arena_layout.block_tables_offset,
        &block_table,
    );

    let caps = DeviceCaps::gb10(arena_layout.attention_workspace_bytes as u64);
    let front_base = front.device_address();
    let addresses = pipeline.scheduler().arena().addresses();
    let mut stream = CudaStreamOwner::create().map_err(io::Error::other)?;
    let mut fence = QsaPipelineCudaFence::create().map_err(io::Error::other)?;

    let zero = pipeline
        .scheduler_mut()
        .begin_workspace_zero()
        .map_err(io::Error::other)?;
    // SAFETY: the zero lease owns this fixed workspace range.
    unsafe {
        stream.memset_async(
            zero.arena().addresses().attention_workspace,
            0,
            arena_layout.attention_workspace_bytes,
        )
    }
    .map_err(io::Error::other)?;
    fence
        .record_workspace_zero(&mut stream, zero)
        .map_err(io::Error::other)?;
    assert_eq!(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
        QsaPipelineCudaCompletion::WorkspaceReady
    );

    let prep = pipeline
        .scheduler_mut()
        .begin_token(1)
        .map_err(io::Error::other)?;
    assert_eq!(prep.stage(), QsaPipelineStage::IndexPrep);
    let prep_args = QsaIndexPrepArgs {
        struct_size: std::mem::size_of::<QsaIndexPrepArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaIndexPrepPlan::qwen38_flash(1, 0, INDEX_STATE_SLOTS as u32, 1, 1),
        qk: device_pointer(front_base, layout.qk)?.cast_const(),
        q_output: device_pointer(front_base, layout.index_query)?,
        q_norm_weight: device_pointer(front_base, layout.q_weight)?.cast_const(),
        k_norm_weight: device_pointer(front_base, layout.k_weight)?.cast_const(),
        cos_sin_cache: device_pointer(front_base, layout.cos_sin)?
            .cast::<f32>()
            .cast_const(),
        cos_sin_rows: ROTARY_ROWS as u64,
        axis_map: device_pointer(front_base, layout.axis_map)?
            .cast::<i32>()
            .cast_const(),
        positions: device_pointer(front_base, layout.positions)?
            .cast::<i64>()
            .cast_const(),
        positions_stride: 1,
        cache_locs: device_pointer(front_base, layout.cache_locs)?
            .cast::<i64>()
            .cast_const(),
        key_state: device_pointer(front_base, layout.index_key_state)?,
        rope_positions: device_pointer(front_base, layout.rope_positions)?.cast::<i64>(),
        group_locs: std::ptr::null(),
        write_locs: std::ptr::null(),
        compressed_keys: std::ptr::null_mut(),
        eps: 1.0e-6,
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    if let Err(error) =
        status_result(unsafe { sparkserve_qsa_index_prep_launch(&caps, &prep_args) })
    {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, prep)
        .map_err(io::Error::other)?;
    let prep_ready = stage_ready(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
    )?;

    let score = pipeline
        .scheduler_mut()
        .begin_next(prep_ready)
        .map_err(io::Error::other)?;
    assert_eq!(score.stage(), QsaPipelineStage::Score);
    let score_args = QsaScoreArgs {
        struct_size: std::mem::size_of::<QsaScoreArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaScorePlan::qwen38_flash(1, COMPRESSED_PAGES as u32, COMPRESSED_MAX_PAGES as u32),
        query: device_pointer(front_base, layout.index_query)?.cast_const(),
        key_cache: device_pointer(front_base, layout.compressed_key_cache)?.cast_const(),
        page_table: device_pointer(front_base, layout.compressed_page_table)?
            .cast::<i32>()
            .cast_const(),
        context_lengths: device_pointer(front_base, layout.compressed_lengths)?
            .cast::<i32>()
            .cast_const(),
        logits: device_pointer(front_base, layout.logits)?.cast::<f32>(),
        score_scale: (INDEX_HEAD_DIM as f32).sqrt(),
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    if let Err(error) = status_result(unsafe { sparkserve_qsa_score_launch(&caps, &score_args) }) {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, score)
        .map_err(io::Error::other)?;
    let score_ready = stage_ready(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
    )?;

    let topk = pipeline
        .scheduler_mut()
        .begin_next(score_ready)
        .map_err(io::Error::other)?;
    assert_eq!(topk.stage(), QsaPipelineStage::BlockTopk);
    let topk_args = QsaTopkArgs {
        struct_size: std::mem::size_of::<QsaTopkArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaTopkPlan::qwen38_flash(1, SCORE_COLUMNS as u32, SCORE_COLUMNS as u64),
        scores: device_pointer(front_base, layout.logits)?
            .cast::<f32>()
            .cast_const(),
        row_starts: device_pointer(front_base, layout.row_start)?
            .cast::<i32>()
            .cast_const(),
        lengths: device_pointer(front_base, layout.compressed_lengths)?
            .cast::<i32>()
            .cast_const(),
        indices: device_pointer(front_base, layout.block_indices)?.cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    if let Err(error) = status_result(unsafe { sparkserve_qsa_topk_launch(&caps, &topk_args) }) {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, topk)
        .map_err(io::Error::other)?;
    let topk_ready = stage_ready(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
    )?;

    let expand = pipeline
        .scheduler_mut()
        .begin_next(topk_ready)
        .map_err(io::Error::other)?;
    assert_eq!(expand.stage(), QsaPipelineStage::SelectionExpand);
    let expand_args = QsaExpandArgs {
        struct_size: std::mem::size_of::<QsaExpandArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaExpandPlan::qwen38_flash(1),
        block_indices: device_pointer(front_base, layout.block_indices)?
            .cast::<i32>()
            .cast_const(),
        query_positions: device_pointer(front_base, layout.query_positions)?
            .cast::<i64>()
            .cast_const(),
        sequence_lengths: device_pointer(front_base, layout.sequence_lengths)?
            .cast::<i32>()
            .cast_const(),
        logical_indices: device_pointer(front_base, layout.logical_indices)?.cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    if let Err(error) = status_result(unsafe { sparkserve_qsa_expand_launch(&caps, &expand_args) })
    {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, expand)
        .map_err(io::Error::other)?;
    let expand_ready = stage_ready(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
    )?;

    let pack = pipeline
        .scheduler_mut()
        .begin_next(expand_ready)
        .map_err(io::Error::other)?;
    assert_eq!(pack.stage(), QsaPipelineStage::KvPack);
    let pack_args = QsaKvPackArgs {
        struct_size: std::mem::size_of::<QsaKvPackArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaKvPackPlan::qwen38_flash(1, KV_STATE_SLOTS as u32, 1, REQUEST_STRIDE as u32),
        key_state: device_pointer(front_base, layout.full_key_state)?.cast_const(),
        value_state: device_pointer(front_base, layout.full_value_state)?.cast_const(),
        req_to_token: device_pointer(front_base, layout.request_to_token)?
            .cast::<i32>()
            .cast_const(),
        request_indices: device_pointer(front_base, layout.request_indices)?
            .cast::<i32>()
            .cast_const(),
        logical_indices: device_pointer(front_base, layout.logical_indices)?
            .cast::<i32>()
            .cast_const(),
        sequence_lengths: device_pointer(front_base, layout.sequence_lengths)?
            .cast::<i32>()
            .cast_const(),
        valid_counts: raw_pointer(addresses.valid_counts)?.cast::<i32>(),
        packed_key: raw_pointer(addresses.packed_key)?,
        packed_value: raw_pointer(addresses.packed_value)?,
        cuda_stream: stream.raw(),
    };
    if let Err(error) = status_result(unsafe { sparkserve_qsa_kv_pack_launch(&caps, &pack_args) }) {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, pack)
        .map_err(io::Error::other)?;
    let pack_ready = stage_ready(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
    )?;

    let decode = pipeline
        .scheduler_mut()
        .begin_next(pack_ready)
        .map_err(io::Error::other)?;
    assert_eq!(decode.stage(), QsaPipelineStage::Decode);
    let decode_args = QsaDecodeArgs {
        struct_size: std::mem::size_of::<QsaDecodeArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaDecodePlan::qwen38_flash(1, 48),
        query: device_pointer(front_base, layout.attention_query)?.cast_const(),
        packed_key: raw_pointer(addresses.packed_key)?.cast_const(),
        packed_value: raw_pointer(addresses.packed_value)?.cast_const(),
        block_tables: raw_pointer(addresses.block_tables)?
            .cast::<i32>()
            .cast_const(),
        sequence_lengths: raw_pointer(addresses.valid_counts)?
            .cast::<i32>()
            .cast_const(),
        output: raw_pointer(addresses.attention_output)?,
        workspace: raw_pointer(addresses.attention_workspace)?,
        workspace_bytes: arena_layout.attention_workspace_bytes as u64,
        bmm1_scale: 0.0625,
        bmm2_scale: 1.0,
        cuda_stream: stream.raw(),
    };
    if let Err(error) = status_result(unsafe { sparkserve_qsa_decode_launch(&caps, &decode_args) })
    {
        return Err(io::Error::other(error));
    }
    fence
        .record_stage(&mut stream, decode)
        .map_err(io::Error::other)?;
    assert_eq!(
        fence
            .wait(pipeline.scheduler_mut())
            .map_err(io::Error::other)?,
        QsaPipelineCudaCompletion::TokenComplete
    );

    // SAFETY: the final fence completed every front-end and arena operation.
    let payload = unsafe { front.host_payload_mut() }.map_err(io::Error::other)?;
    let actual_index_query =
        &payload[layout.index_query..layout.index_query + expected_index_query.len()];
    let actual_index_state =
        &payload[layout.index_key_state..layout.index_key_state + expected_index_state.len()];
    let actual_rope = &payload[layout.rope_positions..layout.rope_positions + expected_rope.len()];
    let actual_logits = &payload[layout.logits..layout.logits + expected_logits.len()];
    let actual_blocks =
        &payload[layout.block_indices..layout.block_indices + expected_blocks.len()];
    let actual_logical =
        &payload[layout.logical_indices..layout.logical_indices + expected_logical.len()];
    let front_mismatches = [
        element_mismatches(actual_index_query, &expected_index_query, 2),
        element_mismatches(actual_index_state, &expected_index_state, 2),
        element_mismatches(actual_rope, &expected_rope, 8),
        element_mismatches(
            &actual_logits[..COMPRESSED_LENGTH * 4],
            &expected_logits[..COMPRESSED_LENGTH * 4],
            4,
        ),
        usize::from(sorted_i32(actual_blocks) != sorted_i32(&expected_blocks)),
        usize::from(sorted_i32(actual_logical) != sorted_i32(&expected_logical)),
    ];

    // SAFETY: token completion released the pack/XQA arena.
    let arena = unsafe { pipeline.host_payload_mut() }.map_err(io::Error::other)?;
    let actual_packed_key = &arena[arena_layout.packed_key_offset
        ..arena_layout.packed_key_offset + arena_layout.packed_key_bytes];
    let actual_packed_value = &arena[arena_layout.packed_value_offset
        ..arena_layout.packed_value_offset + arena_layout.packed_value_bytes];
    let actual_valid = &arena[arena_layout.valid_counts_offset
        ..arena_layout.valid_counts_offset + arena_layout.valid_counts_bytes];
    let actual_attention = &arena[arena_layout.attention_output_offset
        ..arena_layout.attention_output_offset + arena_layout.attention_output_bytes];
    // Radix top-k promises a selected set, not a stable order. Verify that the
    // compacted K/V rows match this execution's selection exactly, then allow
    // the small BF16 reduction-order difference in the downstream attention.
    let exact_order_mismatches = element_mismatches(actual_logical, &expected_logical, 4);
    let arena_mismatches = [
        ordered_pack_mismatches(
            actual_packed_key,
            &full_key_state,
            &request_to_token,
            actual_logical,
        ),
        ordered_pack_mismatches(
            actual_packed_value,
            &full_value_state,
            &request_to_token,
            actual_logical,
        ),
        usize::from(actual_valid != expected_valid),
    ];
    let (attention_exact_mismatches, attention_max_abs) =
        bf16_difference(actual_attention, &expected_attention);
    println!(
        "Rust QSA chain mismatches q/state/rope/score/block-set/token-set/packed-k/packed-v/length: \
         {}/{}/{}/{}/{}/{}/{}/{}/{}; selection-order differences: {}; attention BF16 exact/max-abs: {}/{}",
        front_mismatches[0],
        front_mismatches[1],
        front_mismatches[2],
        front_mismatches[3],
        front_mismatches[4],
        front_mismatches[5],
        arena_mismatches[0],
        arena_mismatches[1],
        arena_mismatches[2],
        exact_order_mismatches,
        attention_exact_mismatches,
        attention_max_abs,
    );
    if front_mismatches.into_iter().sum::<usize>() + arena_mismatches.into_iter().sum::<usize>()
        != 0
        || attention_max_abs > 0.03125
    {
        return Err(io::Error::other(
            "Rust QSA decode chain differs from oracle",
        ));
    }
    let stats = pipeline.scheduler().stats();
    if stats.tokens_completed != 1 || stats.stages_completed != 6 {
        return Err(io::Error::other(
            "Rust QSA lease machine did not complete six stages",
        ));
    }

    stream.close().map_err(io::Error::other)?;
    front.close().map_err(io::Error::other)?;
    drop(pipeline);
    Ok(())
}
