use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::cuda::{CudaEventOwner, CudaStreamOwner, status_result};
use sparkserve_runtime::ffi::{
    COHERENT_REGION_PREFAULT, DeviceCaps, QsaExpandArgs, QsaExpandPlan, QsaIndexPrepArgs,
    QsaIndexPrepPlan, QsaScoreArgs, QsaScorePlan, QsaTopkArgs, QsaTopkPlan,
    sparkserve_qsa_expand_launch, sparkserve_qsa_index_prep_launch, sparkserve_qsa_score_launch,
    sparkserve_qsa_topk_launch,
};
use std::ffi::c_void;
use std::io;
use std::path::{Path, PathBuf};

const ALIGNMENT: usize = 256;
const INDEX_TOKENS: usize = 37;
const INDEX_GROUPS: usize = 9;
const INDEX_STATE_SLOTS: usize = 128;
const INDEX_COMPRESSED_SLOTS: usize = 64;
const INDEX_HEAD_DIM: usize = 128;
const INDEX_QUERY_HEADS: usize = 4;
const INDEX_QUERY_HEADS_PADDED: usize = 8;
const INDEX_POSITION_ROWS: usize = 512;
const TOPK_ROWS: usize = 4;
const TOPK_COLUMNS: usize = 65_536;
const TOPK_WIDTH: usize = 512;
const EXPAND_ROWS: usize = 6;
const EXPAND_BLOCK_TOPK: usize = 512;
const EXPAND_FINAL_TOPK: usize = 2051;
const SCORE_BATCH: usize = 3;
const SCORE_HEADS: usize = 8;
const SCORE_HEAD_DIM: usize = 128;
const SCORE_PAGES: usize = 41;
const SCORE_PAGE_SIZE: usize = 16;
const SCORE_MAX_PAGES: usize = 17;
const SCORE_MAX_MODEL_LEN: usize = SCORE_MAX_PAGES * SCORE_PAGE_SIZE;

#[derive(Default)]
struct LayoutBuilder {
    next: usize,
}

impl LayoutBuilder {
    fn field(&mut self, bytes: usize) -> io::Result<usize> {
        let offset = align(self.next)?;
        self.next = offset
            .checked_add(bytes)
            .ok_or_else(|| io::Error::other("coherent QSA fixture layout overflow"))?;
        Ok(offset)
    }

    fn finish(self) -> io::Result<usize> {
        align(self.next)
    }
}

#[derive(Clone, Copy)]
struct IndexLayout {
    qk: usize,
    q_weight: usize,
    k_weight: usize,
    cos_sin: usize,
    axis_map: usize,
    positions: usize,
    cache_locs: usize,
    group_locs: usize,
    write_locs: usize,
    q_output: usize,
    key_state: usize,
    rope_positions: usize,
    compressed_keys: usize,
    total_bytes: usize,
}

impl IndexLayout {
    fn qwen() -> io::Result<Self> {
        let mut layout = LayoutBuilder::default();
        let qk = layout.field(INDEX_TOKENS * (INDEX_QUERY_HEADS + 1) * INDEX_HEAD_DIM * 2)?;
        let q_weight = layout.field(INDEX_HEAD_DIM * 2)?;
        let k_weight = layout.field(INDEX_HEAD_DIM * 2)?;
        let cos_sin = layout.field(INDEX_POSITION_ROWS * INDEX_HEAD_DIM * 4)?;
        let axis_map = layout.field(INDEX_HEAD_DIM / 2 * 4)?;
        let positions = layout.field(INDEX_TOKENS * 8)?;
        let cache_locs = layout.field(INDEX_TOKENS * 8)?;
        let group_locs = layout.field(INDEX_GROUPS * 4 * 4)?;
        let write_locs = layout.field(INDEX_GROUPS * 4)?;
        let q_output =
            layout.field(INDEX_TOKENS * INDEX_QUERY_HEADS_PADDED * INDEX_HEAD_DIM * 2)?;
        let key_state = layout.field(INDEX_STATE_SLOTS * INDEX_HEAD_DIM * 2)?;
        let rope_positions = layout.field(INDEX_STATE_SLOTS * 3 * 8)?;
        let compressed_keys = layout.field(INDEX_COMPRESSED_SLOTS * INDEX_HEAD_DIM * 2)?;
        let total_bytes = layout.finish()?;
        Ok(Self {
            qk,
            q_weight,
            k_weight,
            cos_sin,
            axis_map,
            positions,
            cache_locs,
            group_locs,
            write_locs,
            q_output,
            key_state,
            rope_positions,
            compressed_keys,
            total_bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct TopkLayout {
    scores: usize,
    row_starts: usize,
    lengths: usize,
    indices: usize,
    total_bytes: usize,
}

impl TopkLayout {
    fn qwen() -> io::Result<Self> {
        let mut layout = LayoutBuilder::default();
        let scores = layout.field(TOPK_ROWS * TOPK_COLUMNS * 4)?;
        let row_starts = layout.field(TOPK_ROWS * 4)?;
        let lengths = layout.field(TOPK_ROWS * 4)?;
        let indices = layout.field(TOPK_ROWS * TOPK_WIDTH * 4)?;
        let total_bytes = layout.finish()?;
        Ok(Self {
            scores,
            row_starts,
            lengths,
            indices,
            total_bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct ExpandLayout {
    block_indices: usize,
    query_positions: usize,
    sequence_lengths: usize,
    logical_indices: usize,
    total_bytes: usize,
}

impl ExpandLayout {
    fn qwen() -> io::Result<Self> {
        let mut layout = LayoutBuilder::default();
        let block_indices = layout.field(EXPAND_ROWS * EXPAND_BLOCK_TOPK * 4)?;
        let query_positions = layout.field(EXPAND_ROWS * 8)?;
        let sequence_lengths = layout.field(EXPAND_ROWS * 4)?;
        let logical_indices = layout.field(EXPAND_ROWS * EXPAND_FINAL_TOPK * 4)?;
        let total_bytes = layout.finish()?;
        Ok(Self {
            block_indices,
            query_positions,
            sequence_lengths,
            logical_indices,
            total_bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct ScoreLayout {
    query: usize,
    key_cache: usize,
    page_table: usize,
    context_lengths: usize,
    logits: usize,
    total_bytes: usize,
}

impl ScoreLayout {
    fn qwen() -> io::Result<Self> {
        let mut layout = LayoutBuilder::default();
        let query = layout.field(SCORE_BATCH * SCORE_HEADS * SCORE_HEAD_DIM * 2)?;
        let key_cache = layout.field(SCORE_PAGES * SCORE_PAGE_SIZE * SCORE_HEAD_DIM * 2)?;
        let page_table = layout.field(SCORE_BATCH * SCORE_MAX_PAGES * 4)?;
        let context_lengths = layout.field(SCORE_BATCH * 4)?;
        let logits = layout.field(SCORE_BATCH * SCORE_MAX_MODEL_LEN * 4)?;
        let total_bytes = layout.finish()?;
        Ok(Self {
            query,
            key_cache,
            page_table,
            context_lengths,
            logits,
            total_bytes,
        })
    }
}

fn align(value: usize) -> io::Result<usize> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|sum| sum / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| io::Error::other("coherent QSA fixture alignment overflow"))
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
    let address = usize::try_from(address)
        .map_err(|_| io::Error::other("device address does not fit usize"))?;
    Ok(address as *mut c_void)
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn element_mismatches(left: &[u8], right: &[u8], width: usize) -> usize {
    left.chunks_exact(width)
        .zip(right.chunks_exact(width))
        .filter(|(left, right)| left != right)
        .count()
}

fn run_index_prep(
    root: &Path,
    caps: &DeviceCaps,
    stream: &mut CudaStreamOwner,
    event: &mut CudaEventOwner,
) -> io::Result<()> {
    let layout = IndexLayout::qwen()?;
    let qk = read_fixture(
        root,
        "qk_bf16.bin",
        INDEX_TOKENS * (INDEX_QUERY_HEADS + 1) * INDEX_HEAD_DIM * 2,
    )?;
    let q_weight = read_fixture(root, "q_weight_bf16.bin", INDEX_HEAD_DIM * 2)?;
    let k_weight = read_fixture(root, "k_weight_bf16.bin", INDEX_HEAD_DIM * 2)?;
    let cos_sin = read_fixture(
        root,
        "cos_sin_f32.bin",
        INDEX_POSITION_ROWS * INDEX_HEAD_DIM * 4,
    )?;
    let axis_map = read_fixture(root, "axis_map_i32.bin", INDEX_HEAD_DIM / 2 * 4)?;
    let positions = read_fixture(root, "positions_i64.bin", INDEX_TOKENS * 8)?;
    let cache_locs = read_fixture(root, "cache_locs_i64.bin", INDEX_TOKENS * 8)?;
    let group_locs = read_fixture(root, "group_locs_i32.bin", INDEX_GROUPS * 4 * 4)?;
    let write_locs = read_fixture(root, "write_locs_i32.bin", INDEX_GROUPS * 4)?;
    let expected_q = read_fixture(
        root,
        "q_output_bf16.bin",
        INDEX_TOKENS * INDEX_QUERY_HEADS_PADDED * INDEX_HEAD_DIM * 2,
    )?;
    let expected_state = read_fixture(
        root,
        "key_state_bf16.bin",
        INDEX_STATE_SLOTS * INDEX_HEAD_DIM * 2,
    )?;
    let expected_rope = read_fixture(root, "rope_positions_i64.bin", INDEX_STATE_SLOTS * 3 * 8)?;
    let expected_compressed = read_fixture(
        root,
        "compressed_bf16.bin",
        INDEX_COMPRESSED_SLOTS * INDEX_HEAD_DIM * 2,
    )?;

    let mut region = CoherentRegionOwner::slab(
        u64::try_from(layout.total_bytes).map_err(|_| io::Error::other("layout overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: no CUDA work references this freshly allocated mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.qk, &qk);
    copy_at(payload, layout.q_weight, &q_weight);
    copy_at(payload, layout.k_weight, &k_weight);
    copy_at(payload, layout.cos_sin, &cos_sin);
    copy_at(payload, layout.axis_map, &axis_map);
    copy_at(payload, layout.positions, &positions);
    copy_at(payload, layout.cache_locs, &cache_locs);
    copy_at(payload, layout.group_locs, &group_locs);
    copy_at(payload, layout.write_locs, &write_locs);

    let base = region.device_address();
    let args = QsaIndexPrepArgs {
        struct_size: std::mem::size_of::<QsaIndexPrepArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaIndexPrepPlan::qwen38_flash(
            INDEX_TOKENS as u32,
            INDEX_GROUPS as u32,
            INDEX_STATE_SLOTS as u32,
            INDEX_COMPRESSED_SLOTS as u32,
            1,
        ),
        qk: device_pointer(base, layout.qk)?.cast_const(),
        q_output: device_pointer(base, layout.q_output)?,
        q_norm_weight: device_pointer(base, layout.q_weight)?.cast_const(),
        k_norm_weight: device_pointer(base, layout.k_weight)?.cast_const(),
        cos_sin_cache: device_pointer(base, layout.cos_sin)?
            .cast::<f32>()
            .cast_const(),
        cos_sin_rows: INDEX_POSITION_ROWS as u64,
        axis_map: device_pointer(base, layout.axis_map)?
            .cast::<i32>()
            .cast_const(),
        positions: device_pointer(base, layout.positions)?
            .cast::<i64>()
            .cast_const(),
        positions_stride: INDEX_TOKENS as u64,
        cache_locs: device_pointer(base, layout.cache_locs)?
            .cast::<i64>()
            .cast_const(),
        key_state: device_pointer(base, layout.key_state)?,
        rope_positions: device_pointer(base, layout.rope_positions)?.cast::<i64>(),
        group_locs: device_pointer(base, layout.group_locs)?
            .cast::<i32>()
            .cast_const(),
        write_locs: device_pointer(base, layout.write_locs)?
            .cast::<i32>()
            .cast_const(),
        compressed_keys: device_pointer(base, layout.compressed_keys)?,
        eps: 1.0e-6,
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    // SAFETY: every pointer targets a live coherent mapping and remains stable
    // until the recorded event is synchronized below.
    if let Err(error) = status_result(unsafe { sparkserve_qsa_index_prep_launch(caps, &args) }) {
        let _ = stream.synchronize();
        return Err(io::Error::other(error));
    }
    event.record(stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;

    // SAFETY: event completion excludes all CUDA access to this mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    let q = &payload[layout.q_output..layout.q_output + expected_q.len()];
    let state = &payload[layout.key_state..layout.key_state + expected_state.len()];
    let rope = &payload[layout.rope_positions..layout.rope_positions + expected_rope.len()];
    let compressed =
        &payload[layout.compressed_keys..layout.compressed_keys + expected_compressed.len()];
    let mismatches = [
        element_mismatches(q, &expected_q, 2),
        element_mismatches(state, &expected_state, 2),
        element_mismatches(rope, &expected_rope, 8),
        element_mismatches(compressed, &expected_compressed, 2),
    ];
    println!(
        "Rust QSA prep mismatches q/state/rope/compressed: {}/{}/{}/{}",
        mismatches[0], mismatches[1], mismatches[2], mismatches[3]
    );
    if mismatches.into_iter().sum::<usize>() != 0 {
        return Err(io::Error::other("Rust QSA prep differs from oracle"));
    }
    region.close().map_err(io::Error::other)
}

fn run_topk(
    root: &Path,
    caps: &DeviceCaps,
    stream: &mut CudaStreamOwner,
    event: &mut CudaEventOwner,
) -> io::Result<()> {
    let layout = TopkLayout::qwen()?;
    let scores = read_fixture(root, "scores_f32.bin", TOPK_ROWS * TOPK_COLUMNS * 4)?;
    let row_starts = read_fixture(root, "row_starts_i32.bin", TOPK_ROWS * 4)?;
    let lengths = read_fixture(root, "lengths_i32.bin", TOPK_ROWS * 4)?;
    let expected = read_fixture(root, "indices_i32.bin", TOPK_ROWS * TOPK_WIDTH * 4)?;

    let mut region = CoherentRegionOwner::slab(
        u64::try_from(layout.total_bytes).map_err(|_| io::Error::other("layout overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: no CUDA work references this freshly allocated mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.scores, &scores);
    copy_at(payload, layout.row_starts, &row_starts);
    copy_at(payload, layout.lengths, &lengths);

    let base = region.device_address();
    let args = QsaTopkArgs {
        struct_size: std::mem::size_of::<QsaTopkArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaTopkPlan::qwen38_flash(TOPK_ROWS as u32, TOPK_COLUMNS as u32, TOPK_COLUMNS as u64),
        scores: device_pointer(base, layout.scores)?
            .cast::<f32>()
            .cast_const(),
        row_starts: device_pointer(base, layout.row_starts)?
            .cast::<i32>()
            .cast_const(),
        lengths: device_pointer(base, layout.lengths)?
            .cast::<i32>()
            .cast_const(),
        indices: device_pointer(base, layout.indices)?.cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    // SAFETY: every pointer targets a live coherent mapping and remains stable
    // until the recorded event is synchronized below.
    if let Err(error) = status_result(unsafe { sparkserve_qsa_topk_launch(caps, &args) }) {
        let _ = stream.synchronize();
        return Err(io::Error::other(error));
    }
    event.record(stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;

    // SAFETY: event completion excludes all CUDA access to this mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    let mut actual = i32_values(&payload[layout.indices..layout.indices + expected.len()]);
    let expected = i32_values(&expected);
    let mut mismatched_rows = 0;
    for row in 0..TOPK_ROWS {
        let range = row * TOPK_WIDTH..(row + 1) * TOPK_WIDTH;
        actual[range.clone()].sort_unstable();
        let mut expected_row = expected[range].to_vec();
        expected_row.sort_unstable();
        mismatched_rows +=
            usize::from(actual[row * TOPK_WIDTH..(row + 1) * TOPK_WIDTH] != expected_row);
    }
    println!("Rust QSA radix top-k selected-set mismatched rows: {mismatched_rows}");
    if mismatched_rows != 0 {
        return Err(io::Error::other("Rust QSA top-k differs from oracle"));
    }
    region.close().map_err(io::Error::other)
}

fn run_score(
    root: &Path,
    caps: &DeviceCaps,
    stream: &mut CudaStreamOwner,
    event: &mut CudaEventOwner,
) -> io::Result<()> {
    let layout = ScoreLayout::qwen()?;
    let query = read_fixture(
        root,
        "q_bf16.bin",
        SCORE_BATCH * SCORE_HEADS * SCORE_HEAD_DIM * 2,
    )?;
    let key_cache = read_fixture(
        root,
        "k_cache_bf16.bin",
        SCORE_PAGES * SCORE_PAGE_SIZE * SCORE_HEAD_DIM * 2,
    )?;
    let page_table = read_fixture(
        root,
        "page_table_i32.bin",
        SCORE_BATCH * SCORE_MAX_PAGES * 4,
    )?;
    let context_lengths = read_fixture(root, "context_lens_i32.bin", SCORE_BATCH * 4)?;
    let expected = read_fixture(
        root,
        "logits_f32.bin",
        SCORE_BATCH * SCORE_MAX_MODEL_LEN * 4,
    )?;
    let lengths = i32_values(&context_lengths);

    let mut region = CoherentRegionOwner::slab(
        u64::try_from(layout.total_bytes).map_err(|_| io::Error::other("layout overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: no CUDA work references this freshly allocated mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.query, &query);
    copy_at(payload, layout.key_cache, &key_cache);
    copy_at(payload, layout.page_table, &page_table);
    copy_at(payload, layout.context_lengths, &context_lengths);
    for value in payload[layout.logits..layout.logits + expected.len()].chunks_exact_mut(4) {
        value.copy_from_slice(&f32::NAN.to_ne_bytes());
    }

    let base = region.device_address();
    let args = QsaScoreArgs {
        struct_size: std::mem::size_of::<QsaScoreArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaScorePlan::qwen38_flash(
            SCORE_BATCH as u32,
            SCORE_PAGES as u32,
            SCORE_MAX_PAGES as u32,
        ),
        query: device_pointer(base, layout.query)?.cast_const(),
        key_cache: device_pointer(base, layout.key_cache)?.cast_const(),
        page_table: device_pointer(base, layout.page_table)?
            .cast::<i32>()
            .cast_const(),
        context_lengths: device_pointer(base, layout.context_lengths)?
            .cast::<i32>()
            .cast_const(),
        logits: device_pointer(base, layout.logits)?.cast::<f32>(),
        score_scale: (SCORE_HEAD_DIM as f32).sqrt(),
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    // SAFETY: every pointer targets a live coherent mapping and remains stable
    // until the recorded event is synchronized below.
    if let Err(error) = status_result(unsafe { sparkserve_qsa_score_launch(caps, &args) }) {
        let _ = stream.synchronize();
        return Err(io::Error::other(error));
    }
    event.record(stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;

    // SAFETY: event completion excludes all CUDA access to this mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    let actual = &payload[layout.logits..layout.logits + expected.len()];
    let mut mismatches = 0;
    let mut compared = 0;
    for (row, &length) in lengths.iter().enumerate() {
        let length = usize::try_from(length).map_err(|_| io::Error::other("negative length"))?;
        let begin = row * SCORE_MAX_MODEL_LEN * 4;
        let end = begin + length * 4;
        mismatches += element_mismatches(&actual[begin..end], &expected[begin..end], 4);
        compared += length;
    }
    println!("Rust QSA score mismatches: {mismatches}/{compared}");
    if mismatches != 0 {
        return Err(io::Error::other("Rust QSA score differs from oracle"));
    }
    region.close().map_err(io::Error::other)
}

fn run_expand(
    root: &Path,
    caps: &DeviceCaps,
    stream: &mut CudaStreamOwner,
    event: &mut CudaEventOwner,
) -> io::Result<()> {
    let layout = ExpandLayout::qwen()?;
    let block_indices = read_fixture(
        root,
        "block_indices_i32.bin",
        EXPAND_ROWS * EXPAND_BLOCK_TOPK * 4,
    )?;
    let query_positions = read_fixture(root, "query_positions_i64.bin", EXPAND_ROWS * 8)?;
    let sequence_lengths = read_fixture(root, "sequence_lengths_i32.bin", EXPAND_ROWS * 4)?;
    let expected = read_fixture(
        root,
        "logical_indices_i32.bin",
        EXPAND_ROWS * EXPAND_FINAL_TOPK * 4,
    )?;

    let mut region = CoherentRegionOwner::slab(
        u64::try_from(layout.total_bytes).map_err(|_| io::Error::other("layout overflow"))?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: no CUDA work references this freshly allocated mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.block_indices, &block_indices);
    copy_at(payload, layout.query_positions, &query_positions);
    copy_at(payload, layout.sequence_lengths, &sequence_lengths);

    let base = region.device_address();
    let args = QsaExpandArgs {
        struct_size: std::mem::size_of::<QsaExpandArgs>() as u32,
        abi_version: sparkserve_runtime::kernel::KERNEL_ABI_VERSION,
        plan: QsaExpandPlan::qwen38_flash(EXPAND_ROWS as u32),
        block_indices: device_pointer(base, layout.block_indices)?
            .cast::<i32>()
            .cast_const(),
        query_positions: device_pointer(base, layout.query_positions)?
            .cast::<i64>()
            .cast_const(),
        sequence_lengths: device_pointer(base, layout.sequence_lengths)?
            .cast::<i32>()
            .cast_const(),
        logical_indices: device_pointer(base, layout.logical_indices)?.cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    // SAFETY: every pointer targets a live coherent mapping and remains stable
    // until the recorded event is synchronized below.
    if let Err(error) = status_result(unsafe { sparkserve_qsa_expand_launch(caps, &args) }) {
        let _ = stream.synchronize();
        return Err(io::Error::other(error));
    }
    event.record(stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;

    // SAFETY: event completion excludes all CUDA access to this mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    let actual = &payload[layout.logical_indices..layout.logical_indices + expected.len()];
    let mismatches = element_mismatches(actual, &expected, 4);
    println!("Rust QSA block expansion mismatches: {mismatches}");
    if mismatches != 0 {
        return Err(io::Error::other(
            "Rust QSA block expansion differs from oracle",
        ));
    }
    region.close().map_err(io::Error::other)
}

fn main() -> io::Result<()> {
    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let index_fixture = arguments.next().ok_or_else(|| {
        io::Error::other(
            "usage: qsa_frontend_smoke <index-fixture> <topk-fixture> <expand-fixture> <score-fixture>",
        )
    })?;
    let topk_fixture = arguments.next().ok_or_else(|| {
        io::Error::other(
            "usage: qsa_frontend_smoke <index-fixture> <topk-fixture> <expand-fixture> <score-fixture>",
        )
    })?;
    let expand_fixture = arguments.next().ok_or_else(|| {
        io::Error::other(
            "usage: qsa_frontend_smoke <index-fixture> <topk-fixture> <expand-fixture> <score-fixture>",
        )
    })?;
    let score_fixture = arguments.next().ok_or_else(|| {
        io::Error::other(
            "usage: qsa_frontend_smoke <index-fixture> <topk-fixture> <expand-fixture> <score-fixture>",
        )
    })?;
    if arguments.next().is_some() {
        return Err(io::Error::other(
            "usage: qsa_frontend_smoke <index-fixture> <topk-fixture> <expand-fixture> <score-fixture>",
        ));
    }

    let caps = DeviceCaps::gb10(0);
    let mut stream = CudaStreamOwner::create().map_err(io::Error::other)?;
    let mut event = CudaEventOwner::create().map_err(io::Error::other)?;
    run_index_prep(&index_fixture, &caps, &mut stream, &mut event)?;
    run_score(&score_fixture, &caps, &mut stream, &mut event)?;
    run_topk(&topk_fixture, &caps, &mut stream, &mut event)?;
    run_expand(&expand_fixture, &caps, &mut stream, &mut event)?;
    event.close().map_err(io::Error::other)?;
    stream.close().map_err(io::Error::other)
}
