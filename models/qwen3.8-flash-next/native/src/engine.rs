//! Batch-one native Qwen3.8 Flash-Next execution owner.
//!
//! The example entry point retains the same-token hot-path benchmark, while
//! `QwenNativeEngine` exposes persistent sequence state for the standalone
//! decoder and OpenAI service integration.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_void};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use crate::checkpoint::{FlashNextCheckpoint, load_flash_next_checkpoint};
use crate::coherent::CoherentRegionOwner;
use crate::cuda::{CudaBlasOwner, CudaStreamOwner};
use crate::ffi::{
    DeviceCaps, GdnBlockArgs, GdnBlockPlan, GdnDecodeArgs, GdnDecodePlan, GroupedNvfp4Args,
    GroupedNvfp4Plan, GroupedNvfp4WeightView, MhcArgs, MhcPlan, MoeGateArgs, MoeGatePlan,
    MoeJoinArgs, MoeJoinPlan, MoeRouteArgs, MoeRoutePlan, Nvfp4MatrixView, PleGatherArgs,
    PleGatherPlan, PleRowFragment, QsaDecodeArgs, QsaDecodePlan, QsaExpandArgs,
    QsaExpandPlan, QsaIndexPrepArgs, QsaIndexPrepPlan, QsaKvPackArgs, QsaKvPackPlan,
    QsaScoreArgs, QsaScorePlan, QsaTopkArgs, QsaTopkPlan,
    QWEN_DECODE_GLUE_ABI_VERSION, QWEN_GDN_AUX_ABI_VERSION,
    QWEN_PLE_BLOCK_ABI_VERSION, QWEN_QSA_BLOCK_ABI_VERSION, QwenBf16ToF32Args,
    QwenDecodeGlueArgs, QwenLmHeadArgs, QwenPleBlockArgs,
    QwenQsaFinishArgs, QwenQsaProjectArgs, SegmentedNvfp4QuantizeArgs,
    SegmentedNvfp4QuantizePlan, SegmentedSiluNvfp4Args, SegmentedSiluNvfp4Plan,
    SharedExpertArgs, SharedExpertPlan, Status,
    flash_qwen_runtime_gdn_finish as flash_gdn_block_finish_launch,
    flash_qwen_runtime_gdn_prepare as flash_gdn_block_prepare_launch,
    flash_qwen_runtime_gdn_decode as flash_gdn_decode_launch,
    flash_qwen_runtime_grouped_nvfp4 as flash_grouped_nvfp4_launch,
    flash_qwen_runtime_mhc_combine as flash_mhc_combine_launch,
    flash_qwen_runtime_mhc_mix as flash_mhc_mix_launch,
    flash_qwen_runtime_moe_gate as flash_moe_gate_launch,
    flash_qwen_runtime_moe_join as flash_moe_join_launch,
    flash_qwen_runtime_moe_dispatch as flash_moe_route_dispatch,
    flash_qwen_runtime_moe_finalize as flash_moe_route_finalize,
    flash_qwen_runtime_ple_gather as flash_ple_gather_launch,
    flash_qwen_add_hyper_launch,
    flash_qwen_runtime_bf16_to_f32 as flash_qwen_bf16_to_f32_launch,
    flash_qwen_lm_head_launch,
    flash_qwen_ple_block_launch, flash_qwen_qsa_finish_launch,
    flash_qwen_qsa_project_launch,
    flash_qwen_repeat_embedding_launch,
    flash_qwen_runtime_segmented_quantize as flash_segmented_nvfp4_quantize_launch,
    flash_qwen_runtime_segmented_silu as flash_segmented_silu_nvfp4_launch,
    flash_qwen_runtime_shared_expert as flash_shared_expert_launch,
    flash_qsa_decode_launch, flash_qsa_expand_launch,
    flash_qsa_index_prep_launch, flash_qsa_kv_pack_launch,
    flash_qsa_score_launch, flash_qsa_topk_launch,
};
use crate::kernel::{
    GroupedNvfp4Spec, KERNEL_ABI_VERSION, SegmentedNvfp4QuantizeSpec,
    SegmentedSiluNvfp4Spec,
};
use crate::fabric::{ExpertKey, ExpertLoad, ExpertSlotAddress};
use crate::qwen_expert_cache::{
    QwenExpertFileLoader, QwenExpertHotCache, QwenPreparedExpertCache,
    QwenPreparedExpertStats,
};
use crate::qwen_ple::decode_row_ids;
use crate::qwen_weights::{FlashNextWeightMaps, QwenTensorView};
use crate::routing::RoutePlan;
use crate::storage::{FixedPleCache, PleIndex};
use crate::tokenizer::{NativeQwenTokenizer, QWEN_MODEL_MAX_LENGTH};

const HIDDEN: u64 = 2560;
const HC: u64 = 4;
const HYPER: u64 = HIDDEN * HC;
const LOWRANK: u64 = 320;
const LAYERS: u32 = 48;
const QK_HEADS: u64 = 16;
const VALUE_HEADS: u64 = 48;
const HEAD_DIM: u64 = 128;
const QK_WIDTH: u64 = QK_HEADS * HEAD_DIM;
const VALUE_WIDTH: u64 = VALUE_HEADS * HEAD_DIM;
const GDN_CONV_WIDTH: u64 = 2 * QK_WIDTH + VALUE_WIDTH;
const QUERY_HEADS: u64 = 24;
const KV_HEADS: u64 = 2;
const QSA_HEAD_DIM: u64 = 256;
const QUERY_WIDTH: u64 = QUERY_HEADS * QSA_HEAD_DIM;
const KV_WIDTH: u64 = KV_HEADS * QSA_HEAD_DIM;
const INDEX_WIDTH: u64 = 640;
const INDEX_HEAD_DIM: u64 = 128;
const ROTARY_DIM: u32 = 64;
const INTERMEDIATE: u64 = 640;
const TOP_K: u32 = 10;
const ACTIVE_EXPERTS: u32 = 10;
const PREFILL_CHUNK_TOKENS: u64 = 16;
// Keep every donor group active. FlashInfer's SM120 grouped kernel can accept
// empty groups in its metadata, but the current binary has shown a late,
// route-dependent illegal-instruction failure when six spare groups are
// present. Ten slots still preserve zero-copy hits: misses overwrite experts
// that are not selected by the current token.
const LAYER_EXPERT_SLOTS: u32 = ACTIVE_EXPERTS;
// A private prepared range per layer keeps likely follow-on routes in the
// unified-memory tier while the donor grouped-GEMM ABI remains ten groups.
const PREPARED_SLOTS_PER_LAYER: u32 = 32;
const PREPARED_EXPERT_SLOTS: u32 = LAYERS * PREPARED_SLOTS_PER_LAYER;
const WORKSPACE: u64 = 32 * 1024 * 1024;
const QSA_WORKSPACE: u64 = 128 * 1024 * 1024;
const VOCABULARY: u64 = 248_320;
const MOE_PADDED_ROWS: u64 = ACTIVE_EXPERTS as u64 * 4;
const MOE_SCALE_ROWS: u64 = LAYER_EXPERT_SLOTS as u64 * 128;
const QSA_LAYERS: u64 = 12;
const QSA_COMPRESS_RATIO: u64 = 4;
const QSA_COMPRESSED_SLOTS: u64 = QWEN_MODEL_MAX_LENGTH as u64 / QSA_COMPRESS_RATIO;
const QSA_COMPRESSED_PAGE_SIZE: u64 = 16;
const QSA_COMPRESSED_PAGES: u64 = QSA_COMPRESSED_SLOTS / QSA_COMPRESSED_PAGE_SIZE;
const QSA_BLOCK_TOPK: u64 = 512;
const QSA_FINAL_TOPK: u64 = 2051;
const QSA_PACKED_TOKENS: u64 = 2112;
const QSA_XQA_PAGES: u64 = 33;

/// Fixed-address, batch-one decode scratch. Production execution allocates
/// these regions once and reuses them for every layer and token; no donor
/// kernel owns an allocation.
struct QwenDecodeArena {
    mhc_normed: CoherentRegionOwner,
    mhc_down: CoherentRegionOwner,
    mhc_activated: CoherentRegionOwner,
    mhc_up: CoherentRegionOwner,
    mixed: CoherentRegionOwner,
    hyper_mid: CoherentRegionOwner,
    attention: CoherentRegionOwner,
    moe_output: CoherentRegionOwner,
    gdn_a_log: CoherentRegionOwner,
    gdn_dt: CoherentRegionOwner,
    gdn_projected_qkv: CoherentRegionOwner,
    gdn_projected_z: CoherentRegionOwner,
    gdn_projected_b: CoherentRegionOwner,
    gdn_projected_a: CoherentRegionOwner,
    gdn_convolved: CoherentRegionOwner,
    gdn_core: CoherentRegionOwner,
    gdn_gated: CoherentRegionOwner,
    state_index: CoherentRegionOwner,
    qsa_projected_q: CoherentRegionOwner,
    qsa_projected_k: CoherentRegionOwner,
    qsa_query: CoherentRegionOwner,
    qsa_key: CoherentRegionOwner,
    qsa_value: CoherentRegionOwner,
    qsa_gate: CoherentRegionOwner,
    qsa_index_qk: CoherentRegionOwner,
    qsa_cos_sin: CoherentRegionOwner,
    qsa_positions: CoherentRegionOwner,
    qsa_cache_locs: CoherentRegionOwner,
    qsa_axis_map: CoherentRegionOwner,
    qsa_index_query: CoherentRegionOwner,
    qsa_compressed_page_table: CoherentRegionOwner,
    qsa_compressed_lengths: CoherentRegionOwner,
    qsa_logits: CoherentRegionOwner,
    qsa_row_start: CoherentRegionOwner,
    qsa_block_indices: CoherentRegionOwner,
    qsa_query_positions: CoherentRegionOwner,
    qsa_sequence_lengths: CoherentRegionOwner,
    qsa_logical_indices: CoherentRegionOwner,
    qsa_request_to_token: CoherentRegionOwner,
    qsa_request_indices: CoherentRegionOwner,
    qsa_group_locs: CoherentRegionOwner,
    qsa_write_locs: CoherentRegionOwner,
    qsa_packed_key: CoherentRegionOwner,
    qsa_packed_value: CoherentRegionOwner,
    qsa_valid_counts: CoherentRegionOwner,
    qsa_xqa_block_table: CoherentRegionOwner,
    qsa_attention_workspace: CoherentRegionOwner,
    qsa_raw_attention: CoherentRegionOwner,
    qsa_gated: CoherentRegionOwner,
    router_logits: CoherentRegionOwner,
    route_weights: CoherentRegionOwner,
    route_ids: CoherentRegionOwner,
    route_map: CoherentRegionOwner,
    m_indptr: CoherentRegionOwner,
    packed_input: CoherentRegionOwner,
    input_fp4: CoherentRegionOwner,
    input_scales: CoherentRegionOwner,
    gate_up: CoherentRegionOwner,
    down_input: CoherentRegionOwner,
    down_scales: CoherentRegionOwner,
    expert_output: CoherentRegionOwner,
    int_workspace: CoherentRegionOwner,
    float_workspace: CoherentRegionOwner,
    shared_gate_up: CoherentRegionOwner,
    shared_activated: CoherentRegionOwner,
    shared_output: CoherentRegionOwner,
    ple_fragments: CoherentRegionOwner,
    ple_embedding: CoherentRegionOwner,
    ple_key: CoherentRegionOwner,
    ple_value: CoherentRegionOwner,
    ple_gated: CoherentRegionOwner,
    ple_normed: CoherentRegionOwner,
    ple_delta: CoherentRegionOwner,
    final_dummy: CoherentRegionOwner,
    final_hidden: CoherentRegionOwner,
    final_combined: CoherentRegionOwner,
    logits: CoherentRegionOwner,
}

impl QwenDecodeArena {
    fn create() -> Result<Self, Box<dyn std::error::Error>> {
        let mut qsa_cos_sin = slab(
            u64::try_from(QWEN_MODEL_MAX_LENGTH)? * u64::from(ROTARY_DIM) * 4,
        )?;
        initialize_rope_cache(&mut qsa_cos_sin)?;
        let mut qsa_compressed_page_table = slab(QSA_COMPRESSED_PAGES * 4)?;
        write_identity_i32(&mut qsa_compressed_page_table, QSA_COMPRESSED_PAGES)?;
        let mut qsa_request_to_token = slab(u64::try_from(QWEN_MODEL_MAX_LENGTH)? * 4)?;
        write_identity_i32(
            &mut qsa_request_to_token,
            u64::try_from(QWEN_MODEL_MAX_LENGTH)?,
        )?;
        let mut qsa_xqa_block_table = slab(QSA_XQA_PAGES * 4)?;
        write_identity_i32(&mut qsa_xqa_block_table, QSA_XQA_PAGES)?;
        Ok(Self {
            mhc_normed: slab(PREFILL_CHUNK_TOKENS * HYPER * 2)?,
            mhc_down: slab(PREFILL_CHUNK_TOKENS * LOWRANK * 2)?,
            mhc_activated: slab(PREFILL_CHUNK_TOKENS * LOWRANK * 2)?,
            mhc_up: slab(PREFILL_CHUNK_TOKENS * HYPER * 2)?,
            mixed: slab(PREFILL_CHUNK_TOKENS * HIDDEN * 2)?,
            hyper_mid: slab(PREFILL_CHUNK_TOKENS * HYPER * 2)?,
            attention: slab(PREFILL_CHUNK_TOKENS * HIDDEN * 2)?,
            moe_output: slab(PREFILL_CHUNK_TOKENS * HIDDEN * 2)?,
            gdn_a_log: slab(VALUE_HEADS * 4)?,
            gdn_dt: slab(VALUE_HEADS * 4)?,
            gdn_projected_qkv: slab(PREFILL_CHUNK_TOKENS * GDN_CONV_WIDTH * 2)?,
            gdn_projected_z: slab(PREFILL_CHUNK_TOKENS * VALUE_WIDTH * 2)?,
            gdn_projected_b: slab(PREFILL_CHUNK_TOKENS * VALUE_HEADS * 2)?,
            gdn_projected_a: slab(PREFILL_CHUNK_TOKENS * VALUE_HEADS * 2)?,
            gdn_convolved: slab(PREFILL_CHUNK_TOKENS * GDN_CONV_WIDTH * 2)?,
            gdn_core: slab(PREFILL_CHUNK_TOKENS * VALUE_WIDTH * 2)?,
            gdn_gated: slab(PREFILL_CHUNK_TOKENS * VALUE_WIDTH * 2)?,
            state_index: slab(4)?,
            qsa_projected_q: slab(2 * QUERY_WIDTH * 2)?,
            qsa_projected_k: slab(KV_WIDTH * 2)?,
            qsa_query: slab(QUERY_WIDTH * 2)?,
            qsa_key: slab(KV_WIDTH * 2)?,
            qsa_value: slab(KV_WIDTH * 2)?,
            qsa_gate: slab(QUERY_WIDTH * 2)?,
            qsa_index_qk: slab(INDEX_WIDTH * 2)?,
            qsa_cos_sin,
            qsa_positions: slab(8)?,
            qsa_cache_locs: slab(8)?,
            qsa_axis_map: slab(u64::from(ROTARY_DIM / 2) * 4)?,
            qsa_index_query: slab(8 * INDEX_HEAD_DIM * 2)?,
            qsa_compressed_page_table,
            qsa_compressed_lengths: slab(4)?,
            qsa_logits: slab(QSA_COMPRESSED_SLOTS * 4)?,
            qsa_row_start: slab(4)?,
            qsa_block_indices: slab(QSA_BLOCK_TOPK * 4)?,
            qsa_query_positions: slab(8)?,
            qsa_sequence_lengths: slab(4)?,
            qsa_logical_indices: slab(QSA_FINAL_TOPK * 4)?,
            qsa_request_to_token,
            qsa_request_indices: slab(4)?,
            qsa_group_locs: slab(QSA_COMPRESS_RATIO * 4)?,
            qsa_write_locs: slab(4)?,
            qsa_packed_key: slab(QSA_PACKED_TOKENS * KV_WIDTH * 2)?,
            qsa_packed_value: slab(QSA_PACKED_TOKENS * KV_WIDTH * 2)?,
            qsa_valid_counts: slab(4)?,
            qsa_xqa_block_table,
            qsa_attention_workspace: slab(QSA_WORKSPACE)?,
            qsa_raw_attention: slab(QUERY_WIDTH * 2)?,
            qsa_gated: slab(QUERY_WIDTH * 2)?,
            router_logits: slab(PREFILL_CHUNK_TOKENS * 512 * 2)?,
            route_weights: slab(PREFILL_CHUNK_TOKENS * u64::from(TOP_K) * 4)?,
            route_ids: slab(PREFILL_CHUNK_TOKENS * u64::from(TOP_K) * 4)?,
            route_map: slab(u64::from(TOP_K) * 4)?,
            m_indptr: slab((u64::from(LAYER_EXPERT_SLOTS) + 1) * 4)?,
            packed_input: slab(MOE_PADDED_ROWS * HIDDEN * 2)?,
            input_fp4: slab(MOE_PADDED_ROWS * HIDDEN / 2)?,
            input_scales: slab(MOE_SCALE_ROWS * HIDDEN / 16)?,
            gate_up: slab(MOE_PADDED_ROWS * 2 * INTERMEDIATE * 2)?,
            down_input: slab(MOE_PADDED_ROWS * INTERMEDIATE / 2)?,
            down_scales: slab(MOE_SCALE_ROWS * INTERMEDIATE / 16)?,
            expert_output: slab(MOE_PADDED_ROWS * HIDDEN * 2)?,
            int_workspace: slab(WORKSPACE)?,
            float_workspace: slab(WORKSPACE)?,
            shared_gate_up: slab(2 * INTERMEDIATE * 2)?,
            shared_activated: slab(INTERMEDIATE * 2)?,
            shared_output: slab(HIDDEN * 2)?,
            ple_fragments: slab(u64::try_from(std::mem::size_of::<PleRowFragment>() * 16)?)?,
            ple_embedding: slab(HIDDEN * 2)?,
            ple_key: slab(HYPER * 2)?,
            ple_value: slab(HIDDEN * 2)?,
            ple_gated: slab(HYPER * 2)?,
            ple_normed: slab(HYPER * 2)?,
            ple_delta: slab(HYPER * 2)?,
            final_dummy: slab(HC * HYPER * 2)?,
            final_hidden: slab(HIDDEN * 2)?,
            final_combined: slab(HYPER * 2)?,
            logits: slab(VOCABULARY * 4)?,
        })
    }
}

/// One sequence-independent PLE storage owner. Field order is intentional:
/// the fixed reader and its registered `io_uring` are dropped before the
/// coherent region that backs their stable host/device aliases.
struct QwenPleRuntime {
    cache: FixedPleCache<'static>,
    index: PleIndex,
    multipliers: [i64; 3],
    sizes: [i64; 16],
    offsets: [i64; 16],
    _region: Box<CoherentRegionOwner>,
}

impl QwenPleRuntime {
    fn create(
        checkpoint: &FlashNextCheckpoint,
        model_root: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let index = PleIndex::decode(&std::fs::read(
            model_root.join(".spark.c/ple.ssple"),
        )?)?;
        let multipliers = read_checkpoint_i64(
            checkpoint,
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        )?
        .try_into()
        .map_err(|_| "bad PLE multipliers")?;
        let sizes = read_checkpoint_i64(
            checkpoint,
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
        )?
        .try_into()
        .map_err(|_| "bad PLE sizes")?;
        let offsets = read_checkpoint_i64(
            checkpoint,
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        )?
        .try_into()
        .map_err(|_| "bad PLE offsets")?;

        let region = Box::new(slab(4 * 1024 * 1024)?);
        // SAFETY: `region` is boxed before this reference is formed, so the
        // native mapping and the embedded view remain at stable addresses.
        // `_region` is declared after `cache`, which guarantees that the ring
        // and its borrowed fixed buffer are destroyed before the mapping.
        let view: &'static crate::ffi::CoherentRegionView =
            unsafe { &*(region.view() as *const _) };
        let cache = unsafe { FixedPleCache::open_coherent(&index, model_root, view, 32)? };
        Ok(Self {
            cache,
            index,
            multipliers,
            sizes,
            offsets,
            _region: region,
        })
    }
}

/// Aligned, CUDA-visible copies of only the BF16 tensors consumed by cuBLAS.
/// Safetensors payload offsets are not guaranteed to meet cuBLAS alignment,
/// while raw CUDA pack/gather kernels can continue reading the original mmap.
/// A tensor is staged at most once and remains resident for all later tokens.
struct QwenResidentWeights {
    tensors: BTreeMap<String, CoherentRegionOwner>,
    bytes: u64,
}

/// Stable identities for the ten donor groups while one layer processes a
/// short prefill bucket. Experts shared by adjacent tokens stay in place;
/// only genuine misses are repacked into slots not used by the current top-k.
struct QwenLayerExpertSlots {
    slot_experts: [Option<i32>; LAYER_EXPERT_SLOTS as usize],
    last_used: [u64; LAYER_EXPERT_SLOTS as usize],
    tick: u64,
}

struct QwenLayerExpertPlan {
    loads: Vec<ExpertLoad>,
    physical: Vec<u32>,
}

impl QwenLayerExpertSlots {
    fn new() -> Self {
        Self {
            slot_experts: [None; LAYER_EXPERT_SLOTS as usize],
            last_used: [0; LAYER_EXPERT_SLOTS as usize],
            tick: 0,
        }
    }

    fn plan(
        &mut self,
        layer: u32,
        logical: &[i32],
    ) -> Result<QwenLayerExpertPlan, Box<dyn std::error::Error>> {
        let requested = logical.iter().copied().collect::<BTreeSet<_>>();
        if requested.len() != ACTIVE_EXPERTS as usize {
            return Err("Qwen top-k experts are not unique".into());
        }
        let layer = u16::try_from(layer)?;
        let mut loads = Vec::new();
        for expert in requested.iter().copied() {
            let expert_id = u16::try_from(expert)?;
            if self.slot_experts.contains(&Some(expert)) {
                continue;
            }
            let slot = self
                .slot_experts
                .iter()
                .position(Option::is_none)
                .or_else(|| {
                    self.slot_experts
                        .iter()
                        .enumerate()
                        .filter(|(_, resident)| {
                            resident.is_some_and(|resident| !requested.contains(&resident))
                        })
                        .min_by_key(|(slot, _)| (self.last_used[*slot], *slot))
                        .map(|(slot, _)| slot)
                })
                .ok_or("Qwen expert slot planner cannot retain the current top-k")?;
            let evicts = self.slot_experts[slot]
                .map(|resident| -> Result<ExpertKey, std::num::TryFromIntError> {
                    Ok(ExpertKey {
                        layer,
                        expert: u16::try_from(resident)?,
                    })
                })
                .transpose()?;
            self.slot_experts[slot] = Some(expert);
            loads.push(ExpertLoad {
                key: ExpertKey {
                    layer,
                    expert: expert_id,
                },
                address: ExpertSlotAddress {
                    slot: u32::try_from(slot)?,
                    byte_offset: 0,
                },
                evicts,
            });
        }
        self.tick = self.tick.checked_add(1).ok_or("Qwen expert LRU tick overflow")?;
        let physical = logical
            .iter()
            .map(|expert| {
                let slot = self
                    .slot_experts
                    .iter()
                    .position(|resident| resident == &Some(*expert))
                    .ok_or("Qwen routed expert is absent from the hot bank")?;
                self.last_used[slot] = self.tick;
                Ok(u32::try_from(slot)?)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok(QwenLayerExpertPlan { loads, physical })
    }
}

impl QwenResidentWeights {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            bytes: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get(
        &mut self,
        maps: &mut FlashNextWeightMaps,
        stream: &mut CudaStreamOwner,
        name: &str,
        dtype: &str,
        shape: &[u64],
        bytes: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if !self.tensors.contains_key(name) {
            let source = checked_tensor(maps, name, dtype, shape, bytes)?;
            let output = slab(bytes)?;
            unsafe {
                stream.memcpy_async(
                    output.device_address(),
                    source.device_address,
                    usize::try_from(bytes)?,
                )?;
            }
            self.tensors.insert(name.to_owned(), output);
            self.bytes = self.bytes.checked_add(bytes).ok_or("resident weight bytes overflow")?;
        }
        Ok(self
            .tensors
            .get(name)
            .expect("resident tensor inserted")
            .device_address())
    }

    #[allow(clippy::too_many_arguments)]
    fn merged_bf16_pair(
        &mut self,
        maps: &mut FlashNextWeightMaps,
        stream: &mut CudaStreamOwner,
        key: &str,
        first_name: &str,
        second_name: &str,
        shape: &[u64],
        tensor_bytes: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if !self.tensors.contains_key(key) {
            let first = checked_tensor(maps, first_name, "BF16", shape, tensor_bytes)?;
            let second = checked_tensor(maps, second_name, "BF16", shape, tensor_bytes)?;
            let output = slab(tensor_bytes.checked_mul(2).ok_or("merged weight size overflow")?)?;
            unsafe {
                stream.memcpy_async(
                    output.device_address(),
                    first.device_address,
                    usize::try_from(tensor_bytes)?,
                )?;
                stream.memcpy_async(
                    output.device_address() + tensor_bytes,
                    second.device_address,
                    usize::try_from(tensor_bytes)?,
                )?;
            }
            self.tensors.insert(key.to_owned(), output);
            self.bytes = self
                .bytes
                .checked_add(tensor_bytes * 2)
                .ok_or("resident weight bytes overflow")?;
        }
        Ok(self
            .tensors
            .get(key)
            .expect("merged resident tensor inserted")
            .device_address())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QwenNativeStep {
    pub token: u32,
    pub elapsed_seconds: f64,
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub expert_evictions: u64,
}

/// One CUDA owner thread creates and exclusively uses this value. It remains
/// intentionally `!Send`/`!Sync` through its coherent-region and CUDA guards;
/// the HTTP backend communicates with that owner through channels.
pub struct QwenNativeEngine {
    stream: CudaStreamOwner,
    blas: CudaBlasOwner,
    caps: DeviceCaps,
    weight_maps: FlashNextWeightMaps,
    resident: QwenResidentWeights,
    hidden: CoherentRegionOwner,
    hyper_a: CoherentRegionOwner,
    hyper_b: CoherentRegionOwner,
    gdn_conv_states: CoherentRegionOwner,
    gdn_temporal_states: CoherentRegionOwner,
    ple_state: CoherentRegionOwner,
    ple: QwenPleRuntime,
    qsa_index_key_states: CoherentRegionOwner,
    qsa_rope_positions: CoherentRegionOwner,
    qsa_compressed_keys: CoherentRegionOwner,
    qsa_full_key_states: CoherentRegionOwner,
    qsa_full_value_states: CoherentRegionOwner,
    hot_experts: QwenExpertHotCache,
    prepared_experts: QwenPreparedExpertCache,
    expert_loader: QwenExpertFileLoader,
    expert_packs: u64,
    arena: QwenDecodeArena,
    token_history: Vec<u32>,
}

impl QwenNativeEngine {
    pub fn create(model_root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let checkpoint = load_flash_next_checkpoint(model_root)?;
        let ple = QwenPleRuntime::create(&checkpoint, model_root)?;
        let expert_loader = QwenExpertFileLoader::new(&checkpoint);
        let weight_maps = FlashNextWeightMaps::new(&checkpoint, 0);
        let mut hot_experts = QwenExpertHotCache::create(0)?;
        let mut prepared_experts = QwenPreparedExpertCache::create(PREPARED_EXPERT_SLOTS, 0)?;
        let prefault_started = Instant::now();
        let prefault_bytes = hot_experts
            .prefault()?
            .checked_add(prepared_experts.prefault()?)
            .ok_or("Qwen expert prefault byte count overflow")?;
        eprintln!(
            "Qwen expert arena prefault: {:.3} GiB in {:.3} s",
            prefault_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            prefault_started.elapsed().as_secs_f64(),
        );
        let mut engine = Self {
            stream: CudaStreamOwner::create()?,
            blas: CudaBlasOwner::create()?,
            caps: DeviceCaps::gb10(QSA_WORKSPACE),
            weight_maps,
            resident: QwenResidentWeights::new(),
            hidden: slab(PREFILL_CHUNK_TOKENS * HIDDEN * 2)?,
            hyper_a: slab(PREFILL_CHUNK_TOKENS * HYPER * 2)?,
            hyper_b: slab(PREFILL_CHUNK_TOKENS * HYPER * 2)?,
            gdn_conv_states: slab(36 * GDN_CONV_WIDTH * 3 * 2)?,
            gdn_temporal_states: slab(36 * VALUE_HEADS * HEAD_DIM * HEAD_DIM * 2)?,
            ple_state: slab(HYPER * 9 * 2)?,
            ple,
            qsa_index_key_states: slab(
                QSA_LAYERS
                    * u64::try_from(QWEN_MODEL_MAX_LENGTH)?
                    * INDEX_HEAD_DIM
                    * 2,
            )?,
            qsa_rope_positions: slab(
                QSA_LAYERS * u64::try_from(QWEN_MODEL_MAX_LENGTH)? * 3 * 8,
            )?,
            qsa_compressed_keys: slab(
                QSA_LAYERS * QSA_COMPRESSED_SLOTS * INDEX_HEAD_DIM * 2,
            )?,
            qsa_full_key_states: slab(
                QSA_LAYERS
                    * u64::try_from(QWEN_MODEL_MAX_LENGTH)?
                    * KV_WIDTH
                    * 2,
            )?,
            qsa_full_value_states: slab(
                QSA_LAYERS
                    * u64::try_from(QWEN_MODEL_MAX_LENGTH)?
                    * KV_WIDTH
                    * 2,
            )?,
            hot_experts,
            prepared_experts,
            expert_loader,
            expert_packs: 0,
            arena: QwenDecodeArena::create()?,
            token_history: Vec::new(),
        };
        engine.reset_sequence()?;
        if std::env::var_os("FLASH_QWEN_WARM_EXPERT_SOURCE").is_some() {
            let started = Instant::now();
            let bytes = engine.expert_loader.warm_expert_source()?;
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "Qwen expert mmap warmup: {:.3} GiB in {:.3} s ({:.3} GiB/s)",
                bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                elapsed,
                bytes as f64 / 1024.0 / 1024.0 / 1024.0
                    / elapsed.max(f64::MIN_POSITIVE),
            );
        }
        Ok(engine)
    }

    pub fn reset_sequence(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.stream.memset_async(
                self.gdn_conv_states.device_address(),
                0,
                self.gdn_conv_states.payload_bytes()?,
            )?;
            self.stream.memset_async(
                self.gdn_temporal_states.device_address(),
                0,
                self.gdn_temporal_states.payload_bytes()?,
            )?;
            self.stream.memset_async(
                self.ple_state.device_address(),
                0,
                self.ple_state.payload_bytes()?,
            )?;
            self.stream.memset_async(
                self.arena.qsa_attention_workspace.device_address(),
                0,
                self.arena.qsa_attention_workspace.payload_bytes()?,
            )?;
        }
        self.token_history.clear();
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        input_token: u32,
        verbose: bool,
    ) -> Result<QwenNativeStep, Box<dyn std::error::Error>> {
        self.forward_tokens(&[input_token], verbose)
    }

    /// Advance one fixed AOT prefill bucket. Layers remain outermost so their
    /// weights and mapped expert shard are reused across the whole short chunk.
    pub fn forward_tokens(
        &mut self,
        input_tokens: &[u32],
        verbose: bool,
    ) -> Result<QwenNativeStep, Box<dyn std::error::Error>> {
        if !matches!(input_tokens.len(), 1 | 2 | 4 | 8 | 16)
            || input_tokens
                .iter()
                .any(|token| u64::from(*token) >= VOCABULARY)
            || self.token_history.len() + input_tokens.len() > QWEN_MODEL_MAX_LENGTH
        {
            return Err("Qwen prefill bucket or input token is invalid".into());
        }
        let start_position = self.token_history.len();
        self.token_history.extend_from_slice(input_tokens);
        let num_tokens = u32::try_from(input_tokens.len())?;

        let embedding = self.resident.get(
            &mut self.weight_maps,
            &mut self.stream,
            "model.language_model.embed_tokens.weight",
            "BF16",
            &[VOCABULARY, HIDDEN],
            VOCABULARY * HIDDEN * 2,
        )?;
        for (offset, input_token) in input_tokens.iter().copied().enumerate() {
            let row = u64::try_from(offset)?;
            unsafe {
                self.stream.memcpy_async(
                    self.hidden.device_address() + row * HIDDEN * 2,
                    embedding + u64::from(input_token) * HIDDEN * 2,
                    usize::try_from(HIDDEN * 2)?,
                )?;
            }
            glue(
                "Qwen embedding repeat",
                self.hidden.device_address() + row * HIDDEN * 2,
                self.hyper_a.device_address() + row * HYPER * 2,
                &self.stream,
                flash_qwen_repeat_embedding_launch,
            )?;
        }

        let expert_stats_before = self.prepared_experts.stats();
        let started = Instant::now();
        for layer in 0..LAYERS {
            let (current, next) = if layer.is_multiple_of(2) {
                (&self.hyper_a, &self.hyper_b)
            } else {
                (&self.hyper_b, &self.hyper_a)
            };
            if layer == 1 {
                for token_offset in 0..input_tokens.len() {
                    run_ple(
                        &mut self.weight_maps,
                        &mut self.resident,
                        &mut self.ple,
                        &self.token_history[..start_position + token_offset + 1],
                        current.device_address() + u64::try_from(token_offset)? * HYPER * 2,
                        &self.ple_state,
                        &mut self.arena,
                        &mut self.stream,
                        &self.blas,
                        &self.caps,
                    )?;
                }
            }
            run_layer(
                &mut self.weight_maps,
                &mut self.resident,
                layer,
                current,
                next,
                &self.gdn_conv_states,
                &self.gdn_temporal_states,
                &self.qsa_index_key_states,
                &self.qsa_rope_positions,
                &self.qsa_compressed_keys,
                &self.qsa_full_key_states,
                &self.qsa_full_value_states,
                u32::try_from(start_position)?,
                num_tokens,
                &mut self.hot_experts,
                &mut self.prepared_experts,
                &mut self.expert_loader,
                &mut self.expert_packs,
                &mut self.arena,
                &mut self.stream,
                &self.blas,
                &self.caps,
                verbose,
            )?;
            if verbose {
                println!(
                    "Qwen positions {}..{} layer {layer}/47 complete",
                    start_position,
                    start_position + input_tokens.len() - 1,
                );
            }
        }
        let final_hyper_base = if LAYERS.is_multiple_of(2) {
            self.hyper_a.device_address()
        } else {
            self.hyper_b.device_address()
        };
        let final_hyper = final_hyper_base
            + u64::try_from(input_tokens.len() - 1)? * HYPER * 2;
        let token = finish_logits(
            &mut self.weight_maps,
            &mut self.resident,
            final_hyper,
            &self.arena,
            &mut self.stream,
            &self.blas,
            &self.caps,
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let expert_stats_after = self.prepared_experts.stats();
        Ok(QwenNativeStep {
            token,
            elapsed_seconds,
            expert_hits: expert_stats_after.hits - expert_stats_before.hits,
            expert_misses: expert_stats_after.misses - expert_stats_before.misses,
            expert_evictions: expert_stats_after.evictions - expert_stats_before.evictions,
        })
    }

    pub fn resident_weight_bytes(&self) -> u64 {
        self.resident.bytes
    }

    pub fn resident_weight_tensors(&self) -> usize {
        self.resident.tensors.len()
    }

    pub fn expert_stats(&self) -> QwenPreparedExpertStats {
        self.prepared_experts.stats()
    }
}

pub fn smoke_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_first_token <model-root> [input-token-id] [replays]"));
    let input_token = arguments
        .next()
        .map_or(Ok(9707_u32), |value| value.parse::<u32>())?;
    let replays = arguments
        .next()
        .map_or(Ok(1_u32), |value| value.parse::<u32>())?;
    if arguments.next().is_some()
        || u64::from(input_token) >= VOCABULARY
        || replays == 0
    {
        return Err("invalid Qwen input token or replay count".into());
    }
    let model_root = Path::new(&model);
    let mut engine = QwenNativeEngine::create(model_root)?;
    let tokenizer = NativeQwenTokenizer::from_model_root(model_root)?;
    for replay in 0..replays {
        engine.reset_sequence()?;
        let step = engine.forward_token(input_token, true)?;
        let text = tokenizer.decode(&[step.token], false)?;
        println!(
            "Qwen replay {replay} native token: {} {text:?}; {:.3} s ({:.3} token/s single-token replay)",
            step.token,
            step.elapsed_seconds,
            1.0 / step.elapsed_seconds,
        );
        println!(
            "Qwen replay {replay} GPU expert packs: {} resident, {} hits, {} packs, {} evictions",
            engine.expert_stats().resident,
            step.expert_hits,
            step.expert_misses,
            step.expert_evictions,
        );
    }
    println!(
        "Qwen resident cuBLAS weights: {} tensors, {:.3} GiB",
        engine.resident_weight_tensors(),
        engine.resident_weight_bytes() as f64 / 1024.0 / 1024.0 / 1024.0
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    maps: &mut FlashNextWeightMaps,
    resident: &mut QwenResidentWeights,
    layer: u32,
    hyper_input: &CoherentRegionOwner,
    hyper_output: &CoherentRegionOwner,
    gdn_conv_states: &CoherentRegionOwner,
    gdn_temporal_states: &CoherentRegionOwner,
    qsa_index_key_states: &CoherentRegionOwner,
    qsa_rope_positions: &CoherentRegionOwner,
    qsa_compressed_keys: &CoherentRegionOwner,
    qsa_full_key_states: &CoherentRegionOwner,
    qsa_full_value_states: &CoherentRegionOwner,
    start_position: u32,
    num_tokens: u32,
    hot_experts: &mut QwenExpertHotCache,
    prepared_experts: &mut QwenPreparedExpertCache,
    expert_loader: &mut QwenExpertFileLoader,
    expert_packs: &mut u64,
    arena: &mut QwenDecodeArena,
    stream: &mut CudaStreamOwner,
    blas: &CudaBlasOwner,
    caps: &DeviceCaps,
    trace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let attn_prefix = format!("model.language_model.layers.{layer}.attn_hyper_connection");
    let attn_weights = load_mhc_weights(maps, resident, stream, &attn_prefix)?;
    let attn_mhc = mhc_args(
        num_tokens,
        hyper_input.device_address(),
        arena.attention.device_address(),
        arena.hyper_mid.device_address(),
        &arena.mixed,
        &arena.mhc_normed,
        &arena.mhc_down,
        &arena.mhc_activated,
        &arena.mhc_up,
        &attn_weights,
        blas,
        stream,
    );
    native("Qwen attention mHC mix", unsafe {
        flash_mhc_mix_launch(caps, &attn_mhc)
    })?;
    trace_stage(stream, trace, layer, "attention mHC mix")?;
    if (layer + 1).is_multiple_of(4) {
        for token_offset in 0..num_tokens {
            run_qsa(
                maps,
                resident,
                layer,
                start_position + token_offset,
                arena.mixed.device_address() + u64::from(token_offset) * HIDDEN * 2,
                arena.attention.device_address() + u64::from(token_offset) * HIDDEN * 2,
                qsa_index_key_states,
                qsa_rope_positions,
                qsa_compressed_keys,
                qsa_full_key_states,
                qsa_full_value_states,
                arena,
                stream,
                blas,
                caps,
            )?;
            // QSA reuses scalar metadata and one-token scratch. Complete the
            // prior token before the CPU updates those coherent locations.
            stream.synchronize()?;
        }
    } else {
        run_gdn(
            maps,
            resident,
            layer,
            num_tokens,
            gdn_conv_states,
            gdn_temporal_states,
            arena,
            stream,
            blas,
            caps,
        )?;
    }
    trace_stage(stream, trace, layer, "attention")?;
    native("Qwen attention mHC combine", unsafe {
        flash_mhc_combine_launch(caps, &attn_mhc)
    })?;
    trace_stage(stream, trace, layer, "attention mHC combine")?;

    let mlp_prefix = format!("model.language_model.layers.{layer}.mlp_hyper_connection");
    let mlp_weights = load_mhc_weights(maps, resident, stream, &mlp_prefix)?;
    let mlp_mhc = mhc_args(
        num_tokens,
        arena.hyper_mid.device_address(),
        arena.moe_output.device_address(),
        hyper_output.device_address(),
        &arena.mixed,
        &arena.mhc_normed,
        &arena.mhc_down,
        &arena.mhc_activated,
        &arena.mhc_up,
        &mlp_weights,
        blas,
        stream,
    );
    native("Qwen MLP mHC mix", unsafe {
        flash_mhc_mix_launch(caps, &mlp_mhc)
    })?;
    trace_stage(stream, trace, layer, "MLP mHC mix")?;
    run_moe(
        maps,
        resident,
        layer,
        num_tokens,
        arena.mixed.device_address(),
        arena.moe_output.device_address(),
        arena,
        hot_experts,
        prepared_experts,
        expert_loader,
        expert_packs,
        stream,
        blas,
        caps,
        trace,
    )?;
    trace_stage(stream, trace, layer, "MoE")?;
    native("Qwen MLP mHC combine", unsafe {
        flash_mhc_combine_launch(caps, &mlp_mhc)
    })?;
    trace_stage(stream, trace, layer, "MLP mHC combine")?;
    Ok(())
}

fn trace_stage(
    stream: &mut CudaStreamOwner,
    trace: bool,
    layer: u32,
    stage: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if trace {
        stream.synchronize()?;
        eprintln!("Qwen trace layer {layer}: {stage} complete");
    }
    Ok(())
}

struct MhcWeights {
    norm: u64,
    down: u64,
    up: u64,
    inject: u64,
}

fn load_mhc_weights(
    maps: &mut FlashNextWeightMaps,
    resident: &mut QwenResidentWeights,
    stream: &mut CudaStreamOwner,
    prefix: &str,
) -> Result<MhcWeights, Box<dyn std::error::Error>> {
    Ok(MhcWeights {
        norm: resident.get(
            maps,
            stream,
            &format!("{prefix}.hc_norm.weight"),
            "BF16",
            &[HYPER],
            HYPER * 2,
        )?,
        down: resident.get(
            maps,
            stream,
            &format!("{prefix}.input_mix_weight_down.weight"),
            "BF16",
            &[LOWRANK, HYPER],
            LOWRANK * HYPER * 2,
        )?,
        up: resident.get(
            maps,
            stream,
            &format!("{prefix}.input_mix_weight_up.weight"),
            "BF16",
            &[HYPER, LOWRANK],
            HYPER * LOWRANK * 2,
        )?,
        inject: resident.get(
            maps,
            stream,
            &format!("{prefix}.block_inject_weight.weight"),
            "BF16",
            &[HC, HYPER],
            HC * HYPER * 2,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
fn mhc_args(
    num_tokens: u32,
    hyper_input: u64,
    block_output: u64,
    combined_output: u64,
    mixed: &CoherentRegionOwner,
    normed: &CoherentRegionOwner,
    down: &CoherentRegionOwner,
    activated: &CoherentRegionOwner,
    up: &CoherentRegionOwner,
    weights: &MhcWeights,
    blas: &CudaBlasOwner,
    stream: &CudaStreamOwner,
) -> MhcArgs {
    MhcArgs {
        struct_size: size::<MhcArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: MhcPlan::qwen38_flash(num_tokens),
        hyper_input: ptr(hyper_input),
        norm_weight: ptr(weights.norm),
        mix_down_weight: ptr(weights.down),
        mix_up_weight: ptr(weights.up),
        inject_weight: ptr(weights.inject),
        block_output: ptr(block_output),
        normed: ptr_mut(normed.device_address()),
        mix_down: ptr_mut(down.device_address()),
        mix_activated: ptr_mut(activated.device_address()),
        mix_up: ptr_mut(up.device_address()),
        mixed_output: ptr_mut(mixed.device_address()),
        combined_output: ptr_mut(combined_output),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gdn(
    maps: &mut FlashNextWeightMaps,
    resident: &mut QwenResidentWeights,
    layer: u32,
    num_tokens: u32,
    conv_states: &CoherentRegionOwner,
    temporal_states: &CoherentRegionOwner,
    arena: &QwenDecodeArena,
    stream: &mut CudaStreamOwner,
    blas: &CudaBlasOwner,
    caps: &DeviceCaps,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}.linear_attn");
    let qkv = resident.get(maps, stream, &format!("{prefix}.in_proj_qkv.weight"), "BF16", &[GDN_CONV_WIDTH, HIDDEN], GDN_CONV_WIDTH * HIDDEN * 2)?;
    let z = resident.get(maps, stream, &format!("{prefix}.in_proj_z.weight"), "BF16", &[VALUE_WIDTH, HIDDEN], VALUE_WIDTH * HIDDEN * 2)?;
    let b = resident.get(maps, stream, &format!("{prefix}.in_proj_b.weight"), "BF16", &[VALUE_HEADS, HIDDEN], VALUE_HEADS * HIDDEN * 2)?;
    let a = resident.get(maps, stream, &format!("{prefix}.in_proj_a.weight"), "BF16", &[VALUE_HEADS, HIDDEN], VALUE_HEADS * HIDDEN * 2)?;
    let conv = resident.get(maps, stream, &format!("{prefix}.conv1d.weight"), "BF16", &[GDN_CONV_WIDTH, 1, 4], GDN_CONV_WIDTH * 4 * 2)?;
    let norm = resident.get(maps, stream, &format!("{prefix}.norm.weight"), "BF16", &[HEAD_DIM], HEAD_DIM * 2)?;
    let out = resident.get(maps, stream, &format!("{prefix}.out_proj.weight"), "BF16", &[HIDDEN, VALUE_WIDTH], HIDDEN * VALUE_WIDTH * 2)?;
    let a_log_bf16 = resident.get(maps, stream, &format!("{prefix}.A_log"), "BF16", &[VALUE_HEADS], VALUE_HEADS * 2)?;
    let dt_bf16 = resident.get(maps, stream, &format!("{prefix}.dt_bias"), "BF16", &[VALUE_HEADS], VALUE_HEADS * 2)?;
    convert_bf16(stream, a_log_bf16, &arena.gdn_a_log, VALUE_HEADS)?;
    convert_bf16(stream, dt_bf16, &arena.gdn_dt, VALUE_HEADS)?;
    let gdn_ordinal = u64::from(layer - layer / 4);
    let conv_state = conv_states.device_address() + gdn_ordinal * GDN_CONV_WIDTH * 3 * 2;
    let temporal_state = temporal_states.device_address()
        + gdn_ordinal * VALUE_HEADS * HEAD_DIM * HEAD_DIM * 2;
    let block = GdnBlockArgs {
        struct_size: size::<GdnBlockArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: GdnBlockPlan::qwen38_flash_decode(num_tokens), hidden_states: ptr(arena.mixed.device_address()),
        in_proj_qkv_weight: ptr(qkv), in_proj_z_weight: ptr(z),
        in_proj_b_weight: ptr(b), in_proj_a_weight: ptr(a),
        conv_weight: ptr(conv), gated_norm_weight: ptr(norm),
        out_proj_weight: ptr(out), conv_state_pool: ptr_mut(conv_state),
        state_indices: ptr(arena.state_index.device_address()).cast::<i32>(), projected_qkv: ptr_mut(arena.gdn_projected_qkv.device_address()),
        projected_z: ptr_mut(arena.gdn_projected_z.device_address()), projected_b: ptr_mut(arena.gdn_projected_b.device_address()),
        projected_a: ptr_mut(arena.gdn_projected_a.device_address()), convolved_qkv: ptr_mut(arena.gdn_convolved.device_address()),
        gdn_core_output: ptr(arena.gdn_core.device_address()), gated_norm_output: ptr_mut(arena.gdn_gated.device_address()),
        attention_output: ptr_mut(arena.attention.device_address()), cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    let recurrence = GdnDecodeArgs {
        struct_size: size::<GdnDecodeArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: GdnDecodePlan::qwen38_flash_decode(1, 1), q: ptr(arena.gdn_convolved.device_address()),
        k: ptr(arena.gdn_convolved.device_address() + u64::from(num_tokens) * QK_WIDTH * 2),
        v: ptr(arena.gdn_convolved.device_address() + 2 * u64::from(num_tokens) * QK_WIDTH * 2),
        a: ptr(arena.gdn_projected_a.device_address()), b: ptr(arena.gdn_projected_b.device_address()),
        a_log: ptr(arena.gdn_a_log.device_address()).cast::<f32>(), dt_bias: ptr(arena.gdn_dt.device_address()).cast::<f32>(),
        state_pool: ptr_mut(temporal_state), state_indices: ptr(arena.state_index.device_address()).cast::<i32>(),
        output: ptr_mut(arena.gdn_core.device_address()), scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        sequence_length: num_tokens, cuda_stream: stream.raw(),
    };
    native("Qwen GDN prepare", unsafe { flash_gdn_block_prepare_launch(caps, &block) })?;
    native("Qwen GDN recurrence", unsafe { flash_gdn_decode_launch(caps, &recurrence) })?;
    native("Qwen GDN finish", unsafe { flash_gdn_block_finish_launch(caps, &block) })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_qsa(
    maps: &mut FlashNextWeightMaps,
    resident: &mut QwenResidentWeights,
    layer: u32,
    position: u32,
    hidden_states: u64,
    attention_output: u64,
    index_key_states: &CoherentRegionOwner,
    rope_positions: &CoherentRegionOwner,
    compressed_keys: &CoherentRegionOwner,
    full_key_states: &CoherentRegionOwner,
    full_value_states: &CoherentRegionOwner,
    arena: &mut QwenDecodeArena,
    stream: &mut CudaStreamOwner,
    blas: &CudaBlasOwner,
    caps: &DeviceCaps,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}.self_attn");
    write_i64_scalar(&mut arena.qsa_positions, i64::from(position))?;
    let q = resident.get(maps, stream, &format!("{prefix}.q_proj.weight"), "BF16", &[2 * QUERY_WIDTH, HIDDEN], 2 * QUERY_WIDTH * HIDDEN * 2)?;
    let k = resident.get(maps, stream, &format!("{prefix}.k_proj.weight"), "BF16", &[KV_WIDTH, HIDDEN], KV_WIDTH * HIDDEN * 2)?;
    let v = resident.get(maps, stream, &format!("{prefix}.v_proj.weight"), "BF16", &[KV_WIDTH, HIDDEN], KV_WIDTH * HIDDEN * 2)?;
    let index = resident.get(maps, stream, &format!("{prefix}.indexer.index_qk_proj.weight"), "BF16", &[INDEX_WIDTH, HIDDEN], INDEX_WIDTH * HIDDEN * 2)?;
    let q_norm = resident.get(maps, stream, &format!("{prefix}.q_norm.weight"), "BF16", &[QSA_HEAD_DIM], QSA_HEAD_DIM * 2)?;
    let k_norm = resident.get(maps, stream, &format!("{prefix}.k_norm.weight"), "BF16", &[QSA_HEAD_DIM], QSA_HEAD_DIM * 2)?;
    let index_q_norm = resident.get(maps, stream, &format!("{prefix}.indexer.q_layernorm.weight"), "BF16", &[INDEX_HEAD_DIM], INDEX_HEAD_DIM * 2)?;
    let index_k_norm = resident.get(maps, stream, &format!("{prefix}.indexer.k_layernorm.weight"), "BF16", &[INDEX_HEAD_DIM], INDEX_HEAD_DIM * 2)?;
    let out = resident.get(maps, stream, &format!("{prefix}.o_proj.weight"), "BF16", &[HIDDEN, QUERY_WIDTH], HIDDEN * QUERY_WIDTH * 2)?;

    let qsa_ordinal = u64::from(layer / 4);
    if qsa_ordinal >= QSA_LAYERS {
        return Err("Qwen QSA layer ordinal exceeds the persistent state".into());
    }
    let context_capacity = u64::try_from(QWEN_MODEL_MAX_LENGTH)?;
    let position_u64 = u64::from(position);
    let sequence_length = position.checked_add(1).ok_or("Qwen QSA position overflow")?;
    let compressed_length = sequence_length / u32::try_from(QSA_COMPRESS_RATIO)?;
    let groups = u32::from(sequence_length.is_multiple_of(u32::try_from(QSA_COMPRESS_RATIO)?));
    let index_state_base = index_key_states.device_address()
        + qsa_ordinal * context_capacity * INDEX_HEAD_DIM * 2;
    let rope_positions_base = rope_positions.device_address()
        + qsa_ordinal * context_capacity * 3 * 8;
    let compressed_keys_base = compressed_keys.device_address()
        + qsa_ordinal * QSA_COMPRESSED_SLOTS * INDEX_HEAD_DIM * 2;
    let full_key_base = full_key_states.device_address()
        + qsa_ordinal * context_capacity * KV_WIDTH * 2;
    let full_value_base = full_value_states.device_address()
        + qsa_ordinal * context_capacity * KV_WIDTH * 2;

    write_i64_scalar(&mut arena.qsa_cache_locs, i64::from(position))?;
    write_i64_scalar(&mut arena.qsa_query_positions, i64::from(position))?;
    write_i32(&mut arena.qsa_sequence_lengths, &[i32::try_from(sequence_length)?])?;
    write_i32(&mut arena.qsa_compressed_lengths, &[i32::try_from(compressed_length)?])?;
    write_i32(&mut arena.qsa_row_start, &[0])?;
    write_i32(&mut arena.qsa_request_indices, &[0])?;
    if groups != 0 {
        let first = position.checked_sub(3).ok_or("Qwen QSA compression group underflow")?;
        write_i32(
            &mut arena.qsa_group_locs,
            &[
                i32::try_from(first)?,
                i32::try_from(first + 1)?,
                i32::try_from(first + 2)?,
                i32::try_from(position)?,
            ],
        )?;
        write_i32(
            &mut arena.qsa_write_locs,
            &[i32::try_from(compressed_length - 1)?],
        )?;
    }

    let project = QwenQsaProjectArgs {
        struct_size: size::<QwenQsaProjectArgs>(), abi_version: QWEN_QSA_BLOCK_ABI_VERSION,
        tokens: 1, rotary_dim: ROTARY_DIM, cos_sin_stride: u64::from(ROTARY_DIM),
        hidden_states: ptr(hidden_states), q_weight: ptr(q),
        k_weight: ptr(k), v_weight: ptr(v),
        index_qk_weight: ptr(index), q_norm_weight: ptr(q_norm),
        k_norm_weight: ptr(k_norm), cos_sin_cache: ptr(arena.qsa_cos_sin.device_address()).cast::<f32>(),
        positions: ptr(arena.qsa_positions.device_address()).cast::<i64>(), projected_q: ptr_mut(arena.qsa_projected_q.device_address()),
        projected_k: ptr_mut(arena.qsa_projected_k.device_address()), query: ptr_mut(arena.qsa_query.device_address()),
        key: ptr_mut(arena.qsa_key.device_address()), value: ptr_mut(arena.qsa_value.device_address()),
        gate: ptr_mut(arena.qsa_gate.device_address()), index_qk: ptr_mut(arena.qsa_index_qk.device_address()),
        cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen QSA project", unsafe { flash_qwen_qsa_project_launch(&project) })?;

    unsafe {
        stream.memcpy_async(
            full_key_base + position_u64 * KV_WIDTH * 2,
            arena.qsa_key.device_address(),
            usize::try_from(KV_WIDTH * 2)?,
        )?;
        stream.memcpy_async(
            full_value_base + position_u64 * KV_WIDTH * 2,
            arena.qsa_value.device_address(),
            usize::try_from(KV_WIDTH * 2)?,
        )?;
    }

    let index_prep = QsaIndexPrepArgs {
        struct_size: size::<QsaIndexPrepArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaIndexPrepPlan::qwen38_flash_with_rotary(
            1,
            groups,
            u32::try_from(QWEN_MODEL_MAX_LENGTH)?,
            u32::try_from(QSA_COMPRESSED_SLOTS)?,
            1,
            ROTARY_DIM,
        ),
        qk: ptr(arena.qsa_index_qk.device_address()),
        q_output: ptr_mut(arena.qsa_index_query.device_address()),
        q_norm_weight: ptr(index_q_norm),
        k_norm_weight: if groups == 0 { std::ptr::null() } else { ptr(index_k_norm) },
        cos_sin_cache: ptr(arena.qsa_cos_sin.device_address()).cast::<f32>(),
        cos_sin_rows: context_capacity,
        axis_map: ptr(arena.qsa_axis_map.device_address()).cast::<i32>(),
        positions: ptr(arena.qsa_positions.device_address()).cast::<i64>(),
        positions_stride: 1,
        cache_locs: ptr(arena.qsa_cache_locs.device_address()).cast::<i64>(),
        key_state: ptr_mut(index_state_base),
        rope_positions: ptr_mut(rope_positions_base).cast::<i64>(),
        group_locs: if groups == 0 { std::ptr::null() } else { ptr(arena.qsa_group_locs.device_address()).cast::<i32>() },
        write_locs: if groups == 0 { std::ptr::null() } else { ptr(arena.qsa_write_locs.device_address()).cast::<i32>() },
        compressed_keys: if groups == 0 { std::ptr::null_mut() } else { ptr_mut(compressed_keys_base) },
        eps: 1.0e-6,
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA index prep", unsafe {
        flash_qsa_index_prep_launch(caps, &index_prep)
    })?;

    let max_pages = compressed_length
        .div_ceil(u32::try_from(QSA_COMPRESSED_PAGE_SIZE)?)
        .max(1);
    let score_plan = QsaScorePlan::qwen38_flash(
        1,
        u32::try_from(QSA_COMPRESSED_PAGES)?,
        max_pages,
    );
    let score = QsaScoreArgs {
        struct_size: size::<QsaScoreArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: score_plan,
        query: ptr(arena.qsa_index_query.device_address()),
        key_cache: ptr(compressed_keys_base),
        page_table: ptr(arena.qsa_compressed_page_table.device_address()).cast::<i32>(),
        context_lengths: ptr(arena.qsa_compressed_lengths.device_address()).cast::<i32>(),
        logits: ptr_mut(arena.qsa_logits.device_address()).cast::<f32>(),
        score_scale: (INDEX_HEAD_DIM as f32).sqrt(),
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA index score", unsafe {
        flash_qsa_score_launch(caps, &score)
    })?;

    let topk = QsaTopkArgs {
        struct_size: size::<QsaTopkArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaTopkPlan::qwen38_flash(
            1,
            score_plan.max_model_len,
            u64::from(score_plan.max_model_len),
        ),
        scores: ptr(arena.qsa_logits.device_address()).cast::<f32>(),
        row_starts: ptr(arena.qsa_row_start.device_address()).cast::<i32>(),
        lengths: ptr(arena.qsa_compressed_lengths.device_address()).cast::<i32>(),
        indices: ptr_mut(arena.qsa_block_indices.device_address()).cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA block top-k", unsafe {
        flash_qsa_topk_launch(caps, &topk)
    })?;

    let expand = QsaExpandArgs {
        struct_size: size::<QsaExpandArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaExpandPlan::qwen38_flash(1),
        block_indices: ptr(arena.qsa_block_indices.device_address()).cast::<i32>(),
        query_positions: ptr(arena.qsa_query_positions.device_address()).cast::<i64>(),
        sequence_lengths: ptr(arena.qsa_sequence_lengths.device_address()).cast::<i32>(),
        logical_indices: ptr_mut(arena.qsa_logical_indices.device_address()).cast::<i32>(),
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA token expansion", unsafe {
        flash_qsa_expand_launch(caps, &expand)
    })?;

    let pack = QsaKvPackArgs {
        struct_size: size::<QsaKvPackArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaKvPackPlan::qwen38_flash(
            1,
            u32::try_from(QWEN_MODEL_MAX_LENGTH)?,
            1,
            u32::try_from(QWEN_MODEL_MAX_LENGTH)?,
        ),
        key_state: ptr(full_key_base),
        value_state: ptr(full_value_base),
        req_to_token: ptr(arena.qsa_request_to_token.device_address()).cast::<i32>(),
        request_indices: ptr(arena.qsa_request_indices.device_address()).cast::<i32>(),
        logical_indices: ptr(arena.qsa_logical_indices.device_address()).cast::<i32>(),
        sequence_lengths: ptr(arena.qsa_sequence_lengths.device_address()).cast::<i32>(),
        valid_counts: ptr_mut(arena.qsa_valid_counts.device_address()).cast::<i32>(),
        packed_key: ptr_mut(arena.qsa_packed_key.device_address()),
        packed_value: ptr_mut(arena.qsa_packed_value.device_address()),
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA selected K/V pack", unsafe {
        flash_qsa_kv_pack_launch(caps, &pack)
    })?;

    let decode = QsaDecodeArgs {
        struct_size: size::<QsaDecodeArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaDecodePlan::qwen38_flash(1, 48),
        query: ptr(arena.qsa_query.device_address()),
        packed_key: ptr(arena.qsa_packed_key.device_address()),
        packed_value: ptr(arena.qsa_packed_value.device_address()),
        block_tables: ptr(arena.qsa_xqa_block_table.device_address()).cast::<i32>(),
        sequence_lengths: ptr(arena.qsa_valid_counts.device_address()).cast::<i32>(),
        output: ptr_mut(arena.qsa_raw_attention.device_address()),
        workspace: ptr_mut(arena.qsa_attention_workspace.device_address()),
        workspace_bytes: QSA_WORKSPACE,
        bmm1_scale: 1.0 / (QSA_HEAD_DIM as f32).sqrt(),
        bmm2_scale: 1.0,
        cuda_stream: stream.raw(),
    };
    native("Qwen QSA sparse decode", unsafe {
        flash_qsa_decode_launch(caps, &decode)
    })?;

    let finish = QwenQsaFinishArgs {
        struct_size: size::<QwenQsaFinishArgs>(), abi_version: QWEN_QSA_BLOCK_ABI_VERSION,
        tokens: 1, reserved: 0, attention_output: ptr(arena.qsa_raw_attention.device_address()),
        gate: ptr(arena.qsa_gate.device_address()), out_weight: ptr(out),
        gated_output: ptr_mut(arena.qsa_gated.device_address()), output: ptr_mut(attention_output),
        cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen QSA finish", unsafe { flash_qwen_qsa_finish_launch(&finish) })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_moe(
    maps: &mut FlashNextWeightMaps, resident: &mut QwenResidentWeights,
    layer: u32, num_tokens: u32, hidden_states: u64, moe_output: u64,
    arena: &mut QwenDecodeArena,
    hot_experts: &mut QwenExpertHotCache,
    prepared_experts: &mut QwenPreparedExpertCache,
    expert_loader: &mut QwenExpertFileLoader,
    expert_packs: &mut u64,
    stream: &mut CudaStreamOwner, blas: &CudaBlasOwner, caps: &DeviceCaps,
    trace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    let router = resident.get(maps, stream, &format!("{prefix}.gate.weight"), "BF16", &[512, HIDDEN], 512 * HIDDEN * 2)?;
    let gate = MoeGateArgs {
        struct_size: size::<MoeGateArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: MoeGatePlan::qwen38_flash(num_tokens), hidden_states: ptr(hidden_states),
        router_weight: ptr(router), router_logits: ptr_mut(arena.router_logits.device_address()),
        topk_weights: ptr_mut(arena.route_weights.device_address()).cast::<f32>(), topk_ids: ptr_mut(arena.route_ids.device_address()).cast::<i32>(),
        cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    let router_started = Instant::now();
    native("Qwen router", unsafe { flash_moe_gate_launch(caps, &gate) })?;
    stream.synchronize()?;
    if trace { eprintln!("Qwen trace layer {layer} MoE: router complete"); }
    let router_seconds = router_started.elapsed().as_secs_f64();
    let route_count = usize::try_from(num_tokens.checked_mul(TOP_K).ok_or("Qwen route count overflow")?)?;
    let logical = read_i32_region(&arena.route_ids, route_count)?;
    for token_routes in logical.chunks_exact(TOP_K as usize) {
        let unique = token_routes.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != ACTIVE_EXPERTS as usize {
            return Err("Qwen top-k experts are not unique".into());
        }
    }
    let union = logical.iter().copied().collect::<BTreeSet<_>>();
    let union_keys = union
        .into_iter()
        .map(|expert| Ok(ExpertKey {
            layer: u16::try_from(layer)?,
            expert: u16::try_from(expert)?,
        }))
        .collect::<Result<Vec<_>, std::num::TryFromIntError>>()?;
    let prepare_started = Instant::now();
    expert_loader.prefetch_experts(&union_keys)?;
    let prefetch_seconds = prepare_started.elapsed().as_secs_f64();
    if trace || prefetch_seconds >= 0.25 {
        eprintln!(
            "Qwen layer {layer} route union: {} experts, router {:.3} s, prefetch {:.3} s",
            union_keys.len(), router_seconds, prefetch_seconds,
        );
    }
    let mut expert_slots = QwenLayerExpertSlots::new();
    for token_offset in 0..num_tokens {
        let route_begin = usize::try_from(token_offset * TOP_K)?;
        let route_end = route_begin + TOP_K as usize;
        let slot_plan = expert_slots.plan(layer, &logical[route_begin..route_end])?;
        run_moe_token(
            maps,
            resident,
            layer,
            hidden_states + u64::from(token_offset) * HIDDEN * 2,
            moe_output + u64::from(token_offset) * HIDDEN * 2,
            arena.route_weights.device_address()
                + u64::from(token_offset) * u64::from(TOP_K) * 4,
            &slot_plan.physical,
            &slot_plan.loads,
            arena,
            hot_experts,
            prepared_experts,
            expert_loader,
            expert_packs,
            stream,
            blas,
            caps,
            trace,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_moe_token(
    maps: &mut FlashNextWeightMaps, resident: &mut QwenResidentWeights,
    layer: u32, hidden_states: u64, moe_output: u64, route_weights: u64,
    physical: &[u32], loads: &[ExpertLoad], arena: &mut QwenDecodeArena,
    hot_experts: &mut QwenExpertHotCache,
    prepared_experts: &mut QwenPreparedExpertCache,
    expert_loader: &mut QwenExpertFileLoader,
    expert_packs: &mut u64,
    stream: &mut CudaStreamOwner, blas: &CudaBlasOwner, caps: &DeviceCaps,
    trace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let moe_started = Instant::now();
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    let pack_started = Instant::now();
    let misses_before = prepared_experts.stats().misses;
    unsafe {
        prepared_experts.prepare_and_promote_layer(
            expert_loader,
            hot_experts,
            loads,
            PREPARED_SLOTS_PER_LAYER,
        )?;
    }
    let prepared_fills = prepared_experts.stats().misses - misses_before;
    *expert_packs = expert_packs
        .checked_add(prepared_fills)
        .ok_or("Qwen GPU expert pack counter overflow")?;
    if trace {
        eprintln!(
            "Qwen trace layer {layer} MoE: {prepared_fills} prepared fills, {} promotions",
            loads.len(),
        );
    }
    let pack_submit_seconds = pack_started.elapsed().as_secs_f64();
    if pack_submit_seconds >= 0.25 {
        eprintln!(
            "Qwen layer {layer} prepared expert fill/promote: {:.3} s for {} experts",
            pack_submit_seconds,
            loads.len(),
        );
    }
    let route = RoutePlan::build(1, TOP_K, LAYER_EXPERT_SLOTS, physical)?;
    if route.grouped.total_rows > MOE_PADDED_ROWS || route.grouped.input_scale_rows > MOE_SCALE_ROWS {
        return Err("Qwen decode route exceeds fixed arena".into());
    }
    write_u32(&mut arena.route_map, &route.route_to_packed_row)?;
    write_i32(&mut arena.m_indptr, &route.grouped.m_indptr)?;
    let route_args = MoeRouteArgs {
        struct_size: size::<MoeRouteArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: MoeRoutePlan::from(route.kernel_spec(HIDDEN)?), token_input: ptr(hidden_states),
        route_to_packed_row: ptr(arena.route_map.device_address()).cast::<u32>(), packed_input: ptr_mut(arena.packed_input.device_address()),
        route_weights: ptr(route_weights).cast::<f32>(), packed_expert_output: ptr(arena.expert_output.device_address()),
        token_output: ptr_mut(moe_output), token_input_row_stride_bytes: HIDDEN * 2,
        packed_row_stride_bytes: HIDDEN * 2, expert_output_row_stride_bytes: HIDDEN * 2, cuda_stream: stream.raw(),
    };
    native("Qwen route dispatch", unsafe { flash_moe_route_dispatch(caps, &route_args) })?;
    trace_stage(stream, trace, layer, "MoE route dispatch")?;
    let active_rows = route.expert_rows.iter().map(|rows| i32::try_from(*rows).unwrap()).collect::<Vec<_>>();
    let views = hot_experts.views();
    let quantize = SegmentedNvfp4QuantizeArgs {
        struct_size: size::<SegmentedNvfp4QuantizeArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: SegmentedNvfp4QuantizePlan::from(SegmentedNvfp4QuantizeSpec::from_grouped_layout(&route.grouped, HIDDEN)?),
        input: ptr(arena.packed_input.device_address()), input_global_scales: ptr(views.w13_input_global_scales).cast::<f32>(),
        active_rows_host: active_rows.as_ptr(), m_indptr_host: route.grouped.m_indptr.as_ptr(),
        scale_row_offsets_host: route.grouped.scale_row_offsets.as_ptr(), packed_output: ptr_mut(arena.input_fp4.device_address()),
        output_scales: ptr_mut(arena.input_scales.device_address()), input_row_stride_bytes: HIDDEN * 2,
        output_row_stride_bytes: HIDDEN / 2, scale_row_stride_bytes: HIDDEN / 16, cuda_stream: stream.raw(),
    };
    native("Qwen routed quantize", unsafe { flash_segmented_nvfp4_quantize_launch(caps, &quantize) })?;
    trace_stage(stream, trace, layer, "MoE routed quantize")?;
    let w13 = grouped_args(GroupedNvfp4Plan::from(GroupedNvfp4Spec::qwen_expert_projection(&route.grouped, 2 * INTERMEDIATE, HIDDEN)?),
        arena.input_fp4.device_address(), arena.input_scales.device_address(), views.w13_weights, views.w13_scales,
        arena.m_indptr.device_address(), views.w13_alpha, arena.gate_up.device_address(), arena.int_workspace.device_address(),
        arena.float_workspace.device_address(), 2 * INTERMEDIATE, HIDDEN, stream.raw());
    native("Qwen grouped gate/up", unsafe { flash_grouped_nvfp4_launch(caps, &w13) })?;
    trace_stage(stream, trace, layer, "MoE grouped gate/up")?;
    let silu = SegmentedSiluNvfp4Args {
        struct_size: size::<SegmentedSiluNvfp4Args>(), abi_version: KERNEL_ABI_VERSION,
        plan: SegmentedSiluNvfp4Plan::from(SegmentedSiluNvfp4Spec::from_grouped_layout(&route.grouped, INTERMEDIATE)?),
        input: ptr(arena.gate_up.device_address()), input_global_scales: ptr(views.w2_input_global_scales).cast::<f32>(),
        active_rows_host: active_rows.as_ptr(), m_indptr_host: route.grouped.m_indptr.as_ptr(),
        scale_row_offsets_host: route.grouped.scale_row_offsets.as_ptr(), packed_output: ptr_mut(arena.down_input.device_address()),
        output_scales: ptr_mut(arena.down_scales.device_address()), input_row_stride_bytes: 2 * INTERMEDIATE * 2,
        output_row_stride_bytes: INTERMEDIATE / 2, scale_row_stride_bytes: INTERMEDIATE / 16, cuda_stream: stream.raw(),
    };
    native("Qwen SiLU quantize", unsafe { flash_segmented_silu_nvfp4_launch(caps, &silu) })?;
    trace_stage(stream, trace, layer, "MoE SiLU quantize")?;
    let w2 = grouped_args(GroupedNvfp4Plan::from(GroupedNvfp4Spec::qwen_expert_projection(&route.grouped, HIDDEN, INTERMEDIATE)?),
        arena.down_input.device_address(), arena.down_scales.device_address(), views.w2_weights, views.w2_scales,
        arena.m_indptr.device_address(), views.w2_alpha, arena.expert_output.device_address(), arena.int_workspace.device_address(),
        arena.float_workspace.device_address(), HIDDEN, INTERMEDIATE, stream.raw());
    native("Qwen grouped down", unsafe { flash_grouped_nvfp4_launch(caps, &w2) })?;
    trace_stage(stream, trace, layer, "MoE grouped down")?;
    native("Qwen route finalize", unsafe { flash_moe_route_finalize(caps, &route_args) })?;
    trace_stage(stream, trace, layer, "MoE route finalize")?;

    let shared_gate_up = resident.merged_bf16_pair(
        maps,
        stream,
        &format!("{prefix}.shared_expert.gate_up_merged"),
        &format!("{prefix}.shared_expert.gate_proj.weight"),
        &format!("{prefix}.shared_expert.up_proj.weight"),
        &[INTERMEDIATE, HIDDEN],
        INTERMEDIATE * HIDDEN * 2,
    )?;
    let shared_down = resident.get(maps, stream, &format!("{prefix}.shared_expert.down_proj.weight"), "BF16", &[HIDDEN, INTERMEDIATE], HIDDEN * INTERMEDIATE * 2)?;
    let shared_gate_weight = resident.get(maps, stream, &format!("{prefix}.shared_expert_gate.weight"), "BF16", &[1, HIDDEN], HIDDEN * 2)?;
    let shared = SharedExpertArgs {
        struct_size: size::<SharedExpertArgs>(), abi_version: KERNEL_ABI_VERSION, plan: SharedExpertPlan::qwen38_flash(1),
        hidden_states: ptr(hidden_states), gate_up_weight: ptr(shared_gate_up),
        down_weight: ptr(shared_down), shared_gate_weight: ptr(shared_gate_weight),
        gate_up: ptr_mut(arena.shared_gate_up.device_address()), activated: ptr_mut(arena.shared_activated.device_address()),
        shared_gate: std::ptr::null_mut(), output: ptr_mut(arena.shared_output.device_address()), cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen shared expert", unsafe { flash_shared_expert_launch(caps, &shared) })?;
    trace_stage(stream, trace, layer, "MoE shared expert")?;
    let join = MoeJoinArgs {
        struct_size: size::<MoeJoinArgs>(), abi_version: KERNEL_ABI_VERSION, plan: MoeJoinPlan::qwen38_flash(1),
        hidden_states: ptr(hidden_states), shared_gate_weight: ptr(shared_gate_weight),
        shared_output: ptr(arena.shared_output.device_address()), routed_output: ptr_mut(moe_output), cuda_stream: stream.raw(),
    };
    native("Qwen MoE join", unsafe { flash_moe_join_launch(caps, &join) })?;
    trace_stage(stream, trace, layer, "MoE join")?;
    stream.synchronize()?;
    let moe_seconds = moe_started.elapsed().as_secs_f64();
    if moe_seconds >= 0.25 {
        eprintln!(
            "Qwen layer {layer} MoE token stages: {:.3} s",
            moe_seconds,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_ple(
    maps: &mut FlashNextWeightMaps,
    resident: &mut QwenResidentWeights,
    ple_runtime: &mut QwenPleRuntime,
    token_history: &[u32],
    hyper: u64, state: &CoherentRegionOwner,
    arena: &mut QwenDecodeArena,
    stream: &mut CudaStreamOwner, blas: &CudaBlasOwner, caps: &DeviceCaps,
) -> Result<(), Box<dyn std::error::Error>> {
    let QwenPleRuntime {
        cache,
        index,
        multipliers,
        sizes,
        offsets,
        ..
    } = ple_runtime;
    let history = token_history
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let rows = decode_row_ids(&history, 248_044, *multipliers, *sizes, *offsets)?;
    let batch = cache.fetch_rows(index, &rows)?;
    let mut fragments = [PleRowFragment { first_offset_bytes: 0, second_offset_bytes: 0, first_bytes: 0, second_bytes: 0 }; 16];
    batch.write_kernel_fragments(&mut fragments)?;
    let bytes = unsafe { std::slice::from_raw_parts(fragments.as_ptr().cast::<u8>(), std::mem::size_of_val(&fragments)) };
    unsafe { arena.ple_fragments.host_payload_mut()? }.copy_from_slice(bytes);
    let gather = PleGatherArgs {
        struct_size: size::<PleGatherArgs>(), abi_version: KERNEL_ABI_VERSION,
        plan: PleGatherPlan::qwen38_flash(16), coherent_base: batch.device_base().ok_or("PLE cache not CUDA-visible")?.as_ptr(),
        fragments: arena.ple_fragments.device_address() as usize as *const PleRowFragment,
        output: ptr_mut(arena.ple_embedding.device_address()), output_row_stride_bytes: 320,
        scale_bf16_bits: index.scale_bf16_bits, reserved16: 0, reserved32: 0, cuda_stream: stream.raw(),
    };
    native("Qwen PLE gather", unsafe { flash_ple_gather_launch(caps, &gather) })?;
    let prefix = "model.language_model.layers.1.ple";
    let key_weight = resident.get(maps, stream, &format!("{prefix}.key_proj.weight"), "BF16", &[HYPER, HIDDEN], HYPER * HIDDEN * 2)?;
    let value_weight = resident.get(maps, stream, &format!("{prefix}.value_proj.weight"), "BF16", &[HIDDEN, HIDDEN], HIDDEN * HIDDEN * 2)?;
    let norm_key = resident.get(maps, stream, &format!("{prefix}.norm_key.weight"), "BF16", &[HYPER], HYPER * 2)?;
    let norm_query = resident.get(maps, stream, &format!("{prefix}.norm_query.weight"), "BF16", &[HYPER], HYPER * 2)?;
    let norm_conv = resident.get(maps, stream, &format!("{prefix}.norm_conv.weight"), "BF16", &[HYPER], HYPER * 2)?;
    let conv = resident.get(maps, stream, &format!("{prefix}.conv1d.weight"), "BF16", &[HYPER, 1, 4], HYPER * 4 * 2)?;
    let ple = QwenPleBlockArgs {
        struct_size: size::<QwenPleBlockArgs>(), abi_version: QWEN_PLE_BLOCK_ABI_VERSION, tokens: 1, reserved: 0,
        hidden_states: ptr(hyper), embedding: ptr(arena.ple_embedding.device_address()),
        key_weight: ptr(key_weight), value_weight: ptr(value_weight),
        norm_key_weight: ptr(norm_key), norm_query_weight: ptr(norm_query),
        norm_conv_weight: ptr(norm_conv), conv_weight: ptr(conv),
        conv_state: ptr_mut(state.device_address()), key_scratch: ptr_mut(arena.ple_key.device_address()),
        value_scratch: ptr_mut(arena.ple_value.device_address()), gated_scratch: ptr_mut(arena.ple_gated.device_address()),
        normed_scratch: ptr_mut(arena.ple_normed.device_address()), output: ptr_mut(arena.ple_delta.device_address()),
        cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen PLE block", unsafe { flash_qwen_ple_block_launch(&ple) })?;
    glue("Qwen PLE add", arena.ple_delta.device_address(), hyper, stream, flash_qwen_add_hyper_launch)?;
    stream.synchronize()?;
    Ok(())
}

fn finish_logits(
    maps: &mut FlashNextWeightMaps, resident: &mut QwenResidentWeights,
    hyper: u64,
    arena: &QwenDecodeArena, stream: &mut CudaStreamOwner, blas: &CudaBlasOwner,
    caps: &DeviceCaps,
) -> Result<u32, Box<dyn std::error::Error>> {
    let prefix = "model.language_model.hyper_connection_mixer";
    let norm = resident.get(maps, stream, &format!("{prefix}.hc_norm.weight"), "BF16", &[HYPER], HYPER * 2)?;
    let down = resident.get(maps, stream, &format!("{prefix}.input_mix_weight_down.weight"), "BF16", &[LOWRANK, HYPER], LOWRANK * HYPER * 2)?;
    let up = resident.get(maps, stream, &format!("{prefix}.input_mix_weight_up.weight"), "BF16", &[HYPER, LOWRANK], HYPER * LOWRANK * 2)?;
    let args = MhcArgs {
        struct_size: size::<MhcArgs>(), abi_version: KERNEL_ABI_VERSION, plan: MhcPlan::qwen38_flash(1),
        hyper_input: ptr(hyper), norm_weight: ptr(norm),
        mix_down_weight: ptr(down), mix_up_weight: ptr(up),
        inject_weight: ptr(arena.final_dummy.device_address()), block_output: ptr(arena.final_hidden.device_address()),
        normed: ptr_mut(arena.mhc_normed.device_address()), mix_down: ptr_mut(arena.mhc_down.device_address()),
        mix_activated: ptr_mut(arena.mhc_activated.device_address()), mix_up: ptr_mut(arena.mhc_up.device_address()),
        mixed_output: ptr_mut(arena.final_hidden.device_address()), combined_output: ptr_mut(arena.final_combined.device_address()),
        cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen final hyper mix", unsafe { flash_mhc_mix_launch(caps, &args) })?;
    let lm_head = resident.get(maps, stream, "lm_head.weight", "BF16", &[VOCABULARY, HIDDEN], VOCABULARY * HIDDEN * 2)?;
    let head = QwenLmHeadArgs {
        struct_size: size::<QwenLmHeadArgs>(), abi_version: QWEN_DECODE_GLUE_ABI_VERSION,
        vocabulary: VOCABULARY as u32, hidden_size: HIDDEN as u32,
        hidden_states: ptr(arena.final_hidden.device_address()), weight: ptr(lm_head),
        logits: ptr_mut(arena.logits.device_address()).cast::<f32>(), cublas_handle: blas.raw(), cuda_stream: stream.raw(),
    };
    native("Qwen LM head", unsafe { flash_qwen_lm_head_launch(&head) })?;
    stream.synchronize()?;
    let values = unsafe { arena.logits.host_payload()? };
    let mut best = (0_u32, f32::NEG_INFINITY);
    for (index, bytes) in values.chunks_exact(4).enumerate() {
        let value = f32::from_ne_bytes(bytes.try_into()?);
        if value > best.1 { best = (u32::try_from(index)?, value); }
    }
    Ok(best.0)
}

fn checked_tensor(
    maps: &mut FlashNextWeightMaps, name: &str, dtype: &str, shape: &[u64], bytes: u64,
) -> Result<QwenTensorView, Box<dyn std::error::Error>> {
    let tensor = maps.tensor(name, 1)?;
    if tensor.dtype != dtype || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(format!("unexpected tensor geometry for {name}: {} {:?} {}", tensor.dtype, tensor.shape, tensor.data_bytes).into());
    }
    Ok(tensor)
}

fn read_checkpoint_i64(checkpoint: &FlashNextCheckpoint, name: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let tensor = checkpoint.tensor(name)?;
    if tensor.dtype != "I64" || !tensor.data_bytes.is_multiple_of(8) { return Err(format!("{name} is not i64").into()); }
    let file = std::fs::File::open(checkpoint.plan.root.join(&tensor.relative_file))?;
    let mut bytes = vec![0_u8; usize::try_from(tensor.data_bytes)?]; file.read_exact_at(&mut bytes, tensor.absolute_offset)?;
    Ok(bytes.chunks_exact(8).map(|value| i64::from_le_bytes(value.try_into().unwrap())).collect())
}

fn initialize_rope_cache(region: &mut CoherentRegionOwner) -> Result<(), Box<dyn std::error::Error>> {
    let payload = unsafe { region.host_payload_mut()? };
    let rotary_dim = usize::try_from(ROTARY_DIM)?;
    let half = rotary_dim / 2;
    let row_bytes = rotary_dim * 4;
    if payload.len() != QWEN_MODEL_MAX_LENGTH * row_bytes {
        return Err("Qwen RoPE cache size is invalid".into());
    }
    for position in 0..QWEN_MODEL_MAX_LENGTH {
        let row = position * row_bytes;
        for dimension in 0..half {
            let exponent = (2 * dimension) as f32 / rotary_dim as f32;
            let frequency = 1.0_f32 / 10_000_000.0_f32.powf(exponent);
            let angle = position as f32 * frequency;
            payload[row + dimension * 4..row + dimension * 4 + 4]
                .copy_from_slice(&angle.cos().to_ne_bytes());
            let sine = row + (half + dimension) * 4;
            payload[sine..sine + 4].copy_from_slice(&angle.sin().to_ne_bytes());
        }
    }
    Ok(())
}

fn write_i64_scalar(
    region: &mut CoherentRegionOwner,
    value: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = unsafe { region.host_payload_mut()? };
    if output.len() != 8 {
        return Err("Qwen i64 scalar slab size mismatch".into());
    }
    output.copy_from_slice(&value.to_ne_bytes());
    Ok(())
}

fn convert_bf16(stream: &CudaStreamOwner, input: u64, output: &CoherentRegionOwner, elements: u64) -> Result<(), Box<dyn std::error::Error>> {
    let args = QwenBf16ToF32Args { struct_size: size::<QwenBf16ToF32Args>(), abi_version: QWEN_GDN_AUX_ABI_VERSION,
        input_bf16: ptr(input).cast::<u16>(), output_f32: ptr_mut(output.device_address()).cast::<f32>(), elements, cuda_stream: stream.raw() };
    native("Qwen BF16 conversion", unsafe { flash_qwen_bf16_to_f32_launch(&args) })
}

fn glue(
    stage: &str, input: u64, output: u64, stream: &CudaStreamOwner,
    launch: unsafe extern "C" fn(*const QwenDecodeGlueArgs) -> Status,
) -> Result<(), Box<dyn std::error::Error>> {
    let args = QwenDecodeGlueArgs { struct_size: size::<QwenDecodeGlueArgs>(), abi_version: QWEN_DECODE_GLUE_ABI_VERSION,
        input: ptr(input), output: ptr_mut(output), cuda_stream: stream.raw() };
    native(stage, unsafe { launch(&args) })
}

#[allow(clippy::too_many_arguments)]
fn grouped_args(plan: GroupedNvfp4Plan, input: u64, input_scales: u64, weights: u64,
    weight_scales: u64, m_indptr: u64, alpha: u64, output: u64, int_workspace: u64,
    float_workspace: u64, n: u64, k: u64, stream: *mut c_void) -> GroupedNvfp4Args {
    GroupedNvfp4Args {
        struct_size: size::<GroupedNvfp4Args>(), abi_version: KERNEL_ABI_VERSION, plan,
        input: Nvfp4MatrixView { packed_data: ptr(input), block_scales: ptr(input_scales), packed_row_stride_bytes: k / 2, scale_row_stride_bytes: k / 16 },
        weights: GroupedNvfp4WeightView { packed_data: ptr(weights), block_scales: ptr(weight_scales), packed_group_stride_bytes: n * k / 2, scale_group_stride_bytes: n * k / 16 },
        m_indptr: ptr(m_indptr).cast::<i32>(), alpha_device: ptr(alpha).cast::<f32>(), output: ptr_mut(output),
        output_row_stride_bytes: n * 2, int_workspace: ptr_mut(int_workspace), int_workspace_bytes: WORKSPACE,
        float_workspace: ptr_mut(float_workspace), float_workspace_bytes: WORKSPACE, cuda_stream: stream,
    }
}

fn read_i32_region(
    region: &CoherentRegionOwner,
    elements: usize,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let bytes = elements.checked_mul(4).ok_or("i32 read size overflow")?;
    let payload = unsafe { region.host_payload()? };
    if bytes > payload.len() {
        return Err("i32 read exceeds coherent region".into());
    }
    Ok(payload[..bytes]
        .chunks_exact(4)
        .map(|value| i32::from_ne_bytes(value.try_into().unwrap()))
        .collect())
}

fn write_u32(region: &mut CoherentRegionOwner, values: &[u32]) -> Result<(), Box<dyn std::error::Error>> { write_words(region, values.iter().map(|value| value.to_ne_bytes())) }
fn write_i32(region: &mut CoherentRegionOwner, values: &[i32]) -> Result<(), Box<dyn std::error::Error>> { write_words(region, values.iter().map(|value| value.to_ne_bytes())) }
fn write_identity_i32(region: &mut CoherentRegionOwner, count: u64) -> Result<(), Box<dyn std::error::Error>> {
    let output = unsafe { region.host_payload_mut()? };
    let expected = usize::try_from(count.checked_mul(4).ok_or("identity slab size overflow")?)?;
    if output.len() != expected { return Err("identity slab size mismatch".into()); }
    for (index, word) in output.chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&i32::try_from(index)?.to_ne_bytes());
    }
    Ok(())
}
fn write_words<const N: usize>(region: &mut CoherentRegionOwner, values: impl Iterator<Item = [u8; N]>) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new(); for value in values { bytes.extend_from_slice(&value); }
    let output = unsafe { region.host_payload_mut()? }; if output.len() != bytes.len() { return Err("word slab size mismatch".into()); } output.copy_from_slice(&bytes); Ok(())
}

fn slab(bytes: u64) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> { Ok(CoherentRegionOwner::slab(bytes, 256, 0)?) }
fn size<T>() -> u32 { u32::try_from(std::mem::size_of::<T>()).unwrap() }
fn ptr(address: u64) -> *const c_void { address as usize as *const c_void }
fn ptr_mut(address: u64) -> *mut c_void { address as usize as *mut c_void }
fn native(stage: &str, status: Status) -> Result<(), Box<dyn std::error::Error>> {
    if status.code == 0 { return Ok(()); }
    let message = if status.message.is_null() { "native error".to_owned() } else { unsafe { CStr::from_ptr(status.message) }.to_string_lossy().into_owned() };
    Err(format!("{stage}: {message} (status {})", status.code).into())
}
