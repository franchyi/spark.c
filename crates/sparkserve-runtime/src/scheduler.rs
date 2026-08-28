use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fabric::{
    ExpertCacheStats, ExpertKey, ExpertLoad, ExpertResidencyPlan, ExpertSlotAddress, FabricError,
    FixedExpertCache,
};
use crate::routing::{RouteError, RoutePlan};

const ARENA_ALIGNMENT: u64 = 256;
static NEXT_MOE_STEP_LEASE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    pub sequence_slots: u32,
    pub max_context_tokens: u32,
    pub max_prefill_tokens: u32,
    pub prefill_buckets: Vec<u32>,
    pub kv_bytes_per_token: u64,
    pub recurrent_state_bytes: u64,
    pub shared_workspace_bytes: u64,
    pub fabric_committed_bytes: u64,
    pub safety_reserve_bytes: u64,
    pub physical_memory_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceArena {
    pub slot: u32,
    pub offset: u64,
    pub bytes: u64,
    pub kv_offset: u64,
    pub kv_bytes: u64,
    pub recurrent_state_offset: u64,
    pub recurrent_state_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaPlan {
    pub sequence_stride_bytes: u64,
    pub sequence_arenas_bytes: u64,
    pub shared_workspace_offset: u64,
    pub shared_workspace_bytes: u64,
    pub runtime_committed_bytes: u64,
    pub total_committed_bytes: u64,
    sequence_slots: u32,
    kv_bytes: u64,
    recurrent_state_bytes: u64,
}

impl ArenaPlan {
    pub fn build(config: &SchedulerConfig) -> Result<Self, SchedulerError> {
        validate_config(config)?;
        let kv_bytes = checked_mul(
            u64::from(config.max_context_tokens),
            config.kv_bytes_per_token,
        )?;
        let state_offset = align_up(kv_bytes, ARENA_ALIGNMENT)?;
        let sequence_stride_bytes = align_up(
            checked_add(state_offset, config.recurrent_state_bytes)?,
            ARENA_ALIGNMENT,
        )?;
        let sequence_arenas_bytes =
            checked_mul(sequence_stride_bytes, u64::from(config.sequence_slots))?;
        let shared_workspace_offset = align_up(sequence_arenas_bytes, ARENA_ALIGNMENT)?;
        let runtime_committed_bytes = checked_add(
            shared_workspace_offset,
            align_up(config.shared_workspace_bytes, ARENA_ALIGNMENT)?,
        )?;
        let total_committed_bytes = checked_add(
            checked_add(config.fabric_committed_bytes, runtime_committed_bytes)?,
            config.safety_reserve_bytes,
        )?;
        if total_committed_bytes > config.physical_memory_bytes {
            return Err(SchedulerError::MemoryBudget {
                required: total_committed_bytes,
                available: config.physical_memory_bytes,
            });
        }
        Ok(Self {
            sequence_stride_bytes,
            sequence_arenas_bytes,
            shared_workspace_offset,
            shared_workspace_bytes: config.shared_workspace_bytes,
            runtime_committed_bytes,
            total_committed_bytes,
            sequence_slots: config.sequence_slots,
            kv_bytes,
            recurrent_state_bytes: config.recurrent_state_bytes,
        })
    }

    pub fn sequence(&self, slot: u32) -> Result<SequenceArena, SchedulerError> {
        if slot >= self.sequence_slots {
            return Err(SchedulerError::InvalidSlot(slot));
        }
        let offset = checked_mul(u64::from(slot), self.sequence_stride_bytes)?;
        let recurrent_state_offset =
            checked_add(offset, align_up(self.kv_bytes, ARENA_ALIGNMENT)?)?;
        Ok(SequenceArena {
            slot,
            offset,
            bytes: self.sequence_stride_bytes,
            kv_offset: offset,
            kv_bytes: self.kv_bytes,
            recurrent_state_offset,
            recurrent_state_bytes: self.recurrent_state_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphShape {
    Decode,
    Prefill { bucket_tokens: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSpec {
    pub id: u64,
    pub prompt_tokens: u32,
    pub max_new_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceLease {
    pub request_id: u64,
    pub slot: u32,
    pub generation: u64,
    pub prompt_tokens: u32,
    pub max_new_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveSequence {
    lease: SequenceLease,
    generated_tokens: u32,
}

pub struct StaticScheduler {
    config: SchedulerConfig,
    arena: ArenaPlan,
    free_slots: VecDeque<u32>,
    slot_generations: Vec<u64>,
    active: BTreeMap<u64, ActiveSequence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeSchedulerConfig {
    pub layers: u16,
    pub num_experts: u32,
    pub top_k: u32,
    pub hidden_size: u64,
    pub expert_slots: u32,
    pub expert_slot_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentExpert {
    pub expert: u32,
    pub address: ExpertSlotAddress,
}

/// A routed layer whose expert destinations are reserved but not yet visible.
/// The storage thread reads `loads()`, fills those fixed slots, then returns the
/// plan to `commit_step`. A failed read simply drops this value.
#[must_use]
#[derive(Debug)]
pub struct PendingMoeStep {
    layer: u16,
    route: Option<RoutePlan>,
    active_experts: Vec<u32>,
    residency: Option<ExpertResidencyPlan>,
    scheduler_lease: Arc<AtomicU64>,
    lease_id: u64,
}

impl PendingMoeStep {
    pub fn layer(&self) -> u16 {
        self.layer
    }

    pub fn route(&self) -> &RoutePlan {
        self.route.as_ref().expect("pending route remains owned")
    }

    pub fn loads(&self) -> &[ExpertLoad] {
        self.residency
            .as_ref()
            .expect("pending residency remains owned")
            .loads()
    }

    pub fn expert_addresses(&self) -> &[ExpertSlotAddress] {
        self.residency
            .as_ref()
            .expect("pending residency remains owned")
            .addresses()
    }

    /// Exact two-phase residency transaction consumed by fixed-buffer storage
    /// backends. It remains unpublishable until `commit_step` consumes `self`.
    pub fn residency_plan(&self) -> &ExpertResidencyPlan {
        self.residency
            .as_ref()
            .expect("pending residency remains owned")
    }
}

impl Drop for PendingMoeStep {
    fn drop(&mut self) {
        let _ = self.scheduler_lease.compare_exchange(
            self.lease_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadyMoeStep {
    pub layer: u16,
    /// Logical model-expert route retained for telemetry and correctness.
    pub route: RoutePlan,
    /// Physical cache-slot route consumed by dispatch/grouped-GEMM/finalize.
    /// Group `g` addresses fixed expert slot `g`; no weight gather is needed.
    pub execution_route: RoutePlan,
    pub experts: Vec<ResidentExpert>,
}

/// The policy layer for out-of-core MoE. Kernel code never selects cache
/// victims or changes pointers: Rust reserves fixed addresses transactionally
/// from the exact top-k route, while the I/O engine decides how to fill them.
pub struct RoutedMoeScheduler {
    config: MoeSchedulerConfig,
    cache: FixedExpertCache,
    pending_lease: Arc<AtomicU64>,
}

impl RoutedMoeScheduler {
    pub fn new(config: MoeSchedulerConfig) -> Result<Self, MoeScheduleError> {
        if config.layers == 0
            || config.num_experts == 0
            || config.num_experts > 512
            || config.top_k == 0
            || config.top_k > config.num_experts
            || config.top_k > 32
            || config.hidden_size == 0
            || config.hidden_size % 8 != 0
        {
            return Err(MoeScheduleError::InvalidConfig);
        }
        let cache = FixedExpertCache::new(config.expert_slots, config.expert_slot_bytes)?;
        Ok(Self {
            config,
            cache,
            pending_lease: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn prepare_step(
        &self,
        layer: u16,
        num_tokens: u32,
        expert_ids: &[u32],
    ) -> Result<PendingMoeStep, MoeScheduleError> {
        if layer >= self.config.layers {
            return Err(MoeScheduleError::LayerOutOfRange {
                layer,
                layers: self.config.layers,
            });
        }
        let lease_id = NEXT_MOE_STEP_LEASE.fetch_add(1, Ordering::Relaxed);
        if lease_id == 0 {
            return Err(MoeScheduleError::InvalidConfig);
        }
        if self
            .pending_lease
            .compare_exchange(0, lease_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(MoeScheduleError::PendingStep);
        }
        let result = (|| {
            let route = RoutePlan::build(
                num_tokens,
                self.config.top_k,
                self.config.num_experts,
                expert_ids,
            )?;
            route.kernel_spec(self.config.hidden_size)?;
            let active_experts: Vec<u32> = route
                .expert_rows
                .iter()
                .enumerate()
                .filter_map(|(expert, rows)| (*rows != 0).then_some(expert as u32))
                .collect();
            let keys = active_experts
                .iter()
                .map(|expert| ExpertKey {
                    layer,
                    expert: *expert as u16,
                })
                .collect::<Vec<_>>();
            let residency = self.cache.prepare(&keys)?;
            Ok(PendingMoeStep {
                layer,
                route: Some(route),
                active_experts,
                residency: Some(residency),
                scheduler_lease: Arc::clone(&self.pending_lease),
                lease_id,
            })
        })();
        if result.is_err() {
            let _ = self.pending_lease.compare_exchange(
                lease_id,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        result
    }

    /// Consume the INT32 expert ids written by SGLang's top-k ABI after its
    /// CUDA completion event. On GB10 these ids can be read through the CPU
    /// alias of the same coherent allocation; no device-to-host copy or second
    /// route tensor is required.
    pub fn prepare_gate_step(
        &self,
        layer: u16,
        num_tokens: u32,
        expert_ids: &[i32],
    ) -> Result<PendingMoeStep, MoeScheduleError> {
        let mut converted = Vec::with_capacity(expert_ids.len());
        for (route, expert) in expert_ids.iter().copied().enumerate() {
            converted.push(u32::try_from(expert).map_err(|_| {
                MoeScheduleError::Route(RouteError::NegativeExpert { route, expert })
            })?);
        }
        self.prepare_step(layer, num_tokens, &converted)
    }

    pub fn commit_step(
        &mut self,
        mut pending: PendingMoeStep,
    ) -> Result<ReadyMoeStep, MoeScheduleError> {
        if !Arc::ptr_eq(&self.pending_lease, &pending.scheduler_lease)
            || self.pending_lease.load(Ordering::Acquire) != pending.lease_id
        {
            return Err(MoeScheduleError::ForeignStep);
        }
        let residency = pending
            .residency
            .take()
            .ok_or(MoeScheduleError::ForeignStep)?;
        let addresses = self.cache.commit(residency)?;
        if addresses.len() != pending.active_experts.len() {
            return Err(MoeScheduleError::ResidencyMismatch);
        }
        let experts: Vec<ResidentExpert> = std::mem::take(&mut pending.active_experts)
            .into_iter()
            .zip(addresses)
            .map(|(expert, address)| ResidentExpert { expert, address })
            .collect();
        let route = pending.route.take().ok_or(MoeScheduleError::ForeignStep)?;
        let mut slot_ids = Vec::with_capacity(route.route_experts.len());
        for logical_expert in &route.route_experts {
            let placement = experts
                .binary_search_by_key(logical_expert, |placement| placement.expert)
                .ok()
                .and_then(|index| experts.get(index))
                .ok_or(MoeScheduleError::ResidencyMismatch)?;
            slot_ids.push(placement.address.slot);
        }
        let execution_route = RoutePlan::build(
            route.num_tokens,
            route.top_k,
            self.config.expert_slots,
            &slot_ids,
        )?;
        execution_route.kernel_spec(self.config.hidden_size)?;
        Ok(ReadyMoeStep {
            layer: pending.layer,
            route,
            execution_route,
            experts,
        })
    }

    pub fn cache_stats(&self) -> ExpertCacheStats {
        self.cache.stats
    }

    pub fn cache_capacity_bytes(&self) -> Result<u64, MoeScheduleError> {
        Ok(self.cache.capacity_bytes()?)
    }
}

impl StaticScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        let arena = ArenaPlan::build(&config)?;
        let free_slots = (0..config.sequence_slots).collect();
        let slot_generations = vec![0; config.sequence_slots as usize];
        Ok(Self {
            config,
            arena,
            free_slots,
            slot_generations,
            active: BTreeMap::new(),
        })
    }

    pub fn arena(&self) -> &ArenaPlan {
        &self.arena
    }

    pub fn graph_for_prefill(&self, tokens: u32) -> Result<GraphShape, SchedulerError> {
        if tokens == 0 || tokens > self.config.max_prefill_tokens {
            return Err(SchedulerError::PrefillLength(tokens));
        }
        let bucket_tokens = self
            .config
            .prefill_buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= tokens)
            .ok_or(SchedulerError::PrefillLength(tokens))?;
        Ok(GraphShape::Prefill { bucket_tokens })
    }

    pub fn admit(&mut self, request: RequestSpec) -> Result<SequenceLease, SchedulerError> {
        if request.prompt_tokens == 0 {
            return Err(SchedulerError::PromptLength(0));
        }
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or(SchedulerError::IntegerOverflow)?;
        if total_tokens > self.config.max_context_tokens {
            return Err(SchedulerError::ContextLimit {
                requested: total_tokens,
                maximum: self.config.max_context_tokens,
            });
        }
        self.graph_for_prefill(request.prompt_tokens)?;
        if self.active.contains_key(&request.id) {
            return Err(SchedulerError::DuplicateRequest(request.id));
        }
        let slot = self
            .free_slots
            .pop_front()
            .ok_or(SchedulerError::NoSequenceSlot)?;
        let generation = self.slot_generations[slot as usize]
            .checked_add(1)
            .ok_or(SchedulerError::IntegerOverflow)?;
        self.slot_generations[slot as usize] = generation;
        let lease = SequenceLease {
            request_id: request.id,
            slot,
            generation,
            prompt_tokens: request.prompt_tokens,
            max_new_tokens: request.max_new_tokens,
        };
        self.active.insert(
            request.id,
            ActiveSequence {
                lease,
                generated_tokens: 0,
            },
        );
        Ok(lease)
    }

    pub fn record_decode(&mut self, lease: SequenceLease) -> Result<GraphShape, SchedulerError> {
        let active = self
            .active
            .get_mut(&lease.request_id)
            .ok_or(SchedulerError::StaleLease)?;
        if active.lease != lease {
            return Err(SchedulerError::StaleLease);
        }
        if active.generated_tokens >= active.lease.max_new_tokens {
            return Err(SchedulerError::GenerationLimit);
        }
        active.generated_tokens += 1;
        Ok(GraphShape::Decode)
    }

    pub fn release(&mut self, lease: SequenceLease) -> Result<(), SchedulerError> {
        let active = self
            .active
            .get(&lease.request_id)
            .ok_or(SchedulerError::StaleLease)?;
        if active.lease != lease {
            return Err(SchedulerError::StaleLease);
        }
        self.active.remove(&lease.request_id);
        self.free_slots.push_back(lease.slot);
        Ok(())
    }

    pub fn active_sequences(&self) -> usize {
        self.active.len()
    }
}

fn validate_config(config: &SchedulerConfig) -> Result<(), SchedulerError> {
    if config.sequence_slots == 0
        || config.max_context_tokens == 0
        || config.max_prefill_tokens == 0
        || config.kv_bytes_per_token == 0
        || config.recurrent_state_bytes == 0
        || config.physical_memory_bytes == 0
    {
        return Err(SchedulerError::InvalidConfig);
    }
    if config.max_prefill_tokens > config.max_context_tokens
        || config.prefill_buckets.is_empty()
        || config.prefill_buckets.last().copied() != Some(config.max_prefill_tokens)
        || config.prefill_buckets.contains(&0)
        || config
            .prefill_buckets
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(SchedulerError::InvalidPrefillBuckets);
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, SchedulerError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        checked_add(value, alignment - remainder)
    }
}

fn checked_add(left: u64, right: u64) -> Result<u64, SchedulerError> {
    left.checked_add(right)
        .ok_or(SchedulerError::IntegerOverflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, SchedulerError> {
    left.checked_mul(right)
        .ok_or(SchedulerError::IntegerOverflow)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    InvalidConfig,
    InvalidPrefillBuckets,
    IntegerOverflow,
    MemoryBudget { required: u64, available: u64 },
    InvalidSlot(u32),
    PrefillLength(u32),
    PromptLength(u32),
    ContextLimit { requested: u32, maximum: u32 },
    DuplicateRequest(u64),
    NoSequenceSlot,
    StaleLease,
    GenerationLimit,
}

impl Display for SchedulerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid scheduler configuration"),
            Self::InvalidPrefillBuckets => formatter.write_str("invalid prefill graph buckets"),
            Self::IntegerOverflow => formatter.write_str("scheduler size overflow"),
            Self::MemoryBudget {
                required,
                available,
            } => write!(
                formatter,
                "scheduler requires {required} committed bytes, only {available} are available"
            ),
            Self::InvalidSlot(slot) => write!(formatter, "invalid sequence slot {slot}"),
            Self::PrefillLength(tokens) => write!(formatter, "unsupported prefill length {tokens}"),
            Self::PromptLength(tokens) => write!(formatter, "invalid prompt length {tokens}"),
            Self::ContextLimit { requested, maximum } => write!(
                formatter,
                "request needs {requested} tokens, context limit is {maximum}"
            ),
            Self::DuplicateRequest(id) => write!(formatter, "request {id} is already active"),
            Self::NoSequenceSlot => formatter.write_str("no fixed sequence slot is available"),
            Self::StaleLease => formatter.write_str("stale or unknown sequence lease"),
            Self::GenerationLimit => formatter.write_str("request generation limit reached"),
        }
    }
}

impl std::error::Error for SchedulerError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoeScheduleError {
    InvalidConfig,
    PendingStep,
    ForeignStep,
    LayerOutOfRange { layer: u16, layers: u16 },
    ResidencyMismatch,
    Route(RouteError),
    Fabric(FabricError),
}

impl From<RouteError> for MoeScheduleError {
    fn from(error: RouteError) -> Self {
        Self::Route(error)
    }
}

impl From<FabricError> for MoeScheduleError {
    fn from(error: FabricError) -> Self {
        Self::Fabric(error)
    }
}

impl Display for MoeScheduleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => {
                formatter.write_str("invalid routed MoE scheduler configuration")
            }
            Self::PendingStep => {
                formatter.write_str("one routed MoE residency step is already pending")
            }
            Self::ForeignStep => {
                formatter.write_str("routed MoE step belongs to another scheduler")
            }
            Self::LayerOutOfRange { layer, layers } => {
                write!(
                    formatter,
                    "MoE layer {layer} is outside the {layers}-layer model"
                )
            }
            Self::ResidencyMismatch => {
                formatter.write_str("expert residency result does not match active routes")
            }
            Self::Route(error) => write!(formatter, "{error}"),
            Self::Fabric(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for MoeScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            sequence_slots: 2,
            max_context_tokens: 8_192,
            max_prefill_tokens: 4_096,
            prefill_buckets: vec![128, 512, 1_024, 2_048, 4_096],
            kv_bytes_per_token: 4_096,
            recurrent_state_bytes: 3 * 1024 * 1024,
            shared_workspace_bytes: 64 * 1024 * 1024,
            fabric_committed_bytes: 80 * 1024 * 1024,
            safety_reserve_bytes: 16 * 1024 * 1024,
            physical_memory_bytes: 512 * 1024 * 1024,
        }
    }

    #[test]
    fn arena_addresses_do_not_move_when_slots_are_reused() {
        let mut scheduler = StaticScheduler::new(config()).expect("valid scheduler");
        let first = scheduler
            .admit(RequestSpec {
                id: 1,
                prompt_tokens: 100,
                max_new_tokens: 20,
            })
            .expect("first lease");
        let _second = scheduler
            .admit(RequestSpec {
                id: 2,
                prompt_tokens: 100,
                max_new_tokens: 20,
            })
            .expect("second lease");
        let address = scheduler.arena().sequence(first.slot).expect("arena");
        scheduler.release(first).expect("release");
        let reused = scheduler
            .admit(RequestSpec {
                id: 3,
                prompt_tokens: 200,
                max_new_tokens: 20,
            })
            .expect("reused lease");
        assert_eq!(first.slot, reused.slot);
        assert_ne!(first.generation, reused.generation);
        assert_eq!(
            address,
            scheduler.arena().sequence(reused.slot).expect("arena")
        );
        assert_eq!(
            scheduler.record_decode(first),
            Err(SchedulerError::StaleLease)
        );
    }

    #[test]
    fn selects_smallest_prefill_graph_and_enforces_context() {
        let mut scheduler = StaticScheduler::new(config()).expect("valid scheduler");
        assert_eq!(
            scheduler.graph_for_prefill(513),
            Ok(GraphShape::Prefill {
                bucket_tokens: 1_024
            })
        );
        assert_eq!(
            scheduler.admit(RequestSpec {
                id: 1,
                prompt_tokens: 4_096,
                max_new_tokens: 4_097,
            }),
            Err(SchedulerError::ContextLimit {
                requested: 8_193,
                maximum: 8_192,
            })
        );
    }

    #[test]
    fn refuses_to_start_when_the_full_fixed_plan_does_not_fit() {
        let mut config = config();
        config.physical_memory_bytes = 1;
        assert!(matches!(
            StaticScheduler::new(config),
            Err(SchedulerError::MemoryBudget { .. })
        ));
    }

    #[test]
    fn lease_cannot_generate_past_its_limit() {
        let mut scheduler = StaticScheduler::new(config()).expect("valid scheduler");
        let lease = scheduler
            .admit(RequestSpec {
                id: 7,
                prompt_tokens: 10,
                max_new_tokens: 1,
            })
            .expect("lease");
        assert_eq!(scheduler.record_decode(lease), Ok(GraphShape::Decode));
        assert_eq!(
            scheduler.record_decode(lease),
            Err(SchedulerError::GenerationLimit)
        );
    }

    #[test]
    fn routed_step_reserves_only_active_experts_then_publishes_atomically() {
        let mut scheduler = RoutedMoeScheduler::new(MoeSchedulerConfig {
            layers: 48,
            num_experts: 512,
            top_k: 4,
            hidden_size: 2560,
            expert_slots: 8,
            expert_slot_bytes: 4096,
        })
        .expect("scheduler");
        let pending = scheduler
            .prepare_step(3, 2, &[9, 1, 7, 4, 7, 9, 11, 2])
            .expect("pending");
        assert_eq!(pending.route().active_experts(), 6);
        assert_eq!(pending.loads().len(), 6);
        assert_eq!(scheduler.cache_stats(), ExpertCacheStats::default());

        let ready = scheduler.commit_step(pending).expect("commit after I/O");
        assert_eq!(ready.experts.len(), 6);
        assert_eq!(
            ready
                .experts
                .iter()
                .map(|placement| placement.expert)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 7, 9, 11]
        );
        assert_eq!(scheduler.cache_stats().misses, 6);
        assert_eq!(ready.execution_route.num_experts, 8);
        assert_eq!(
            ready.execution_route.route_experts,
            vec![4, 0, 3, 2, 3, 4, 5, 1]
        );
        assert_eq!(ready.execution_route.active_experts(), 6);
        assert_eq!(ready.execution_route.grouped.total_rows, 24);

        let hot = scheduler
            .prepare_step(3, 1, &[11, 9, 7, 1])
            .expect("hot route");
        assert!(hot.loads().is_empty());
        scheduler.commit_step(hot).expect("hot commit");
        assert_eq!(scheduler.cache_stats().hits, 4);
    }

    #[test]
    fn sglang_gate_ids_feed_the_same_transactional_route_without_a_shadow_tensor() {
        let scheduler = RoutedMoeScheduler::new(MoeSchedulerConfig {
            layers: 48,
            num_experts: 512,
            top_k: 4,
            hidden_size: 2560,
            expert_slots: 8,
            expert_slot_bytes: 4096,
        })
        .expect("scheduler");
        let pending = scheduler
            .prepare_gate_step(3, 2, &[9, 1, 7, 4, 7, 9, 11, 2])
            .expect("SGLang gate handoff");
        assert_eq!(pending.route().active_experts(), 6);
        assert_eq!(pending.route().useful_rows(), 8);
        drop(pending);
        assert!(matches!(
            scheduler.prepare_gate_step(3, 1, &[-1, 2, 3, 4]),
            Err(MoeScheduleError::Route(RouteError::NegativeExpert {
                route: 0,
                expert: -1
            }))
        ));
    }

    #[test]
    fn failed_expert_io_does_not_poison_the_next_route() {
        let mut scheduler = RoutedMoeScheduler::new(MoeSchedulerConfig {
            layers: 2,
            num_experts: 4,
            top_k: 2,
            hidden_size: 2560,
            expert_slots: 2,
            expert_slot_bytes: 1024,
        })
        .expect("scheduler");
        let failed = scheduler
            .prepare_step(0, 1, &[0, 1])
            .expect("failed read plan");
        drop(failed);
        assert_eq!(scheduler.cache_stats(), ExpertCacheStats::default());

        let retry = scheduler.prepare_step(0, 1, &[0, 1]).expect("retry");
        scheduler.commit_step(retry).expect("commit retry");
        assert_eq!(scheduler.cache_stats().misses, 2);
    }

    #[test]
    fn only_one_nvme_fill_can_target_fixed_expert_slots_at_a_time() {
        let scheduler = RoutedMoeScheduler::new(MoeSchedulerConfig {
            layers: 2,
            num_experts: 4,
            top_k: 2,
            hidden_size: 2560,
            expert_slots: 2,
            expert_slot_bytes: 1024,
        })
        .expect("scheduler");
        let first = scheduler.prepare_step(0, 1, &[0, 1]).expect("first");
        assert!(matches!(
            scheduler.prepare_step(0, 1, &[2, 3]),
            Err(MoeScheduleError::PendingStep)
        ));
        drop(first);
        assert!(scheduler.prepare_step(0, 1, &[2, 3]).is_ok());
    }

    #[test]
    fn invalid_route_releases_the_pending_fill_lease() {
        let scheduler = RoutedMoeScheduler::new(MoeSchedulerConfig {
            layers: 1,
            num_experts: 4,
            top_k: 2,
            hidden_size: 2560,
            expert_slots: 2,
            expert_slot_bytes: 1024,
        })
        .expect("scheduler");
        assert!(matches!(
            scheduler.prepare_step(0, 1, &[1, 1]),
            Err(MoeScheduleError::Route(RouteError::DuplicateExpert { .. }))
        ));
        assert!(scheduler.prepare_step(0, 1, &[1, 2]).is_ok());
    }
}
