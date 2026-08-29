use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_EXPERT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementKind {
    /// One file-backed representation is prefaulted and retained. There is no
    /// second CUDA weight allocation.
    DirectResident,
    /// Only a bounded set of source pages is committed in fixed cache slots.
    SparsePageCache,
    /// Encoded expert blocks remain quantized in fixed cache slots.
    QuantizedExpertCache,
    /// Runtime-owned KV/state/workspace arena.
    FixedArena,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionSpec {
    pub name: String,
    pub kind: PlacementKind,
    pub source_bytes: u64,
    pub committed_bytes: u64,
    pub alignment: u64,
    pub keeps_shadow_copy: bool,
}

impl RegionSpec {
    pub fn direct_resident(name: impl Into<String>, bytes: u64, alignment: u64) -> Self {
        Self {
            name: name.into(),
            kind: PlacementKind::DirectResident,
            source_bytes: bytes,
            committed_bytes: bytes,
            alignment,
            keeps_shadow_copy: false,
        }
    }

    pub fn sparse_cache(
        name: impl Into<String>,
        source_bytes: u64,
        cache_bytes: u64,
        alignment: u64,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PlacementKind::SparsePageCache,
            source_bytes,
            committed_bytes: cache_bytes,
            alignment,
            keeps_shadow_copy: false,
        }
    }

    pub fn expert_cache(
        name: impl Into<String>,
        source_bytes: u64,
        cache_bytes: u64,
        alignment: u64,
    ) -> Self {
        Self {
            name: name.into(),
            kind: PlacementKind::QuantizedExpertCache,
            source_bytes,
            committed_bytes: cache_bytes,
            alignment,
            keeps_shadow_copy: false,
        }
    }

    pub fn fixed_arena(name: impl Into<String>, bytes: u64, alignment: u64) -> Self {
        Self {
            name: name.into(),
            kind: PlacementKind::FixedArena,
            source_bytes: 0,
            committed_bytes: bytes,
            alignment,
            keeps_shadow_copy: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionPlacement {
    pub spec: RegionSpec,
    pub fixed_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FabricPlan {
    pub regions: Vec<RegionPlacement>,
    pub committed_bytes: u64,
    pub safety_reserve_bytes: u64,
    pub total_budgeted_bytes: u64,
    pub physical_memory_bytes: u64,
}

impl FabricPlan {
    pub fn build(
        physical_memory_bytes: u64,
        safety_reserve_bytes: u64,
        regions: Vec<RegionSpec>,
    ) -> Result<Self, FabricError> {
        if physical_memory_bytes == 0 || regions.is_empty() {
            return Err(FabricError::InvalidPlan);
        }
        let mut names = BTreeSet::new();
        let mut offset = 0_u64;
        let mut placements = Vec::with_capacity(regions.len());
        for spec in regions {
            validate_region(&spec)?;
            if !names.insert(spec.name.clone()) {
                return Err(FabricError::DuplicateRegion(spec.name));
            }
            offset = align_up(offset, spec.alignment)?;
            let fixed_offset = offset;
            offset = offset
                .checked_add(spec.committed_bytes)
                .ok_or(FabricError::IntegerOverflow)?;
            placements.push(RegionPlacement { spec, fixed_offset });
        }
        let committed_bytes = offset;
        let total_budgeted_bytes = committed_bytes
            .checked_add(safety_reserve_bytes)
            .ok_or(FabricError::IntegerOverflow)?;
        if total_budgeted_bytes > physical_memory_bytes {
            return Err(FabricError::MemoryBudget {
                required: total_budgeted_bytes,
                available: physical_memory_bytes,
            });
        }
        Ok(Self {
            regions: placements,
            committed_bytes,
            safety_reserve_bytes,
            total_budgeted_bytes,
            physical_memory_bytes,
        })
    }

    pub fn region(&self, name: &str) -> Option<&RegionPlacement> {
        self.regions.iter().find(|region| region.spec.name == name)
    }
}

fn validate_region(spec: &RegionSpec) -> Result<(), FabricError> {
    if spec.name.is_empty()
        || spec.committed_bytes == 0
        || spec.alignment == 0
        || !spec.alignment.is_power_of_two()
    {
        return Err(FabricError::InvalidRegion(spec.name.clone()));
    }
    if spec.keeps_shadow_copy {
        return Err(FabricError::ShadowCopy(spec.name.clone()));
    }
    match spec.kind {
        PlacementKind::DirectResident => {
            if spec.source_bytes == 0 || spec.source_bytes != spec.committed_bytes {
                return Err(FabricError::InvalidRegion(spec.name.clone()));
            }
        }
        PlacementKind::SparsePageCache | PlacementKind::QuantizedExpertCache => {
            if spec.source_bytes == 0 || spec.committed_bytes > spec.source_bytes {
                return Err(FabricError::InvalidRegion(spec.name.clone()));
            }
        }
        PlacementKind::FixedArena => {
            if spec.source_bytes != 0 {
                return Err(FabricError::InvalidRegion(spec.name.clone()));
            }
        }
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, FabricError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(FabricError::IntegerOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSlotAddress {
    pub slot: u32,
    pub byte_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertLoad {
    pub key: ExpertKey,
    pub address: ExpertSlotAddress,
    pub evicts: Option<ExpertKey>,
}

/// A two-phase expert residency decision. The I/O layer fills every `load`
/// destination first and commits only after all reads and checksums succeed.
/// Dropping a plan leaves the cache unchanged.
#[must_use]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertResidencyPlan {
    cache_id: u64,
    base_version: u64,
    step: u64,
    requested_unique: Vec<ExpertKey>,
    resident_hits: u64,
    addresses: Vec<ExpertSlotAddress>,
    loads: Vec<ExpertLoad>,
}

impl ExpertResidencyPlan {
    pub fn addresses(&self) -> &[ExpertSlotAddress] {
        &self.addresses
    }

    pub fn loads(&self) -> &[ExpertLoad] {
        &self.loads
    }

    pub fn is_ready(&self) -> bool {
        self.loads.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpertSlot {
    key: Option<ExpertKey>,
    last_used_step: u64,
}

/// A fixed-address cache for encoded GGUF expert blocks. Loading changes the
/// contents of a slot, never its address, so captured graphs can keep pointers.
pub struct FixedExpertCache {
    cache_id: u64,
    version: u64,
    slot_bytes: u64,
    slots: Vec<ExpertSlot>,
    locations: BTreeMap<ExpertKey, usize>,
    step: u64,
    pub stats: ExpertCacheStats,
}

impl FixedExpertCache {
    pub fn new(slot_count: u32, slot_bytes: u64) -> Result<Self, FabricError> {
        if slot_count == 0 || slot_bytes == 0 {
            return Err(FabricError::InvalidCache);
        }
        let cache_id = NEXT_EXPERT_CACHE_ID.fetch_add(1, Ordering::Relaxed);
        if cache_id == 0 {
            return Err(FabricError::IntegerOverflow);
        }
        Ok(Self {
            cache_id,
            version: 0,
            slot_bytes,
            slots: vec![
                ExpertSlot {
                    key: None,
                    last_used_step: 0,
                };
                slot_count as usize
            ],
            locations: BTreeMap::new(),
            step: 0,
            stats: ExpertCacheStats::default(),
        })
    }

    pub fn capacity_bytes(&self) -> Result<u64, FabricError> {
        self.slot_bytes
            .checked_mul(self.slots.len() as u64)
            .ok_or(FabricError::IntegerOverflow)
    }

    /// Choose fixed destinations without making them visible as resident.
    /// Every currently requested expert is protected from eviction, and one
    /// slot is reserved at most once within the returned transaction.
    pub fn prepare(&self, requested: &[ExpertKey]) -> Result<ExpertResidencyPlan, FabricError> {
        let requested_set: BTreeSet<_> = requested.iter().copied().collect();
        if requested_set.is_empty() {
            return Err(FabricError::InvalidCache);
        }
        if requested_set.len() > self.slots.len() {
            return Err(FabricError::WorkingSet {
                requested: requested_set.len(),
                slots: self.slots.len(),
            });
        }
        let step = self
            .step
            .checked_add(1)
            .ok_or(FabricError::IntegerOverflow)?;
        let mut placements = BTreeMap::new();
        let mut reserved_slots = BTreeSet::new();
        let mut resident_hits = 0_u64;
        for key in requested_set.iter().copied() {
            if let Some(slot_index) = self.locations.get(&key).copied() {
                placements.insert(key, slot_index);
                resident_hits = resident_hits.saturating_add(1);
            }
        }

        let mut loads = Vec::new();
        for key in requested_set.iter().copied() {
            if placements.contains_key(&key) {
                continue;
            }
            let slot_index = self
                .slots
                .iter()
                .enumerate()
                .find(|(index, slot)| slot.key.is_none() && !reserved_slots.contains(index))
                .map(|(index, _)| index)
                .or_else(|| {
                    self.slots
                        .iter()
                        .enumerate()
                        .filter(|(index, slot)| {
                            !reserved_slots.contains(index)
                                && slot
                                    .key
                                    .map(|resident| !requested_set.contains(&resident))
                                    .unwrap_or(true)
                        })
                        .min_by_key(|(index, slot)| (slot.last_used_step, *index))
                        .map(|(index, _)| index)
                })
                .ok_or(FabricError::WorkingSet {
                    requested: requested_set.len(),
                    slots: self.slots.len(),
                })?;
            reserved_slots.insert(slot_index);
            placements.insert(key, slot_index);
            loads.push(ExpertLoad {
                key,
                address: self.address(slot_index)?,
                evicts: self.slots[slot_index].key,
            });
        }

        let addresses = requested
            .iter()
            .map(|key| {
                let slot_index = placements
                    .get(key)
                    .copied()
                    .ok_or(FabricError::InvalidCache)?;
                self.address(slot_index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ExpertResidencyPlan {
            cache_id: self.cache_id,
            base_version: self.version,
            step,
            requested_unique: requested_set.into_iter().collect(),
            resident_hits,
            addresses,
            loads,
        })
    }

    /// Publish a completely loaded plan. A concurrent or out-of-order plan is
    /// rejected before it can overwrite newer residency state.
    pub fn commit(
        &mut self,
        plan: ExpertResidencyPlan,
    ) -> Result<Vec<ExpertSlotAddress>, FabricError> {
        if plan.cache_id != self.cache_id || plan.base_version != self.version {
            return Err(FabricError::StalePlan);
        }
        let next_version = self
            .version
            .checked_add(1)
            .ok_or(FabricError::IntegerOverflow)?;
        for load in &plan.loads {
            let slot = self
                .slots
                .get(load.address.slot as usize)
                .ok_or(FabricError::InvalidCache)?;
            if slot.key != load.evicts {
                return Err(FabricError::StalePlan);
            }
        }
        for load in &plan.loads {
            let slot_index = load.address.slot as usize;
            if let Some(evicted) = self.slots[slot_index].key {
                self.locations.remove(&evicted);
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
            self.slots[slot_index] = ExpertSlot {
                key: Some(load.key),
                last_used_step: plan.step,
            };
            self.locations.insert(load.key, slot_index);
        }
        for key in &plan.requested_unique {
            let slot_index = self
                .locations
                .get(key)
                .copied()
                .ok_or(FabricError::InvalidCache)?;
            self.slots[slot_index].last_used_step = plan.step;
        }
        self.stats.hits = self.stats.hits.saturating_add(plan.resident_hits);
        self.stats.misses = self.stats.misses.saturating_add(plan.loads.len() as u64);
        self.stats.bytes_loaded = self
            .stats
            .bytes_loaded
            .saturating_add(self.slot_bytes.saturating_mul(plan.loads.len() as u64));
        self.step = plan.step;
        self.version = next_version;
        Ok(plan.addresses)
    }

    /// Synchronous compatibility helper used by tests and resident baselines.
    /// Production NVMe code uses `prepare`, fills the returned slots, and then
    /// calls `commit`.
    pub fn resolve(
        &mut self,
        requested: &[ExpertKey],
    ) -> Result<Vec<ExpertSlotAddress>, FabricError> {
        let plan = self.prepare(requested)?;
        self.commit(plan)
    }

    fn address(&self, slot_index: usize) -> Result<ExpertSlotAddress, FabricError> {
        let byte_offset = self
            .slot_bytes
            .checked_mul(slot_index as u64)
            .ok_or(FabricError::IntegerOverflow)?;
        Ok(ExpertSlotAddress {
            slot: u32::try_from(slot_index).map_err(|_| FabricError::IntegerOverflow)?,
            byte_offset,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FabricError {
    InvalidPlan,
    InvalidRegion(String),
    DuplicateRegion(String),
    ShadowCopy(String),
    IntegerOverflow,
    MemoryBudget { required: u64, available: u64 },
    InvalidCache,
    StalePlan,
    WorkingSet { requested: usize, slots: usize },
}

impl Display for FabricError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlan => formatter.write_str("invalid one-copy fabric plan"),
            Self::InvalidRegion(name) => write!(formatter, "invalid fabric region {name}"),
            Self::DuplicateRegion(name) => write!(formatter, "duplicate fabric region {name}"),
            Self::ShadowCopy(name) => write!(formatter, "region {name} violates the one-copy rule"),
            Self::IntegerOverflow => formatter.write_str("fabric size overflow"),
            Self::MemoryBudget {
                required,
                available,
            } => write!(
                formatter,
                "fabric requires {required} committed bytes, only {available} are available"
            ),
            Self::InvalidCache => formatter.write_str("invalid fixed expert cache"),
            Self::StalePlan => formatter.write_str("stale expert residency transaction"),
            Self::WorkingSet { requested, slots } => write!(
                formatter,
                "expert working set needs {requested} slots, cache has {slots}"
            ),
        }
    }
}

impl std::error::Error for FabricError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_copy_plan_counts_cache_capacity_not_whole_sparse_source() {
        let gib = 1024_u64 * 1024 * 1024;
        let plan = FabricPlan::build(
            120 * gib,
            8 * gib,
            vec![
                RegionSpec::direct_resident("qwen-resident", 72 * gib, 2 * 1024 * 1024),
                RegionSpec::sparse_cache("ple", 48 * gib, 2 * gib, 64 * 1024),
                RegionSpec::fixed_arena("runtime", 8 * gib, 256),
            ],
        )
        .expect("bounded one-copy plan");
        assert!(plan.total_budgeted_bytes <= 120 * gib);
        assert_eq!(
            plan.region("ple").expect("PLE").spec.committed_bytes,
            2 * gib
        );
    }

    #[test]
    fn rejects_weight_shadow_copies() {
        let mut region = RegionSpec::direct_resident("weights", 1024, 256);
        region.keeps_shadow_copy = true;
        assert_eq!(
            FabricPlan::build(4096, 0, vec![region]),
            Err(FabricError::ShadowCopy("weights".into()))
        );
    }

    #[test]
    fn expert_slots_keep_fixed_addresses_and_lru_evict() {
        let mut cache = FixedExpertCache::new(2, 4096).expect("cache");
        let a = ExpertKey {
            layer: 0,
            expert: 1,
        };
        let b = ExpertKey {
            layer: 0,
            expert: 2,
        };
        let c = ExpertKey {
            layer: 1,
            expert: 3,
        };
        let first = cache.resolve(&[a, b]).expect("warm");
        let hit = cache.resolve(&[a]).expect("hit");
        assert_eq!(first[0], hit[0]);
        let replacement = cache.resolve(&[a, c]).expect("replace b");
        assert_eq!(replacement[0], hit[0]);
        assert_eq!(cache.stats.hits, 2);
        assert_eq!(cache.stats.misses, 3);
        assert_eq!(cache.stats.evictions, 1);
        assert_eq!(cache.stats.bytes_loaded, 3 * 4096);
    }

    #[test]
    fn refuses_a_route_larger_than_the_fixed_cache() {
        let mut cache = FixedExpertCache::new(1, 4096).expect("cache");
        let error = cache
            .resolve(&[
                ExpertKey {
                    layer: 0,
                    expert: 1,
                },
                ExpertKey {
                    layer: 0,
                    expert: 2,
                },
            ])
            .expect_err("working set must fit");
        assert!(matches!(error, FabricError::WorkingSet { .. }));
    }

    #[test]
    fn dropped_residency_plan_does_not_publish_unread_expert_bytes() {
        let mut cache = FixedExpertCache::new(1, 4096).expect("cache");
        let expert = ExpertKey {
            layer: 3,
            expert: 7,
        };
        let abandoned = cache.prepare(&[expert]).expect("prepare failed I/O");
        assert_eq!(abandoned.loads().len(), 1);
        assert_eq!(cache.stats, ExpertCacheStats::default());
        drop(abandoned);

        let retry = cache.prepare(&[expert]).expect("retry");
        assert_eq!(retry.loads().len(), 1);
        let addresses = cache.commit(retry).expect("publish loaded bytes");
        assert_eq!(addresses[0].byte_offset, 0);
        assert_eq!(cache.stats.misses, 1);
        assert_eq!(cache.stats.bytes_loaded, 4096);
    }

    #[test]
    fn stale_residency_transaction_cannot_overwrite_a_newer_commit() {
        let mut cache = FixedExpertCache::new(1, 4096).expect("cache");
        let a = ExpertKey {
            layer: 0,
            expert: 1,
        };
        let b = ExpertKey {
            layer: 0,
            expert: 2,
        };
        let first = cache.prepare(&[a]).expect("first");
        let stale = cache.prepare(&[b]).expect("parallel stale plan");
        cache.commit(first).expect("first commit");
        assert_eq!(cache.commit(stale), Err(FabricError::StalePlan));
        assert_eq!(cache.resolve(&[a]).expect("a remains resident")[0].slot, 0);
        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.misses, 1);
    }
}
