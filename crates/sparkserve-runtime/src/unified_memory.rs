//! Hard admission ceiling for GB10's shared CPU/GPU physical memory.
//!
//! SparkServe does not treat an address-space mapping as free memory. The
//! fixed fabric plan is charged before startup, while request-local transient
//! bytes are reserved transactionally. Linux residency is sampled at every
//! admission so file-backed RSS, anonymous RSS, system pressure, and cgroup
//! pressure can reject work before an allocation or CUDA graph launch.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnifiedMemorySnapshot {
    pub process_rss_bytes: u64,
    pub process_anon_bytes: u64,
    pub process_file_bytes: u64,
    pub system_available_bytes: u64,
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
}

pub trait UnifiedMemoryProbe: Send + Sync + 'static {
    fn snapshot(&self) -> Result<UnifiedMemorySnapshot, UnifiedMemoryError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnifiedMemoryConfig {
    /// Maximum memory SparkServe may make resident, excluding the OS reserve.
    pub hard_ceiling_bytes: u64,
    /// Resident weights, fixed caches, KV/state arenas, and workspaces.
    pub planned_fixed_bytes: u64,
    /// Minimum physical/cgroup headroom retained for the OS and driver.
    pub safety_reserve_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnifiedMemoryStats {
    pub admissions: u64,
    pub rejections: u64,
    pub active_requests: u64,
    pub reserved_transient_bytes: u64,
    pub peak_reserved_transient_bytes: u64,
    pub peak_process_rss_bytes: u64,
    pub peak_projected_bytes: u64,
    pub latest: UnifiedMemorySnapshot,
}

pub struct UnifiedMemoryGuard {
    inner: Arc<GuardInner>,
}

impl UnifiedMemoryGuard {
    pub fn new(
        config: UnifiedMemoryConfig,
        probe: Arc<dyn UnifiedMemoryProbe>,
    ) -> Result<Self, UnifiedMemoryError> {
        if config.hard_ceiling_bytes == 0
            || config.planned_fixed_bytes == 0
            || config.planned_fixed_bytes > config.hard_ceiling_bytes
            || config
                .hard_ceiling_bytes
                .checked_add(config.safety_reserve_bytes)
                .is_none()
        {
            return Err(UnifiedMemoryError::InvalidConfig);
        }
        Ok(Self {
            inner: Arc::new(GuardInner {
                config,
                probe,
                state: Mutex::new(GuardState::default()),
            }),
        })
    }

    /// Reserve the only request-dependent bytes before admission. A reservation
    /// is released by `Drop`, including error and client-disconnect paths.
    pub fn try_admit(
        &self,
        request_id: u64,
        transient_bytes: u64,
    ) -> Result<UnifiedMemoryReservation, UnifiedMemoryError> {
        if request_id == 0 {
            return Err(UnifiedMemoryError::InvalidRequestId);
        }
        let snapshot = self.inner.probe.snapshot()?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| UnifiedMemoryError::Poisoned)?;
        if state.reservations.contains_key(&request_id) {
            return Err(UnifiedMemoryError::DuplicateRequest(request_id));
        }
        let reserved = state
            .stats
            .reserved_transient_bytes
            .checked_add(transient_bytes)
            .ok_or(UnifiedMemoryError::IntegerOverflow)?;
        let planned = self
            .inner
            .config
            .planned_fixed_bytes
            .checked_add(reserved)
            .ok_or(UnifiedMemoryError::IntegerOverflow)?;
        let measured = snapshot
            .process_rss_bytes
            .checked_add(reserved)
            .ok_or(UnifiedMemoryError::IntegerOverflow)?;
        let projected = planned.max(measured);
        if projected > self.inner.config.hard_ceiling_bytes {
            state.reject(snapshot, projected);
            return Err(UnifiedMemoryError::HardCeiling {
                projected,
                ceiling: self.inner.config.hard_ceiling_bytes,
            });
        }
        let required_available = self
            .inner
            .config
            .safety_reserve_bytes
            .checked_add(transient_bytes)
            .ok_or(UnifiedMemoryError::IntegerOverflow)?;
        if snapshot.system_available_bytes < required_available {
            state.reject(snapshot, projected);
            return Err(UnifiedMemoryError::SystemPressure {
                available: snapshot.system_available_bytes,
                required: required_available,
            });
        }
        if let (Some(current), Some(limit)) =
            (snapshot.cgroup_current_bytes, snapshot.cgroup_limit_bytes)
        {
            let projected_cgroup = current
                .checked_add(transient_bytes)
                .and_then(|value| value.checked_add(self.inner.config.safety_reserve_bytes))
                .ok_or(UnifiedMemoryError::IntegerOverflow)?;
            if projected_cgroup > limit {
                state.reject(snapshot, projected);
                return Err(UnifiedMemoryError::CgroupPressure {
                    projected: projected_cgroup,
                    limit,
                });
            }
        }

        state.reservations.insert(request_id, transient_bytes);
        state.stats.admissions = state.stats.admissions.saturating_add(1);
        state.stats.active_requests = state.reservations.len() as u64;
        state.stats.reserved_transient_bytes = reserved;
        state.stats.peak_reserved_transient_bytes =
            state.stats.peak_reserved_transient_bytes.max(reserved);
        state.observe(snapshot, projected);
        Ok(UnifiedMemoryReservation {
            inner: Arc::clone(&self.inner),
            request_id,
            active: true,
        })
    }

    pub fn stats(&self) -> Result<UnifiedMemoryStats, UnifiedMemoryError> {
        self.inner
            .state
            .lock()
            .map(|state| state.stats)
            .map_err(|_| UnifiedMemoryError::Poisoned)
    }
}

struct GuardInner {
    config: UnifiedMemoryConfig,
    probe: Arc<dyn UnifiedMemoryProbe>,
    state: Mutex<GuardState>,
}

#[derive(Default)]
struct GuardState {
    reservations: BTreeMap<u64, u64>,
    stats: UnifiedMemoryStats,
}

impl GuardState {
    fn observe(&mut self, snapshot: UnifiedMemorySnapshot, projected: u64) {
        self.stats.latest = snapshot;
        self.stats.peak_process_rss_bytes = self
            .stats
            .peak_process_rss_bytes
            .max(snapshot.process_rss_bytes);
        self.stats.peak_projected_bytes = self.stats.peak_projected_bytes.max(projected);
    }

    fn reject(&mut self, snapshot: UnifiedMemorySnapshot, projected: u64) {
        self.stats.rejections = self.stats.rejections.saturating_add(1);
        self.observe(snapshot, projected);
    }
}

#[must_use]
pub struct UnifiedMemoryReservation {
    inner: Arc<GuardInner>,
    request_id: u64,
    active: bool,
}

impl UnifiedMemoryReservation {
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    pub fn transient_bytes(&self) -> Result<u64, UnifiedMemoryError> {
        self.inner
            .state
            .lock()
            .map_err(|_| UnifiedMemoryError::Poisoned)?
            .reservations
            .get(&self.request_id)
            .copied()
            .ok_or(UnifiedMemoryError::UnknownRequest(self.request_id))
    }

    pub fn release(mut self) -> Result<(), UnifiedMemoryError> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> Result<(), UnifiedMemoryError> {
        if !self.active {
            return Ok(());
        }
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| UnifiedMemoryError::Poisoned)?;
        let bytes = state
            .reservations
            .remove(&self.request_id)
            .ok_or(UnifiedMemoryError::UnknownRequest(self.request_id))?;
        state.stats.reserved_transient_bytes = state
            .stats
            .reserved_transient_bytes
            .checked_sub(bytes)
            .ok_or(UnifiedMemoryError::IntegerOverflow)?;
        state.stats.active_requests = state.reservations.len() as u64;
        self.active = false;
        Ok(())
    }
}

impl Drop for UnifiedMemoryReservation {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnifiedMemoryError {
    InvalidConfig,
    InvalidRequestId,
    DuplicateRequest(u64),
    UnknownRequest(u64),
    IntegerOverflow,
    Poisoned,
    Probe(String),
    HardCeiling { projected: u64, ceiling: u64 },
    SystemPressure { available: u64, required: u64 },
    CgroupPressure { projected: u64, limit: u64 },
}

impl Display for UnifiedMemoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid unified-memory guard config"),
            Self::InvalidRequestId => formatter.write_str("request id must be nonzero"),
            Self::DuplicateRequest(id) => {
                write!(formatter, "request {id} already has a memory reservation")
            }
            Self::UnknownRequest(id) => write!(formatter, "request {id} has no memory reservation"),
            Self::IntegerOverflow => formatter.write_str("unified-memory accounting overflow"),
            Self::Poisoned => formatter.write_str("unified-memory accounting lock was poisoned"),
            Self::Probe(error) => write!(formatter, "cannot measure unified memory: {error}"),
            Self::HardCeiling { projected, ceiling } => write!(
                formatter,
                "request projects {projected} resident bytes above the {ceiling}-byte hard ceiling"
            ),
            Self::SystemPressure {
                available,
                required,
            } => write!(
                formatter,
                "system has {available} bytes available but request and reserve require {required}"
            ),
            Self::CgroupPressure { projected, limit } => write!(
                formatter,
                "cgroup projects {projected} bytes above its {limit}-byte limit"
            ),
        }
    }
}

impl std::error::Error for UnifiedMemoryError {}

#[cfg(target_os = "linux")]
pub struct LinuxUnifiedMemoryProbe;

#[cfg(target_os = "linux")]
impl UnifiedMemoryProbe for LinuxUnifiedMemoryProbe {
    fn snapshot(&self) -> Result<UnifiedMemorySnapshot, UnifiedMemoryError> {
        let smaps = std::fs::read_to_string("/proc/self/smaps_rollup")
            .map_err(|error| UnifiedMemoryError::Probe(error.to_string()))?;
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| UnifiedMemoryError::Probe(error.to_string()))?;
        let (cgroup_current_bytes, cgroup_limit_bytes) = read_cgroup_v2()?;
        parse_linux_snapshot(&smaps, &meminfo, cgroup_current_bytes, cgroup_limit_bytes)
    }
}

#[cfg(target_os = "linux")]
fn read_cgroup_v2() -> Result<(Option<u64>, Option<u64>), UnifiedMemoryError> {
    use std::path::PathBuf;

    let cgroup = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|error| UnifiedMemoryError::Probe(error.to_string()))?;
    let Some(relative) = cgroup.lines().find_map(|line| line.strip_prefix("0::")) else {
        return Ok((None, None));
    };
    let mut root = PathBuf::from("/sys/fs/cgroup");
    let relative = relative.trim().trim_start_matches('/');
    if !relative.is_empty() {
        root.push(relative);
    }
    let current = read_optional_u64(root.join("memory.current"))?;
    let maximum = match std::fs::read_to_string(root.join("memory.max")) {
        Ok(value) if value.trim() == "max" => None,
        Ok(value) => Some(value.trim().parse::<u64>().map_err(|error| {
            UnifiedMemoryError::Probe(format!("invalid cgroup memory.max: {error}"))
        })?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(UnifiedMemoryError::Probe(error.to_string())),
    };
    Ok((current, maximum))
}

#[cfg(target_os = "linux")]
fn read_optional_u64(path: std::path::PathBuf) -> Result<Option<u64>, UnifiedMemoryError> {
    match std::fs::read_to_string(path) {
        Ok(value) => value
            .trim()
            .parse::<u64>()
            .map(Some)
            .map_err(|error| UnifiedMemoryError::Probe(error.to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(UnifiedMemoryError::Probe(error.to_string())),
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_snapshot(
    smaps: &str,
    meminfo: &str,
    cgroup_current_bytes: Option<u64>,
    cgroup_limit_bytes: Option<u64>,
) -> Result<UnifiedMemorySnapshot, UnifiedMemoryError> {
    Ok(UnifiedMemorySnapshot {
        process_rss_bytes: parse_kib_field(smaps, "Rss:")?,
        process_anon_bytes: parse_kib_field(smaps, "Pss_Anon:")?,
        process_file_bytes: parse_kib_field(smaps, "Pss_File:")?,
        system_available_bytes: parse_kib_field(meminfo, "MemAvailable:")?,
        cgroup_current_bytes,
        cgroup_limit_bytes,
    })
}

#[cfg(target_os = "linux")]
fn parse_kib_field(payload: &str, name: &str) -> Result<u64, UnifiedMemoryError> {
    let line = payload
        .lines()
        .find(|line| line.starts_with(name))
        .ok_or_else(|| UnifiedMemoryError::Probe(format!("missing {name}")))?;
    let kib = line[name.len()..]
        .split_whitespace()
        .next()
        .ok_or_else(|| UnifiedMemoryError::Probe(format!("missing value for {name}")))?
        .parse::<u64>()
        .map_err(|error| UnifiedMemoryError::Probe(error.to_string()))?;
    kib.checked_mul(1024)
        .ok_or(UnifiedMemoryError::IntegerOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct FixedProbe(UnifiedMemorySnapshot);

    impl UnifiedMemoryProbe for FixedProbe {
        fn snapshot(&self) -> Result<UnifiedMemorySnapshot, UnifiedMemoryError> {
            Ok(self.0)
        }
    }

    fn snapshot() -> UnifiedMemorySnapshot {
        UnifiedMemorySnapshot {
            process_rss_bytes: 60,
            process_anon_bytes: 20,
            process_file_bytes: 40,
            system_available_bytes: 50,
            cgroup_current_bytes: Some(70),
            cgroup_limit_bytes: Some(120),
        }
    }

    fn guard(snapshot: UnifiedMemorySnapshot) -> UnifiedMemoryGuard {
        UnifiedMemoryGuard::new(
            UnifiedMemoryConfig {
                hard_ceiling_bytes: 100,
                planned_fixed_bytes: 80,
                safety_reserve_bytes: 10,
            },
            Arc::new(FixedProbe(snapshot)),
        )
        .expect("guard")
    }

    #[test]
    fn reservation_is_transactional_and_drop_releases_it() {
        let guard = guard(snapshot());
        let lease = guard.try_admit(1, 5).expect("admit");
        assert_eq!(lease.transient_bytes(), Ok(5));
        let stats = guard.stats().expect("stats");
        assert_eq!(stats.active_requests, 1);
        assert_eq!(stats.reserved_transient_bytes, 5);
        assert_eq!(stats.latest.process_file_bytes, 40);
        drop(lease);
        let stats = guard.stats().expect("released stats");
        assert_eq!(stats.active_requests, 0);
        assert_eq!(stats.reserved_transient_bytes, 0);
    }

    #[test]
    fn planned_fixed_bytes_charge_lazy_mappings_before_they_fault() {
        let guard = guard(snapshot());
        let error = match guard.try_admit(1, 21) {
            Ok(_) => panic!("request above the fixed plan was admitted"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            UnifiedMemoryError::HardCeiling {
                projected: 101,
                ceiling: 100
            }
        );
        assert_eq!(guard.stats().expect("stats").rejections, 1);
    }

    #[test]
    fn measured_rss_can_be_stricter_than_the_static_plan() {
        let mut measured = snapshot();
        measured.process_rss_bytes = 99;
        let error = match guard(measured).try_admit(1, 2) {
            Ok(_) => panic!("request above measured RSS was admitted"),
            Err(error) => error,
        };
        assert!(matches!(error, UnifiedMemoryError::HardCeiling { .. }));
    }

    #[test]
    fn system_and_cgroup_reserves_fail_closed() {
        let mut low_system = snapshot();
        low_system.system_available_bytes = 10;
        assert!(matches!(
            guard(low_system).try_admit(1, 1),
            Err(UnifiedMemoryError::SystemPressure { .. })
        ));

        let mut low_cgroup = snapshot();
        low_cgroup.cgroup_current_bytes = Some(115);
        assert!(matches!(
            guard(low_cgroup).try_admit(1, 1),
            Err(UnifiedMemoryError::CgroupPressure { .. })
        ));
    }

    #[test]
    fn duplicate_request_cannot_double_reserve() {
        let guard = guard(snapshot());
        let _lease = guard.try_admit(7, 1).expect("first");
        assert!(matches!(
            guard.try_admit(7, 1),
            Err(UnifiedMemoryError::DuplicateRequest(7))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_rollups_without_counting_virtual_mapping_size() {
        let parsed = parse_linux_snapshot(
            "Rss:                1024 kB\nPss_Anon:            256 kB\nPss_File:            700 kB\n",
            "MemTotal:       100000 kB\nMemAvailable:    50000 kB\n",
            Some(1234),
            Some(5678),
        )
        .expect("snapshot");
        assert_eq!(parsed.process_rss_bytes, 1024 * 1024);
        assert_eq!(parsed.process_anon_bytes, 256 * 1024);
        assert_eq!(parsed.process_file_bytes, 700 * 1024);
        assert_eq!(parsed.system_available_bytes, 50_000 * 1024);
        assert_eq!(parsed.cgroup_current_bytes, Some(1234));
        assert_eq!(parsed.cgroup_limit_bytes, Some(5678));
    }
}
