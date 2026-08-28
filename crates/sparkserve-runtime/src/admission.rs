//! Atomic request admission across fixed sequence slots and unified memory.

use std::fmt::{Display, Formatter};

use crate::scheduler::{GraphShape, RequestSpec, SchedulerError, SequenceLease, StaticScheduler};
use crate::unified_memory::{
    UnifiedMemoryError, UnifiedMemoryGuard, UnifiedMemoryReservation, UnifiedMemoryStats,
};

pub struct RuntimeAdmission {
    scheduler: StaticScheduler,
    memory: UnifiedMemoryGuard,
}

impl RuntimeAdmission {
    pub fn new(scheduler: StaticScheduler, memory: UnifiedMemoryGuard) -> Self {
        Self { scheduler, memory }
    }

    /// Acquire a fixed sequence arena first, then charge transient unified
    /// memory. A memory rejection rolls the arena back before returning.
    pub fn admit(
        &mut self,
        request: RequestSpec,
        transient_bytes: u64,
    ) -> Result<AdmittedRequest, AdmissionError> {
        let sequence = self.scheduler.admit(request.clone())?;
        let memory = match self.memory.try_admit(request.id, transient_bytes) {
            Ok(memory) => memory,
            Err(error) => {
                self.scheduler
                    .release(sequence)
                    .map_err(AdmissionError::Rollback)?;
                return Err(AdmissionError::Memory(error));
            }
        };
        Ok(AdmittedRequest { sequence, memory })
    }

    pub fn record_decode(
        &mut self,
        request: &AdmittedRequest,
    ) -> Result<GraphShape, AdmissionError> {
        self.scheduler
            .record_decode(request.sequence)
            .map_err(AdmissionError::Scheduler)
    }

    pub fn release(&mut self, request: AdmittedRequest) -> Result<(), AdmissionError> {
        self.scheduler.release(request.sequence)?;
        request.memory.release()?;
        Ok(())
    }

    pub fn active_sequences(&self) -> usize {
        self.scheduler.active_sequences()
    }

    pub fn memory_stats(&self) -> Result<UnifiedMemoryStats, AdmissionError> {
        self.memory.stats().map_err(AdmissionError::Memory)
    }
}

#[must_use]
pub struct AdmittedRequest {
    sequence: SequenceLease,
    memory: UnifiedMemoryReservation,
}

impl AdmittedRequest {
    pub fn sequence(&self) -> SequenceLease {
        self.sequence
    }

    pub fn request_id(&self) -> u64 {
        self.sequence.request_id
    }

    pub fn transient_bytes(&self) -> Result<u64, UnifiedMemoryError> {
        self.memory.transient_bytes()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    Scheduler(SchedulerError),
    Memory(UnifiedMemoryError),
    Rollback(SchedulerError),
}

impl From<SchedulerError> for AdmissionError {
    fn from(error: SchedulerError) -> Self {
        Self::Scheduler(error)
    }
}

impl From<UnifiedMemoryError> for AdmissionError {
    fn from(error: UnifiedMemoryError) -> Self {
        Self::Memory(error)
    }
}

impl Display for AdmissionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scheduler(error) => {
                write!(formatter, "request scheduler rejected admission: {error}")
            }
            Self::Memory(error) => write!(
                formatter,
                "unified-memory guard rejected admission: {error}"
            ),
            Self::Rollback(error) => write!(
                formatter,
                "cannot roll back sequence admission after memory rejection: {error}"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::scheduler::SchedulerConfig;
    use crate::unified_memory::{UnifiedMemoryConfig, UnifiedMemoryProbe, UnifiedMemorySnapshot};

    use super::*;

    struct FixedProbe;

    impl UnifiedMemoryProbe for FixedProbe {
        fn snapshot(&self) -> Result<UnifiedMemorySnapshot, UnifiedMemoryError> {
            Ok(UnifiedMemorySnapshot {
                process_rss_bytes: 60,
                process_anon_bytes: 20,
                process_file_bytes: 40,
                system_available_bytes: 50,
                cgroup_current_bytes: None,
                cgroup_limit_bytes: None,
            })
        }
    }

    fn runtime() -> RuntimeAdmission {
        let scheduler = StaticScheduler::new(SchedulerConfig {
            sequence_slots: 1,
            max_context_tokens: 128,
            max_prefill_tokens: 64,
            prefill_buckets: vec![16, 32, 64],
            kv_bytes_per_token: 16,
            recurrent_state_bytes: 256,
            shared_workspace_bytes: 256,
            fabric_committed_bytes: 1024,
            safety_reserve_bytes: 256,
            physical_memory_bytes: 8192,
        })
        .expect("scheduler");
        let memory = UnifiedMemoryGuard::new(
            UnifiedMemoryConfig {
                hard_ceiling_bytes: 100,
                planned_fixed_bytes: 80,
                safety_reserve_bytes: 10,
            },
            Arc::new(FixedProbe),
        )
        .expect("memory guard");
        RuntimeAdmission::new(scheduler, memory)
    }

    fn request(id: u64) -> RequestSpec {
        RequestSpec {
            id,
            prompt_tokens: 16,
            max_new_tokens: 2,
        }
    }

    #[test]
    fn memory_rejection_rolls_back_the_fixed_sequence_slot() {
        let mut runtime = runtime();
        assert!(matches!(
            runtime.admit(request(1), 21),
            Err(AdmissionError::Memory(
                UnifiedMemoryError::HardCeiling { .. }
            ))
        ));
        assert_eq!(runtime.active_sequences(), 0);
        let admitted = runtime.admit(request(2), 1).expect("slot was returned");
        assert_eq!(admitted.sequence().slot, 0);
    }

    #[test]
    fn release_returns_both_sequence_and_memory_leases() {
        let mut runtime = runtime();
        let admitted = runtime.admit(request(1), 5).expect("admit");
        assert_eq!(runtime.record_decode(&admitted), Ok(GraphShape::Decode));
        assert_eq!(admitted.transient_bytes(), Ok(5));
        runtime.release(admitted).expect("release");
        assert_eq!(runtime.active_sequences(), 0);
        assert_eq!(
            runtime
                .memory_stats()
                .expect("stats")
                .reserved_transient_bytes,
            0
        );
    }

    #[test]
    fn full_sequence_scheduler_never_creates_a_memory_reservation() {
        let mut runtime = runtime();
        let first = runtime.admit(request(1), 1).expect("first");
        assert!(matches!(
            runtime.admit(request(2), 1),
            Err(AdmissionError::Scheduler(SchedulerError::NoSequenceSlot))
        ));
        let stats = runtime.memory_stats().expect("stats");
        assert_eq!(stats.admissions, 1);
        assert_eq!(stats.active_requests, 1);
        runtime.release(first).expect("release");
    }
}
