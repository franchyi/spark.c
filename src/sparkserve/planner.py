from __future__ import annotations

from dataclasses import asdict, dataclass

GIB = 1024**3


@dataclass(frozen=True)
class ModelProfile:
    key: str
    checkpoint_gib: float
    resident_weights_gib: float
    sparse_store_gib: float = 0.0
    architecture: str = ""
    quantization: str = ""


@dataclass(frozen=True)
class MemoryPlan:
    model: str
    system_gib: float
    resident_weights_gib: float
    sparse_cache_gib: float
    kv_cache_gib: float
    runtime_gib: float
    safety_gib: float
    required_gib: float
    headroom_gib: float
    fits: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# The payload total comes from the hf-mirror model API on 2026-08-27. Tensor
# class sizes come from the standalone Rust safetensors-header scan: text-only
# base weights exclude 4.856 GiB of deferred MTP and 0.836 GiB of ignored vision
# tensors. Keeping PLE in FP8 avoids the 2x BF16 expansion.
_FLASH_CHECKPOINT_GIB = 135_253_622_894 / GIB
_FLASH_RESIDENT_BASE_GIB = 77_843_712_026 / GIB
_FLASH_PLE_FP8_GIB = 51_200_245_760 / GIB

PROFILES = {
    "qwen38-27b-nvfp4": ModelProfile(
        key="qwen38-27b-nvfp4",
        checkpoint_gib=23.77,
        resident_weights_gib=24.5,
        architecture="Qwen3.8 27B dense",
        quantization="NVFP4 routed/dense linears; BF16 exceptions",
    ),
    "qwen38-flash-next-nvfp4": ModelProfile(
        key="qwen38-flash-next-nvfp4",
        checkpoint_gib=_FLASH_CHECKPOINT_GIB,
        resident_weights_gib=_FLASH_RESIDENT_BASE_GIB,
        sparse_store_gib=_FLASH_PLE_FP8_GIB,
        architecture="Qwen3.8 Flash-Next: 36 GDN + 12 QSA + 512-expert MoE + sparse PLE",
        quantization="NVFP4 routed experts; FP8 PLE; BF16 attention/shared paths",
    ),
}


def plan_memory(
    profile: ModelProfile,
    *,
    system_gib: float = 121.0,
    sparse_cache_gib: float = 2.0,
    kv_cache_gib: float = 8.0,
    runtime_gib: float = 12.0,
    safety_gib: float = 8.0,
) -> MemoryPlan:
    if not profile.sparse_store_gib:
        sparse_cache_gib = 0.0
    values = (system_gib, sparse_cache_gib, kv_cache_gib, runtime_gib, safety_gib)
    if any(value < 0 for value in values):
        raise ValueError("memory sizes cannot be negative")

    required = (
        profile.resident_weights_gib
        + sparse_cache_gib
        + kv_cache_gib
        + runtime_gib
        + safety_gib
    )
    return MemoryPlan(
        model=profile.key,
        system_gib=round(system_gib, 2),
        resident_weights_gib=round(profile.resident_weights_gib, 2),
        sparse_cache_gib=round(sparse_cache_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        runtime_gib=round(runtime_gib, 2),
        safety_gib=round(safety_gib, 2),
        required_gib=round(required, 2),
        headroom_gib=round(system_gib - required, 2),
        fits=required <= system_gib,
    )
