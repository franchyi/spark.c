const GIB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug)]
pub struct ModelProfile {
    pub key: &'static str,
    pub resident_weight_bytes: u64,
    pub sparse_store_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryPlan {
    pub required_gib: f64,
    pub headroom_gib: f64,
    pub fits: bool,
}

pub const QWEN38_27B_NVFP4: ModelProfile = ModelProfile {
    key: "qwen38-27b-nvfp4",
    resident_weight_bytes: 26_306_150_400,
    sparse_store_bytes: 0,
};

pub const QWEN38_FLASH_NEXT_NVFP4: ModelProfile = ModelProfile {
    key: "qwen38-flash-next-nvfp4",
    resident_weight_bytes: 77_843_712_026,
    sparse_store_bytes: 51_200_245_760,
};

pub fn profile(key: &str) -> Option<ModelProfile> {
    match key {
        "qwen38-27b-nvfp4" => Some(QWEN38_27B_NVFP4),
        "qwen38-flash-next-nvfp4" => Some(QWEN38_FLASH_NEXT_NVFP4),
        _ => None,
    }
}
pub fn plan_memory(
    model: ModelProfile,
    system_gib: f64,
    sparse_cache_gib: f64,
    kv_cache_gib: f64,
    runtime_gib: f64,
    safety_gib: f64,
) -> Result<MemoryPlan, &'static str> {
    if [
        system_gib,
        sparse_cache_gib,
        kv_cache_gib,
        runtime_gib,
        safety_gib,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("memory sizes must be finite and non-negative");
    }

    let sparse_cache = if model.sparse_store_bytes == 0 {
        0.0
    } else {
        sparse_cache_gib
    };
    let resident_gib = model.resident_weight_bytes as f64 / GIB;
    let required_gib = resident_gib + sparse_cache + kv_cache_gib + runtime_gib + safety_gib;
    Ok(MemoryPlan {
        required_gib,
        headroom_gib: system_gib - required_gib,
        fits: required_gib <= system_gib,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_flash_next_plan_fits() {
        let plan =
            plan_memory(QWEN38_FLASH_NEXT_NVFP4, 121.0, 2.0, 8.0, 12.0, 8.0).expect("valid plan");
        assert!(plan.fits);
        assert!(plan.required_gib > 102.0 && plan.required_gib < 103.0);
    }

    #[test]
    fn materialized_ple_does_not_fit() {
        let ple_gib = QWEN38_FLASH_NEXT_NVFP4.sparse_store_bytes as f64 / GIB;
        let plan = plan_memory(QWEN38_FLASH_NEXT_NVFP4, 121.0, ple_gib, 8.0, 12.0, 8.0)
            .expect("valid plan");
        assert!(!plan.fits);
    }
}
