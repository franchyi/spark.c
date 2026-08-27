use std::fmt;

pub const KERNEL_ABI_VERSION: u32 = 1;
pub const NVFP4_GROUP_SIZE: u32 = 16;
pub const WEIGHT_N_ALIGNMENT: u64 = 32;
pub const WEIGHT_K_ALIGNMENT: u64 = 64;
pub const WEIGHT_SCALE_N_ALIGNMENT: u64 = 128;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Invalid = 0,
    BFloat16 = 1,
    Nvfp4E2m1Packed = 2,
    Fp8E4m3 = 3,
    Float32 = 4,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleLayout {
    Invalid = 0,
    Linear = 1,
    Cutlass128x4 = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelBackend {
    Auto = 0,
    FlashInferMmFp4 = 1,
    CutlassSm121 = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseNvfp4Spec {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub padded_n: u64,
    pub padded_k: u64,
    pub scale_padded_n: u64,
    pub group_size: u32,
    pub input_scale_layout: ScaleLayout,
    pub weight_scale_layout: ScaleLayout,
    pub output_dtype: DataType,
    pub requested_backend: KernelBackend,
}

impl DenseNvfp4Spec {
    pub fn native(m: u64, n: u64, k: u64) -> Result<Self, KernelContractError> {
        Ok(Self {
            m,
            n,
            k,
            padded_n: align_up(n, WEIGHT_N_ALIGNMENT)?,
            padded_k: align_up(k, WEIGHT_K_ALIGNMENT)?,
            scale_padded_n: align_up(n, WEIGHT_SCALE_N_ALIGNMENT)?,
            group_size: NVFP4_GROUP_SIZE,
            input_scale_layout: ScaleLayout::Cutlass128x4,
            weight_scale_layout: ScaleLayout::Cutlass128x4,
            output_dtype: DataType::BFloat16,
            requested_backend: KernelBackend::Auto,
        })
    }

    pub fn validate(self) -> Result<(), KernelContractError> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(KernelContractError::ZeroDimension);
        }
        if self.group_size != NVFP4_GROUP_SIZE {
            return Err(KernelContractError::UnsupportedGroupSize(self.group_size));
        }
        if self.padded_n < self.n || self.padded_k < self.k {
            return Err(KernelContractError::PaddingSmallerThanLogical);
        }
        if self.scale_padded_n < self.padded_n {
            return Err(KernelContractError::ScalePaddingSmallerThanWeight);
        }
        if self.padded_n % WEIGHT_N_ALIGNMENT != 0
            || self.padded_k % WEIGHT_K_ALIGNMENT != 0
            || self.scale_padded_n % WEIGHT_SCALE_N_ALIGNMENT != 0
        {
            return Err(KernelContractError::NativeAlignment);
        }
        if self.input_scale_layout != ScaleLayout::Cutlass128x4
            || self.weight_scale_layout != ScaleLayout::Cutlass128x4
        {
            return Err(KernelContractError::UnsupportedScaleLayout);
        }
        if self.output_dtype != DataType::BFloat16 {
            return Err(KernelContractError::UnsupportedOutputType);
        }
        self.buffer_requirements()?;
        Ok(())
    }

    pub fn buffer_requirements(self) -> Result<DenseNvfp4Buffers, KernelContractError> {
        let packed_input_bytes = checked_product(&[self.m, self.padded_k / 2])?;
        let packed_weight_bytes = checked_product(&[self.padded_n, self.padded_k / 2])?;
        let input_scale_bytes =
            checked_product(&[self.m, self.padded_k / u64::from(self.group_size)])?;
        let weight_scale_bytes = checked_product(&[
            self.scale_padded_n,
            self.padded_k / u64::from(self.group_size),
        ])?;
        let output_bytes = checked_product(&[self.m, self.n, 2])?;
        Ok(DenseNvfp4Buffers {
            packed_input_bytes,
            input_scale_bytes,
            packed_weight_bytes,
            weight_scale_bytes,
            output_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseNvfp4Buffers {
    pub packed_input_bytes: u64,
    pub input_scale_bytes: u64,
    pub packed_weight_bytes: u64,
    pub weight_scale_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCaps {
    pub sm: u32,
    pub supports_fp4_tensor_cores: bool,
}

impl DeviceCaps {
    pub const GB10: Self = Self {
        sm: 121,
        supports_fp4_tensor_cores: true,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelCandidate {
    pub id: &'static str,
    pub backend: KernelBackend,
    pub source_revision: &'static str,
    pub linked: bool,
    pub min_sm: u32,
    pub exact_sm: Option<u32>,
    pub priority: u8,
}

pub const DENSE_NVFP4_CANDIDATES: [KernelCandidate; 2] = [
    KernelCandidate {
        id: "flashinfer-mm-fp4",
        backend: KernelBackend::FlashInferMmFp4,
        source_revision: "906181e3f4cf4bcc81835fb480db4011bbd80b62",
        linked: false,
        min_sm: 100,
        exact_sm: None,
        priority: 100,
    },
    KernelCandidate {
        id: "cutlass-sm121-nvfp4",
        backend: KernelBackend::CutlassSm121,
        source_revision: "unfrozen-candidate",
        linked: false,
        min_sm: 121,
        exact_sm: Some(121),
        priority: 50,
    },
];

pub fn select_dense_nvfp4_candidate(
    spec: DenseNvfp4Spec,
    caps: DeviceCaps,
) -> Result<KernelCandidate, KernelContractError> {
    spec.validate()?;
    if !caps.supports_fp4_tensor_cores || caps.sm < 100 {
        return Err(KernelContractError::DeviceUnsupported(caps.sm));
    }

    DENSE_NVFP4_CANDIDATES
        .iter()
        .copied()
        .filter(|candidate| {
            caps.sm >= candidate.min_sm
                && candidate.exact_sm.is_none_or(|exact| exact == caps.sm)
                && (spec.requested_backend == KernelBackend::Auto
                    || spec.requested_backend == candidate.backend)
        })
        .max_by_key(|candidate| candidate.priority)
        .ok_or(KernelContractError::NoKernelCandidate)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelContractError {
    ZeroDimension,
    DimensionOverflow,
    UnsupportedGroupSize(u32),
    PaddingSmallerThanLogical,
    ScalePaddingSmallerThanWeight,
    NativeAlignment,
    UnsupportedScaleLayout,
    UnsupportedOutputType,
    DeviceUnsupported(u32),
    NoKernelCandidate,
}

impl fmt::Display for KernelContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "M, N, and K must be non-zero"),
            Self::DimensionOverflow => write!(formatter, "kernel buffer size overflow"),
            Self::UnsupportedGroupSize(size) => {
                write!(formatter, "NVFP4 group size must be 16, got {size}")
            }
            Self::PaddingSmallerThanLogical => {
                write!(formatter, "padded shape is smaller than the logical shape")
            }
            Self::ScalePaddingSmallerThanWeight => {
                write!(formatter, "scale-padded N is smaller than packed-weight N")
            }
            Self::NativeAlignment => {
                write!(
                    formatter,
                    "weight N, weight K, and scale N must align to 32, 64, and 128"
                )
            }
            Self::UnsupportedScaleLayout => {
                write!(formatter, "expected CUTLASS 128x4 block-scale layout")
            }
            Self::UnsupportedOutputType => write!(formatter, "expected BF16 output"),
            Self::DeviceUnsupported(sm) => {
                write!(
                    formatter,
                    "SM{sm} does not expose native NVFP4 Tensor Cores"
                )
            }
            Self::NoKernelCandidate => write!(formatter, "no kernel candidate matches the plan"),
        }
    }
}

impl std::error::Error for KernelContractError {}

fn align_up(value: u64, alignment: u64) -> Result<u64, KernelContractError> {
    if value == 0 {
        return Err(KernelContractError::ZeroDimension);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or(KernelContractError::DimensionOverflow)
}

fn checked_product(values: &[u64]) -> Result<u64, KernelContractError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(KernelContractError::DimensionOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_plan_pads_to_flashinfer_alignment() {
        let spec = DenseNvfp4Spec::native(1, 4100, 4097).expect("valid shape");
        assert_eq!(spec.padded_n, 4128);
        assert_eq!(spec.padded_k, 4160);
        assert_eq!(spec.scale_padded_n, 4224);
        spec.validate().expect("valid native plan");
    }

    #[test]
    fn aligned_qwen_shape_has_exact_buffer_requirements() {
        let spec = DenseNvfp4Spec::native(1, 4096, 4096).expect("valid shape");
        assert_eq!(
            spec.buffer_requirements().expect("sizes fit"),
            DenseNvfp4Buffers {
                packed_input_bytes: 2048,
                input_scale_bytes: 256,
                packed_weight_bytes: 8_388_608,
                weight_scale_bytes: 1_048_576,
                output_bytes: 8192,
            }
        );
    }

    #[test]
    fn rejects_wrong_scale_layout_before_launch() {
        let mut spec = DenseNvfp4Spec::native(1, 4096, 4096).expect("valid shape");
        spec.weight_scale_layout = ScaleLayout::Linear;
        assert_eq!(
            spec.validate(),
            Err(KernelContractError::UnsupportedScaleLayout)
        );
    }

    #[test]
    fn selects_oracle_kernel_source_for_gb10() {
        let spec = DenseNvfp4Spec::native(1, 4096, 4096).expect("valid shape");
        let candidate = select_dense_nvfp4_candidate(spec, DeviceCaps::GB10).expect("candidate");
        assert_eq!(candidate.id, "flashinfer-mm-fp4");
        assert!(!candidate.linked);
    }

    #[test]
    fn rejects_pre_blackwell_device() {
        let spec = DenseNvfp4Spec::native(1, 4096, 4096).expect("valid shape");
        assert_eq!(
            select_dense_nvfp4_candidate(
                spec,
                DeviceCaps {
                    sm: 90,
                    supports_fp4_tensor_cores: false,
                }
            ),
            Err(KernelContractError::DeviceUnsupported(90))
        );
    }
}
