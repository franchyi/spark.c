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
    Int32 = 5,
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
    FlashInferGroupMmFp4 = 3,
    FlashInferCuteSiluNvfp4 = 4,
    FlashInferCuteNvfp4Quantize = 5,
    FlashInferMoeRoute = 6,
    SglangPleGather = 7,
    SglangQsaTopk = 8,
    SglangQsaIndexPrep = 9,
    SglangQsaKvPack = 10,
    FlashInferXqaDecode = 11,
    SglangQsaExpand = 12,
    TilelangQsaScore = 13,
    SglangCublasMoeGate = 14,
    SglangCublasSharedExpert = 15,
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
        // CUTLASS stores scale factors in 512-byte tiles: 128 logical rows by
        // four K-block columns. Even M=1 therefore owns a full 128-row scale
        // tile; using the logical M*K/16 size would under-allocate by 128x.
        let input_scale_bytes =
            swizzled_scale_bytes(self.m, self.padded_k / u64::from(self.group_size))?;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupedExpertLayout {
    pub m_indptr: Vec<i32>,
    pub scale_row_offsets: Vec<u64>,
    pub total_rows: u64,
    pub input_scale_rows: u64,
}

impl GroupedExpertLayout {
    /// Build FlashInfer's routed-row layout from one unpadded token count per
    /// expert. The four-row GEMM padding and 128-row scale padding are kept
    /// separate because they are different physical address spaces.
    pub fn from_expert_rows(expert_rows: &[u32]) -> Result<Self, KernelContractError> {
        if expert_rows.is_empty() || expert_rows.len() > 512 {
            return Err(KernelContractError::InvalidExpertCount(expert_rows.len()));
        }

        let mut m_indptr = Vec::with_capacity(expert_rows.len() + 1);
        let mut scale_row_offsets = Vec::with_capacity(expert_rows.len());
        m_indptr.push(0);
        let mut total_rows = 0_u64;
        let mut input_scale_rows = 0_u64;
        for (expert_index, rows) in expert_rows.iter().copied().enumerate() {
            let padded_rows = if rows == 0 {
                0
            } else {
                align_up(u64::from(rows), 4)?
            };
            let scale_offset_numerator = total_rows
                .checked_add(
                    u64::try_from(expert_index)
                        .map_err(|_| KernelContractError::DimensionOverflow)?
                        .checked_mul(127)
                        .ok_or(KernelContractError::DimensionOverflow)?,
                )
                .ok_or(KernelContractError::DimensionOverflow)?;
            let scale_offset = scale_offset_numerator / 128 * 128;
            scale_row_offsets.push(scale_offset);
            // Keep even an empty expert's pointer inside an allocated scale
            // tile. This lets a fixed 512-expert CUDA graph remain valid as
            // the set of hot experts changes between batches.
            let expert_scale_rows = if padded_rows == 0 {
                128
            } else {
                align_up(padded_rows, 128)?
            };
            input_scale_rows = input_scale_rows.max(
                scale_offset
                    .checked_add(expert_scale_rows)
                    .ok_or(KernelContractError::DimensionOverflow)?,
            );
            total_rows = total_rows
                .checked_add(padded_rows)
                .ok_or(KernelContractError::DimensionOverflow)?;
            m_indptr.push(
                i32::try_from(total_rows)
                    .map_err(|_| KernelContractError::RoutedRowCountOverflow)?,
            );
        }
        if total_rows == 0 {
            return Err(KernelContractError::ZeroDimension);
        }
        Ok(Self {
            m_indptr,
            scale_row_offsets,
            total_rows,
            input_scale_rows,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedNvfp4Spec {
    pub num_groups: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub n: u64,
    pub k: u64,
    pub group_size: u32,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub swap_ab: bool,
    pub input_scale_layout: ScaleLayout,
    pub weight_scale_layout: ScaleLayout,
    pub output_dtype: DataType,
    pub requested_backend: KernelBackend,
}

impl GroupedNvfp4Spec {
    pub fn qwen_expert_projection(
        layout: &GroupedExpertLayout,
        n: u64,
        k: u64,
    ) -> Result<Self, KernelContractError> {
        let num_groups = u32::try_from(layout.scale_row_offsets.len())
            .map_err(|_| KernelContractError::DimensionOverflow)?;
        let spec = Self {
            num_groups,
            total_rows: layout.total_rows,
            input_scale_rows: layout.input_scale_rows,
            n,
            k,
            group_size: NVFP4_GROUP_SIZE,
            tile_m: 128,
            tile_n: 128,
            tile_k: 256,
            swap_ab: false,
            input_scale_layout: ScaleLayout::Cutlass128x4,
            weight_scale_layout: ScaleLayout::Cutlass128x4,
            output_dtype: DataType::BFloat16,
            requested_backend: KernelBackend::FlashInferGroupMmFp4,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(self) -> Result<(), KernelContractError> {
        if self.num_groups == 0 || self.num_groups > 512 {
            return Err(KernelContractError::InvalidExpertCount(
                self.num_groups as usize,
            ));
        }
        if self.total_rows == 0 || self.n == 0 || self.k == 0 {
            return Err(KernelContractError::ZeroDimension);
        }
        if self.total_rows % 4 != 0
            || self.input_scale_rows < self.total_rows
            || self.input_scale_rows % 128 != 0
        {
            return Err(KernelContractError::InvalidRoutedPadding);
        }
        if self.n % 128 != 0 || self.k % 128 != 0 {
            return Err(KernelContractError::GroupedAlignment);
        }
        if self.group_size != NVFP4_GROUP_SIZE {
            return Err(KernelContractError::UnsupportedGroupSize(self.group_size));
        }
        if self.tile_m != 128 || self.tile_n != 128 || self.tile_k != 256 || self.swap_ab {
            return Err(KernelContractError::UnsupportedGroupedTactic);
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

    pub fn buffer_requirements(self) -> Result<GroupedNvfp4Buffers, KernelContractError> {
        Ok(GroupedNvfp4Buffers {
            packed_input_bytes: checked_product(&[self.total_rows, self.k / 2])?,
            input_scale_bytes: checked_product(&[
                self.input_scale_rows,
                self.k / u64::from(self.group_size),
            ])?,
            packed_weight_bytes: checked_product(&[
                u64::from(self.num_groups),
                self.n,
                self.k / 2,
            ])?,
            weight_scale_bytes: checked_product(&[
                u64::from(self.num_groups),
                self.n,
                self.k / u64::from(self.group_size),
            ])?,
            output_bytes: checked_product(&[self.total_rows, self.n, 2])?,
            int_workspace_bytes: 32 * 1024 * 1024,
            float_workspace_bytes: 32 * 1024 * 1024,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedNvfp4Buffers {
    pub packed_input_bytes: u64,
    pub input_scale_bytes: u64,
    pub packed_weight_bytes: u64,
    pub weight_scale_bytes: u64,
    pub output_bytes: u64,
    pub int_workspace_bytes: u64,
    pub float_workspace_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiluNvfp4Spec {
    pub num_experts: u32,
    pub rows_per_expert: u32,
    pub hidden_size: u64,
    pub group_size: u32,
    pub input_dtype: DataType,
    pub output_scale_layout: ScaleLayout,
    pub requested_backend: KernelBackend,
}

impl SiluNvfp4Spec {
    pub fn qwen_flash(num_experts: u32, rows_per_expert: u32) -> Result<Self, KernelContractError> {
        let spec = Self {
            num_experts,
            rows_per_expert,
            hidden_size: 640,
            group_size: NVFP4_GROUP_SIZE,
            input_dtype: DataType::BFloat16,
            output_scale_layout: ScaleLayout::Cutlass128x4,
            requested_backend: KernelBackend::FlashInferCuteSiluNvfp4,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(self) -> Result<(), KernelContractError> {
        if self.num_experts == 0 || self.num_experts > 512 {
            return Err(KernelContractError::InvalidExpertCount(
                self.num_experts as usize,
            ));
        }
        if self.rows_per_expert == 0 || self.rows_per_expert % 4 != 0 {
            return Err(KernelContractError::InvalidRoutedPadding);
        }
        if self.hidden_size == 0 || self.hidden_size % 128 != 0 {
            return Err(KernelContractError::GroupedAlignment);
        }
        if self.group_size != NVFP4_GROUP_SIZE {
            return Err(KernelContractError::UnsupportedGroupSize(self.group_size));
        }
        if self.input_dtype != DataType::BFloat16 {
            return Err(KernelContractError::UnsupportedInputType);
        }
        if self.output_scale_layout != ScaleLayout::Cutlass128x4 {
            return Err(KernelContractError::UnsupportedScaleLayout);
        }
        if !matches!(
            self.requested_backend,
            KernelBackend::Auto | KernelBackend::FlashInferCuteSiluNvfp4
        ) {
            return Err(KernelContractError::UnsupportedFusedTactic);
        }
        self.buffer_requirements()?;
        Ok(())
    }

    pub fn buffer_requirements(self) -> Result<SiluNvfp4Buffers, KernelContractError> {
        let experts = u64::from(self.num_experts);
        let rows = u64::from(self.rows_per_expert);
        let scale_rows = align_up(rows, 128)?;
        let scale_columns = align_up(self.hidden_size / u64::from(self.group_size), 4)?;
        Ok(SiluNvfp4Buffers {
            input_bytes: checked_product(&[experts, rows, self.hidden_size, 4])?,
            packed_output_bytes: checked_product(&[experts, rows, self.hidden_size / 2])?,
            output_scale_bytes: checked_product(&[experts, scale_rows, scale_columns])?,
            global_scale_bytes: checked_product(&[experts, 4])?,
            active_rows_bytes: checked_product(&[experts, 4])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiluNvfp4Buffers {
    pub input_bytes: u64,
    pub packed_output_bytes: u64,
    pub output_scale_bytes: u64,
    pub global_scale_bytes: u64,
    pub active_rows_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedSiluNvfp4Spec {
    pub num_experts: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub hidden_size: u64,
    pub group_size: u32,
    pub input_dtype: DataType,
    pub output_scale_layout: ScaleLayout,
    pub requested_backend: KernelBackend,
}

impl SegmentedSiluNvfp4Spec {
    pub fn from_grouped_layout(
        layout: &GroupedExpertLayout,
        hidden_size: u64,
    ) -> Result<Self, KernelContractError> {
        let spec = Self {
            num_experts: u32::try_from(layout.m_indptr.len() - 1)
                .map_err(|_| KernelContractError::DimensionOverflow)?,
            total_rows: layout.total_rows,
            input_scale_rows: layout.input_scale_rows,
            hidden_size,
            group_size: NVFP4_GROUP_SIZE,
            input_dtype: DataType::BFloat16,
            output_scale_layout: ScaleLayout::Cutlass128x4,
            requested_backend: KernelBackend::FlashInferCuteSiluNvfp4,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(self) -> Result<(), KernelContractError> {
        if self.num_experts == 0 || self.num_experts > 512 {
            return Err(KernelContractError::InvalidExpertCount(
                self.num_experts as usize,
            ));
        }
        if self.total_rows == 0
            || self.total_rows % 4 != 0
            || self.input_scale_rows < self.total_rows
            || self.input_scale_rows % 128 != 0
        {
            return Err(KernelContractError::InvalidRoutedPadding);
        }
        if self.hidden_size == 0 || self.hidden_size % 128 != 0 {
            return Err(KernelContractError::GroupedAlignment);
        }
        if self.group_size != NVFP4_GROUP_SIZE {
            return Err(KernelContractError::UnsupportedGroupSize(self.group_size));
        }
        if self.input_dtype != DataType::BFloat16 {
            return Err(KernelContractError::UnsupportedInputType);
        }
        if self.output_scale_layout != ScaleLayout::Cutlass128x4 {
            return Err(KernelContractError::UnsupportedScaleLayout);
        }
        if !matches!(
            self.requested_backend,
            KernelBackend::Auto | KernelBackend::FlashInferCuteSiluNvfp4
        ) {
            return Err(KernelContractError::UnsupportedFusedTactic);
        }
        self.buffer_requirements()?;
        Ok(())
    }

    pub fn buffer_requirements(self) -> Result<SegmentedSiluNvfp4Buffers, KernelContractError> {
        Ok(SegmentedSiluNvfp4Buffers {
            input_bytes: checked_product(&[self.total_rows, self.hidden_size, 4])?,
            packed_output_bytes: checked_product(&[self.total_rows, self.hidden_size / 2])?,
            output_scale_bytes: checked_product(&[
                self.input_scale_rows,
                self.hidden_size / u64::from(self.group_size),
            ])?,
            global_scale_bytes: checked_product(&[u64::from(self.num_experts), 4])?,
            host_metadata_bytes: checked_product(&[u64::from(self.num_experts), 16])?
                .checked_add(4)
                .ok_or(KernelContractError::DimensionOverflow)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedSiluNvfp4Buffers {
    pub input_bytes: u64,
    pub packed_output_bytes: u64,
    pub output_scale_bytes: u64,
    pub global_scale_bytes: u64,
    pub host_metadata_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedNvfp4QuantizeSpec {
    pub num_experts: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub hidden_size: u64,
    pub group_size: u32,
    pub input_dtype: DataType,
    pub output_scale_layout: ScaleLayout,
    pub requested_backend: KernelBackend,
}

impl SegmentedNvfp4QuantizeSpec {
    pub fn from_grouped_layout(
        layout: &GroupedExpertLayout,
        hidden_size: u64,
    ) -> Result<Self, KernelContractError> {
        let spec = Self {
            num_experts: u32::try_from(layout.m_indptr.len() - 1)
                .map_err(|_| KernelContractError::DimensionOverflow)?,
            total_rows: layout.total_rows,
            input_scale_rows: layout.input_scale_rows,
            hidden_size,
            group_size: NVFP4_GROUP_SIZE,
            input_dtype: DataType::BFloat16,
            output_scale_layout: ScaleLayout::Cutlass128x4,
            requested_backend: KernelBackend::FlashInferCuteNvfp4Quantize,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(self) -> Result<(), KernelContractError> {
        if self.num_experts == 0 || self.num_experts > 512 {
            return Err(KernelContractError::InvalidExpertCount(
                self.num_experts as usize,
            ));
        }
        if self.total_rows == 0
            || self.total_rows % 4 != 0
            || self.input_scale_rows < self.total_rows
            || self.input_scale_rows % 128 != 0
        {
            return Err(KernelContractError::InvalidRoutedPadding);
        }
        if self.hidden_size == 0 || self.hidden_size % 128 != 0 {
            return Err(KernelContractError::GroupedAlignment);
        }
        if self.group_size != NVFP4_GROUP_SIZE {
            return Err(KernelContractError::UnsupportedGroupSize(self.group_size));
        }
        if self.input_dtype != DataType::BFloat16 {
            return Err(KernelContractError::UnsupportedInputType);
        }
        if self.output_scale_layout != ScaleLayout::Cutlass128x4 {
            return Err(KernelContractError::UnsupportedScaleLayout);
        }
        if !matches!(
            self.requested_backend,
            KernelBackend::Auto | KernelBackend::FlashInferCuteNvfp4Quantize
        ) {
            return Err(KernelContractError::UnsupportedFusedTactic);
        }
        self.buffer_requirements()?;
        Ok(())
    }

    pub fn buffer_requirements(self) -> Result<SegmentedNvfp4QuantizeBuffers, KernelContractError> {
        Ok(SegmentedNvfp4QuantizeBuffers {
            input_bytes: checked_product(&[self.total_rows, self.hidden_size, 2])?,
            packed_output_bytes: checked_product(&[self.total_rows, self.hidden_size / 2])?,
            output_scale_bytes: checked_product(&[
                self.input_scale_rows,
                self.hidden_size / u64::from(self.group_size),
            ])?,
            global_scale_bytes: checked_product(&[u64::from(self.num_experts), 4])?,
            host_metadata_bytes: checked_product(&[u64::from(self.num_experts), 16])?
                .checked_add(4)
                .ok_or(KernelContractError::DimensionOverflow)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedNvfp4QuantizeBuffers {
    pub input_bytes: u64,
    pub packed_output_bytes: u64,
    pub output_scale_bytes: u64,
    pub global_scale_bytes: u64,
    pub host_metadata_bytes: u64,
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
    InvalidExpertCount(usize),
    RoutedRowCountOverflow,
    InvalidRoutedPadding,
    GroupedAlignment,
    UnsupportedGroupedTactic,
    UnsupportedInputType,
    UnsupportedFusedTactic,
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
            Self::InvalidExpertCount(count) => {
                write!(
                    formatter,
                    "grouped NVFP4 requires 1..=512 experts, got {count}"
                )
            }
            Self::RoutedRowCountOverflow => {
                write!(
                    formatter,
                    "routed expert rows exceed the INT32 kernel index range"
                )
            }
            Self::InvalidRoutedPadding => write!(
                formatter,
                "routed rows require four-row GEMM padding and 128-row scale padding"
            ),
            Self::GroupedAlignment => {
                write!(formatter, "grouped NVFP4 N and K must align to 128")
            }
            Self::UnsupportedGroupedTactic => write!(
                formatter,
                "linked grouped NVFP4 tactic is 128x128x256 with swap_ab=false"
            ),
            Self::UnsupportedInputType => write!(formatter, "expected BF16 input"),
            Self::UnsupportedFusedTactic => {
                write!(formatter, "expected the FlashInfer CuTe fused NVFP4 donor")
            }
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

fn swizzled_scale_bytes(rows: u64, scale_columns: u64) -> Result<u64, KernelContractError> {
    let row_tiles = rows
        .checked_add(127)
        .ok_or(KernelContractError::DimensionOverflow)?
        / 128;
    let column_tiles = scale_columns
        .checked_add(3)
        .ok_or(KernelContractError::DimensionOverflow)?
        / 4;
    checked_product(&[row_tiles, column_tiles, 512])
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
                input_scale_bytes: 32_768,
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

    #[test]
    fn routed_expert_layout_separates_gemm_and_scale_padding() {
        let layout = GroupedExpertLayout::from_expert_rows(&[1, 0, 5]).expect("layout");
        assert_eq!(layout.m_indptr, vec![0, 4, 4, 12]);
        assert_eq!(layout.scale_row_offsets, vec![0, 128, 256]);
        assert_eq!(layout.total_rows, 12);
        assert_eq!(layout.input_scale_rows, 384);
    }

    #[test]
    fn qwen_grouped_projection_sizes_include_all_experts() {
        let mut rows = vec![0; 512];
        rows[3] = 1;
        rows[300] = 2;
        let layout = GroupedExpertLayout::from_expert_rows(&rows).expect("layout");
        let spec = GroupedNvfp4Spec::qwen_expert_projection(&layout, 640, 2560)
            .expect("Qwen gate projection");
        let buffers = spec.buffer_requirements().expect("buffer sizes");
        assert_eq!(spec.num_groups, 512);
        assert_eq!(spec.total_rows, 8);
        assert_eq!(spec.input_scale_rows, 65_024);
        assert_eq!(buffers.packed_weight_bytes, 419_430_400);
        assert_eq!(buffers.weight_scale_bytes, 52_428_800);
        assert_eq!(
            buffers.int_workspace_bytes + buffers.float_workspace_bytes,
            64 * 1024 * 1024
        );
    }

    #[test]
    fn qwen_fused_activation_preserves_per_expert_scale_tiles() {
        let spec = SiluNvfp4Spec::qwen_flash(10, 4).expect("fused activation");
        assert_eq!(
            spec.buffer_requirements().expect("buffer sizes"),
            SiluNvfp4Buffers {
                input_bytes: 102_400,
                packed_output_bytes: 12_800,
                output_scale_bytes: 51_200,
                global_scale_bytes: 40,
                active_rows_bytes: 40,
            }
        );
    }

    #[test]
    fn segmented_activation_reuses_the_grouped_row_layout_without_copying() {
        let layout = GroupedExpertLayout::from_expert_rows(&[4, 0, 2]).expect("layout");
        let spec = SegmentedSiluNvfp4Spec::from_grouped_layout(&layout, 640)
            .expect("segmented activation");
        assert_eq!(spec.total_rows, 8);
        assert_eq!(spec.input_scale_rows, 384);
        assert_eq!(
            spec.buffer_requirements().expect("buffers"),
            SegmentedSiluNvfp4Buffers {
                input_bytes: 20_480,
                packed_output_bytes: 2_560,
                output_scale_bytes: 15_360,
                global_scale_bytes: 12,
                host_metadata_bytes: 52,
            }
        );
    }

    #[test]
    fn qwen_input_quantizer_reuses_the_rust_expert_layout() {
        let layout = GroupedExpertLayout::from_expert_rows(&[2, 0, 3]).expect("layout");
        let spec = SegmentedNvfp4QuantizeSpec::from_grouped_layout(&layout, 2560)
            .expect("segmented quantizer");
        assert_eq!(spec.total_rows, 8);
        assert_eq!(spec.input_scale_rows, 384);
        assert_eq!(
            spec.buffer_requirements().expect("buffers"),
            SegmentedNvfp4QuantizeBuffers {
                input_bytes: 40_960,
                packed_output_bytes: 10_240,
                output_scale_bytes: 61_440,
                global_scale_bytes: 12,
                host_metadata_bytes: 52,
            }
        );
    }

    #[test]
    fn routed_scale_arena_covers_a_large_last_expert() {
        let layout = GroupedExpertLayout::from_expert_rows(&[1, 257]).expect("layout");
        assert_eq!(layout.m_indptr, vec![0, 4, 264]);
        assert_eq!(layout.scale_row_offsets, vec![0, 128]);
        assert_eq!(layout.input_scale_rows, 512);
    }
}
