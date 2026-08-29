use std::path::Path;
use std::path::PathBuf;

use spark_flash_next::checkpoint::load_flash_next_checkpoint;
use spark_flash_next::fabric::{ExpertKey, ExpertLoad, ExpertSlotAddress};
use spark_flash_next::qwen_expert_cache::{
    QwenExpertFileLoader, QwenExpertHotCache, QwenPreparedExpertCache,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: qwen_expert_cache_smoke <model-root>"));
    let fixture = std::env::args().nth(2).map(PathBuf::from);
    let checkpoint = load_flash_next_checkpoint(Path::new(&model))?;
    let mut loader = QwenExpertFileLoader::new(&checkpoint);
    let mut cache = QwenExpertHotCache::create(0)?;
    let mut prepared = QwenPreparedExpertCache::create(16, 0)?;
    let mut experts = if let Some(fixture) = &fixture {
        std::fs::read(fixture.join("route_experts_i32.bin"))?
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .map(|expert| u16::try_from(expert).expect("nonnegative Qwen expert"))
            .collect::<Vec<_>>()
    } else {
        vec![0, 127]
    };
    if fixture.is_some() {
        experts.sort_unstable();
    }
    let loads = experts
        .iter()
        .enumerate()
        .map(|(slot, expert)| ExpertLoad {
            key: ExpertKey {
                layer: 0,
                expert: *expert,
            },
            address: ExpertSlotAddress {
                slot: slot as u32,
                byte_offset: slot as u64 * 2_764_800,
            },
            evicts: None,
        })
        .collect::<Vec<_>>();
    unsafe { prepared.prepare_and_promote(&mut loader, &mut cache, &loads)?; }
    if let Some(fixture) = fixture {
        let host = unsafe { cache.host_payloads()? };
        compare_prefix(host.w13_weights, &fixture.join("w13_fp4.bin"))?;
        compare_prefix(host.w2_weights, &fixture.join("w2_fp4.bin"))?;
        compare_prefix(host.w13_scales, &fixture.join("w13_scales.bin"))?;
        compare_prefix(host.w2_scales, &fixture.join("w2_scales.bin"))?;
        compare_prefix(
            host.w13_input_global_scales,
            &fixture.join("w13_global_scales_f32.bin"),
        )?;
        compare_prefix(host.w13_alpha, &fixture.join("w13_alpha_f32.bin"))?;
        compare_prefix(
            host.w2_input_global_scales,
            &fixture.join("down_global_scales_f32.bin"),
        )?;
        compare_prefix(host.w2_alpha, &fixture.join("w2_alpha_f32.bin"))?;
    }
    println!(
        "Qwen prepared expert promotion passed: {} fills, {} resident",
        loads.len(),
        prepared.stats().resident,
    );
    unsafe { prepared.prepare_and_promote(&mut loader, &mut cache, &loads)?; }
    let stats = prepared.stats();
    if stats.hits != loads.len() as u64 || stats.misses != loads.len() as u64 {
        return Err(format!("unexpected prepared cache stats: {stats:?}").into());
    }
    println!("Qwen prepared expert replay passed: {} cache hits", stats.hits);
    Ok(())
}

fn compare_prefix(actual: &[u8], expected_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let expected = std::fs::read(expected_path)?;
    if actual.get(..expected.len()) != Some(expected.as_slice()) {
        return Err(format!("real checkpoint pack differs from {}", expected_path.display()).into());
    }
    Ok(())
}
