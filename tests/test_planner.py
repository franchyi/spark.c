from sparkserve.planner import PROFILES, plan_memory


def test_flash_next_sparse_plan_fits_single_spark() -> None:
    plan = plan_memory(PROFILES["qwen38-flash-next-nvfp4"])
    assert plan.fits
    assert 77 < plan.resident_weights_gib < 79
    assert plan.required_gib < 110


def test_materializing_the_ple_does_not_fit() -> None:
    profile = PROFILES["qwen38-flash-next-nvfp4"]
    plan = plan_memory(profile, sparse_cache_gib=profile.sparse_store_gib)
    assert not plan.fits


def test_dense_model_ignores_sparse_cache() -> None:
    plan = plan_memory(PROFILES["qwen38-27b-nvfp4"], sparse_cache_gib=99)
    assert plan.sparse_cache_gib == 0
