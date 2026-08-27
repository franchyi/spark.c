from sparkserve.manifest import classify_file, summarize_model_payload


def test_classify_checkpoint_files() -> None:
    assert classify_file("model-plefp8-00000.safetensors") == "ple"
    assert classify_file("layer-00000-experts-0000-0127.safetensors") == "expert"
    assert classify_file("model-bf16-00001.safetensors") == "other"


def test_summarize_hf_payload() -> None:
    payload = {
        "sha": "abc123",
        "siblings": [
            {"rfilename": "layer-experts.safetensors", "size": 100},
            {"rfilename": "model-plefp8.safetensors", "lfs": {"size": 70}},
            {"rfilename": "config.json", "size": 30},
        ],
    }
    summary = summarize_model_payload(payload, "owner/model")
    assert summary.revision == "abc123"
    assert summary.file_count == 3
    assert summary.total_bytes == 200
    assert summary.expert_bytes == 100
    assert summary.ple_bytes == 70
    assert summary.other_bytes == 30
