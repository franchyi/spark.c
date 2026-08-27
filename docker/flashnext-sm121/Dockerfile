ARG BASE_IMAGE=docker.1ms.run/lmsysorg/sglang:qwen38flashnext
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="SparkServe SGLang Flash-Next SM121 oracle"
LABEL org.opencontainers.image.description="Day-zero Qwen3.8 Flash-Next image with the GB10 QSA sparse-decode capability guard enabled"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang/pull/36497"

# The day-zero image only admits SM100 to the TRT-LLM sparse-decode path even
# though its bundled kernel works on consumer Blackwell SM120/SM121. It also
# contains Qwen4-Exp's text-only implementation but omits the architecture from
# the corresponding validation allowlist.
RUN target=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py && \
    sed -i \
      's/from sglang.srt.utils import is_sm100_supported/from sglang.srt.utils import is_sm100_supported, is_sm120_supported/' \
      "$target" && \
    sed -i \
      's/if not is_sm100_supported():/if not (is_sm100_supported() or is_sm120_supported()):/' \
      "$target" && \
    grep -Fq 'if not (is_sm100_supported() or is_sm120_supported()):' "$target" && \
    args=/sgl-workspace/sglang/python/sglang/srt/server_args.py && \
    sed -i \
      's/LANGUAGE_MODEL_ONLY_ARCHITECTURES = ("MuseGlimmerForConditionalGeneration",)/LANGUAGE_MODEL_ONLY_ARCHITECTURES = ("MuseGlimmerForConditionalGeneration", "Qwen4ExpForConditionalGeneration")/' \
      "$args" && \
    grep -Fq '"Qwen4ExpForConditionalGeneration")' "$args"
