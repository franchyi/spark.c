"""Opt-in SGLang oracle adapter for SparkServe's exact-FP8 NVMe PLE store."""

from __future__ import annotations

import os


def _install_nvme_ple() -> None:
    index_path = os.environ.get("SPARKSERVE_PLE_INDEX")
    if not index_path:
        return

    import torch
    from torch import nn

    from sparkserve.ple_store import PleIndex, PleReader
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.models import qwen4_exp
    from sglang.srt.utils import logger

    if getattr(qwen4_exp, "_sparkserve_nvme_ple_installed", False):
        return

    model_root = os.environ.get("SPARKSERVE_MODEL_ROOT", "/model")
    cache_mib = int(os.environ.get("SPARKSERVE_PLE_CACHE_MIB", "512"))
    workers = int(os.environ.get("SPARKSERVE_PLE_WORKERS", "16"))
    index = PleIndex.read(index_path)
    cache_pages = cache_mib * 1024**2 // index.page_bytes
    if cache_pages <= 0:
        raise RuntimeError("SPARKSERVE_PLE_CACHE_MIB is smaller than one page")

    original_pinned_class = qwen4_exp.Qwen4ExpPinnedHostEmbedding

    class SparkServeNvmePleEmbedding(VocabParallelEmbedding):
        """Read exact FP8 rows from NVMe instead of materializing the PLE table."""

        _COPIED_ATTRIBUTES = original_pinned_class._COPIED_ATTRIBUTES

        def __init__(self, embedding: VocabParallelEmbedding) -> None:
            nn.Module.__init__(self)
            if embedding.tp_size != 1:
                raise NotImplementedError(
                    "SparkServe NVMe PLE oracle currently requires TP=1"
                )
            if embedding.weight.dtype != torch.float8_e4m3fn:
                raise TypeError("SparkServe NVMe PLE requires exact FP8-E4M3 storage")
            for name in self._COPIED_ATTRIBUTES:
                setattr(self, name, getattr(embedding, name))
            if self.org_vocab_size != index.total_rows:
                raise RuntimeError(
                    f"PLE index has {index.total_rows} rows, model expects "
                    f"{self.org_vocab_size}"
                )
            self.quant_method = None
            self.register_parameter("weight", None)
            self.register_buffer(
                "weight_scale", embedding.weight_scale, persistent=True
            )
            del embedding.weight
            torch.cuda.empty_cache()
            self._reader = PleReader(
                index,
                model_root,
                cache_pages=cache_pages,
                workers=workers,
            )
            logger.info(
                "SparkServe NVMe PLE enabled: %.3f GiB exact FP8, %d MiB cache, "
                "%d workers",
                index.total_rows * index.row_bytes / 1024**3,
                cache_mib,
                workers,
            )

        def allocate_output(self, shape, device):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        def gather(self, input_ids: torch.Tensor, out=None) -> torch.Tensor:
            expected_shape = (*input_ids.shape, self.embedding_dim)
            output = (
                self.allocate_output(expected_shape, input_ids.device)
                if out is None
                else out
            )
            if tuple(output.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE output shape: {tuple(output.shape)} != {expected_shape}"
                )
            if output.dtype != torch.bfloat16 or output.device != input_ids.device:
                raise ValueError("PLE output must be BF16 on the input-id device")

            flat_ids = input_ids.reshape(-1).long()
            if flat_ids.numel() == 0:
                return output
            rows = self._reader.fetch_rows(flat_ids.detach().cpu().tolist())
            row_bytes = torch.frombuffer(bytearray(rows), dtype=torch.uint8)
            fp8_rows = row_bytes.view(torch.float8_e4m3fn).reshape(expected_shape)
            output.copy_(fp8_rows.to(device=input_ids.device, dtype=torch.bfloat16))
            return output

        def reduce(self, output: torch.Tensor) -> torch.Tensor:
            return output

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.gather(input_ids)

    original_load_weights = qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights

    def load_weights_without_resident_ple(self, weights):
        filtered = (
            (name, weight)
            for name, weight in weights
            if ".ngram_embedding.shard_" not in name
        )
        return original_load_weights(self, filtered)

    qwen4_exp.Qwen4ExpPinnedHostEmbedding = SparkServeNvmePleEmbedding
    qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights = (
        load_weights_without_resident_ple
    )
    qwen4_exp._sparkserve_nvme_ple_installed = True
    logger.info("installed SparkServe exact-FP8 NVMe PLE oracle adapter")


_install_nvme_ple()
