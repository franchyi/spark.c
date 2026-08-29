from array import array
from pathlib import Path
import tempfile
import unittest

from scripts import q27_continuation_parity as parity


def _step(logits: array, token_ids: list[int], greedy_token_id: int | None = None) -> dict:
    return {
        "greedy_token_id": token_ids[0] if greedy_token_id is None else greedy_token_id,
        "top5": [
            {"token_id": token_id, "logit_fp32": float(logits[token_id])}
            for token_id in token_ids
        ],
    }


class TopFiveValidationTest(unittest.TestCase):
    def test_preserves_untied_order(self) -> None:
        logits = array("f", [-5.0, 9.0, 2.0, 7.0, 8.0, 1.0, 6.0])

        parity._validate_top5(_step(logits, [1, 4, 3, 6, 2]), logits, "untied")

    def test_uses_lowest_token_id_for_maximum_tie(self) -> None:
        logits = array("f", [10.0, 3.0, 10.0, 9.0, 8.0, 7.0, 6.0])
        expected = [0, 2, 3, 4, 5]

        self.assertEqual(parity._top_indices(logits, parity.TOP_K), expected)
        parity._validate_top5(_step(logits, expected), logits, "maximum-tie")

        with self.assertRaisesRegex(
            parity.InputError, "deterministic value/token-id ordering"
        ):
            parity._validate_top5(
                _step(logits, [2, 0, 3, 4, 5], greedy_token_id=2),
                logits,
                "maximum-tie",
            )

    def test_uses_lowest_token_id_across_cutoff_tie(self) -> None:
        logits = array(
            "f", [0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0]
        )
        expected = [10, 9, 8, 7, 1]

        self.assertEqual(parity._top_indices(logits, parity.TOP_K), expected)
        parity._validate_top5(_step(logits, expected), logits, "cutoff-tie")

        with self.assertRaisesRegex(
            parity.InputError, "deterministic value/token-id ordering"
        ):
            parity._validate_top5(
                _step(logits, [10, 9, 8, 7, 4]),
                logits,
                "cutoff-tie",
            )


class NativeOutputParsingTest(unittest.TestCase):
    def _write(self, payload: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(payload)
        return Path(temporary.name)

    def test_accepts_one_or_multiple_fields_per_line(self) -> None:
        path = self._write(
            "q27_eager=native\n"
            "input_token=248045\n"
            "top1_token=8678 top1_logit=23.064886093\n"
        )

        self.assertEqual(
            parity._parse_native_output(path),
            {
                "q27_eager": "native",
                "input_token": "248045",
                "top1_token": "8678",
                "top1_logit": "23.064886093",
            },
        )

    def test_rejects_duplicate_field_across_compound_line(self) -> None:
        path = self._write(
            "q27_eager=native\n"
            "top1_token=8678 top1_logit=23.0\n"
            "top1_token=846\n"
        )

        with self.assertRaisesRegex(parity.InputError, "duplicate native output key"):
            parity._parse_native_output(path)


if __name__ == "__main__":
    unittest.main()
