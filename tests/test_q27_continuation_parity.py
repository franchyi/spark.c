from array import array
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


if __name__ == "__main__":
    unittest.main()
