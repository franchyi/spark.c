import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import q27_chatml_trace_parity as parity


def _record(
    record_id: int,
    prompt: list[int],
    generated: list[int],
    finish: str = "length",
    terminal: int | None = None,
) -> dict:
    return {
        "schema": parity.TRACE_SCHEMA,
        "record_id": record_id,
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "finish_reason": finish,
        "terminal_stop_token_id": terminal,
    }


class ChatmlTraceParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_trace(self, records: list[dict], *, final_newline: bool = True) -> Path:
        path = self.root / "native.jsonl"
        payload = "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
        if final_newline:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
        return path

    def _write_oracle(self, cases: list[dict]) -> Path:
        path = self.root / "oracle.json"
        path.write_text(
            json.dumps({"schema": parity.ORACLE_SCHEMA, "cases": cases}), encoding="utf-8"
        )
        return path

    def _write_suite(self, trace: Path, oracle: Path) -> Path:
        path = self.root / "suite.json"
        path.write_text(
            json.dumps(
                {
                    "schema": parity.SUITE_SCHEMA,
                    "native_trace": trace.name,
                    "oracle_artifact": oracle.name,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_exact_ordered_cases_pass(self) -> None:
        trace = self._write_trace(
            [
                _record(0, [248045, 10], [8678, 198]),
                _record(1, [248045, 20], [], "stop", 248046),
            ]
        )
        oracle = self._write_oracle(
            [
                {
                    "name": "chatml-a",
                    "prompt_token_ids": [248045, 10],
                    "generated_token_ids": [8678, 198],
                    "finish_reason": "length",
                    "terminal_stop_token_id": None,
                },
                {
                    "name": "chatml-b",
                    "prompt_token_ids": [248045, 20],
                    "generated_token_ids": [],
                    "finish_reason": "stop",
                    "terminal_stop_token_id": 248046,
                },
            ]
        )

        report = parity._compare_suite(self._write_suite(trace, oracle))

        self.assertTrue(report["passed"])
        self.assertEqual([case["name"] for case in report["cases"]], ["chatml-a", "chatml-b"])

    def test_exact_prompt_and_generation_mismatches_fail(self) -> None:
        trace = self._write_trace([_record(0, [1, 2], [3, 4])])
        oracle = self._write_oracle(
            [
                {
                    "name": "mismatch",
                    "prompt_token_ids": [1, 9],
                    "generated_token_ids": [3, 8],
                    "finish_reason": "length",
                }
            ]
        )

        report = parity._compare_suite(self._write_suite(trace, oracle))

        self.assertFalse(report["passed"])
        self.assertEqual(report["cases"][0]["prompt_difference"]["index"], 1)
        self.assertEqual(report["cases"][0]["generated_difference"]["index"], 1)

    def test_rejects_permissive_mode_and_symlink(self) -> None:
        trace = self._write_trace([_record(0, [1], [2])])
        trace.chmod(0o640)
        with self.assertRaisesRegex(parity.InputError, "more permissive than 0600"):
            parity._read_trace(trace)

        trace.chmod(0o600)
        link = self.root / "trace-link.jsonl"
        link.symlink_to(trace)
        with self.assertRaises(parity.InputError):
            parity._read_trace(link)

    def test_rejects_partial_empty_corrupt_and_trailing_jsonl(self) -> None:
        path = self.root / "native.jsonl"
        cases = {
            "partial": json.dumps(_record(0, [1], [2])).encode(),
            "empty-line": (json.dumps(_record(0, [1], [2])) + "\n\n").encode(),
            "corrupt": b"{not-json}\n",
            "trailing": (json.dumps(_record(0, [1], [2])) + " garbage\n").encode(),
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                path.write_bytes(payload)
                path.chmod(0o600)
                with self.assertRaises(parity.InputError):
                    parity._read_trace(path)

    def test_rejects_noncontiguous_ids_u32_violations_and_stop_inconsistency(self) -> None:
        invalid = [
            [_record(1, [1], [2])],
            [_record(0, [True], [2])],
            [_record(0, [1], [-1])],
            [_record(0, [1], [2], "stop", None)],
            [_record(0, [1], [2], "length", 3)],
        ]
        for records in invalid:
            with self.subTest(records=records):
                trace = self._write_trace(records)
                with self.assertRaises(parity.InputError):
                    parity._read_trace(trace)

    def test_rejects_record_and_byte_limit_without_large_fixtures(self) -> None:
        trace = self._write_trace([_record(0, [1], [2]), _record(1, [3], [4])])
        with self.assertRaisesRegex(parity.InputError, "exceeds 1 records"):
            parity._read_trace(trace, max_records=1)
        with self.assertRaisesRegex(parity.InputError, "exceeds 1 bytes"):
            parity._read_trace(trace, max_bytes=1)

    def test_rejects_duplicate_keys(self) -> None:
        trace = self.root / "native.jsonl"
        trace.write_text(
            '{"schema":"sparkserve.q27.token-trace.v1","schema":"duplicate"}\n',
            encoding="utf-8",
        )
        trace.chmod(0o600)
        with self.assertRaisesRegex(parity.InputError, "duplicate JSON key"):
            parity._read_trace(trace)


if __name__ == "__main__":
    unittest.main()
