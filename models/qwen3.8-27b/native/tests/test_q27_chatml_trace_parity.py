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

    def _write_pinned_oracle(self, cases: list[dict]) -> Path:
        path = self.root / "pinned-oracle.json"
        provenance = dict(parity.EXPECTED_PINNED_ORACLE)
        provenance.update(
            {
                "repo_digests": [],
                "flashinfer_version": "test",
                "launch_command": [],
                "model_info": {},
            }
        )
        raw_cases = []
        for index, case in enumerate(cases):
            name = case["name"]
            request_path = self.root / name / "request.json"
            response_path = self.root / name / "response.json"
            request_path.parent.mkdir()
            request = {"model": "q27-chatml-oracle", "messages": []}
            request.update(parity.EXPECTED_PINNED_REQUEST)
            request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            selected = [[0.0, token, str(token)] for token in case["output_token_ids"]]
            top = [[[0.0, token, str(token)]] for token in case["output_token_ids"]]
            response = {
                "id": f"case-{index}",
                "choices": [
                    {
                        "prompt_token_ids": case["prompt_token_ids"],
                        "token_ids": case["output_token_ids"],
                        "finish_reason": case["finish_reason"],
                        "matched_stop": case["matched_stop"],
                        "meta_info": {
                            "output_token_logprobs": selected,
                            "output_top_logprobs": top,
                        },
                    }
                ],
            }
            response_path.write_text(json.dumps(response) + "\n", encoding="utf-8")
            raw_cases.append(
                {
                    **case,
                    "request": str(request_path.relative_to(self.root)),
                    "response": str(response_path.relative_to(self.root)),
                    "id": f"case-{index}",
                    "text": "",
                }
            )
        path.write_text(
            json.dumps(
                {
                    "schema": parity.PINNED_ORACLE_SCHEMA,
                    "captured_at_utc": "2026-08-29T00:00:00Z",
                    "oracle": provenance,
                    "common_request": parity.EXPECTED_PINNED_REQUEST,
                    "cases": raw_cases,
                }
            )
            + "\n",
            encoding="utf-8",
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

    def test_pinned_oracle_manifest_and_response_fields_are_crosschecked(self) -> None:
        cases = [
            {
                "name": f"case-{index}",
                "prompt_token_ids": [248045, index],
                "output_token_ids": [100 + index],
                "finish_reason": "length",
                "matched_stop": None,
            }
            for index in range(3)
        ]
        trace = self._write_trace(
            [
                _record(index, case["prompt_token_ids"], case["output_token_ids"])
                for index, case in enumerate(cases)
            ]
        )
        oracle = self._write_pinned_oracle(cases)

        report = parity._compare_suite(self._write_suite(trace, oracle))

        self.assertTrue(report["passed"])
        response_path = self.root / "case-1" / "response.json"
        response = json.loads(response_path.read_text())
        response["choices"][0]["token_ids"] = [999]
        response_path.write_text(json.dumps(response), encoding="utf-8")
        with self.assertRaisesRegex(parity.InputError, "response output ids differ"):
            parity._compare_suite(self._write_suite(trace, oracle))

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
            '{"schema":"q27.q27.token-trace.v1","schema":"duplicate"}\n',
            encoding="utf-8",
        )
        trace.chmod(0o600)
        with self.assertRaisesRegex(parity.InputError, "duplicate JSON key"):
            parity._read_trace(trace)


if __name__ == "__main__":
    unittest.main()
