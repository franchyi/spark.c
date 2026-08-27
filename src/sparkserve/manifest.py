from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ManifestSummary:
    repo: str
    revision: str | None
    file_count: int
    total_bytes: int
    expert_bytes: int
    ple_bytes: int
    other_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_file(name: str) -> str:
    lowered = name.lower()
    if "ple" in lowered or "ngram" in lowered:
        return "ple"
    if "expert" in lowered:
        return "expert"
    return "other"


def summarize_model_payload(payload: dict[str, object], repo: str) -> ManifestSummary:
    totals = {"expert": 0, "ple": 0, "other": 0}
    siblings = payload.get("siblings", [])
    if not isinstance(siblings, list):
        raise ValueError("model API response has no siblings list")

    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("rfilename", ""))
        size = item.get("size", 0)
        if not size and isinstance(item.get("lfs"), dict):
            size = item["lfs"].get("size", 0)
        totals[classify_file(name)] += int(size or 0)

    return ManifestSummary(
        repo=repo,
        revision=str(payload.get("sha")) if payload.get("sha") else None,
        file_count=len(siblings),
        total_bytes=sum(totals.values()),
        expert_bytes=totals["expert"],
        ple_bytes=totals["ple"],
        other_bytes=totals["other"],
    )


def fetch_manifest(
    repo: str, *, endpoint: str = "https://hf-mirror.com", timeout: float = 30.0
) -> ManifestSummary:
    safe_repo = quote(repo, safe="/")
    url = f"{endpoint.rstrip('/')}/api/models/{safe_repo}?blobs=true"
    request = Request(url, headers={"User-Agent": "sparkserve/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return summarize_model_payload(payload, repo)
