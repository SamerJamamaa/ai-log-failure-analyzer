"""Generates the bundled synthetic sample dataset (data/sample_workflow_logs.csv).

Record composition per workflow is constructed deterministically (not purely
sampled) so every detection rule in src/anomalies.py has something concrete
to catch: recurring errors of each required category, a handful of
abnormally long executions, and one workflow ("WhatsApp Lead Intake") whose
failure rate deliberately jumps in its second chronological half. No real
customer data is used anywhere — all names, IDs and messages are synthetic.
"""

import os
import random
from datetime import datetime, timedelta

RNG_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_workflow_logs.csv")

ERROR_MESSAGES = {
    "APITimeoutError": "Request to external API timed out before completion.",
    "AuthenticationError": "Authentication with the upstream service failed (invalid or expired credentials).",
    "InvalidJSONSchemaError": "Response payload did not match the expected JSON schema.",
    "DuplicateRecordError": "Record already exists; skipped to avoid duplicate processing.",
}

# Each workflow spec fully controls its own record composition:
#   size:                total execution count for this workflow
#   duration:            (mean, std) seconds for a normal (non-outlier) execution
#   failures:            ordered list of error_type strings to inject as failures
#   slow_count:          number of deliberately abnormal-duration (but successful) executions
#   slow_multiplier:     how many std above the mean the slow executions land at
#   degrade_second_half: if True, all but one failure are concentrated in the
#                         second half of the timeline instead of spread evenly
#                         (drives the failure-rate-increase detection rule)
WORKFLOW_SPECS = [
    {
        "name": "WhatsApp Lead Intake",
        "size": 55,
        "duration": (4.0, 1.0),
        "failures": ["APITimeoutError"] * 12 + ["AuthenticationError"] * 2,
        "slow_count": 0,
        "slow_multiplier": 0,
        "degrade_second_half": True,
    },
    {
        "name": "Invoice Parser",
        "size": 48,
        "duration": (8.0, 2.0),
        "failures": ["InvalidJSONSchemaError"] * 3 + ["DuplicateRecordError"] * 1,
        "slow_count": 4,
        "slow_multiplier": 4.5,
        "degrade_second_half": False,
    },
    {
        "name": "CRM Synchronization",
        "size": 45,
        "duration": (3.0, 0.8),
        "failures": ["AuthenticationError"] * 4 + ["DuplicateRecordError"] * 1,
        "slow_count": 0,
        "slow_multiplier": 0,
        "degrade_second_half": False,
    },
    {
        "name": "Customer Support Classifier",
        "size": 52,
        "duration": (5.0, 1.5),
        "failures": ["DuplicateRecordError"] * 4 + ["APITimeoutError"] * 1,
        "slow_count": 3,
        "slow_multiplier": 4.0,
        "degrade_second_half": False,
    },
]

TOTAL_RECORD_COUNT = sum(spec["size"] for spec in WORKFLOW_SPECS)
assert TOTAL_RECORD_COUNT == 200, f"WORKFLOW_SPECS sizes must sum to exactly 200, got {TOTAL_RECORD_COUNT}"

TOTAL_SPAN_DAYS = 20


def _clipped_normal(rng: random.Random, mean: float, std: float, max_std: float = 1.5) -> float:
    """A "normal-looking" duration, clipped so baseline points never
    accidentally cross into slow-execution-outlier territory."""
    value = rng.gauss(mean, std)
    value = max(mean - max_std * std, min(mean + max_std * std, value))
    return round(max(0.5, value), 2)


def _interleave_evenly(base: list[dict], inserts: list[dict]) -> list[dict]:
    """Distributes `inserts` evenly by position throughout `base`, so a small
    number of failures/outliers doesn't cluster at one end of the timeline."""
    if not inserts:
        return list(base)
    result = list(base)
    step = len(result) / (len(inserts) + 1)
    for offset, item in enumerate(inserts, start=1):
        position = min(len(result), round(step * offset))
        result.insert(position, item)
    return result


def _make_success(rng: random.Random, mean: float, std: float) -> dict:
    return {
        "status": "success",
        "error_type": "",
        "error_message": "",
        "duration_seconds": _clipped_normal(rng, mean, std),
    }


def _make_slow_success(rng: random.Random, mean: float, std: float, multiplier: float) -> dict:
    duration = round(mean + multiplier * std + rng.uniform(0, std), 2)
    return {"status": "success", "error_type": "", "error_message": "", "duration_seconds": duration}


def _make_failure(rng: random.Random, mean: float, std: float, error_type: str) -> dict:
    return {
        "status": "failure",
        "error_type": error_type,
        "error_message": ERROR_MESSAGES[error_type],
        "duration_seconds": _clipped_normal(rng, mean, std),
    }


def _build_records(spec: dict, rng: random.Random) -> list[dict]:
    mean, std = spec["duration"]
    n_failures = len(spec["failures"])
    n_success = spec["size"] - n_failures - spec["slow_count"]

    successes = [_make_success(rng, mean, std) for _ in range(n_success)]
    slow_executions = [_make_slow_success(rng, mean, std, spec["slow_multiplier"]) for _ in range(spec["slow_count"])]
    failures = [_make_failure(rng, mean, std, error_type) for error_type in spec["failures"]]

    if spec["degrade_second_half"]:
        half = spec["size"] // 2
        first_half_failures, second_half_failures = failures[:1], failures[1:]
        first_half = _interleave_evenly(successes[: half - len(first_half_failures)], first_half_failures)
        remaining_successes = successes[half - len(first_half_failures) :] + slow_executions
        second_half = _interleave_evenly(remaining_successes, second_half_failures)
        return first_half + second_half

    return _interleave_evenly(successes, failures + slow_executions)


def _assign_timestamps(records: list[dict], workflow_name: str, start_id: int) -> list[dict]:
    start_time = datetime.now() - timedelta(days=TOTAL_SPAN_DAYS)
    step = (TOTAL_SPAN_DAYS * 24 * 60) / len(records)
    rng = random.Random(RNG_SEED + start_id)  # deterministic jitter, independent of composition RNG

    rows = []
    for i, record in enumerate(records):
        jitter_minutes = rng.uniform(0, step * 0.8)
        timestamp = start_time + timedelta(minutes=i * step + jitter_minutes)
        rows.append(
            {
                "execution_id": f"exec_{start_id + i:06d}",
                "workflow_name": workflow_name,
                "start_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                **record,
            }
        )
    return rows


def generate() -> list[dict]:
    rng = random.Random(RNG_SEED)
    all_rows: list[dict] = []
    next_id = 1
    for spec in WORKFLOW_SPECS:
        records = _build_records(spec, rng)
        all_rows.extend(_assign_timestamps(records, spec["name"], next_id))
        next_id += spec["size"]
    all_rows.sort(key=lambda r: r["start_time"])
    return all_rows


def main() -> None:
    import pandas as pd

    df = pd.DataFrame(generate())
    output_path = os.path.abspath(OUTPUT_PATH)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
