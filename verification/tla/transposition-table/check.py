from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlc_runner import SuiteSpec, run_suite_cli

MODEL_DIR = Path(__file__).resolve().parent

MUTANTS = {
    "skip_seq2_check": "NoTornRead",
    "accept_odd_seq": "NoTornRead",
    "publish_before_payload": "NoTornRead",
    "ignore_bound_none": "BoundNoneNeverHits",
    "clear_without_cas": "SingleWriterPerSlot",
    "exceed_retry_budget": "BoundedRetries",
    "livelock_reject_loop": "Temporal",
}

DEEP_OVERRIDES = {
    "MaxReaderOps": "2",
}

SUITE = SuiteSpec(
    name="transposition-table",
    model_dir=MODEL_DIR,
    tla_module="TranspositionTable",
    mutants=MUTANTS,
    deep_overrides=DEEP_OVERRIDES,
)


def main() -> int:
    return run_suite_cli(SUITE)


if __name__ == "__main__":
    sys.exit(main())
