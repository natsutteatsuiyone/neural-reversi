"""Model-check SplitLifecycle.qnt with TLC and replay witnesses through Lean SplitTapeReplay."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import run_witness_suite


def main():
    run_witness_suite(
        __doc__,
        "split-lifecycle",
        "SplitLifecycle",
        "inv",
        ("NeuralReversi.PvsParallel.Interleaving.SplitTapeReplay",),
        "ReplaySplitLifecycle.lean",
        [
            ("step", "noNormalDone"),
            ("backwardAlpha", "noBackwardAlpha"),
            ("reusedEpoch", "noReusedEpochHit"),
            ("missedWrappedAbort", "noMissedWrappedAbort"),
            ("reuseScoreRead", "noReuseScoreRead"),
        ],
    )


if __name__ == "__main__":
    main()
