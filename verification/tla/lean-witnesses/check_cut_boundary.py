"""Model-check CutBoundary.qnt with TLC and replay witnesses through Lean CutClosure / JournalResident."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import run_witness_suite


def main():
    run_witness_suite(
        __doc__,
        "cut-boundary",
        "CutBoundary",
        "inv",
        (
            "NeuralReversi.PvsParallel.Interleaving.CutClosure",
            "NeuralReversi.PvsParallel.Interleaving.MultiSessionCarry",
        ),
        "ReplayCutBoundary.lean",
        [
            ("step", "noNormalDone"),
            ("hbOnlyCut", "noHbOnlyDrop"),
            ("undrainedHandoff", "noUndrainedHandoff"),
        ],
    )


if __name__ == "__main__":
    main()
