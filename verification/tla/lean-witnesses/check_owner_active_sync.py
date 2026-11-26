"""Check ownerActive (split lock) vs foreign book_into (worker lock) synchronization."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import run_witness_suite


def main():
    run_witness_suite(
        __doc__,
        "owner-active-sync",
        "OwnerActiveSync",
        "safe",
        ("NeuralReversi.PvsParallel.Interleaving.ActivePointer",),
        "ReplayOwnerActiveSync.lean",
        [
            ("step", "noNormalWithIntervener"),
            ("raceBook", "noDataRace"),
            ("wrongLock", "noWrongLock"),
            ("earlyRelease", "noEarlyDispatch"),
        ],
    )


if __name__ == "__main__":
    main()
