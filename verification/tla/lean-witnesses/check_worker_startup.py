"""Check initial and late booking publication across a reused split-point bank."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import run_witness_suite


def main():
    run_witness_suite(
        __doc__,
        "worker-startup",
        "WorkerStartup",
        "safe",
        (
            "NeuralReversi.PvsParallel.Interleaving.ActivePointer",
            "NeuralReversi.PvsParallel.Interleaving.SlotSynchronization",
            "NeuralReversi.PvsParallel.Interleaving.TaskAncestry",
            "NeuralReversi.PvsParallel.SplitSchedulerMir.Startup",
        ),
        "ReplayWorkerStartup.lean",
        [
            ("step", "noInitialChain"),
            ("step", "noLateChain"),
            ("oldClone", "noOldAlias"),
            ("wrongWorker", "noMissingLock"),
            ("wrongWorker", "noSelfHandoff"),
        ],
        reject_errors={
            "oldClone": "generation or address mismatch",
            "wrongWorker": "unexpected target or guard sequence",
        },
    )


if __name__ == "__main__":
    main()
