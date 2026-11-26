"""Check activation identity when overlapping workers repeat identical requests."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import build_lean, compile_model, prepare, replay_lean, tlc


def main():
    quint, work, jar = prepare(__doc__, "call-returns")
    compile_model(quint, work, HERE / "CallReturns.qnt")
    build_lean("NeuralReversi.PvsParallel.Interleaving.CallReturns")
    tlc(work, jar, "CallReturns", "step", "safe")
    for policy, invariant, mode in [
        ("step", "noOverlap", "step"),
        ("firstInParent", "correctScore", "firstInParent"),
        ("firstAcrossParents", "correctScore", "firstAcrossParents"),
        ("firstAcrossParents", "identityOnEqualScores", "equalAcrossParents"),
    ]:
        path, _ = tlc(work, jar, "CallReturns", policy, invariant, witness=True)
        replay_lean("ReplayCallReturns.lean", mode, path)
        if mode == "equalAcrossParents":
            replay_lean("ReplayCallReturns.lean", "step", path,
                        expected_error="equal score from a different invocation")
            print("The correct reader rejected the wrong invocation with identical score.", flush=True)
    print("PASS")


if __name__ == "__main__":
    main()
