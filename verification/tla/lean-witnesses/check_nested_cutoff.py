"""Check nested cutoff assignments, then replay the TLC state sequences in Lean."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import build_lean, compile_model, prepare, replay_lean, tlc


def main():
    quint, work, jar = prepare(__doc__, "nested-cutoff")
    module = "NestedCutoff"
    compile_model(quint, work, HERE / f"{module}.qnt")
    build_lean("NeuralReversi.PvsParallel.SlotFrameMir.Program",
               "NeuralReversi.Threading.SplitReturn.Observation")
    tlc(work, jar, module, "step", "safe")
    for mode, prop in [
        ("step", "noReuse"), ("step", "noLocalOnly"), ("step", "noRestoredMark"),
        ("staleAssignment", "noStaleAlias"), ("staleAssignment", "noStaleStop"),
    ]:
        path, states = tlc(work, jar, module, mode, prop, witness=True)
        print(prop, flush=True)
        replay_lean("ReplayNestedCutoff.lean", mode, path)
        if mode == "staleAssignment":
            replay_lean("ReplayNestedCutoff.lean", "step", path,
                        expected_error="assignment, flag or cache register differs")
            end = states[-1]
            # The consequence differs when the new parent actually remained clear.
            if (end["nestedStop"] == end["expectedNested"]) != (prop == "noStaleAlias"):
                raise RuntimeError("mutation did not distinguish identical and different cutoff returns")
    print("PASS")


if __name__ == "__main__":
    main()
