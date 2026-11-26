"""Check parent-mask zero RF identities and replay TLC witnesses against Lean."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import build_lean, compile_model, prepare, replay_lean, tlc


def main():
    quint, work, jar = prepare(__doc__, "parent-mask-zero")
    name = "ParentMaskZero"
    compile_model(quint, work, HERE / f"{name}.qnt")
    for mode, invariant in [("step", "unique"), ("step", "latestPhysical"), ("raw", "safe"),
                            ("ownerRaw", "ownerOrdered")]:
        print(mode, invariant, flush=True)
        tlc(work, jar, name, mode, invariant)
    build_lean("NeuralReversi.PvsParallel.Interleaving.Mask",
               "NeuralReversi.PvsParallel.Interleaving.SlotSynchronization")
    for mode, invariant in [("step", "noCurrentFinish"), ("raw", "unique"), ("raw", "latestPhysical"),
                            ("ownerRaw", "noOwnerOrdered"), ("valueOnly", "noPriorFinish"),
                            ("valueOnly", "noNextClear")]:
        path, _ = tlc(work, jar, name, mode, invariant, witness=True)
        print(mode, invariant, flush=True)
        replay_lean("ReplayParentMaskZero.lean", mode, path)
    print("PASS")


if __name__ == "__main__":
    main()
