"""Check TT relaxed-load readback and replay TLC witnesses against Lean."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import build_lean, compile_model, prepare, replay_lean, tlc


def replay(work, label, mode, state, expected_error=None):
    path = work / f"{label}-read.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    output = replay_lean("ReplayTTReadback.lean", mode, path,
                         expected_error=expected_error, quiet=True)
    if expected_error is None:
        print(f"{label}: read from {state['source']}; {output}", flush=True)


def main():
    quint, work, jar = prepare(__doc__, "tt-readback")
    compile_model(quint, work, HERE / "TTReadback.qnt")
    build_lean("NeuralReversi.PvsParallel.Interleaving.AtomicReadback")
    tlc(work, jar, "TTReadback", "step", "exact")
    for mode, prop in [
        ("step", "noStale"), ("step", "noAmbiguity"), ("step", "noBracket"),
        ("latestOnly", "keepsStale"), ("omitUpper", "respectsBracket"),
    ]:
        _, states = tlc(work, jar, "TTReadback", mode, prop, witness=True)
        observed = states[-1]
        replay(work, prop, mode, observed)
        if prop == "noAmbiguity":
            other = next(i for i in observed["candidates"] if i > 0 and i != observed["source"])
            replay(work, "other-equal-source", mode, {**observed, "source": other})
        if prop == "noBracket":
            recovered = {**observed, "source": observed["cut"] + 1}
            replay(work, "forbidden-equal-source", mode, recovered, "Lean excluded the semantic source")
    print("PASS")


if __name__ == "__main__":
    main()
