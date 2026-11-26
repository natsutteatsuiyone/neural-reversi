"""Compile the Quint score-slot model to TLA+, exhaust it with TLC, and replay its witnesses in Lean."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from quint_tlc import LEAN, REPO, build_lean, compile_model, prepare, replay_lean, tlc


def check_rust():
    # These extractors check the mask ordering, the relaxed field accesses,
    # the physical slot and the actual owner return tuple in the pinned Rust MIR.
    with tempfile.TemporaryDirectory(prefix="score-reuse-mir-") as temp:
        dump = Path(temp) / "mir"
        subprocess.run(["cargo", "rustc", "-p", "reversi-core", "--locked", "--offline",
                        "--lib", "--profile", "check", "--target-dir", temp, "--",
                        "-Zdump-mir=SimplifyCfg-initial", f"-Zdump-mir-dir={dump}",
                        "-Zdump-mir-exclude-alloc-bytes", "-Zdump-mir-exclude-pass-number",
                        "--emit=metadata"], cwd=REPO, check=True)
        for script, program in [("extract_split_activation_mir.py", "SplitActivationMir"),
                                ("extract_split_scheduler_mir.py", "SplitSchedulerMir"),
                                ("extract_split_worker_mir.py", "SplitWorkerMir")]:
            subprocess.run([sys.executable, str(LEAN / "scripts" / script), "--mir", str(dump),
                            "--check", str(LEAN / "NeuralReversi/PvsParallel" / program / "Program.lean")],
                           cwd=REPO, check=True)


def check(work, jar, step, invariant, witness):
    result = tlc(work, jar, "ScoreReuse", step, invariant, witness is not None)
    if result is None:
        return
    snapshots, rows = result
    replay_lean("ReplayScoreReuse.lean", witness, snapshots)
    if witness in ("lastEqual", "firstAcross"):
        replay_lean("ReplayScoreReuse.lean", "tie1", snapshots,
                    expected_error="reported helper did not publish this score")
        print("The normal reader rejected the equal-score publisher substitution.", flush=True)
    if witness == "step":
        controls = [
            ([s for s in rows if (s["generation"], s["phase"]) != (2, "zero")],
             "missing ordered zero wait, choice or read"),
            ([*rows[:-1], {**rows[-1], "source": rows[-1]["last"] + 1}],
             "read crossed the next initializer"),
        ]
        for index, (altered, reason) in enumerate(controls):
            corrupted = work / f"corrupted-{index}.json"
            corrupted.write_text(json.dumps(altered), encoding="utf-8")
            replay_lean("ReplayScoreReuse.lean", witness, corrupted, expected_error=reason)
        print("Replay rejected a missing zero wait and a substituted equal-valued write.", flush=True)


def main():
    quint, work, jar = prepare(__doc__, "score-reuse")
    check_rust()
    compile_model(quint, work, HERE / "ScoreReuse.qnt")
    build_lean("NeuralReversi.PvsParallel.Interleaving.ScoreReuse")
    check(work, jar, "step", "safe", None)
    check(work, jar, "step", "noEqualPayload", "step")
    check(work, jar, "step", "noSeedOnly", "seedOnly")
    check(work, jar, "relaxedWait", "safe", "relaxedWait")
    check(work, jar, "relaxedReset", "safe", "relaxedReset")
    check(work, jar, "unorderedReuse", "noEqualSubstitution", "unorderedReuse")
    check(work, jar, "step", "noFirstHelperWon", "tie1")
    check(work, jar, "step", "noSecondHelperWon", "tie2")
    check(work, jar, "lastEqual", "tieOwner", "lastEqual")
    check(work, jar, "firstAcross", "noPriorPublisher", "firstAcross")
    print("PASS")


if __name__ == "__main__":
    main()
