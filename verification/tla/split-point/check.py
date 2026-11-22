import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

TLA_VERSION = "v1.7.4"
JAR_URL = f"https://github.com/tlaplus/tlaplus/releases/download/{TLA_VERSION}/tla2tools.jar"
JAR_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
JAR = Path.home() / ".cache" / "tlaplus" / TLA_VERSION / "tla2tools.jar"

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[2]
WORK = REPO / "target" / "tla" / "split-point"

MUTANTS = {
    "late_join_no_recheck": "BookedOnlyIntoLiveSplitPoints",
    "leave_keeps_all_helpers_searching": "BookedOnlyIntoLiveSplitPoints",
    "reset_bit_before_merge_counters": "TaskAliveWhileUsed",
    "can_join_ignores_searching": "BookingMatchesThread",
    "leave_keeps_active": "ActiveChainAlive",
    "cutoff_check_only_own_split_point": "CutoffCacheSound",
    "leave_no_cache_reset": "CutoffCacheNoFalseStop",
    "finalize_skips_clear": "CleanTeardown",
    "hold_lock_across_idle_loop": "Deadlock",
}

DEEP_OVERRIDES = {"N": "2", "Budget": "3"}


def ensure_jar() -> None:
    if not JAR.exists():
        JAR.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {JAR_URL}", file=sys.stderr)
        urllib.request.urlretrieve(JAR_URL, JAR)
    digest = hashlib.sha256(JAR.read_bytes()).hexdigest()
    if digest != JAR_SHA256:
        sys.exit(f"{JAR}: sha256 {digest} != {JAR_SHA256}; delete it and rerun")


def run(mutant: str, workers: str, overrides: dict[str, str] | None = None,
        run_name: str | None = None) -> tuple[int, str]:
    out = WORK / (run_name or mutant)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_DIR / "SplitPoint.tla", out)
    cfg = (MODEL_DIR / "SplitPoint.cfg").read_text(encoding="utf-8")
    cfg, n = re.subn(r'Mutant = "[^"]*"', f'Mutant = "{mutant}"', cfg)
    assert n == 1, "SplitPoint.cfg must set Mutant exactly once"
    for key, val in (overrides or {}).items():
        cfg, n = re.subn(rf'\b{key} = \S+', f'{key} = {val}', cfg)
        assert n == 1, f"SplitPoint.cfg must set {key} exactly once"
    (out / "SplitPoint.cfg").write_text(cfg, encoding="utf-8")

    trans = subprocess.run(
        ["java", "-cp", str(JAR), "pcal.trans", "-nocfg", "SplitPoint.tla"],
        cwd=out, capture_output=True, text=True,
    )
    if trans.returncode != 0 or "Translation completed" not in trans.stdout:
        return 1, trans.stdout + trans.stderr

    tlc = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-jar", str(JAR), "-workers", workers,
         "-config", "SplitPoint.cfg", "-metadir", "states", "SplitPoint.tla"],
        cwd=out, capture_output=True, text=True,
    )
    log = tlc.stdout + tlc.stderr
    (out / "tlc.out").write_text(log, encoding="utf-8")
    return tlc.returncode, log


def summary(log: str) -> str:
    keep = [
        line for line in log.splitlines()
        if re.match(r"(Error:|Invariant |Deadlock|Model checking completed|\d+ states generated|"
                    r"The depth of the complete state graph)", line)
    ]
    return "\n".join(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mutants", action="store_true", help="run every mutant; each must fail")
    g.add_argument("--mutant", choices=sorted(MUTANTS), help="run one mutant")
    g.add_argument("--deep", action="store_true",
                   help="run clean model with Budget=3, N=2 (nesting + slot reuse)")
    g.add_argument("--all", action="store_true",
                   help="run clean model, deep (Budget=3, N=2), and every mutant")
    ap.add_argument("--workers", default="auto", help="TLC worker threads (default: auto)")
    args = ap.parse_args()

    ensure_jar()
    if args.all:
        runs: list[tuple[str, str, dict[str, str] | None]] = [
            ("none", "none", None),
            ("deep", "none", DEEP_OVERRIDES),
            *((m, m, None) for m in sorted(MUTANTS)),
        ]
    elif args.deep:
        runs = [("deep", "none", DEEP_OVERRIDES)]
    elif args.mutants:
        runs = [(m, m, None) for m in sorted(MUTANTS)]
    elif args.mutant:
        runs = [(args.mutant, args.mutant, None)]
    else:
        runs = [("none", "none", None)]
    ok = True
    for label, mutant, overrides in runs:
        code, log = run(mutant, args.workers, overrides=overrides, run_name=label)
        print(f"== {label}  (full log: {WORK / label / 'tlc.out'})")
        print(summary(log) or log[-3000:])
        if mutant == "none":
            passed = code == 0 and "No error has been found" in log
            print("PASS" if passed else "FAIL: the model breaks a property (see the trace in the log)")
        else:
            expected = MUTANTS[mutant]
            if expected == "Deadlock":
                passed = code != 0 and "Error: Deadlock reached." in log
            else:
                passed = code != 0 and f"Invariant {expected} is violated" in log
            print(f"PASS: mutant caught by {expected}" if passed
                  else f"FAIL: mutant not caught by {expected}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
