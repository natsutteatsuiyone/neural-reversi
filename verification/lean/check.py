"""Run the Lean audit/policy gates, Lean build/regressions, and Rust differential checks.

Usage (from anywhere inside the repo):
    python verification/lean/check.py [--lean-only] [--move-bound-certificate FILE]

Steps:
1. audit_modules.py --write (generates the audit inventory) and
   check_audit_modules_regression.py
2. check_policy.py and check_policy_regression.py
3. lake build NeuralReversi revmodel, including Audit.lean, CSimpAuditRegression.lean, and
   AuditOwnershipRegression.lean
4. check_policy_lean_regression.py, in parallel with the following optional checks
5. (Unless --lean-only) difftest/ prints cases from the Rust flip, move generation,
   activation, timing, ProbCut, PV, aspiration, Lazy SMP, score range, and
   evaluation cache, aligned-buffer, and pattern-feature routines.
   revmodel compares them with the Lean models.
Exit status 0 only when all steps pass.
"""

import argparse
import subprocess
import sys
from pathlib import Path

LEAN_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = LEAN_DIR / "scripts"
REPO = LEAN_DIR.parents[1]
WORK = REPO / "target" / "verify-lean"
REVMODEL = LEAN_DIR / ".lake" / "build" / "bin" / (
    "revmodel.exe" if sys.platform == "win32" else "revmodel"
)


def run(cmd: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lean-only",
        action="store_true",
        help="Skip the Rust differential checks.",
    )
    parser.add_argument(
        "--move-bound-certificate",
        type=Path,
        help="Check a move-bound LRAT proof for the query of 36 or more legal moves.",
    )
    args = parser.parse_args()

    checks: list[tuple[str, list[str], Path]] = [
        (
            "audit_modules.py",
            [sys.executable, str(SCRIPTS_DIR / "audit_modules.py"), "--write"],
            REPO,
        ),
        (
            "check_audit_modules_regression.py",
            [sys.executable, str(SCRIPTS_DIR / "check_audit_modules_regression.py")],
            REPO,
        ),
        ("check_policy.py", [sys.executable, str(SCRIPTS_DIR / "check_policy.py")], REPO),
        (
            "check_policy_regression.py",
            [sys.executable, str(SCRIPTS_DIR / "check_policy_regression.py")],
            REPO,
        ),
        ("lake build", ["lake", "build", "NeuralReversi", "revmodel"], LEAN_DIR),
    ]

    for label, cmd, cwd in checks:
        if run(cmd, cwd).returncode != 0:
            print(f"FAIL: {label}")
            return 1

    policy_cmd = [sys.executable, str(SCRIPTS_DIR / "check_policy_lean_regression.py")]
    print("$", " ".join(policy_cmd), flush=True)
    with subprocess.Popen(policy_cmd, cwd=REPO) as policy:
        if args.move_bound_certificate is not None:
            proof = args.move_bound_certificate.resolve()
            if run([str(REVMODEL), "move-bound-check", "35", str(proof)], LEAN_DIR).returncode:
                print("FAIL: move-bound certificate")
                return 1

        if not args.lean_only and run_differential_checks() != 0:
            return 1
        if policy.wait() != 0:
            print("FAIL: check_policy_lean_regression.py")
            return 1

    print("PASS")
    return 0


def run_differential_checks() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    cargo_run = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--locked",
        "--manifest-path",
        str(LEAN_DIR / "difftest" / "Cargo.toml"),
        "--target-dir",
        str(WORK / "target"),
    ]
    for name in (
        "flip-scalar-cases",
        "move-cases",
        "activation-cases",
        "endgame-timing-cases",
        "endgame-probcut-cases",
        "midgame-probcut-cases",
        "pv-stack-cases",
        "midgame-aspiration-cases",
        "lazy-smp-cases",
        "finite-score-cases",
        "eval-cache-cases",
        "aligned-buffer-cases",
        "pattern-feature-cases",
    ):
        if name in ("aligned-buffer-cases", "pattern-feature-cases"):
            args = ["--bin", name.replace("-", "_")]
        else:
            args = [] if name == "flip-scalar-cases" else ["--", name]
        cases = WORK / f"{name}.txt"
        with cases.open("w", encoding="utf-8", newline="\n") as f:
            gen = run(cargo_run + args, REPO, stdout=f)
        if gen.returncode != 0:
            print(f"FAIL: generating {name} from the Rust code")
            return 1

        diff = run([str(REVMODEL), name, str(cases)], LEAN_DIR)
        if diff.returncode != 0:
            print(f"FAIL: revmodel {name} (see the error above)")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
