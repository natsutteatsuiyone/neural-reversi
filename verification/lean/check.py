"""Run the Lean audit/policy gates, Lean build/regressions, and Rust differential checks.

Usage (from anywhere inside the repo):
    python verification/lean/check.py [--lean-only] [--move-bound-certificate FILE]
        [--wasm-pattern-features] [--pvs-trace FILE]

Steps:
1. audit_modules.py --write (generates the audit inventory) and
   check_audit_modules_regression.py
2. check_policy.py and check_policy_regression.py
3. lake build NeuralReversi revmodel, including Audit.lean, CSimpAuditRegression.lean, and
   AuditOwnershipRegression.lean
4. check_policy_lean_regression.py, in parallel with the following optional checks
5. (Unless --lean-only) difftest/ prints cases from the Rust flip, move generation,
   activation, timing, fixed endgame, game analysis, OBF, empty list, board hash, move storage and generation, ProbCut, PV, aspiration,
   Lazy SMP, LMR, stop token, in-flight claims, time settings, score range, evaluation cache,
   TT saves and snapshots, aligned-buffer, Windows large-page calls, weight streams, pattern-feature, and search counter routines,
   plus generated pattern-table bytes.
   revmodel compares them with the Lean models.
Exit status 0 only when all steps pass.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
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
    parser.add_argument("--wasm-pattern-features", action="store_true",
                        help="Also run the pattern fixture with wasm32 SIMD in Node.js")
    parser.add_argument("--pvs-trace", type=Path, help="Replay an instrumented exact-solve transcript.")
    parser.add_argument("--pvs-threshold-depth", type=int, metavar="EMPTIES",
                        help="Maximum empties for direct threshold checks with --pvs-trace (default: 8).")
    args = parser.parse_args()
    if args.wasm_pattern_features and args.lean_only:
        parser.error("--wasm-pattern-features requires differential checks")
    if args.pvs_threshold_depth is not None and (args.pvs_trace is None or not 0 <= args.pvs_threshold_depth <= 64):
        parser.error("--pvs-threshold-depth requires --pvs-trace and an integer from 0 to 64")

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

        if args.pvs_trace is not None:
            WORK.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="pvs-trace-", dir=WORK) as directory:
                certificate = Path(directory) / "trace.jsonl"
                normalize = [sys.executable, str(SCRIPTS_DIR / "normalize_pvs_trace.py"),
                             str(args.pvs_trace.resolve()), str(certificate)]
                replay = [str(REVMODEL), "pvs-trace-check", str(certificate)]
                if args.pvs_threshold_depth is not None:
                    replay.append(str(args.pvs_threshold_depth))
                if run(normalize, REPO).returncode or run(replay, LEAN_DIR).returncode:
                    print("FAIL: PVS trace")
                    return 1

        if not args.lean_only and run_differential_checks() != 0:
            return 1
        if args.wasm_pattern_features and run_wasm_pattern_features() != 0:
            return 1
        if policy.wait() != 0:
            print("FAIL: check_policy_lean_regression.py")
            return 1

    print("PASS")
    return 0


def run_differential_checks() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not check_fixture_dependencies():
        return 1
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
    def _cases(kind: str, *names: str) -> list[tuple[str, tuple[list[str], ...]]]:
        if kind == "board":
            return [(n, (["--bin", "board_cases", "--", n],)) for n in names]
        if kind == "bin":
            return [(n, (["--bin", n.replace("-", "_")],)) for n in names]
        return [(n, (["--", n],)) for n in names]

    differential_cases: list[tuple[str, tuple[list[str], ...]]] = [
        ("flip-scalar-cases", ([],)),
        *_cases("default", "move-cases", "activation-cases", "endgame-timing-cases",
                "time-settings-cases", "fixed-endgame-cases"),
        *_cases("board", "game-analysis-cases", "obf-cases", "empty-list-cases", "board-hash-cases",
                "move-storage-cases", "move-generation-cases", "tt-save-cases", "stability-cases"),
        *_cases("default", "endgame-probcut-cases", "midgame-probcut-cases", "pv-stack-cases",
                "midgame-aspiration-cases", "endgame-aspiration-cases", "lazy-smp-cases",
                "lmr-cases", "split-report-cases"),
        *_cases("bin", "stop-token-cases", "in-flight-cases", "sync-primitive-cases"),
        *_cases("default", "finite-score-cases", "eval-cache-cases"),
        *_cases("bin", "aligned-buffer-cases", "large-page-cases"),
        ("weight-stream-cases", (["--bin", "weight_stream_cases", "--features", "weight-stream"],)),
        *_cases("bin", "pattern-feature-cases"),
        ("search-counter-cases", (
            ["--bin", "search_counter_cases"],
            ["--bin", "search_counter_cases", "--features", "search-stats"],
        )),
    ]
    for name, variants in differential_cases:
        cases = WORK / f"{name}.txt"
        with cases.open("w", encoding="utf-8", newline="\n") as f:
            for variant in variants:
                if run(cargo_run + variant, REPO, stdout=f).returncode:
                    print(f"FAIL: generating {name} from the Rust code")
                    return 1

        diff = run([str(REVMODEL), name, str(cases)], LEAN_DIR)
        if diff.returncode != 0:
            print(f"FAIL: revmodel {name} (see the error above)")
            return 1
    return check_generated_pattern_table()


def check_fixture_dependencies() -> bool:
    workspace_lock = (REPO / "Cargo.lock").read_text(encoding="utf-8")
    fixture_lock = (LEAN_DIR / "difftest" / "Cargo.lock").read_text(encoding="utf-8")
    for package in ("rapidhash", "byteorder", "zstd", "zstd-safe", "zstd-sys"):
        pattern = rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"$'
        core = re.findall(pattern, workspace_lock, re.MULTILINE)
        fixture = re.findall(pattern, fixture_lock, re.MULTILINE)
        if len(core) != 1 or core != fixture:
            print(f"FAIL: {package} version differs between reversi-core {core} and its fixture {fixture}")
            return False
    return True


def check_generated_pattern_table() -> int:
    core = REPO / "crates" / "reversi-core"
    with tempfile.TemporaryDirectory(dir=WORK, prefix="generated-table-") as scratch:
        out = Path(scratch)
        builder = out / ("build-core.exe" if sys.platform == "win32" else "build-core")
        blob = out / "eval_feature_u16_sum.bin"
        if run(["rustc", "--edition=2024", "-O", "-o", str(builder), "build.rs"], core).returncode:
            return 1

        little = b""
        for arch, features, order in (
            ("x86_64", "sse2,avx2", "little"),
            ("x86_64", "avx512bw", "little"),
            ("aarch64", "neon", "little"),
            ("aarch64", "neon", "big"),
        ):
            env = dict(os.environ, OUT_DIR=str(out), CARGO_CFG_TARGET_ARCH=arch,
                       CARGO_CFG_TARGET_FEATURE=features, CARGO_CFG_TARGET_ENDIAN=order)
            if run([str(builder)], core, env=env, stdout=subprocess.DEVNULL).returncode:
                return 1
            table = blob.read_bytes()
            if len(table) != 4 * 65536 * 32 * 2:
                print(f"FAIL: invalid pattern table length for {arch}/{features}/{order}")
                return 1
            # Chunk zero, mask 1, diagonal A1-H8. The A1 bit is its leading ternary digit.
            if int.from_bytes(table[72:74], order) != 3 ** 7:
                print(f"FAIL: wrong A1 diagonal contribution for {arch}/{features}/{order}")
                return 1
            if not little:
                little = table
            same = table == little if order == "little" else (
                table[::2] == little[1::2] and table[1::2] == little[::2]
            )
            if not same:
                print(f"FAIL: pattern table bytes differ for {arch}/{features}/{order}")
                return 1

        for arch, features in (("x86_64", "sse2,avx20"), ("aarch64", ""), ("wasm32", "simd128")):
            blob.write_bytes(b"stale")
            env = dict(os.environ, OUT_DIR=str(out), CARGO_CFG_TARGET_ARCH=arch,
                       CARGO_CFG_TARGET_FEATURE=features, CARGO_CFG_TARGET_ENDIAN="little")
            if run([str(builder)], core, env=env, stdout=subprocess.DEVNULL).returncode:
                return 1
            if blob.exists():
                print(f"FAIL: stale pattern table for {arch}/{features}")
                return 1
    print("Generated pattern table: 8,388,608 lanes agree in both target byte orders")
    return 0


def run_wasm_pattern_features() -> int:
    target = "wasm32-unknown-unknown"
    target_dir = WORK / "target-wasm"
    manifest = LEAN_DIR / "difftest" / "Cargo.toml"
    cmd = ["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest),
           "--target", target, "--target-dir", str(target_dir), "--bin", "pattern_feature_cases"]
    env = dict(os.environ, CARGO_ENCODED_RUSTFLAGS="-C\x1ftarget-feature=+simd128")
    if run(cmd, REPO, env=env).returncode:
        return 1
    wasm = target_dir / target / "release" / "pattern_feature_cases.wasm"
    cases = WORK / "wasm-pattern-feature-cases.txt"
    with cases.open("wb") as stream:
        if run(["node", str(SCRIPTS_DIR / "read_wasm_fixture.cjs"), str(wasm)], REPO,
               stdout=stream).returncode:
            return 1
    return run([str(REVMODEL), "pattern-feature-cases", str(cases)], LEAN_DIR).returncode


if __name__ == "__main__":
    sys.exit(main())
