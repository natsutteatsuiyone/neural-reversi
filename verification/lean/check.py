"""Run the Lean audit/policy gates, Lean build/regressions, and Rust differential checks.

Usage (from anywhere inside the repo):
    python verification/lean/check.py [--lean-only] [--move-bound-certificate FILE]
        [--pvs-trace FILE]

Steps:
1. audit_modules.py --write (generates the audit inventory) and
   check_audit_modules_regression.py
2. check_policy.py and check_policy_regression.py
3. lake build NeuralReversi revmodel, including Audit.lean, CSimpAuditRegression.lean, and
   AuditOwnershipRegression.lean
4. check_policy_lean_regression.py, in parallel with the following optional checks
5. (Unless --lean-only) difftest/ prints cases from the Rust flip, move generation,
   activation, timing, fixed endgame, game analysis, OBF, empties, board hash, move storage and generation, ProbCut, PV, aspiration,
   Lazy SMP, LMR, stop token, in-flight claims, time settings, score range, evaluation cache,
   TT saves and snapshots, aligned-buffer, Windows large-page calls, weight streams, pattern-feature, and search counter routines,
   plus generated pattern-table bytes.
   revmodel compares them with the Lean models.
Exit status 0 only when all steps pass.
"""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
REVMODEL = LEAN_DIR / ".lake" / "build" / "bin" / f"revmodel{EXE_SUFFIX}"


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def run(cmd: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, **kw)


def run_captured(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def emit_captured(cmd: list[str], result: subprocess.CompletedProcess[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)


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
    parser.add_argument("--pvs-trace", type=Path, help="Replay an instrumented exact-solve transcript.")
    parser.add_argument("--pvs-threshold-depth", type=int, metavar="EMPTIES",
                        help="Maximum empties for direct threshold checks with --pvs-trace (default: 8).")
    args = parser.parse_args()
    if args.pvs_threshold_depth is not None and (args.pvs_trace is None or not 0 <= args.pvs_threshold_depth <= 64):
        parser.error("--pvs-threshold-depth requires --pvs-trace and an integer from 0 to 64")

    audit_cmd = [sys.executable, str(SCRIPTS_DIR / "audit_modules.py"), "--write"]
    if run(audit_cmd, REPO).returncode != 0:
        print("FAIL: audit_modules.py")
        return 1

    # Overlap independent Python checks, Rust difftest builds/case generation, and
    # generated pattern-table checks with `lake build NeuralReversi revmodel`.
    py_checks: list[tuple[str, list[str], Path]] = [
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
    ]
    lake_cmd = ["lake", "build", "NeuralReversi", "revmodel"]

    case_files: list[tuple[str, Path]] | None = None
    stage1_failed = False
    with ThreadPoolExecutor(max_workers=6) as pool:
        lake_future = pool.submit(run, lake_cmd, LEAN_DIR)
        py_futures = [
            (label, cmd, pool.submit(run_captured, cmd, cwd))
            for label, cmd, cwd in py_checks
        ]
        diff_gen_future = None if args.lean_only else pool.submit(prepare_differential_cases)
        table_future = None if args.lean_only else pool.submit(check_generated_pattern_table)

        for label, cmd, future in py_futures:
            res = future.result()
            emit_captured(cmd, res)
            if res.returncode != 0:
                print(f"FAIL: {label}", flush=True)
                stage1_failed = True

        if lake_future.result().returncode != 0:
            print("FAIL: lake build", flush=True)
            stage1_failed = True

        if diff_gen_future is not None:
            case_files = diff_gen_future.result()
            if case_files is None:
                stage1_failed = True

        if table_future is not None and table_future.result() != 0:
            stage1_failed = True

    if stage1_failed:
        return 1

    policy_cmd = [sys.executable, str(SCRIPTS_DIR / "check_policy_lean_regression.py")]
    print("$", " ".join(policy_cmd), flush=True)
    with subprocess.Popen(policy_cmd, cwd=REPO) as policy:
        stage2_failed = False
        if args.move_bound_certificate is not None:
            proof = args.move_bound_certificate.resolve()
            if run([str(REVMODEL), "move-bound-check", "35", str(proof)], LEAN_DIR).returncode:
                print("FAIL: move-bound certificate")
                stage2_failed = True

        if not stage2_failed and args.pvs_trace is not None:
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
                    stage2_failed = True

        if not stage2_failed and not args.lean_only:
            assert case_files is not None
            if run_revmodel_checks(case_files) != 0:
                stage2_failed = True
        if policy.wait() != 0:
            print("FAIL: check_policy_lean_regression.py")
            stage2_failed = True
        if stage2_failed:
            return 1

    print("PASS")
    return 0


def differential_case_specs() -> list[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]]:
    def _cases(kind: str, *names: str) -> list[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]]:
        if kind == "board":
            return [(n, (("board_cases", (n,)),)) for n in names]
        if kind == "bin":
            return [(n, ((n.replace("-", "_"), ()),)) for n in names]
        return [(n, (("flip-scalar-difftest", (n,)),)) for n in names]

    return [
        ("flip-scalar-cases", (("flip-scalar-difftest", ()),)),
        *_cases("default", "move-cases", "activation-cases", "endgame-timing-cases",
                "time-settings-cases", "fixed-endgame-cases"),
        *_cases("board", "game-analysis-cases", "obf-cases", "empties-cases", "board-hash-cases",
                "move-storage-cases", "move-generation-cases", "tt-save-cases", "stability-cases"),
        *_cases("default", "endgame-probcut-cases", "midgame-probcut-cases", "pv-stack-cases",
                "midgame-aspiration-cases", "endgame-aspiration-cases", "lazy-smp-cases",
                "lmr-cases", "split-report-cases"),
        *_cases("bin", "stop-token-cases", "in-flight-cases", "sync-primitive-cases"),
        *_cases("default", "finite-score-cases", "eval-cache-cases"),
        *_cases("bin", "aligned-buffer-cases", "large-page-cases"),
        ("weight-stream-cases", (("weight_stream_cases", ()),)),
        *_cases("bin", "pattern-feature-cases"),
        ("search-counter-cases", (
            ("search_counter_cases", ()),
            ("search_counter_stats_cases", ()),
        )),
    ]


def difftest_bin(name: str) -> Path:
    return WORK / "target" / "release" / f"{name}{EXE_SUFFIX}"


def build_difftest_bins() -> int:
    cargo_build = [
        "cargo",
        "build",
        "--quiet",
        "--release",
        "--locked",
        "--manifest-path",
        str(LEAN_DIR / "difftest" / "Cargo.toml"),
        "--target-dir",
        str(WORK / "target"),
    ]
    cmds = [
        cargo_build + ["--features", "weight-stream", "--bins"],
        cargo_build + ["--features", "search-stats", "--bin", "search_counter_stats_cases"],
    ]
    with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        futures = [(cmd, pool.submit(run_captured, cmd, REPO)) for cmd in cmds]
        failed = False
        for cmd, future in futures:
            res = future.result()
            emit_captured(cmd, res)
            if res.returncode != 0:
                print("FAIL: building difftest binaries", flush=True)
                failed = True
        return 1 if failed else 0


def _generate_one_case(
    name: str,
    variants: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, Path, int, str]:
    cases = WORK / f"{name}.txt"
    stderr_chunks: list[str] = []
    with cases.open("wb") as stream:
        for bin_name, bin_args in variants:
            res = subprocess.run(
                [str(difftest_bin(bin_name)), *bin_args],
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.PIPE,
            )
            if res.stderr:
                stderr_chunks.append(res.stderr.decode("utf-8", errors="replace"))
            if res.returncode != 0:
                return name, cases, res.returncode, "".join(stderr_chunks)
    return name, cases, 0, "".join(stderr_chunks)


def prepare_differential_cases() -> list[tuple[str, Path]] | None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not check_fixture_dependencies():
        return None
    if build_difftest_bins() != 0:
        return None
    specs = differential_case_specs()
    workers = max(1, min(len(specs), available_cpus()))
    case_files: list[tuple[str, Path]] = []
    failed = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_generate_one_case, name, variants) for name, variants in specs]
        for future in futures:
            name, path, rc, stderr = future.result()
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
            if rc != 0:
                print(f"FAIL: generating {name} from the Rust code", flush=True)
                failed = True
            else:
                case_files.append((name, path))
    return None if failed else case_files


def run_revmodel_checks(case_files: list[tuple[str, Path]]) -> int:
    """Run revmodel over all generated differential case files in one parallel batch."""
    cmd = [
        str(REVMODEL),
        "batch",
        *(arg for name, path in case_files for arg in (name, str(path))),
    ]
    return 0 if run(cmd, LEAN_DIR).returncode == 0 else 1


def run_differential_checks() -> int:
    case_files = prepare_differential_cases()
    if case_files is None:
        return 1
    if run_revmodel_checks(case_files) != 0:
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


def _cached_pattern_table_builder(core: Path) -> Path | None:
    rustc_version = subprocess.check_output(
        ["rustc", "-V"], cwd=core, text=True, encoding="utf-8",
    ).strip()
    digest = hashlib.sha256()
    digest.update(rustc_version.encode("utf-8"))
    for rel in (
        "build.rs",
        "src/eval/pattern_table_data.rs",
        "src/probcut/midgame_params.rs",
    ):
        digest.update(b"\0")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update((core / rel).read_bytes())

    cache_dir = WORK / "pattern-table-builder"
    cache_dir.mkdir(parents=True, exist_ok=True)
    builder = cache_dir / f"build-core-{digest.hexdigest()[:16]}{EXE_SUFFIX}"
    if builder.is_file():
        return builder

    tmp_builder = cache_dir / f"build-core-{digest.hexdigest()[:16]}.{os.getpid()}.tmp{EXE_SUFFIX}"
    cmd = ["rustc", "--edition=2024", "-O", "-o", str(tmp_builder), "build.rs"]
    res = run_captured(cmd, core)
    emit_captured(cmd, res)
    if res.returncode != 0:
        tmp_builder.unlink(missing_ok=True)
        return None
    try:
        os.replace(tmp_builder, builder)
    except OSError:
        tmp_builder.unlink(missing_ok=True)
        if not builder.is_file():
            raise
    return builder


def _swap_u16_bytes(data: bytes) -> bytes:
    words = array("H")
    assert words.itemsize == 2
    words.frombytes(data)
    words.byteswap()
    return words.tobytes()


def check_generated_pattern_table() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    core = REPO / "crates" / "reversi-core"
    builder = _cached_pattern_table_builder(core)
    if builder is None:
        return 1

    positives = (
        ("x86_64", "sse2,avx2", "little"),
        ("x86_64", "avx512bw", "little"),
        ("aarch64", "neon", "little"),
        ("aarch64", "neon", "big"),
    )
    negatives = (
        ("x86_64", "sse2,avx20"),
        ("aarch64", ""),
        ("riscv64", ""),
    )

    def _run_positive(index: int, arch: str, features: str, order: str, base: Path):
        sub = base / f"pos-{index}"
        sub.mkdir()
        env = dict(os.environ, OUT_DIR=str(sub), CARGO_CFG_TARGET_ARCH=arch,
                   CARGO_CFG_TARGET_FEATURE=features, CARGO_CFG_TARGET_ENDIAN=order)
        res = subprocess.run([str(builder)], cwd=core, env=env, stdout=subprocess.DEVNULL)
        if res.returncode != 0:
            return arch, features, order, None, res.returncode
        return arch, features, order, (sub / "eval_feature_u16_sum.bin").read_bytes(), 0

    def _run_negative(index: int, arch: str, features: str, base: Path):
        sub = base / f"neg-{index}"
        sub.mkdir()
        blob = sub / "eval_feature_u16_sum.bin"
        blob.write_bytes(b"stale")
        env = dict(os.environ, OUT_DIR=str(sub), CARGO_CFG_TARGET_ARCH=arch,
                   CARGO_CFG_TARGET_FEATURE=features, CARGO_CFG_TARGET_ENDIAN="little")
        res = subprocess.run([str(builder)], cwd=core, env=env, stdout=subprocess.DEVNULL)
        return arch, features, res.returncode, blob.exists()

    with tempfile.TemporaryDirectory(dir=WORK, prefix="generated-table-") as scratch:
        out = Path(scratch)
        total_jobs = len(positives) + len(negatives)
        with ThreadPoolExecutor(max_workers=max(1, min(total_jobs, available_cpus()))) as pool:
            pos_futures = [
                pool.submit(_run_positive, i, arch, features, order, out)
                for i, (arch, features, order) in enumerate(positives)
            ]
            neg_futures = [
                pool.submit(_run_negative, i, arch, features, out)
                for i, (arch, features) in enumerate(negatives)
            ]

            little = b""
            big_expected = b""
            for future in pos_futures:
                arch, features, order, table, rc = future.result()
                if rc != 0 or table is None:
                    return 1
                if len(table) != 4 * 65536 * 32 * 2:
                    print(f"FAIL: invalid pattern table length for {arch}/{features}/{order}")
                    return 1
                # Chunk zero, mask 1, diagonal A1-H8. The A1 bit is its leading ternary digit.
                if int.from_bytes(table[72:74], order) != 3 ** 7:
                    print(f"FAIL: wrong A1 diagonal contribution for {arch}/{features}/{order}")
                    return 1
                if not little:
                    little = table
                if order == "little":
                    same = table == little
                else:
                    if not big_expected:
                        big_expected = _swap_u16_bytes(little)
                    same = table == big_expected
                if not same:
                    print(f"FAIL: pattern table bytes differ for {arch}/{features}/{order}")
                    return 1

            for future in neg_futures:
                arch, features, rc, exists = future.result()
                if rc != 0:
                    return 1
                if exists:
                    print(f"FAIL: stale pattern table for {arch}/{features}")
                    return 1
    print("Generated pattern table: 8,388,608 lanes agree in both target byte orders", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
