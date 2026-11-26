"""Shared TLC / PlusCal runner for TLA+ verification suites."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

TLA_VERSION = "v1.7.4"
JAR_URL = f"https://github.com/tlaplus/tlaplus/releases/download/{TLA_VERSION}/tla2tools.jar"
JAR_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
JAR = Path.home() / ".cache" / "tlaplus" / TLA_VERSION / "tla2tools.jar"

REPO = Path(__file__).resolve().parents[2]
TARGET_ROOT = REPO / "target" / "tla"


@dataclass(frozen=True)
class SuiteSpec:
    """Declarative specification for a TLC model-checking suite."""

    name: str
    model_dir: Path
    tla_module: str
    mutants: dict[str, str]
    deep_overrides: dict[str, str] | None = None
    pluscal: bool = True
    check_deadlock: bool = True


def ensure_jar() -> Path:
    """Ensure the pinned tla2tools.jar is present and matches its SHA-256."""
    if not JAR.exists():
        JAR.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {JAR_URL}", file=sys.stderr)
        urllib.request.urlretrieve(JAR_URL, JAR)
    digest = hashlib.sha256(JAR.read_bytes()).hexdigest()
    if digest != JAR_SHA256:
        sys.exit(f"{JAR}: sha256 {digest} != {JAR_SHA256}; delete it and rerun")
    return JAR


def _run_one(
    spec: SuiteSpec,
    jar: Path,
    mutant: str,
    workers: str,
    overrides: dict[str, str] | None = None,
    run_name: str | None = None,
) -> tuple[int, str, Path]:
    label = run_name or mutant
    out = TARGET_ROOT / spec.name / label
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    for tla_file in spec.model_dir.glob("*.tla"):
        shutil.copy(tla_file, out)

    cfg_path = spec.model_dir / f"{spec.tla_module}.cfg"
    cfg = cfg_path.read_text(encoding="utf-8")
    if spec.mutants or mutant != "none":
        cfg, n = re.subn(r'Mutant\s*=\s*"[^"]*"', f'Mutant = "{mutant}"', cfg)
        assert n == 1, f"{cfg_path.name} must set Mutant exactly once"
    for key, val in (overrides or {}).items():
        cfg, n = re.subn(rf"\b{re.escape(key)}\s*=\s*\S+", f"{key} = {val}", cfg)
        assert n == 1, f"{cfg_path.name} must set {key} exactly once"
    (out / f"{spec.tla_module}.cfg").write_text(cfg, encoding="utf-8")

    tla_name = f"{spec.tla_module}.tla"
    if spec.pluscal:
        trans = subprocess.run(
            ["java", "-cp", str(jar), "pcal.trans", "-nocfg", tla_name],
            cwd=out,
            capture_output=True,
            text=True,
        )
        if trans.returncode != 0 or "Translation completed" not in trans.stdout:
            log = trans.stdout + trans.stderr
            (out / "tlc.out").write_text(log, encoding="utf-8")
            return 1, log, out

    cmd = [
        "java",
        "-XX:+UseParallelGC",
        "-jar",
        str(jar),
        "-workers",
        str(workers),
        "-config",
        f"{spec.tla_module}.cfg",
        "-metadir",
        "states",
    ]
    if not spec.check_deadlock:
        cmd.append("-deadlock")
    cmd.append(tla_name)

    tlc = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
    log = tlc.stdout + tlc.stderr
    (out / "tlc.out").write_text(log, encoding="utf-8")
    return tlc.returncode, log, out


def _matches_expected(expected: str, log: str) -> bool:
    if expected == "Deadlock":
        return "Error: Deadlock reached." in log
    if expected == "Temporal":
        return "Error: Temporal properties were violated." in log
    if expected == "Assertion":
        return "Error: The first argument of Assert evaluated to FALSE" in log
    if expected.startswith("Property:"):
        prop = expected.split(":", 1)[1]
        return (
            "Error: Temporal properties were violated." in log
            or f"Error: Action property {prop} is violated." in log
            or f"Property {prop} is violated" in log
        )
    return f"Invariant {expected} is violated" in log


def summary(log: str) -> str:
    keep = [
        line
        for line in log.splitlines()
        if re.match(
            r"(Error:|Invariant |Deadlock|Temporal|Action property|"
            r"Model checking completed|\d+ states generated|"
            r"The depth of the complete state graph)",
            line,
        )
    ]
    return "\n".join(keep)


def run_suite(
    spec: SuiteSpec,
    *,
    mode: str = "clean",
    mutant: str | None = None,
    workers: str = "auto",
) -> bool:
    """Run one TLA+ suite in the requested mode and return True on pass."""
    jar = ensure_jar()

    if mutant is not None:
        if mutant not in spec.mutants:
            raise ValueError(f"{spec.name}: unknown mutant {mutant!r}")
        runs: list[tuple[str, str, dict[str, str] | None]] = [(mutant, mutant, None)]
    elif mode == "all":
        runs = [("none", "none", None)]
        if spec.deep_overrides is not None:
            runs.append(("deep", "none", spec.deep_overrides))
        runs.extend((m, m, None) for m in sorted(spec.mutants))
    elif mode == "deep":
        if spec.deep_overrides is None:
            print(f"== [{spec.name}] deep: skipped (no deep_overrides configured)")
            return True
        runs = [("deep", "none", spec.deep_overrides)]
    elif mode == "mutants":
        runs = [(m, m, None) for m in sorted(spec.mutants)]
    elif mode == "clean":
        runs = [("none", "none", None)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    ok = True
    for label, run_mutant, overrides in runs:
        code, log, out = _run_one(
            spec,
            jar,
            run_mutant,
            workers,
            overrides=overrides,
            run_name=label,
        )
        print(f"== [{spec.name}] {label}  (full log: {out / 'tlc.out'})")
        print(summary(log) or log[-3000:])
        if run_mutant == "none":
            passed = code == 0 and "No error has been found" in log
            print(
                "PASS"
                if passed
                else "FAIL: the model breaks a property (see the trace in the log)"
            )
        else:
            expected = spec.mutants[run_mutant]
            passed = code != 0 and _matches_expected(expected, log)
            print(
                f"PASS: mutant caught by {expected}"
                if passed
                else f"FAIL: mutant not caught by {expected}"
            )
        ok &= passed
    return ok


def run_suite_cli(spec: SuiteSpec, argv: list[str] | None = None) -> int:
    """Standard CLI entry point for a suite's check.py."""
    ap = argparse.ArgumentParser(description=f"Run TLC checks for {spec.name}")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--mutants",
        action="store_true",
        help="run every mutant; each must fail with its expected violation",
    )
    if spec.mutants:
        g.add_argument(
            "--mutant",
            choices=sorted(spec.mutants),
            help="run one named mutant",
        )
    deep_help = (
        f"run clean model with deep overrides ({spec.deep_overrides})"
        if spec.deep_overrides is not None
        else "run deep configuration (none configured; skips cleanly)"
    )
    g.add_argument("--deep", action="store_true", help=deep_help)
    g.add_argument(
        "--all",
        action="store_true",
        help="run clean model, deep config (if configured), and every mutant",
    )
    ap.add_argument(
        "--workers",
        default="auto",
        help="TLC worker threads (default: auto)",
    )
    args = ap.parse_args(argv)

    mutant_arg = getattr(args, "mutant", None)
    if args.all:
        mode = "all"
    elif args.deep:
        mode = "deep"
    elif args.mutants:
        mode = "mutants"
    elif mutant_arg is not None:
        mode = "mutant"
    else:
        mode = "clean"

    ok = run_suite(spec, mode=mode, mutant=mutant_arg, workers=args.workers)
    return 0 if ok else 1
