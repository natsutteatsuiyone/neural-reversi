"""Compile a Quint module and exhaust or extract a named TLC witness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
LEAN = REPO / "verification" / "lean"
DEFAULT_WITNESS_DIR = REPO / "verification" / "tla" / "lean-witnesses"
APALACHE = "0.62.4"


def compile_model(quint, work, source):
    name = source.stem
    result = subprocess.run(
        [quint, "compile", str(source), "--target=tlaplus",
         "--apalache-version", APALACHE, "--verbosity=0"],
        cwd=work, stdout=subprocess.PIPE, text=True, check=True,
    )
    # On a cold start the compiler can print server startup messages before the module.
    module = re.search(rf"^-+ MODULE {re.escape(name)} -+\s*$.*?^=+\s*$", result.stdout, re.M | re.S)
    if module is None:
        raise RuntimeError(f"Quint did not emit the TLA+ module:\n{result.stdout}")
    (work / f"{name}.tla").write_text(module[0] + "\n", encoding="utf-8")


def tlc(work, jar, module, step, invariant, witness=False):
    label = f"{step}-{invariant}"
    config = work / f"{label}.cfg"
    config.write_text(f"INIT init\nNEXT {step}\nINVARIANT {invariant}\n", encoding="utf-8")
    trace = work / f"{label}.json"
    result = subprocess.run(
        ["java", "-Xmx512m", "-cp", str(jar), "tlc2.TLC", "-workers", "1",
         "-metadir", f"states-{label}", "-noGenerateSpecTE", "-config", config.name,
         "-dumpTrace", "json", trace.name, f"{module}.tla"],
        cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    (work / f"{label}.log").write_text(result.stdout, encoding="utf-8")
    if not witness:
        if result.returncode or "No error has been found" not in result.stdout:
            raise RuntimeError(f"{label}:\n{result.stdout}")
        print(next(s for s in result.stdout.splitlines() if "distinct states found" in s), flush=True)
        return
    if result.returncode != 12 or f"Invariant {invariant} is violated" not in result.stdout:
        raise RuntimeError(f"{label}: expected the named invariant violation\n{result.stdout}")
    states = json.loads(trace.read_text(encoding="utf-8"))["counterexample"]["state"]
    snapshots = work / f"{label}-snapshots.json"
    rows = [state["s"] for _, state in states]
    snapshots.write_text(json.dumps(rows), encoding="utf-8")
    return snapshots, rows


def prepare(description, group):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quint", default="quint", help="Quint CLI executable")
    parser.add_argument("--apalache-jar", type=Path, help="use an existing Apalache distribution's JAR")
    args = parser.parse_args()
    dest = REPO / "target" / "tla" / group
    dest.mkdir(exist_ok=True, parents=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=dest))
    print(f"Model, TLC logs and witnesses: {work}", flush=True)
    home = Path(os.environ.get("QUINT_HOME", Path.home() / ".quint"))
    jar = (args.apalache_jar or home / f"apalache-dist-{APALACHE}" / "apalache/lib/apalache.jar").resolve()
    return args.quint, work, jar


def build_lean(*modules):
    subprocess.run(["lake", "build", *modules], cwd=LEAN, check=True)


def replay_lean(
    script,
    mode,
    path,
    *,
    model_dir: Path = DEFAULT_WITNESS_DIR,
    expected_error=None,
    expect_reject=False,
    quiet=False,
):
    result = subprocess.run(
        ["lake", "env", "lean", "--run", str(model_dir / script), mode, str(path)],
        cwd=LEAN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if expected_error is not None or expect_reject:
        if result.returncode == 0:
            raise RuntimeError(f"{script} ({mode}) unexpectedly succeeded:\n{result.stdout}")
        if expected_error is not None and expected_error not in result.stdout:
            raise RuntimeError(f"{script} ({mode}) did not report {expected_error!r}:\n{result.stdout}")
        return result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(f"{script} ({mode}) failed (exit {result.returncode}):\n{result.stdout}")
    output = result.stdout.strip()
    if output and not quiet:
        print(output, flush=True)
    return output


def run_witness_suite(
    description,
    group,
    module,
    safety_invariant,
    lean_modules,
    replay_script,
    witnesses,
    *,
    model_dir: Path = DEFAULT_WITNESS_DIR,
    reject_errors=None,
):
    quint, work, jar = prepare(description, group)
    compile_model(quint, work, model_dir / f"{module}.qnt")
    tlc(work, jar, module, "step", safety_invariant)
    build_lean(*lean_modules)
    reject_errors = reject_errors or {}
    for mode, invariant in witnesses:
        path, _ = tlc(work, jar, module, mode, invariant, witness=True)
        print(mode, invariant, flush=True)
        replay_lean(replay_script, mode, path, model_dir=model_dir)
        if mode != "step":
            replay_lean(
                replay_script,
                "step",
                path,
                model_dir=model_dir,
                expected_error=reject_errors.get(mode),
                expect_reject=True,
            )
    print("PASS")
