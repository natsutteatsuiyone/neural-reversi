import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_policy import FORBIDDEN, LEAN_ROOT, strip_comments_and_strings

HEADER = "import Lean\nopen Lean\n"
FIXTURES = [
    ("character", "def quote : Char := '\"'\ndef hidden : Nat := by sorry\n", ["sorry"]),
    ("raw", 'def text : String := r#"a"b"#\ndef hidden : Nat := by sorry\n', ["sorry"]),
    ("nested", 'def hidden : String := s!"{s!"{(by sorry : Nat)}"}"\n', ["sorry"]),
    ("format", 'def hidden : Std.Format := f!"{(by sorry : Nat)}"\n', ["sorry"]),
    ("at", 'def hidden : CoreM Unit := throwErrorAt Syntax.missing "{(by sorry : Nat)}"\n', ["sorry"]),
    ("at_effect", 'def hidden : CoreM Unit := do\n  throwErrorAt (← getRef) "{(by sorry : Nat)}"\n', ["sorry"]),
    ("at_prefix", 'def hidden : CoreM Unit := throwErrorAt (by sorry : Syntax) "ok"\n', ["sorry"]),
    ("at_literal", 'def hidden : CoreM Unit := throwErrorAt (let _ := "{sorry}"; Syntax.missing) "{(by sorry : Nat)}"\n', ["sorry"]),
    ("named", 'def hidden : CoreM Unit := throwNamedError lean.unknownIdentifier "{(by sorry : Nat)}"\n', ["sorry"]),
    ("named_at", 'def hidden : CoreM Unit := throwNamedErrorAt Syntax.missing lean.unknownIdentifier "{(by sorry : Nat)}"\n', ["sorry"]),
    ("trace", 'initialize registerTraceClass `policyRegression\ndef hidden : CoreM Unit := do\n  trace[policyRegression] "{(by sorry : Nat)}"\n', ["sorry"]),
    ("macro_trace", 'def hidden : MacroM Unit := Macro.trace[policyRegression] "{(by sorry : Nat)}"\n', ["sorry"]),
    ("debug", 'def hidden : IO Unit := do\n  dbg_trace "{(by sorry : Nat)}"\n', ["sorry"]),
    ("ordinary", 'def text : String := "{sorry}"\ndef plain : CoreM Unit := Lean.throwErrorAt Syntax.missing "{sorry}"\n', []),
    ("inert", 'def text : String := s!"sorry {("axiom" ++ r#"admit"#)}"\n', []),
]


AUDIT_REJECTION_FIXTURES = [
    (
        "audit_unsafe_def",
        "unsafe def badUnsafe (n : Nat) : Nat := n\n"
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "project declaration must be safe",
    ),
    (
        "audit_partial_def",
        "partial def badPartial (n : Nat) : Nat := badPartial (n + 1)\n"
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "project declaration must be safe",
    ),
    (
        "audit_implemented_by",
        "def okTarget (n : Nat) : Nat := n\n"
        "@[implemented_by okTarget] def badImpl (n : Nat) : Nat := n + 1\n"
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "project implemented_by attribute forbidden",
    ),
    (
        "audit_extern",
        '@[extern "bad_c_fn"] opaque badExtern (n : Nat) : Nat\n'
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "project extern attribute forbidden",
    ),
    (
        "audit_def_sorry",
        "def badSorryDef : Nat := by sorry\n"
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "unapproved axiom sorryAx",
    ),
    (
        "audit_opaque_sorry",
        "opaque badSorryOpaque : Nat := by sorry\n"
        "#eval do\n"
        "  let env ← Lean.getEnv\n"
        "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
        "unapproved axiom sorryAx",
    ),
    (
        "audit_open_status",
        "#eval show Lean.Elab.Command.CommandElabM Unit from do\n"
        "  let env ← Lean.getEnv\n"
        '  let _ ← NeuralReversi.Audit.auditRegistryEntries env [{\n'
        '    id := "reg.open", functions := [], status := "open",\n'
        '    property := "True", theoremName := "True.intro" }]\n',
        "invalid or open status open",
    ),
    (
        "audit_foreign_theorem",
        "#eval show Lean.Elab.Command.CommandElabM Unit from do\n"
        "  let env ← Lean.getEnv\n"
        '  let _ ← NeuralReversi.Audit.auditRegistryEntries env [{\n'
        '    id := "reg.foreign", functions := [], status := "proven",\n'
        '    property := "True", theoremName := "True.intro" }]\n',
        "is not a project declaration",
    ),
]

AUDIT_POSITIVE_FIXTURE = (
    "audit_safe_positive",
    "inductive SafeInd | a | b\n"
    "def safeRec : Nat → Nat\n"
    "  | 0 => 0\n"
    "  | n + 1 => safeRec n\n"
    "#eval do\n"
    "  let env ← Lean.getEnv\n"
    "  let _ ← NeuralReversi.Audit.auditProjectDeclarations [env.mainModule]\n",
)


def expected_lean_version(lean_root: Path) -> str:
    toolchain = (lean_root / "lean-toolchain").read_text(encoding="utf-8").strip()
    version = toolchain.rsplit(":", 1)[-1].lstrip("v")
    if not version:
        raise RuntimeError(f"Could not parse Lean version from lean-toolchain: {toolchain!r}")
    return version


def run_fixture(directory: Path, name: str, source: str, *, strict: bool = False):
    path = directory / f"{name}.lean"
    path.write_text(source, encoding="utf-8")
    warning_mode = "true" if strict else "false"
    return subprocess.run(
        ["lake", "env", "lean", f"-DwarningAsError={warning_mode}", str(path.resolve())],
        cwd=LEAN_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def main() -> None:
    expected_version = expected_lean_version(LEAN_ROOT)
    version = subprocess.check_output(
        ["lake", "env", "lean", "--version"], cwd=LEAN_ROOT,
        text=True, encoding="utf-8", timeout=120,
    )
    if f"version {expected_version}," not in version:
        raise RuntimeError(f"Expected pinned Lean {expected_version}, got {version.strip()}")
    failures = []
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    with tempfile.TemporaryDirectory(prefix="policy-lean-") as tmp, ThreadPoolExecutor(
        max_workers=min(4, cores)
    ) as pool:
        directory = Path(tmp)
        policies = [
            pool.submit(run_fixture, directory, name, HEADER + body)
            for name, body, _ in FIXTURES
        ]
        audit_header = "import NeuralReversi.AuditCore\n"
        rejections = [
            pool.submit(run_fixture, directory, name, audit_header + body)
            for name, body, _ in AUDIT_REJECTION_FIXTURES
        ]
        pos_name, pos_body = AUDIT_POSITIVE_FIXTURE
        positive = pool.submit(run_fixture, directory, pos_name, audit_header + pos_body, strict=True)

        for (name, body, expected), future in zip(FIXTURES, policies):
            actual = FORBIDDEN.findall(strip_comments_and_strings(HEADER + body))
            if actual != expected:
                failures.append(f"{name}: expected {expected}, got {actual}")
            result = future.result()
            if result.returncode:
                failures.append(f"{name}: not a valid Lean fixture\n{result.stdout}{result.stderr}")
            else:
                print(f"POLICY LEAN FIXTURE OK: {name}", flush=True)
        for (name, _, expected_err), future in zip(AUDIT_REJECTION_FIXTURES, rejections):
            result = future.result()
            combined = result.stdout + result.stderr
            if result.returncode == 0 or expected_err not in combined:
                failures.append(
                    f"{name}: expected rejection containing {expected_err!r}, "
                    f"got rc={result.returncode}\n{combined}"
                )
            else:
                print(f"AUDIT CORE REJECTION OK: {name}", flush=True)
        pos_result = positive.result()
        if pos_result.returncode != 0:
            failures.append(
                f"{pos_name}: expected positive audit pass\n{pos_result.stdout}{pos_result.stderr}"
            )
        else:
            print(f"AUDIT CORE POSITIVE OK: {pos_name}", flush=True)
    if failures:
        raise AssertionError("\n\n".join(failures))
    print(
        f"POLICY LEAN REGRESSION OK ({len(FIXTURES)} policy fixtures; "
        f"{len(AUDIT_REJECTION_FIXTURES)} audit rejections + 1 positive control)"
    )


if __name__ == "__main__":
    main()
