import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_policy import FORBIDDEN, LEAN_ROOT, strip_comments_and_strings

REPO_ROOT = LEAN_ROOT.parents[1]
HEADER_CACHE_DIR = REPO_ROOT / "target" / "verify-lean" / "policy-header-cache"
MODULE_FIXTURE_CACHE_DIR = HEADER_CACHE_DIR / "module-fixtures"
CACHE_FORMAT_VERSION = 1

HEADER = "import Lean\nopen Lean\n"
DRIVER_HEADER = "import Lean\nimport NeuralReversi.AuditCore\n"
LEAN_DRIVER = DRIVER_HEADER + """
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let baseEnv ← Lean.getEnv
  let cmdCtx ← read
  unless baseEnv.constants.map₂.isEmpty do
    throwError "policy regression base environment unexpectedly contains local declarations"
  let some specPath ← IO.getEnv "POLICY_SPEC_PATH"
    | throwError "missing POLICY_SPEC_PATH"
  let some outPath ← IO.getEnv "POLICY_OUT_PATH"
    | throwError "missing POLICY_OUT_PATH"
  if let some modDir ← IO.getEnv "POLICY_MODULE_DIR" then
    Lean.searchPathRef.modify (fun sp => System.FilePath.mk modDir :: sp)
  let raw ← IO.FS.readFile specPath
  let items ← IO.ofExcept ((← IO.ofExcept (Lean.Json.parse raw)).getArr?)
  let runFixture := fun (item : Lean.Json) => show IO Lean.Json from do
    let name ← IO.ofExcept (item.getObjValAs? String "name")
    let source ← IO.ofExcept (item.getObjValAs? String "source")
    let strict ← IO.ofExcept (item.getObjValAs? Bool "strict")
    let opts : Lean.Options := Lean.warningAsError.set {} strict
    let env0 := baseEnv.setMainModule (Lean.Name.mkStr2 "PolicyRegression" name)
    let (_, msgs) ← Lean.Elab.process source env0 opts (some s!"{name}.lean")
    let mut hasError := msgs.hasErrors
    let mut rendered : Array String := #[]
    for msg in msgs.toList do
      if strict && msg.severity == .warning then
        hasError := true
      rendered := rendered.push (← msg.toString)
    return Lean.Json.mkObj [
      ("name", Lean.toJson name),
      ("has_error", Lean.toJson hasError),
      ("output", Lean.toJson (String.intercalate "\\n" rendered.toList))
    ]
  let runModuleAudits := fun (audits : Array Lean.Json) => show IO (Array Lean.Json) from do
    if audits.isEmpty then
      return #[]
    let mut parsed : Array (String × Lean.Name) := #[]
    let mut imports : Array Lean.Import := #[]
    for item in audits do
      let name ← IO.ofExcept (item.getObjValAs? String "name")
      let modStr ← IO.ofExcept (item.getObjValAs? String "module")
      let modName := modStr.toName
      parsed := parsed.push (name, modName)
      imports := imports.push { module := modName }
    let importedEnv ← Lean.importModules imports {}
    let st := Lean.Elab.Command.mkState importedEnv {} {}
    let mut out : Array Lean.Json := #[]
    for (name, modName) in parsed do
      let res ← (((NeuralReversi.Audit.auditProjectDeclarations [modName]).run cmdCtx).run st).toBaseIO
      match res with
      | .ok ((thms, decls), _) =>
        out := out.push (Lean.Json.mkObj [
          ("name", Lean.toJson name),
          ("has_error", Lean.toJson false),
          ("output", Lean.toJson s!"thms={thms} decls={decls}")
        ])
      | .error ex =>
        out := out.push (Lean.Json.mkObj [
          ("name", Lean.toJson name),
          ("has_error", Lean.toJson true),
          ("output", Lean.toJson (← ex.toMessageData.toString))
        ])
    return out
  let mut tasks : Array (Task (Except IO.Error Lean.Json)) := #[]
  let mut sequential : Array Lean.Json := #[]
  let mut moduleAudits : Array Lean.Json := #[]
  for item in items do
    if (item.getObjValAs? String "module").toOption.isSome then
      moduleAudits := moduleAudits.push item
    else if (item.getObjValAs? Bool "sequential").toOption.getD false then
      sequential := sequential.push item
    else
      tasks := tasks.push (← IO.asTask (runFixture item))
  let moduleTask ← IO.asTask (runModuleAudits moduleAudits)
  let mut results : Array Lean.Json := #[]
  for task in tasks do
    results := results.push (← IO.ofExcept task.get)
  for entry in (← IO.ofExcept moduleTask.get) do
    results := results.push entry
  for item in sequential do
    results := results.push (← runFixture item)
  let payload := Lean.Json.mkObj [
    ("version", Lean.toJson Lean.versionString),
    ("results", Lean.Json.arr results)
  ]
  IO.FS.writeFile outPath payload.compress
"""
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

MODULE_AUDIT_FIXTURES = [
    (
        "audit_module_private_sorry",
        "ModPrivateSorry",
        "module\nprelude\nimport Init.Prelude\n\nprivate theorem hiddenModuleSorry : True := sorry\n",
        ("unapproved axiom sorryAx", "hiddenModuleSorry"),
    ),
    (
        "audit_module_private_clean",
        "ModPrivateClean",
        "module\nprelude\nimport Init.Prelude\n\nprivate theorem hiddenModuleClean : True := True.intro\n",
        None,
    ),
]


def read_lean_toolchain(lean_root: Path) -> tuple[str, str]:
    toolchain = (lean_root / "lean-toolchain").read_text(encoding="utf-8").strip()
    version = toolchain.rsplit(":", 1)[-1].lstrip("v")
    if not version:
        raise RuntimeError(f"Could not parse Lean version from lean-toolchain: {toolchain!r}")
    return toolchain, version


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_paths() -> tuple[Path, Path, Path]:
    snap = HEADER_CACHE_DIR / "header.snap"
    deps = HEADER_CACHE_DIR / "header.snap.deps"
    meta = HEADER_CACHE_DIR / "header.snap.meta.json"
    return snap, deps, meta


def _invalidate_snapshot_cache() -> None:
    for path in reversed(_cache_paths()):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _collect_snapshot_deps(deps_path: Path, lean_root: Path) -> tuple[list[list[str]], str | None]:
    resolved_root = lean_root.resolve()
    entries = json.loads(deps_path.read_text(encoding="utf-8"))
    project_paths: set[Path] = set()
    toolchain_probe: str | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if not isinstance(value, str):
                continue
            path = Path(value).resolve()
            if path.is_relative_to(resolved_root):
                project_paths.add(path)
            elif toolchain_probe is None:
                toolchain_probe = str(path)
    project_deps = [[str(path), _sha256_file(path)] for path in sorted(project_paths)]
    return project_deps, toolchain_probe


def _valid_cached_snapshot(toolchain: str) -> Path | None:
    snap, deps, meta_path = _cache_paths()
    if not (snap.is_file() and deps.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        meta.get("format") != CACHE_FORMAT_VERSION
        or meta.get("lean_toolchain") != toolchain
        or meta.get("lean_root") != str(LEAN_ROOT.resolve())
        or meta.get("header") != DRIVER_HEADER
    ):
        return None
    toolchain_probe = meta.get("toolchain_probe")
    if isinstance(toolchain_probe, str):
        try:
            if not Path(toolchain_probe).is_file():
                return None
        except OSError:
            return None
    project_deps = meta.get("project_deps")
    if not isinstance(project_deps, list) or not project_deps:
        return None
    for item in project_deps:
        if not (isinstance(item, list) and len(item) == 2):
            return None
        dep_path = Path(item[0])
        try:
            if not dep_path.is_file() or _sha256_file(dep_path) != item[1]:
                return None
        except OSError:
            return None
    return snap


def _save_snapshot_cache(tmp_snap: Path, toolchain: str) -> None:
    tmp_deps = Path(f"{tmp_snap}.deps")
    tmp_meta = HEADER_CACHE_DIR / f"header.{os.getpid()}.tmp.meta.json"
    snap, deps, meta_path = _cache_paths()
    try:
        if not (tmp_snap.is_file() and tmp_deps.is_file()):
            return
        project_deps, toolchain_probe = _collect_snapshot_deps(tmp_deps, LEAN_ROOT)
        if not project_deps:
            return
        meta = {
            "format": CACHE_FORMAT_VERSION,
            "lean_toolchain": toolchain,
            "lean_root": str(LEAN_ROOT.resolve()),
            "header": DRIVER_HEADER,
            "project_deps": project_deps,
            "toolchain_probe": toolchain_probe,
        }
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        # Remove metadata first so concurrent readers treat the swap window as a cache miss.
        meta_path.unlink(missing_ok=True)
        os.replace(tmp_snap, snap)
        os.replace(tmp_deps, deps)
        os.replace(tmp_meta, meta_path)
    except (OSError, ValueError):
        pass
    finally:
        for leftover in (tmp_snap, tmp_deps, tmp_meta):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass


def build_fixture_specs() -> list[dict[str, object]]:
    # Run `trace` strictly after all parallel fixtures so its `registerTraceClass`
    # initializer cannot leak into `macro_trace`.
    specs: list[dict[str, object]] = [
        {
            "name": f"policy_{name}",
            "source": "open Lean\n" + body,
            "strict": False,
            "sequential": name == "trace",
        }
        for name, body, _ in FIXTURES
    ]
    specs.extend(
        {
            "name": name,
            "source": body,
            "strict": False,
            "sequential": False,
        }
        for name, body, _ in AUDIT_REJECTION_FIXTURES
    )
    pos_name, pos_body = AUDIT_POSITIVE_FIXTURE
    specs.append(
        {
            "name": pos_name,
            "source": pos_body,
            "strict": True,
            "sequential": False,
        }
    )
    specs.extend(
        {
            "name": name,
            "module": mod_name,
        }
        for name, mod_name, _, _ in MODULE_AUDIT_FIXTURES
    )
    return specs


def _module_fixture_meta_path() -> Path:
    return MODULE_FIXTURE_CACHE_DIR / "fixtures.meta.json"


def _valid_cached_module_fixtures(toolchain: str) -> Path | None:
    meta_path = _module_fixture_meta_path()
    try:
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        meta.get("format") != CACHE_FORMAT_VERSION
        or meta.get("lean_toolchain") != toolchain
    ):
        return None
    cached_entries = meta.get("fixtures")
    if not isinstance(cached_entries, dict):
        return None
    for _, mod_name, source, _ in MODULE_AUDIT_FIXTURES:
        entry = cached_entries.get(mod_name)
        if not isinstance(entry, dict):
            return None
        expected_src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if entry.get("source_sha256") != expected_src_hash:
            return None
        olean = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean"
        priv = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean.private"
        try:
            if (
                not olean.is_file()
                or not priv.is_file()
                or _sha256_file(olean) != entry.get("olean_sha256")
                or _sha256_file(priv) != entry.get("private_sha256")
            ):
                return None
            server_hash = entry.get("server_sha256")
            if isinstance(server_hash, str):
                server = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean.server"
                if not server.is_file() or _sha256_file(server) != server_hash:
                    return None
        except OSError:
            return None
    return MODULE_FIXTURE_CACHE_DIR


def _compile_single_module_fixture(directory: Path, name: str, mod_name: str, source: str) -> None:
    mod_src = directory / f"{mod_name}.lean"
    mod_olean = directory / f"{mod_name}.olean"
    mod_priv = directory / f"{mod_name}.olean.private"
    mod_src.write_text(source, encoding="utf-8")
    compile_res = subprocess.run(
        [
            "lake", "env", "lean", "-DwarningAsError=false",
            "-R", str(directory),
            "-o", str(mod_olean),
            str(mod_src),
        ],
        cwd=LEAN_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if compile_res.returncode != 0 or not mod_olean.is_file():
        raise RuntimeError(
            f"{name}: failed to compile module fixture {mod_name}\n"
            f"{compile_res.stdout}{compile_res.stderr}"
        )
    if not mod_priv.is_file():
        raise RuntimeError(
            f"{name}: expected `.olean.private` artifact to be emitted at {mod_priv}"
        )


def _save_module_fixture_cache(src_dir: Path, toolchain: str) -> None:
    meta_path = _module_fixture_meta_path()
    pid = os.getpid()
    staged: list[tuple[Path, Path]] = []
    tmp_meta = MODULE_FIXTURE_CACHE_DIR / f"fixtures.{pid}.tmp.meta.json"
    try:
        MODULE_FIXTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fixtures_meta: dict[str, dict[str, str]] = {}
        for _, mod_name, source, _ in MODULE_AUDIT_FIXTURES:
            olean_src = src_dir / f"{mod_name}.olean"
            priv_src = src_dir / f"{mod_name}.olean.private"
            server_src = src_dir / f"{mod_name}.olean.server"
            if not (olean_src.is_file() and priv_src.is_file()):
                return
            entry: dict[str, str] = {
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "olean_sha256": _sha256_file(olean_src),
                "private_sha256": _sha256_file(priv_src),
            }
            tmp_olean = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.{pid}.tmp.olean"
            tmp_priv = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.{pid}.tmp.olean.private"
            tmp_olean.write_bytes(olean_src.read_bytes())
            tmp_priv.write_bytes(priv_src.read_bytes())
            staged.append((tmp_olean, MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean"))
            staged.append((tmp_priv, MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean.private"))
            if server_src.is_file():
                entry["server_sha256"] = _sha256_file(server_src)
                tmp_server = MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.{pid}.tmp.olean.server"
                tmp_server.write_bytes(server_src.read_bytes())
                staged.append((tmp_server, MODULE_FIXTURE_CACHE_DIR / f"{mod_name}.olean.server"))
            fixtures_meta[mod_name] = entry
        meta = {
            "format": CACHE_FORMAT_VERSION,
            "lean_toolchain": toolchain,
            "fixtures": fixtures_meta,
        }
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        # Remove metadata first so concurrent readers treat the swap window as a cache miss.
        meta_path.unlink(missing_ok=True)
        for tmp_path, final_path in staged:
            os.replace(tmp_path, final_path)
        os.replace(tmp_meta, meta_path)
    except (OSError, ValueError):
        pass
    finally:
        for tmp_path, _ in staged:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            tmp_meta.unlink(missing_ok=True)
        except OSError:
            pass


def prepare_module_fixtures(stack: ExitStack, toolchain: str) -> Path:
    """Return a directory containing compiled `.olean` + `.olean.private` files
    for `MODULE_AUDIT_FIXTURES`, reusing the cache when valid and not in CI."""
    if not os.environ.get("CI"):
        cached = _valid_cached_module_fixtures(toolchain)
        if cached is not None:
            return cached
    tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="policy-module-audit-")))
    with ThreadPoolExecutor(max_workers=len(MODULE_AUDIT_FIXTURES)) as pool:
        futs = [
            pool.submit(_compile_single_module_fixture, tmp_dir, name, mod_name, source)
            for name, mod_name, source, _ in MODULE_AUDIT_FIXTURES
        ]
        for fut in futs:
            fut.result()
    if not os.environ.get("CI"):
        _save_module_fixture_cache(tmp_dir, toolchain)
    return tmp_dir


def _invoke_lean_driver(
    driver_path: Path,
    spec_path: Path,
    out_path: Path,
    module_dir: Path,
    extra_args: list[str],
) -> subprocess.CompletedProcess[str]:
    if out_path.exists():
        out_path.unlink()
    env = dict(
        os.environ,
        POLICY_SPEC_PATH=str(spec_path),
        POLICY_OUT_PATH=str(out_path),
        POLICY_MODULE_DIR=str(module_dir),
    )
    return subprocess.run(
        ["lake", "env", "lean", *extra_args, str(driver_path.resolve())],
        cwd=LEAN_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def run_all_fixtures(
    specs: list[dict[str, object]],
    toolchain: str,
    module_dir: Path,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="policy-lean-") as tmp:
        directory = Path(tmp)
        driver_path = directory / "driver.lean"
        spec_path = directory / "spec.json"
        out_path = directory / "out.json"
        driver_path.write_text(LEAN_DRIVER, encoding="utf-8")
        spec_path.write_text(json.dumps(specs), encoding="utf-8")

        cached_snap = _valid_cached_snapshot(toolchain)
        if cached_snap is not None:
            result = _invoke_lean_driver(
                driver_path, spec_path, out_path, module_dir, [f"--incr-load={cached_snap}"]
            )
            if result.returncode == 0 and out_path.is_file():
                return json.loads(out_path.read_text(encoding="utf-8"))
            _invalidate_snapshot_cache()

        # Skip writing a 67 MiB header cache on single-shot CI runners.
        if not os.environ.get("CI"):
            HEADER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_snap = HEADER_CACHE_DIR / f"header.{os.getpid()}.tmp.snap"
            result = _invoke_lean_driver(
                driver_path, spec_path, out_path, module_dir, [f"--incr-header-save={tmp_snap}"]
            )
            if result.returncode == 0 and out_path.is_file():
                _save_snapshot_cache(tmp_snap, toolchain)
                return json.loads(out_path.read_text(encoding="utf-8"))
            for leftover in (tmp_snap, Path(f"{tmp_snap}.deps")):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            combined = result.stdout + result.stderr
            if "incr-header-save" not in combined:
                raise RuntimeError(
                    f"Lean policy regression driver failed (rc={result.returncode}):\n{combined}"
                )

        result = _invoke_lean_driver(driver_path, spec_path, out_path, module_dir, [])
        if result.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"Lean policy regression driver failed (rc={result.returncode}):\n"
                f"{result.stdout}{result.stderr}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> None:
    toolchain, expected_version = read_lean_toolchain(LEAN_ROOT)
    failures: list[str] = []

    # Always run the Python policy scanner on every fixture individually.
    for name, body, expected in FIXTURES:
        actual = FORBIDDEN.findall(strip_comments_and_strings(HEADER + body))
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")

    with ExitStack() as stack:
        module_dir = prepare_module_fixtures(stack, toolchain)
        payload = run_all_fixtures(build_fixture_specs(), toolchain, module_dir)

    actual_version = payload.get("version")
    if actual_version != expected_version:
        raise RuntimeError(f"Expected pinned Lean {expected_version}, got {actual_version!r}")

    results_by_name = {entry["name"]: entry for entry in payload["results"]}

    for name, _, _ in FIXTURES:
        entry = results_by_name.get(f"policy_{name}")
        if entry is None:
            failures.append(f"{name}: missing Lean fixture result")
        elif entry["has_error"]:
            failures.append(f"{name}: not a valid Lean fixture\n{entry['output']}")
        else:
            print(f"POLICY LEAN FIXTURE OK: {name}", flush=True)

    for name, _, expected_err in AUDIT_REJECTION_FIXTURES:
        entry = results_by_name.get(name)
        if entry is None:
            failures.append(f"{name}: missing audit rejection result")
            continue
        combined = entry["output"]
        if not entry["has_error"] or expected_err not in combined:
            failures.append(
                f"{name}: expected rejection containing {expected_err!r}, "
                f"got has_error={entry['has_error']}\n{combined}"
            )
        else:
            print(f"AUDIT CORE REJECTION OK: {name}", flush=True)

    pos_name, _ = AUDIT_POSITIVE_FIXTURE
    pos_entry = results_by_name.get(pos_name)
    if pos_entry is None:
        failures.append(f"{pos_name}: missing positive audit result")
    elif pos_entry["has_error"]:
        failures.append(f"{pos_name}: expected positive audit pass\n{pos_entry['output']}")
    else:
        print(f"AUDIT CORE POSITIVE OK: {pos_name}", flush=True)

    module_rejections = 0
    module_positives = 0
    for name, _, _, expected_errs in MODULE_AUDIT_FIXTURES:
        entry = results_by_name.get(name)
        if entry is None:
            failures.append(f"{name}: missing module audit result")
            continue
        combined = entry["output"]
        if expected_errs is not None:
            module_rejections += 1
            if not entry["has_error"] or any(tok not in combined for tok in expected_errs):
                failures.append(
                    f"{name}: expected rejection containing {expected_errs!r} from `.olean.private`, "
                    f"got has_error={entry['has_error']}\n{combined}"
                )
            else:
                print(f"AUDIT CORE REJECTION OK: {name}", flush=True)
        else:
            module_positives += 1
            if entry["has_error"] or "thms=1 " not in combined:
                failures.append(
                    f"{name}: expected clean module private theorem audit pass (thms=1), "
                    f"got has_error={entry['has_error']}\n{combined}"
                )
            else:
                print(f"AUDIT CORE POSITIVE OK: {name}", flush=True)

    if failures:
        raise AssertionError("\n\n".join(failures))
    print(
        f"POLICY LEAN REGRESSION OK ({len(FIXTURES)} policy fixtures; "
        f"{len(AUDIT_REJECTION_FIXTURES) + module_rejections} audit rejections + "
        f"{1 + module_positives} positive controls)"
    )


if __name__ == "__main__":
    main()
