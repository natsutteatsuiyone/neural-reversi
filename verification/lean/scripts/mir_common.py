"""Shared MIR parsing primitives, compiler dump runner, and extractor CLI."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import fnmatch
import mmap
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading


PASS = "SimplifyCfg-initial"
_LONG_LINE_BYTES = 32768
_ELIDE_SIZE_THRESHOLD = 262144
_MAX_CACHE_FILE_BYTES = 8 * 1024 * 1024
_LONG_BYTE_LITERAL_HEAD = re.compile(rb'^(        _\d+ = const b")')

_PATCH_LOCAL = threading.local()
_ORIGINAL_READ_TEXT = Path.read_text
_READ_TEXT_HOOK_LOCK = threading.Lock()
_READ_TEXT_HOOK_INSTALLED = False

_DUMP_CACHE_LOCK = threading.Lock()
# Maps resolved dump Path -> (entries_tuple[(name, Path, size)], text_cache_dict[(name, elide) -> str])
_DUMP_CACHE = {}


def _get_thread_overrides():
    return getattr(_PATCH_LOCAL, "overrides", None)


def get_patched_mir(path):
    """Return the current thread's overridden MIR text for ``path``, or ``None``."""
    state = _get_thread_overrides()
    if not state:
        return None
    by_resolved, by_name = state
    target = Path(path)
    if target.name not in by_name:
        return None
    return by_resolved.get(target.resolve())


def _thread_aware_read_text(self, *args, **kwargs):
    patched = get_patched_mir(self)
    if patched is not None:
        return patched
    return _ORIGINAL_READ_TEXT(self, *args, **kwargs)


def _ensure_read_text_hook():
    global _READ_TEXT_HOOK_INSTALLED
    if _READ_TEXT_HOOK_INSTALLED:
        return
    with _READ_TEXT_HOOK_LOCK:
        if not _READ_TEXT_HOOK_INSTALLED:
            Path.read_text = _thread_aware_read_text
            _READ_TEXT_HOOK_INSTALLED = True


@contextmanager
def patched_mir_text(path, text):
    """Thread-locally override MIR reads of ``path`` with ``text``."""
    _ensure_read_text_hook()
    target = Path(path)
    resolved = target.resolve()
    name = target.name
    state = _get_thread_overrides()
    if state is None:
        by_resolved, by_name = {}, {}
        _PATCH_LOCAL.overrides = (by_resolved, by_name)
    else:
        by_resolved, by_name = state
    had_prev = resolved in by_resolved
    prev = by_resolved.get(resolved)
    by_resolved[resolved] = text
    by_name[name] = by_name.get(name, 0) + 1
    try:
        yield
    finally:
        if had_prev:
            by_resolved[resolved] = prev
        else:
            by_resolved.pop(resolved, None)
        remaining = by_name[name] - 1
        if remaining:
            by_name[name] = remaining
        else:
            del by_name[name]
        if not by_resolved:
            _PATCH_LOCAL.overrides = None


def _get_dump_entry(dump):
    root = Path(dump).resolve()
    with _DUMP_CACHE_LOCK:
        entry = _DUMP_CACHE.get(root)
    if entry is not None:
        return root, entry
    scanned = []
    for e in os.scandir(root):
        try:
            if e.is_file():
                scanned.append((e.name, Path(e.path), e.stat().st_size))
        except FileNotFoundError:
            continue
    scanned.sort(key=lambda item: item[0])
    with _DUMP_CACHE_LOCK:
        entry = _DUMP_CACHE.setdefault(root, (scanned, {}))
    return root, entry


def clear_dump_cache(dump):
    """Drop cached directory listings and MIR texts for ``dump``."""
    root = Path(dump).resolve()
    with _DUMP_CACHE_LOCK:
        _DUMP_CACHE.pop(root, None)


def seed_dump_cache_subset(src_dump, dst_dump, names):
    """Seed ``dst_dump``'s cache entry from ``src_dump`` for a copied subset of file names."""
    src_root = Path(src_dump).resolve()
    dst_root = Path(dst_dump).resolve()
    wanted = set(names)
    with _DUMP_CACHE_LOCK:
        src_entry = _DUMP_CACHE.get(src_root)
        if src_entry is None:
            return
        src_entries, src_texts = src_entry
        dst_entries = [
            (name, dst_root / name, size)
            for name, _, size in src_entries
            if name in wanted
        ]
        dst_texts = {
            (name, elide): text
            for (name, elide), text in src_texts.items()
            if name in wanted
        }
        _DUMP_CACHE[dst_root] = (dst_entries, dst_texts)


def list_mir_names(dump):
    """Return sorted file names in ``dump`` using the shared directory index."""
    _, (entries, _) = _get_dump_entry(dump)
    return [name for name, _, _ in entries]


def glob_mir(dump, pattern):
    """Cached equivalent of ``list(dump.glob(pattern))`` for flat MIR dump directories."""
    _, (entries, _) = _get_dump_entry(dump)
    matched = set(fnmatch.filter((name for name, _, _ in entries), pattern))
    return [path for name, path, _ in entries if name in matched]


def iter_mir_files(dump, suffix, *, prefix=""):
    """Yield ``Path`` entries in ``dump`` whose names match ``prefix`` and ``suffix``."""
    _, (entries, _) = _get_dump_entry(dump)
    for name, path, _ in entries:
        if name.endswith(suffix) and (not prefix or name.startswith(prefix)):
            yield path


def _validate_and_elide_byte_literals(path):
    """Read a large MIR file via mmap, validating and eliding huge ``const b"..."`` literals.

    Any line longer than 32 KiB must be a pure ``        _<n> = const b"...";`` assignment
    with no unescaped double quotes inside the byte literal. That preserves every MIR
    instruction, call, field projection, type, and identifier while avoiding multi-hundred-MB
    UTF-8 allocations for embedded weight blobs.
    """
    out = []
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as src:
        pos = 0
        total = len(src)
        while pos < total:
            end = src.find(b"\n", pos)
            if end < 0:
                end = total
            line_end = end - 1 if end > pos and src[end - 1] == 13 else end
            if line_end - pos > _LONG_LINE_BYTES:
                head = _LONG_BYTE_LITERAL_HEAD.match(src[pos:pos + 64])
                checked(
                    head is not None and line_end - pos >= head.end() + 2 and src[line_end - 2:line_end] == b'";',
                    f"{path.name}: unrecognized long MIR line",
                )
                quote_open = pos + head.end() - 1
                quote_close = line_end - 2
                # Verify no unescaped quote appears inside the literal and the closing quote is not escaped.
                scan = quote_open + 1
                while True:
                    q = src.find(b'"', scan, quote_close + 1)
                    bs = 0
                    idx = q - 1
                    while idx > quote_open and src[idx] == 92:  # b'\\'
                        bs += 1
                        idx -= 1
                    if q == quote_close:
                        checked(bs % 2 == 0, f"{path.name}: escaped closing quote in long MIR line")
                        break
                    checked(bs % 2 == 1, f"{path.name}: unescaped quote in long MIR line")
                    scan = q + 1
                out.append(head.group(1).decode("utf-8") + '";')
            else:
                out.append(src[pos:line_end].decode("utf-8"))
            pos = end + 1
    return "\n".join(out) + ("\n" if total else "")


def read_mir_text(path, *, elide_byte_literals=False):
    """Read a MIR file using the shared per-dump cache while respecting thread-local patches."""
    patched = get_patched_mir(path)
    if patched is not None:
        return patched
    target = Path(path)
    root, (entries, cache) = _get_dump_entry(target.parent)
    key = (target.name, bool(elide_byte_literals))
    text = cache.get(key)
    if text is not None:
        return text
    size = target.stat().st_size
    if elide_byte_literals and size > _ELIDE_SIZE_THRESHOLD:
        text = _validate_and_elide_byte_literals(target)
        cache[key] = text
        return text
    text = _ORIGINAL_READ_TEXT(target, encoding="utf-8")
    if size <= _MAX_CACHE_FILE_BYTES:
        cache[key] = text
        if elide_byte_literals and size <= _ELIDE_SIZE_THRESHOLD:
            cache[(target.name, False)] = text
    return text


def preload_mir_texts(dump, suffix, *, prefix="", elide_byte_literals=True, max_workers=8):
    """Populate the shared MIR text cache for all files matching ``prefix``/``suffix`` in parallel."""
    root, (entries, cache) = _get_dump_entry(dump)
    targets = [
        (name, path, size)
        for name, path, size in entries
        if name.endswith(suffix) and (not prefix or name.startswith(prefix))
        and (name, bool(elide_byte_literals)) not in cache
    ]
    if not targets:
        return

    def load(item):
        name, path, size = item
        if elide_byte_literals and size > _ELIDE_SIZE_THRESHOLD:
            text = _validate_and_elide_byte_literals(path)
            return name, size, text
        return name, size, _ORIGINAL_READ_TEXT(path, encoding="utf-8")

    # Start huge mmap files first so they overlap with reading the small files.
    targets.sort(key=lambda item: item[2], reverse=True)
    workers = min(max_workers, len(targets))
    if workers <= 1:
        results = [load(item) for item in targets]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(load, targets))
    with _DUMP_CACHE_LOCK:
        for name, size, text in results:
            if elide_byte_literals or size <= _MAX_CACHE_FILE_BYTES:
                cache[(name, bool(elide_byte_literals))] = text
                if elide_byte_literals and size <= _ELIDE_SIZE_THRESHOLD:
                    cache[(name, False)] = text


def iter_mir_texts(dump, suffix, *, prefix="", elide_byte_literals=True):
    """Yield ``(path, text)`` for every MIR file in ``dump`` matching ``prefix``/``suffix``."""
    root, (entries, cache) = _get_dump_entry(dump)
    overrides = _get_thread_overrides()
    by_resolved, by_name = overrides if overrides else (None, None)
    key_flag = bool(elide_byte_literals)
    for name, path, size in entries:
        if not name.endswith(suffix) or (prefix and not name.startswith(prefix)):
            continue
        if by_name and name in by_name:
            patched = by_resolved.get(path)
            if patched is not None:
                yield path, patched
                continue
        text = cache.get((name, key_flag))
        if text is None:
            text = read_mir_text(path, elide_byte_literals=key_flag)
        yield path, text


def _fast_rmtree(path):
    """Remove a flat/shallow temporary MIR dump directory quickly using parallel unlinks."""
    root = Path(path)
    if not root.exists():
        return
    files = []
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
        dirs.append(dirpath)
    if len(files) >= 64:
        with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
            list(pool.map(os.unlink, files))
    else:
        for f in files:
            os.unlink(f)
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError:
            shutil.rmtree(d, ignore_errors=True)

BLOCK = r"^    bb(\d+): \{\n((?:        [^\n]+\n)+)    \}"
CALL = r"(_\d+) = (.+)\((.*)\) -> \[return: bb(\d+), unwind: bb\d+\];"
GOTO = r"goto -> bb(\d+);|false(?:Unwind|Edge) -> \[real: bb(\d+), (?:unwind|imaginary): bb\d+\];"
ASSERT = (
    r'assert\(!((?:move|copy) \(_\d+\.1: bool\)), "attempt to compute [^"]+which would overflow", '
    r'[^\]]+\) -> \[success: bb(\d+), unwind: bb\d+\];'
)


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def blocks(source):
    result = {
        int(m[1]): [line.strip() for line in m[2].splitlines()]
        for m in re.finditer(BLOCK, source, re.M)
    }
    checked(len(result) == len(re.findall(r"^    bb\d+: \{\s*$", source, re.M)), "malformed block")
    return result


@dataclass
class Block:
    statements: list
    end: str
    cleanup: bool


def parse_blocks(path, name, signature):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checked(lines[:2] == [f"// MIR for `{name}` after {PASS}", signature], "unexpected MIR header")
    checked(lines.pop() == "}", "missing function end")
    result = {}
    unit_locals = set()
    index = 2
    while index < len(lines) and not lines[index].startswith("bb"):
        line = lines[index]
        if m := re.fullmatch(r"let (?:mut )?(_\d+): (.*);", line):
            if m[2] == "()":
                unit_locals.add(m[1])
        else:
            checked(re.fullmatch(r"debug \w+ => _\d+;|scope \d+ \{|\}", line), f"unknown declaration: {line}")
        index += 1
    while index < len(lines):
        m = re.fullmatch(r"bb(\d+)( \(cleanup\))?: \{", lines[index])
        checked(m, f"invalid block header {lines[index]}")
        bb = int(m[1])
        checked(bb not in result, f"duplicate bb{bb}")
        index += 1
        instructions = []
        while index < len(lines) and lines[index] != "}":
            line = lines[index]
            index += 1
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", line):
                continue
            if m_unit := re.fullmatch(r"(_\d+) = const \(\);", line):
                checked(m_unit[1] in unit_locals, f"unit write to non-unit local: {line}")
                continue
            instructions.append(line)
        checked(index < len(lines) and instructions, f"incomplete bb{bb}")
        result[bb] = Block(instructions[:-1], instructions[-1], bool(m[2]))
        index += 1
    checked(set(result) == set(range(len(result))), "non-contiguous basic blocks")
    return result


def normal_target(body, dest, cleanup=None):
    checked(int(dest) in body and not body[int(dest)].cleanup, f"invalid normal edge bb{dest}")
    if cleanup is not None:
        checked(body.get(int(cleanup)) == Block([], "resume;", True), f"unexpected unwind bb{cleanup}")
    return int(dest)


def split_call(term):
    return re.fullmatch(r"(_\d+) = (.+)\((.*)\) -> \[return: bb(\d+), unwind: bb(\d+)\];", term)


def const_value(dump, stem):
    """Evaluate the built MIR of an integer `const` item in `reversi_core`.

    Follows referenced constants and the straight-line arithmetic that rustc emits for
    constant expressions. Overflow and division assertions are not evaluated; rustc
    rejects a constant that violates them.
    """
    source = (dump / f"reversi_core.{stem}.built.after.mir").read_text(encoding="utf-8")
    declared = re.search(r"^const .+: ([iu](?:\d+|size)) = \{$", source, re.M)
    checked(declared, f"const {stem} is not an integer constant")
    defs = {}
    for ops in blocks(source).values():
        for op in ops:
            if m := re.fullmatch(r"(_\d+) = (.+);", op):
                checked(m[1] not in defs, f"reassigned {m[1]} in const {stem}")
                defs[m[1]] = m[2]

    def operand(text):
        if m := re.fullmatch(r"const (-?\d+)_[iu](?:\d+|size)", text):
            return int(m[1])
        if m := re.fullmatch(r"const ((?:\w+::)+\w+)", text):
            return const_value(dump, referenced_const(dump, m[1]))
        if m := re.fullmatch(r"(?:copy |move )?(_\d+)", text):
            return value(m[1])
        if m := re.fullmatch(r"(?:copy |move )?\((_\d+)\.0: [iu](?:\d+|size)\)", text):
            return value(m[1])
        raise ValueError(f"unsupported operand in const {stem}: {text}")

    def value(local):
        checked(local in defs, f"undefined {local} in const {stem}")
        rhs = defs[local]
        if m := re.fullmatch(r"(.+) as ([iu](?:\d+|size)) \(IntToInt\)", rhs):
            # Reject a cast that would wrap instead of emulating it.
            return in_range(operand(m[1]), m[2])
        if m := re.fullmatch(r"Neg\((.+)\)", rhs):
            return -operand(m[1])
        if m := re.fullmatch(r"(Add|Sub|Mul|Div|Shl)(?:WithOverflow)?\((.+), (.+)\)", rhs):
            a, b = operand(m[2]), operand(m[3])
            if m[1] == "Div":
                checked(b != 0, f"division by zero in const {stem}")
                quotient = abs(a) // abs(b)
                return quotient if (a < 0) == (b < 0) else -quotient
            return {"Add": a + b, "Sub": a - b, "Mul": a * b, "Shl": a << b}[m[1]]
        return operand(rhs)

    def in_range(number, ty):
        bits = 64 if ty[1:] == "size" else int(ty[1:])
        low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if ty[0] == "i" else (0, (1 << bits) - 1)
        checked(low <= number <= high, f"value {number} does not fit {ty} in const {stem}")
        return number

    return in_range(value("_0"), declared[1])


def referenced_const(dump, path):
    """Return the dump stem of a `reversi_core` constant named by its MIR path."""
    *parents, name = path.split("::")
    modules = []
    for part in parents:
        if not part[0].islower():
            break
        modules.append(part)
    pattern = re.compile(
        rf"reversi_core\.{re.escape('-'.join(modules))}(?:-\{{impl#\d+\}})?-{re.escape(name)}\.built\.after\.mir"
    )
    matches = [fname for fname in list_mir_names(dump) if pattern.fullmatch(fname)]
    checked(len(matches) == 1, f"cannot resolve constant {path}: {matches}")
    return matches[0].removeprefix("reversi_core.").removesuffix(".built.after.mir")


def _unlink_paths(paths):
    def _safe_unlink(p):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

    if not paths:
        return
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        list(pool.map(_safe_unlink, paths))


def _unlink_before_mir(dump):
    """Delete unused ``*.before.mir`` files asynchronously."""
    targets = [
        e.path for e in os.scandir(dump)
        if e.is_file() and e.name.endswith(".before.mir")
    ]
    _unlink_paths(targets)


_BG_CLEANERS_LOCK = threading.Lock()
_BG_CLEANERS = {}


def start_before_mir_cleanup(dump):
    """Start deleting unused ``*.before.mir`` files in ``dump`` in a background thread."""
    root = Path(dump).resolve()
    with _BG_CLEANERS_LOCK:
        if root in _BG_CLEANERS:
            return
        thread = threading.Thread(target=_unlink_before_mir, args=(root,), daemon=True)
        _BG_CLEANERS[root] = thread
        thread.start()


@contextmanager
def dump_mir(repo, filters, *, exclude_alloc_bytes=True, extra_env=None):
    if not isinstance(filters, str):
        filters = "|".join(filters)
    repo_root = Path(repo).resolve()
    verify_dir = repo_root / "target" / "verify-lean"
    verify_dir.mkdir(parents=True, exist_ok=True)
    target_dir = verify_dir / "mir-target"
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    # Disable incremental compilation for the dedicated MIR target so rustc always
    # emits fresh MIR for every function in reversi-core while reusing cached deps.
    env["CARGO_INCREMENTAL"] = "0"
    temporary = Path(tempfile.mkdtemp(prefix="reversi-mir-", dir=verify_dir))
    dump = temporary / "mir"
    try:
        cmd = [
            "cargo", "rustc", "-p", "reversi-core", "--locked", "--offline", "--lib",
            "--profile", "check", "--target-dir", str(target_dir), "--",
            f"-Zdump-mir={filters}", f"-Zdump-mir-dir={dump}",
        ]
        if exclude_alloc_bytes:
            cmd.append("-Zdump-mir-exclude-alloc-bytes")
        cmd.extend(["-Zdump-mir-exclude-pass-number", "--emit=metadata"])
        subprocess.run(cmd, cwd=repo_root, check=True, env=env)
        checked(dump.is_dir() and any(os.scandir(dump)), "rustc produced an empty MIR dump")
        yield dump
    finally:
        with _BG_CLEANERS_LOCK:
            bg_cleaner = _BG_CLEANERS.pop(dump.resolve(), None)
        if bg_cleaner is not None:
            bg_cleaner.join()
        clear_dump_cache(dump)
        _fast_rmtree(temporary)


def run_extractor_cli(
    description,
    producer,
    *,
    filters=None,
    takes_repo=True,
    allow_output=False,
    check_message="generated MIR program differs",
    check_ok_message=None,
    exclude_alloc_bytes=True,
    extra_env=None,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--mir", type=Path, required=filters is None)
    if allow_output:
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--check", type=Path)
        mode.add_argument("--output", type=Path)
    else:
        parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()

    def produce(dump):
        return producer(dump, repo) if takes_repo else producer(dump)

    if args.mir:
        generated = produce(args.mir)
    else:
        with dump_mir(
            repo, filters, exclude_alloc_bytes=exclude_alloc_bytes, extra_env=extra_env
        ) as dump:
            generated = produce(dump)

    if args.check:
        message = check_message(args.check) if callable(check_message) else check_message
        checked(args.check.read_text(encoding="utf-8") == generated, message)
        if check_ok_message:
            print(check_ok_message)
    elif allow_output and args.output:
        args.output.write_text(generated, encoding="utf-8")
    else:
        print(generated, end="")
