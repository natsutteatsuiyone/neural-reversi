"""Shared MIR parsing primitives, compiler dump runner, and extractor CLI."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile


PASS = "SimplifyCfg-initial"

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
    matches = [p.name for p in dump.iterdir() if pattern.fullmatch(p.name)]
    checked(len(matches) == 1, f"cannot resolve constant {path}: {matches}")
    return matches[0].removeprefix("reversi_core.").removesuffix(".built.after.mir")


@contextmanager
def dump_mir(repo, filters, *, exclude_alloc_bytes=True, extra_env=None):
    if not isinstance(filters, str):
        filters = "|".join(filters)
    env = dict(os.environ, **extra_env) if extra_env else None
    with tempfile.TemporaryDirectory() as temporary:
        dump = Path(temporary) / "mir"
        cmd = [
            "cargo", "rustc", "-p", "reversi-core", "--locked", "--offline", "--lib",
            "--profile", "check", "--target-dir", temporary, "--",
            f"-Zdump-mir={filters}", f"-Zdump-mir-dir={dump}",
        ]
        if exclude_alloc_bytes:
            cmd.append("-Zdump-mir-exclude-alloc-bytes")
        cmd.extend(["-Zdump-mir-exclude-pass-number", "--emit=metadata"])
        subprocess.run(cmd, cwd=repo, check=True, env=env)
        yield dump


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
