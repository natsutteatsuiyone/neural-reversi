"""Shared helpers for MIR mutation and regression checks."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import tempfile
from unittest.mock import patch

from mir_common import checked


def replace_once(source, old, new, *, label="", unique=True):
    """Apply a single mutation replacement, verifying the target text is present."""
    if unique:
        checked(source.count(old) == 1, f"missing or ambiguous mutation {label}: {old!r}")
    else:
        checked(old in source, f"missing mutation {label}: {old!r}")
    return source.replace(old, new, 1)


@contextmanager
def patched_mir(path, text):
    """Temporarily override ``Path.read_text`` for ``path`` with ``text``."""
    original_read = Path.read_text
    target = Path(path)
    resolved = target.resolve()

    def read(self, *args, **kwargs):
        if self == target or (self.name == target.name and self.resolve() == resolved):
            return text
        return original_read(self, *args, **kwargs)

    with patch.object(Path, "read_text", read):
        yield


def assert_mutants_rejected(
    dump,
    cases,
    run,
    *,
    baseline=None,
    expected_errors=(ValueError,),
    unique=True,
    error_check=None,
):
    """Run each ``(filename, old, new[, label])`` mutant and require rejection.

    A mutant is rejected if ``run(dump)`` raises one of ``expected_errors`` (and passes
    ``error_check(error, label)`` if given), or—when ``baseline is not None``—returns a
    value different from ``baseline``.
    """
    cache = {}
    for case in cases:
        if len(case) == 3:
            filename, old, new = case
            label = filename
        else:
            filename, old, new, label = case
        path = dump / filename
        if path not in cache:
            cache[path] = path.read_text(encoding="utf-8")
        mutated = replace_once(cache[path], old, new, label=label, unique=unique)
        with patched_mir(path, mutated):
            try:
                result = run(dump)
            except expected_errors as error:
                if error_check is not None:
                    error_check(error, label)
                continue
        if baseline is not None:
            checked(result != baseline, f"accepted mutation ({label}): {old}")
        else:
            raise ValueError(f"accepted mutation ({label}): {old}")
    return len(cases)


def check_mir_projection(
    dump,
    expected,
    render,
    suffix,
    cases,
    *,
    file_filter,
    expected_errors=(ValueError, KeyError, IndexError),
):
    """Render the filtered ``dump`` files as ``expected`` and reject every mutant.

    Returns the number of mutants.
    """
    with tempfile.TemporaryDirectory() as folder:
        copied = Path(folder)
        for path in dump.glob("*" + suffix):
            if file_filter(path):
                (copied / path.name).write_bytes(path.read_bytes())
        checked(render(copied) == expected, "the filtered MIR renders a different program")
        return assert_mutants_rejected(
            copied,
            [(name + suffix, old, new, name) for name, old, new in cases],
            render,
            baseline=expected,
            expected_errors=expected_errors,
            unique=False,
        )


def check_mir_regression(
    dump,
    repo,
    verify,
    cases,
    *,
    expected_errors=(ValueError, KeyError),
    compare_baseline=True,
):
    """Verify ``dump`` and reject every mutant. Returns the number of mutants."""
    baseline = verify(dump, repo)
    return assert_mutants_rejected(
        dump,
        cases,
        lambda d: verify(d, repo),
        baseline=baseline if compare_baseline else None,
        expected_errors=expected_errors,
    )


def run_mir_test_cli(description, check, summary):
    """CLI runner for ``test_*_mir.py`` scripts: ``check(dump, expected)``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mir", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    args = parser.parse_args()
    count = check(args.mir, args.check.read_text(encoding="utf-8"))
    print(summary.format(count=count))


def run_mir_regression_cli(description, check, summary, *, positional_dump=False, default_repo=None):
    """CLI runner for standalone ``check_*_{regression,mutants}.py`` scripts: ``check(dump, repo)``."""
    parser = argparse.ArgumentParser(description=description)
    if positional_dump:
        parser.add_argument("dump", type=Path)
    else:
        parser.add_argument("--mir", dest="dump", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=default_repo or Path.cwd())
    args = parser.parse_args()
    print(summary.format(count=check(args.dump.resolve(), args.repo.resolve())))
