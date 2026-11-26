#!/usr/bin/env python3
"""Compare a cold seven-empty Rust aspiration solve with the state-threaded Lean interpreter."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_serial_cache_trace import MISS, instrument as instrument_cache
from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from check_serial_table_trace import instrument_slots
from normalize_pvs_trace import require

LEAN = REPO / "verification/lean"


def instrument(src):
    instrument_cache(src)
    instrument_slots(src)
    with (src / "transposition_table.rs").open("a") as stream:
        stream.write((FIXTURES / "table_image.rs").read_text())
    source = src / "search/endgame.rs"
    source.write_text(replace_once(source.read_text(), "fn aspiration_search(", "pub(super) fn aspiration_search("))
    source = src / "search/pvs.rs"
    code = source.read_text()
    probe = "    let tt_move = tt_probe_result.best_move();"
    code = replace_once(code, probe, probe + """
    if tracing { eprintln!("HINT {}", tt_move.map(|sq| sq.index() as i64).unwrap_or(-1)); }""")
    sort = "        move_list.sort();"
    # Only initialized squares and ordering keys are read around the real sort.
    rows = "move_list.iter().map(|m| (m.sq.index(), m.value)).collect::<Vec<_>>()"
    code = replace_once(code, sort, f'        if tracing {{ eprintln!("KEY {{:?}}", {rows}); }}\n'
                        + sort + f'\n        if tracing {{ eprintln!("SORTED {{:?}}", {rows}); }}')
    call = "    let result = observed_search::<NT, SS>(ctx, board, depth, alpha, beta, thread, cut_node);"
    code = replace_once(code, call, call + "\n    if tracing && NT::ROOT_NODE { ctx.tt.serial_dump_image(alpha < result && result < beta); }")
    source.write_text(code)


def native(lines):
    stack, stores, roots, children = [], [], [], []
    image, publications, published_slots = [], [], []
    pending = table = None
    last_writer, hint_values = {}, {}
    key_queries, expected_keys, bounds = [], [], []
    issued = loads = cross_trial_hits = 0
    bases, sizes, owners = {}, {}, set()
    for line in lines:
        item = line.split()
        if item[0] == "ALLOC" and item[1] not in ("Deep", "Shallow"):
            require(table is None and item[2:] == ["16", "32"], "unexpected TT allocation")
            table = int(item[1]), int(item[3])
        elif item[0] == "ALLOC":
            require(item[1] not in bases, "worker allocated another table of the same kind")
            bases[item[1]] = item[2]
            sizes[item[1]] = int(item[3])
        elif item[0] == "WORKER":
            owners.add(item[1])
        elif item[0] == "BORROW":
            require(item[1] in owners and item[2:] == [bases["Deep"], bases["Shallow"]],
                    "cache lease changed worker or allocation")
        elif item[0] == "ENTER":
            stack.append((issued, item[1:8]))
            issued += 1
        elif item[0] == "HINT" and item[1] != "-1":
            opening, request = stack[-1]
            require(request[3] == "1", "unexpected TT lookup on a specialized scout")
            hint_values[opening] = item[1]
        elif item[0] == "KEY":
            keys = ast.literal_eval(line.split(" ", 1)[1])
            opening, request = stack[-1]
            for square, key in keys:
                key_queries.append(" ".join([*request[:2], hint_values.get(opening, "-"), str(square)]))
                expected_keys.append(f"KEY {key}")
        elif item[0] == "SORTED":
            sorted_moves = ast.literal_eval(line.split(" ", 1)[1])
            require(sorted_moves == sorted(keys, key=lambda pair: (-pair[1], pair[0])),
                    "the observed library sort has a different tie order on this fixture")
        elif item[0] == "STORE":
            require(pending is None and table is not None and int(item[1]) == table[0], "incorrect TT owner")
            index, offset = divmod(int(item[2]) - table[0], table[1])
            require(offset == 0 and 0 <= index < 32, "unallocated TT slot")
            pending = index, item[-4]
        elif item[0] == "AFTER":
            require(pending is not None, "orphan shared-table publication")
            index, prior_seq = pending
            if item[1] != prior_seq:
                publications.append(" ".join(["TTW", *item[2:5]]))
                published_slots.append(" ".join(["PUBLISH", str(index), *item[1:5]]))
            pending = None
        elif item[0] == "TTC":
            image.append(line)
        elif item[0] == "SAVE":
            require(item[2] == bases[item[1]], "store used another allocation")
            last_writer[item[1], item[4]] = len(roots)
            stores.append(" ".join(["STORE", item[1], *item[6:10]]))
        elif item[0] == "LOAD":
            require(item[2] == bases[item[1]], "probe used another allocation")
            loads += 1
            if int(item[9]) != MISS and last_writer[item[1], item[4]] < len(roots):
                cross_trial_hits += 1
        elif item[0] == "EXIT":
            opening, request = stack.pop()
            bounds.append(" ".join([*request[:2], *request[5:7], item[1]]))
            if stack:
                children.append(" ".join(["ROW", str(opening), *request, item[1], str(len(stores))]))
            else:
                roots.append(" ".join(["ROOT", str(opening), str(issued - 1), *request[5:7], item[1], str(len(stores))]))
    require(len(owners) == 1, "more than one physical worker entered PVS")
    require(not stack and len(roots) > 1 and cross_trial_hits > 0, "expected a cache hit from a failed prior trial")
    require(sizes == {"Deep": 116508, "Shallow": 7281}, f"unexpected worker table lengths {sizes}")
    require(pending is None and len(image) == 32 * len(roots), "incomplete physical table image")
    expected = {"ROOT": roots, "ROW": children, "STORE": stores, "KEY": expected_keys,
                "TTW": publications, "PUBLISH": published_slots, "TTC": image}
    return expected, key_queries, bounds, loads, cross_trial_hits


def compare(label, actual, expected):
    for i, (got, want) in enumerate(zip(actual, expected)):
        require(got == want, f"{label}[{i}]\nLean {got}\nRust {want}")
    require(len(actual) == len(expected), f"{label} length {len(actual)} vs {len(expected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="reusable Cargo target directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="serial-cache-stack-") as tmp:
        folder = Path(tmp)
        engine = capture(folder, (args.target_dir or folder / "target").resolve(),
                         instrument, fixture=FIXTURES / "cache_stack.rs")
        env = {k: v for k, v in os.environ.items() if not k.startswith("REVERSI_TRACE_")}
        env.pop("REVERSI_TABLE_FAULT", None)

        def run_native(overrides):
            trace = subprocess.run([str(engine), str(REPO)], cwd=folder, env=env | overrides, text=True,
                                   stderr=subprocess.PIPE, check=True)
            return native(trace.stderr.splitlines())

        expected, key_queries, bounds, loads, warm = run_native({})
        keys = folder / "keys.txt"
        keys.write_text("\n".join(key_queries) + "\n")
        # Build a native Lean executable against the current project's local artifacts.
        runner = folder / "runner"
        runner.mkdir()
        shutil.copy(LEAN / "lean-toolchain", runner)
        shutil.copy(FIXTURES / "CacheStackComparison.lean.in", runner / "CacheStackComparison.lean")
        (runner / "lakefile.toml").write_text(
            'name = "serialComparison"\n[[require]]\nname = "neuralReversi"\npath = ' + json.dumps(str(LEAN))
            + '\n[[lean_exe]]\nname = "compare"\nroot = "CacheStackComparison"\n')
        subprocess.run(["lake", "build", "compare"], cwd=runner, check=True)
        result = subprocess.run([str(runner / ".lake/build/bin/compare"), str(keys)], cwd=runner,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        actual = {tag: [row for row in result.stdout.splitlines() if row.startswith(tag + " ")] for tag in expected}
        events = result.stderr.splitlines()
        for tag in ("ROW", "PUBLISH"):
            actual[tag] = [row for row in events if row.startswith(tag + " ")]
        require(len(events) == len(actual["ROW"]) + len(actual["PUBLISH"]), "unexpected Lean execution event")
        for tag, wanted in expected.items():
            compare(tag, actual[tag], wanted)

        for fault in ("move", "sequence"):
            challenged, _, _, _, _ = run_native({"REVERSI_TRACE_IMAGE_FAULT": fault})
            for tag in expected:
                if tag != "TTC":
                    compare(tag, actual[tag], challenged[tag])
            try:
                compare("TTC", actual["TTC"], challenged["TTC"])
            except ValueError:
                mismatches = [(x.split(), y.split()) for x, y in zip(actual["TTC"], challenged["TTC"]) if x != y]
                require(len(mismatches) == 1, f"{fault} should corrupt one physical cell")
                a, z = mismatches[0]
                if fault == "sequence":
                    require(a[:2] == z[:2] and a[3:] == z[3:] and int(z[2]) == int(a[2]) - 2,
                            "did not detect a stale sequence")
                else:
                    move_bit = 16 if sys.byteorder == "little" else 40
                    require(a[:5] == z[:5] and int(a[5]) ^ int(z[5]) == 1 << move_bit,
                            "did not detect a changed packed best move")
            else:
                raise ValueError(f"accepted {fault} mutation")
        oracle = folder / "disk-difference"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES / "oracle.rs"), "-o", str(oracle)], check=True)
        subprocess.run([str(oracle)], input="\n".join(bounds) + "\n", text=True, check=True)
        print(f"Matched {len(expected['ROOT'])} root trials, {len(actual['ROW'])} nested PVS returns, "
              f"{len(actual['KEY'])} initialized ordering keys, {len(actual['PUBLISH'])} TT writes with slots and sequence, "
              f"{len(actual['TTC'])} raw TT slots and {len(actual['STORE'])} worker cache writes. "
              f"Rust made {loads} worker-cache probes including {warm} hits from earlier trials. "
              "Rejected changed packed-move and stale-sequence images.")


if __name__ == "__main__":
    main()
