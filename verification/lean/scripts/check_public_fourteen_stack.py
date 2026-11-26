"""Compare all five public search levels and their physical TT publications on a cold fourteen-empty board."""

import argparse
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_endgame_probcut_trace import instrument_decisions
from check_serial_cache_stack import compare
from check_serial_cache_trace import instrument as instrument_cache
from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from check_serial_table_trace import instrument_slots
from normalize_pvs_trace import require


def instrument(src):
    instrument_cache(src)
    instrument_slots(src)
    instrument_decisions(src)
    table = src / "transposition_table.rs"
    with table.open("a") as stream:
        stream.write((FIXTURES / "table_image.rs").read_text())
    pvs = src / "search/pvs.rs"
    s = pvs.read_text()
    flag = "let tracing = SS::IS_ENDGAME && ctx.selectivity == Selectivity::None;"
    require(s.count(flag) == 2, "unexpected frame instrumentation")
    s = s.replace(flag, "let tracing = SS::IS_ENDGAME;", 1).replace(flag, "let tracing = false;")
    entry = "let result = observed_search::<NT, SS>(ctx, board, depth, alpha, beta, thread, cut_node);"
    s = replace_once(s, entry, entry + "\n if tracing && NT::ROOT_NODE { ctx.tt.serial_dump_image("
                     + "ctx.selectivity == Selectivity::None && alpha < result && result < beta); }")
    rows = "move_list.iter().map(|m| (m.sq.index(), m.value)).collect::<Vec<_>>()"
    sort = "move_list.sort();"
    s = replace_once(s, sort, f'if SS::IS_ENDGAME {{ eprintln!("KEYS {{:?}}", {rows}); }}\n'
                     + sort + f'\n if SS::IS_ENDGAME {{ eprintln!("SORTED {{:?}}", {rows}); }}')
    pvs.write_text(s)
    ordering = src / "move_list/ordering.rs"
    s = ordering.read_text()
    for prefix, mode in (("if use_fast {", "fast"), ("if skip_search_ordering {", "skip")):
        s = replace_once(s, prefix, prefix + f'\n if SS::IS_ENDGAME {{ eprintln!("MODE {mode}"); }}')
    search = "let use_cached_child_eval ="
    s = replace_once(s, search, 'if SS::IS_ENDGAME { eprintln!("MODE search"); }\n        ' + search)
    mark = "ctx.undo();"
    s = replace_once(s, mark, mark + '\n if SS::IS_ENDGAME { eprintln!("EST {} {}", mv.sq.index(), mv.value); }')
    ordering.write_text(s)
    end = src / "search/endgame.rs"
    base = "let base_score = estimate_aspiration_base_score(&mut ctx, &board);"
    static = "    midgame::evaluate(ctx, board)\n}"
    evaluated = replace_once(end.read_text(), static, '    let value = midgame::evaluate(ctx, board);\n'
                             + '    eprintln!("SEED_EVAL {}", value.value());\n    value\n}')
    end.write_text(replace_once(evaluated, base, base + '\n eprintln!("BASE {} {} {} {}", '
                                + "board.player().bits(), board.opponent().bits(), ctx.tt.generation(), base_score.value());"))


def normalize(rows, exact_bounds=None, frame_events=None):
    stack, operands, out = [], [], defaultdict(list)
    tables, base, level, trial, estimate = {}, None, None, None, None
    seed_eval = verified = pending = keys = None
    issued = stores = images = 0
    def output(tag, *fields):
        row = " ".join(map(str, [tag, *fields]))
        out[tag].append(row)
        if frame_events is not None and tag in ("ROW", "PUBLISH"):
            frame_events.append(row)
    def operand(tag, *fields):
        operands.append(" ".join(map(str, [tag, *fields])))
    for line in rows:
        tag, *v = line.split()
        if tag == "ALLOC":
            if v[0] in ("Deep", "Shallow"):
                tables[v[0]] = int(v[2])
            else:
                base, tables["tt"], stride = map(int, v)
        elif tag == "SEED_EVAL":
            require(seed_eval is None, "duplicate cold root evaluator")
            seed_eval = int(v[0])
        elif tag == "BASE":
            p, o, gen, score = map(int, v)
            require(tables["tt"] == 16 and gen == 1, "fixture TT generation or layout changed")
            require(seed_eval == score, "cold root estimate did not return the observed evaluation")
            operand("CONFIG", p, o, gen, tables["tt"], tables["Deep"], tables["Shallow"], seed_eval)
        elif tag == "RUNG":
            require(level is None and not stack and v[3] == "0", "unexpected public root line")
            level = int(v[4])
        elif tag == "RUNG_END":
            require(not stack and int(v[4]) == level and trial, "incomplete public rung")
            output("RUNG", level, trial)
            level = trial = None
        elif tag == "ENTER":
            require(level, "PVS call outside a rung")
            stack.append((issued, v[:7]))
            issued += 1
        elif tag == "EXIT":
            i, req = stack.pop()
            if level == 5 and exact_bounds is not None:
                exact_bounds.append(" ".join([*req[:2], *req[5:7], v[0]]))
            if stack:
                output("ROW", i, *req, v[0], stores)
            else:
                output("ROOT", level, i, issued, *req[5:7], v[0], stores)
                trial = v[0]
                require(images == 32, "missing trial snapshot cells")
                images = 0
        elif tag == "MODE":
            operand(tag, stack[-1][0], *v)
        elif tag == "EST":
            operand(tag, stack[-1][0], *v)
        elif tag == "TRY":
            require(v[:2] == stack[-1][1][:2] and int(v[3]) == level, "ProbCut crossed a frame")
        elif tag == "WINDOW":
            estimate = v[0]
        elif tag == "VERIFIED":
            verified, = v
        elif tag == "RESULT":
            require(estimate is not None, "missing calibrated decision")
            operand("PC", stack[-1][0], estimate, verified if verified is not None else "-")
            estimate = verified = None
        elif tag == "KEYS":
            keys = ast.literal_eval(line.split(" ", 1)[1])
        elif tag == "SORTED":
            sorted_moves = ast.literal_eval(line.split(" ", 1)[1])
            require(keys is not None and sorted(keys) == sorted(sorted_moves)
                    and sorted_moves == sorted(sorted_moves, key=lambda m:-m[1]), "native ordering changed key")
            require(sorted_moves == sorted(keys, key=lambda row:(-row[1],row[0])),
                    "fixture's equal-priority moves no longer follow ascending square")
            keys = None
        elif tag == "STORE":
            require(pending is None and int(v[0]) == base, "untracked shared table owner")
            index, offset = divmod(int(v[1])-base, stride)
            require(offset == 0 and 0 <= index < 32, "invalid physical slot")
            pending = index, v[-4]
        elif tag == "AFTER":
            index, before = pending
            if v[0] != before:
                output("PUBLISH", index, *v)
            pending = None
        elif tag == "TTC":
            require(int(v[0]) == images, "reordered physical snapshot")
            images += 1
            output(tag, *v)
        elif tag == "SAVE":
            output("STORE", v[0], *v[5:9])
            stores += 1
        elif tag == "PUBLIC":
            output(tag, *v)
        else:
            require(tag in ("PROBE", "LOOK", "LOAD", "FREE", "BORROW", "WORKER", "DISABLED", "USED"),
                    f"unexpected event {tag}")
    require(not stack and not pending and level is None and len(out["RUNG"]) == 5 and len(out["PUBLIC"]) == 1,
            "incomplete public solve")
    return operands, out


def build(work):
    runner = work / "lean"
    runner.mkdir()
    project = REPO / "verification/lean"
    shutil.copy(project/"lean-toolchain", runner/"lean-toolchain")
    shutil.copy(FIXTURES/"PublicFourteen.lean.in", runner/"PublicFourteen.lean")
    (runner/"lakefile.toml").write_text(
        'name = "publicFourteenComparison"\n[[require]]\nname = "neuralReversi"\npath = '
        + json.dumps(str(project)) + '\n[[lean_exe]]\nname = "compare"\nroot = "PublicFourteen"\n')
    subprocess.run(["lake", "build", "compare"], cwd=runner, check=True)
    return runner/".lake/build/bin/compare"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", type=Path)
    p.add_argument("--artifacts", type=Path, help="Keep raw trace, model operands and outputs")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compare", type=Path, help="Reuse a prebuilt comparison executable")
    mode.add_argument("--capture-only", action="store_true", help="Write captures without building or running Lean")
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-public-fourteen-") as tmp:
        work = Path(tmp)
        artifacts = args.artifacts or work
        artifacts.mkdir(exist_ok=True, parents=True)
        binary = capture(work, args.target_dir.resolve() if args.target_dir else work / "target",
                         instrument, fixture=FIXTURES / "public_fourteen.rs")
        raw = artifacts / "native.trace"
        env = {k:v for k,v in os.environ.items() if not k.startswith("REVERSI_TRACE_") and k != "REVERSI_TABLE_FAULT"}
        with raw.open("w") as stream:
            subprocess.run([str(binary), str(REPO)], env=env, stderr=stream, check=True)
        bounds, frame_events = [], []
        with raw.open() as stream:
            operands, expected = normalize(stream, bounds, frame_events)
        path = artifacts / "operands.txt"
        path.write_text("\n".join(operands)+"\n", newline="\n")
        (artifacts / "expected.txt").write_text("\n".join(row for rows in expected.values() for row in rows)+"\n")
        print(operands[0], dict(Counter(row.split()[0] for row in operands)), {k:len(v) for k,v in expected.items()})
        if args.capture_only:
            return
        oracle = work/"minimax"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES/"oracle.rs"), "-o", str(oracle)],
                       cwd=REPO, check=True)
        subprocess.run([str(oracle)], input="\n".join(bounds)+"\n", text=True, check=True)
        exe = args.compare.resolve() if args.compare else build(work)
        summary = artifacts / "model.trace"
        frames = artifacts / "model-frames.trace"
        # Lean writes nested returns through dbgTrace and completed trials through IO.println.
        # Their buffers may flush in the middle of a line if both share an output file.
        with summary.open("w") as trials, frames.open("w") as nested:
            subprocess.run([str(exe), str(path.resolve())], stdout=trials, stderr=nested, check=True)
        actual = defaultdict(list)
        for trace in (summary, frames):
            with trace.open() as stream:
                for row in stream:
                    row = row.rstrip()
                    parts = row.split()
                    require(parts and parts[0] in expected, f"unexpected Lean output {row}")
                    actual[parts[0]].append(row)
        compare("chronological child returns and TT publications", frames.read_text().splitlines(), frame_events)
        for tag in expected:
            compare(tag, actual[tag], expected[tag])
        for fault in ("move", "sequence"):
            trace = subprocess.run([str(binary), str(REPO)], env=env | {"REVERSI_TRACE_IMAGE_FAULT":fault},
                                   text=True, stderr=subprocess.PIPE, check=True)
            same_inputs, mutated = normalize(trace.stderr.splitlines())
            require(same_inputs == operands, "physical mutation changed an upstream model operand")
            for tag in expected:
                if tag != "TTC":
                    compare(tag, actual[tag], mutated[tag])
            before, after = actual["TTC"], mutated["TTC"]
            changed = [(a.split(),b.split()) for a,b in zip(before, after, strict=True) if a != b]
            require(len(changed) == 1, f"failed to isolate final {fault} corruption")
            a, b = changed[0]
            if fault == "move":
                require(a[:5] == b[:5] and int(a[5]) ^ int(b[5]) == 1 << (16 if sys.byteorder == "little" else 40),
                        "unexpected best-move mutation")
            else:
                require(a[:2] == b[:2] and a[3:] == b[3:] and int(b[2]) == int(a[2])-2,
                        "unexpected sequence mutation")
        print("All seven output streams matched. Rejected changed final best-move and sequence fields.")


if __name__ == "__main__":
    main()
