"""Replay real one-worker exact PVS continuations. From the repository root:

    (cd verification/lean && lake build NeuralReversi.PvsParallel.SerialCalls.TraceReplay)
    python3 verification/lean/scripts/check_serial_calls_trace.py

Requires the two normal evaluation weight files in the checkout and cached Cargo
dependencies. Only the temporary engine copy is instrumented. Pass --target-dir
to reuse a compact scratch Cargo target. Lean uses the existing build above.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from normalize_pvs_trace import require


REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "verification/lean"
FIXTURES = Path(__file__).resolve().parent / "serial_calls"


def replace_once(text, old, new):
    require(text.count(old) == 1, f"missing or ambiguous instrumentation site: {old}")
    return text.replace(old, new)


def instrument(path):
    text = path.read_text()
    begin = text.index("pub fn search<")
    end = text.index("    let all_node", begin)
    signature = text[begin:end]
    active = "let tracing = SS::IS_ENDGAME && ctx.selectivity == Selectivity::None;"
    wrapper = signature.replace("mut depth", "depth").replace("mut alpha", "alpha") + active + """
    let outer = super::context::watch_serial_context(false);
    if tracing {
        eprintln!("ENTER {} {} {} {} {} {} {} {} {}", board.player().bits(), board.opponent().bits(),
                  depth, NT::PV_NODE as u8, cut_node as u8, alpha.value(), beta.value(), NT::ROOT_NODE as u8, ctx.serial_snapshot());
    }
    let result = observed_search::<NT, SS>(ctx, board, depth, alpha, beta, thread, cut_node);
    if tracing { eprintln!("EXIT {} {}", result.value(), ctx.serial_snapshot()); }
    super::context::watch_serial_context(outer);
    result
}

"""
    before, serial = text[:begin], text[end:]
    serial, after = serial.split("pub(super) fn search_split_point", 1)
    for anchor, event in (
        ("ctx.update_pass();", '"SEG pass {} {} 0 -1 {}", alpha.value(), beta.value(), ctx.serial_snapshot()'),
        ("ctx.update(sq, flipped);", '"SEG pre {} {} 0 {} {}", alpha.value(), beta.value(), sq.index(), ctx.serial_snapshot()'),
        ("ctx.update(mv.sq, mv.flipped);", '"SEG owner {} {} {} {} {}", alpha.value(), beta.value(), '
                                           '(move_count == 1) as u8, mv.sq.index(), ctx.serial_snapshot()'),
    ):
        start = 'if tracing { eprintln!(' + event + '); super::context::watch_serial_context(true); }'
        serial = replace_once(serial, anchor, start + '\n        ' + anchor)
    stop = 'if tracing { super::context::watch_serial_context(false); eprintln!("RET {} {}", score.value(), ctx.serial_snapshot()); }'
    serial = replace_once(serial, "ctx.undo_pass();", "ctx.undo_pass();" + '\n        ' + stop)
    # The pre-split loop and the owner loop each undo once.
    require(serial.count("ctx.undo();") == 2, "expected two serial undo sites")
    serial = serial.replace("ctx.undo();", "ctx.undo();" + '\n        ' + stop)
    # This negative control reopens a move between scout and wide. Both children
    # still see identical correct contexts, but the parent ran two physical updates.
    wide = "score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);"
    reopen = """if tracing && move_count > 1 && std::env::var_os("REVERSI_TRACE_REOPEN").is_some() {
                ctx.undo();
                ctx.update(mv.sq, mv.flipped);
            }
            """
    serial = replace_once(serial, wide, reopen + wide)
    path.write_text(before + wrapper + signature.replace("pub fn search", "fn observed_search")
                    + active + "\n" + serial + "pub(super) fn search_split_point" + after)


def instrument_context(src):
    context = src / "search/context.rs"
    code = context.read_text()
    for method in ("update", "undo", "update_pass", "undo_pass"):
        signature = re.compile(rf"    pub fn {method}\([^{{]*?\) \{{.*?^    \}}", re.M | re.S)
        square = "sq.index() as i32" if method == "update" else "-1"
        note = f'        self.serial_change("{method}", {square}, before);\n    }}'
        code, hits = signature.subn(
            lambda m: m[0][:-5].replace("{", "{\n        let before = self.serial_before();", 1) + note, code)

        require(hits == 1, f"unrecognized context function {method}")
    context.write_text(code + (FIXTURES / "snapshot.rs").read_text())


def capture(folder, target, extra_instrument=None, *, fixture=None, serial_trace=True):
    core = folder / "crates/reversi-core"
    shutil.copytree(REPO / "crates/reversi-core", core)
    shutil.copytree(REPO / "crates/eval-weights", folder / "crates/eval-weights")
    manifest = (REPO / "Cargo.toml").read_text().split("[profile.release]")[0]
    manifest, count = re.subn(
        r"members = \[.*?\]",
        'members = ["crates/eval-weights", "crates/reversi-core"]',
        manifest,
        flags=re.S,
    )
    require(count == 1, "expected one workspace member list")
    (folder / "Cargo.toml").write_text(manifest + "\n[profile.dev]\ndebug = 0\nincremental = false\n")
    for name in ("Cargo.lock", "rust-toolchain.toml"):
        shutil.copy(REPO / name, folder / name)
    # The fixture opens the original weights through Eval::with_weight_files. Empty
    # compile-time placeholders avoid embedding another copy into the throwaway binary.
    for weight in REPO.glob("eval*.zst"):
        (folder / weight.name).touch()
    src = core / "src"
    if serial_trace:
        instrument(src / "search/pvs.rs")
        instrument_context(src)
    if extra_instrument is not None:
        extra_instrument(src)
    shutil.copy(fixture or FIXTURES / "fixtures.rs", src / "search/serial_calls_fixture.rs")
    with (src / "search.rs").open("a") as module:
        module.write("\nmod serial_calls_fixture;\npub use serial_calls_fixture::run as run_serial_calls_fixture;\n")
    binary = src / "bin/serial-calls-trace.rs"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text('fn main() { reversi_core::search::run_serial_calls_fixture('
                      'std::path::Path::new(&std::env::args_os().nth(1).unwrap())); }\n')
    # Cargo copies the executable to this capture while it still owns the build lock.
    artifacts = folder / "artifacts"
    subprocess.run(["cargo", "build", "--offline", "-Z", "unstable-options", "--target-dir", str(target),
                    "--artifact-dir", str(artifacts), "--bin", "serial-calls-trace"], cwd=folder, check=True)
    return artifacts / ("serial-calls-trace.exe" if os.name == "nt" else "serial-calls-trace")


def run_engine(binary, folder, mutant=None):
    env = os.environ.copy()
    env.pop("REVERSI_TRACE_REOPEN", None)
    if mutant:
        env[mutant] = "1"
    run = subprocess.run([str(binary), str(REPO)], cwd=folder, stderr=subprocess.PIPE, text=True, env=env)
    require(run.returncode == 0, f"engine exited with {run.returncode}: {run.stderr[-1500:]}")
    return run.stderr


@dataclass
class Frame:
    request: list
    entry: str
    active: str = ""
    restored: str = ""
    segment: list = field(default_factory=list)
    children: list = field(default_factory=list)


def segments(raw):
    stack, rows, context, scores, stats = [], [], [], [], Counter()
    for line in raw.splitlines():
        event, *args = line.split()
        if event == "ENTER":
            *request, root, snapshot = args
            require(len(request) == 7 and (not stack or (stack[-1].segment and root == "0")),
                    f"child outside active parent segment: {line}")
            if stack:
                parent = stack[-1]
                require(parent.active == snapshot and not parent.restored, "child issue outside physical update")
            else:
                context.append(" ".join(["root", *request[:2], snapshot]))
            stack.append(Frame(request, snapshot))
            stats["frames"] += 1
        elif event == "SEG":
            frame = stack[-1]
            role, alpha, beta, first, square, snapshot = args
            require(snapshot == frame.entry, "parent changed its context before update")
            require(not frame.segment and role in ("pass", "pre", "owner"), f"unexpected segment: {line}")
            frame.segment = [role, *frame.request[:5], first, square, alpha, beta]
        elif event == "CTX":
            method, square, prior, snapshot = args
            frame = stack[-1]
            # undo takes no square; the segment pairs it with its update.
            require(frame.segment and (method == "undo" or square == frame.segment[7]),
                    "wrong square at context operation")
            suffix = "_pass" if frame.segment[0] == "pass" else ""
            if method == "update" + suffix:
                require(not frame.active and not frame.children, "duplicate update in one segment")
                require(prior == frame.entry, "update read a different parent state")
                frame.active = snapshot
            else:
                require(method == "undo" + suffix and frame.active and frame.children and not frame.restored,
                        "unexpected operation between child issue and undo")
                require(prior == frame.active, "undo read a different child state")
                require(snapshot == frame.entry, "undo changed a context field")
                frame.restored = snapshot
        elif event == "EXIT":
            word, snapshot = args
            frame = stack.pop()
            require(not frame.segment and snapshot == frame.entry, "child returned a different context")
            scores.append(" ".join([*frame.request[:2], *frame.request[-2:], word]))
            stats["fail-high"] += int(word) >= int(frame.request[-1])
            stats["fail-low"] += int(word) <= int(frame.request[-2])
            if stack:
                stack[-1].children.append([*frame.request, word])
                stats["children"] += 1
        elif event == "RET":
            word, snapshot = args
            frame = stack[-1]
            require(frame.segment and len(frame.children) in (1, 2), "missing or extra child")
            rows.append(" ".join([*frame.segment, word, *(v for req in frame.children for v in req)]))
            require(snapshot == frame.restored, "segment returned without matching physical undo")
            context.append(" ".join(["edge", frame.segment[7], frame.entry, frame.active, frame.restored]))
            frame.active = frame.restored = ""
            stats[frame.segment[0]] += 1
            stats["re-search"] += len(frame.children) == 2
            frame.segment.clear()
            frame.children.clear()
        else:
            raise ValueError(f"unexpected engine trace: {line}")
    require(not stack and all(stats[k] for k in ("pass", "pre", "owner", "re-search", "fail-high", "fail-low")),
            f"incomplete coverage: {stats}")
    return rows, context, scores, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="Optional external scratch Cargo target cache")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-serial-calls-") as tmp:
        folder = Path(tmp)
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        binary = capture(folder, target)
        baseline = run_engine(binary, folder)
        rows, physical, scores, counts = segments(baseline)
        oracle = folder / "minimax"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES / "oracle.rs"), "-o", str(oracle)],
                       cwd=REPO, check=True)
        subprocess.run([str(oracle)], input="\n".join(scores) + "\n", text=True, check=True)
        transcript = folder / "segments.txt"
        transcript.write_text("\n".join(rows) + "\n", newline="\n")
        driver = folder / "Replay.lean"
        driver.write_text("import NeuralReversi.PvsParallel.SerialCalls.TraceReplay\n"
                          "def main := NeuralReversi.PvsParallel.SerialCalls.TraceReplay.run\n")
        subprocess.run(["lake", "env", "lean", "--run", str(driver), str(transcript)],
                       cwd=LEAN, check=True)
        physical_trace = folder / "contexts.txt"
        physical_trace.write_text("\n".join(physical) + "\n", newline="\n")
        driver.write_text("import NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context\n"
                          "def main := NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context.run\n")
        subprocess.run(["lake", "env", "lean", "--run", str(driver), str(physical_trace)],
                       cwd=LEAN, check=True)
        for fault, expected in (
            ("REVERSI_TRACE_REOPEN", "duplicate update in one segment"),
        ):
            altered = run_engine(binary, folder, mutant=fault)
            # These mutations leave request/return scores and chronology unchanged.
            # Only the physical context observations distinguish the two engine runs.
            def calls_only(stream):
                return [line.rsplit(" ", 1)[0] for line in stream.splitlines() if not line.startswith("CTX ")]
            require(calls_only(altered) == calls_only(baseline), f"{fault} changed the numeric search trace")
            try:
                segments(altered)
            except ValueError as err:
                require(str(err) == expected, f"unexpected mutant failure: {err}")
            else:
                raise ValueError(f"accepted native mutation {fault}")
        print("Rejected a duplicate update before wide re-search.")
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
