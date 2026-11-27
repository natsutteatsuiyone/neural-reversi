"""Extract normal worker CFG edges and tag iterator, lock, stop and loop-exit sites."""

import re

from mir_common import CALL, GOTO, blocks, checked
from extract_split_worker_mir import MIR, SIGNATURE


def render(dump, repo):
    source = (dump / MIR).read_text(encoding="utf-8")
    checked(SIGNATURE in source.splitlines(), "split worker signature")
    cfg = blocks(source)
    calls = {pc: m for pc, lines in cfg.items() if (m := re.fullmatch(CALL, lines[-1]))}

    def unique(items, label):
        values = list(items)
        checked(len(values) == 1, f"expected one {label}, found {values}")
        return values[0]

    def call(name):
        pc = unique((k for k, m in calls.items() if m[2] == name), name)
        return pc, calls[pc]

    def switch(pc, guard, zero):
        match = re.fullmatch(r"switchInt\(move " + guard + r"\) -> \[([01]): bb(\d+), otherwise: bb(\d+)\];", cfg[pc][-1])
        checked(match and int(match[1]) == zero, f"unrecognized branch bb{pc} on {guard}")
        return int(match[2]), int(match[3])

    next_at, poll = call("ConcurrentMoveIterator::next")
    guard_at, lock = call("SplitPoint::lock")
    stop_at, stop = call("threading::thread::Thread::should_stop")
    case_at = int(poll[4])
    discr = unique((m[1] for line in cfg[case_at][:-1] if (m := re.fullmatch(r"(_\d+) = discriminant\(" + poll[1] + r"\);", line))), "iterator discriminant")
    hit, empty = switch(case_at, discr, 1)
    stopped_at = int(stop[4])
    still, stopped = switch(stopped_at, stop[1], 0)
    cutoff = unique(re.findall(r"\bdebug cutoff => (_\d+);", source), "cutoff binding")
    final = {}
    for pc, lines in cfg.items():
        for line in lines[:-1]:
            alias = re.fullmatch(r"(_\d+) = copy " + cutoff + r";", line)
            if alias and lines[-1].startswith(f"switchInt(move {alias[1]})"):
                targets = switch(pc, alias[1], 0)
                if all(cfg[t][-1].startswith(f"drop({lock[1]}) -> ") for t in targets):
                    final[pc] = targets
    cutoff_at = unique(final, "final cutoff test dropping the live guard in both arms")
    # The score/field extractor separately follows the operands and both drops.
    special = {
        case_at: ("next", [hit, empty]), guard_at: ("lock", [int(lock[4])]),
        stopped_at: ("stop", [still, stopped]), cutoff_at: ("cutoff", list(final[cutoff_at])),
    }
    checked(len(special) == 4 and next_at != stop_at, "overlapping control sites")

    def instruction(pc, line):
        if pc in special:
            return special[pc]
        if m := re.fullmatch(GOTO, line):
            return "jump", [int(m[1] or m[2])]
        if pc in calls:
            return "jump", [int(calls[pc][4])]
        if m := re.fullmatch(r"switchInt\([^)]*\) -> \[(?:0|1): bb(\d+), otherwise: bb(\d+)\];", line):
            return "jump", [int(m[1]), int(m[2])]
        if m := re.search(r" -> \[(?:return|success): bb(\d+), unwind: bb\d+\];$", line):
            checked(line.startswith(("drop(", "assert(")), f"unsupported normal edge at bb{pc}")
            return "jump", [int(m[1])]
        if line == "return;":
            return "done", []
        checked(re.fullmatch(r'_\d+ = (?:core::panicking::)?panic\(const ".+"\) -> bb\d+;', line), f"unknown terminator at bb{pc}: {line}")
        return "jump", []

    nodes = {pc: instruction(pc, lines[-1]) for pc, lines in cfg.items()}
    size = len(nodes)
    checked(set(nodes) == set(range(size)), "noncontiguous normal blocks")
    phases = {}
    todo = [(0, "poll")]
    while todo:
        pc, phase = todo.pop()
        if pc in phases:
            checked(phase == phases[pc], f"inconsistent phase at bb{pc}")
            continue
        phases[pc] = phase
        op, successors = nodes[pc]
        expected = {"next": "poll", "lock": "issued", "stop": "locked", "cutoff": "locked", "done": "done"}
        if op in expected:
            checked(phase == expected[op], f"{op} in {phase} at bb{pc}")
        outs = {"next": ["issued", "done"], "lock": ["locked"], "stop": ["locked", "done"],
                "cutoff": ["poll", "done"]}.get(op, [phase]*len(successors))
        todo.extend(zip(successors, outs, strict=True))
    checked(len(phases) == size, "unclassified normal block")
    lines=["import NeuralReversi.PvsParallel.WorkerPollMir.Syntax", "",
           "namespace NeuralReversi.PvsParallel.WorkerPollMir", "",
           f"abbrev Block := Fin {size}","",f"def graph : Vector (Op {size} × Phase) {size} := #v["]
    for pc in range(size):
        op, succ = nodes[pc]
        args = (" ["+", ".join(map(str, succ))+"]") if op=="jump" else "".join(f" {i}" for i in succ)
        lines.append(f"  (.{op}{args}, .{phases[pc]})"+("," if pc+1<size else "]"))
    lines.extend(["",f"def code (pc : Fin {size}) := graph[pc].1", f"def phase (pc : Fin {size}) := graph[pc].2","",
                  "end NeuralReversi.PvsParallel.WorkerPollMir",""])
    return "\n".join(lines)

