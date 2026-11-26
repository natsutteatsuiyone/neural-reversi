"""Replay a four-worker exact solve's split lifetimes against the Lean ticket machine."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require

FIXTURES = Path(__file__).resolve().parent / "serial_calls"


def instrument(src):
    thread = src / "search/threading/thread.rs"
    s = thread.read_text()
    anchor = "        let move_iter = ConcurrentMoveIterator::from_offset(req.move_list, req.move_count);"
    s = replace_once(s, anchor, """
        let len = req.move_list.count();
        let parent = task.parent_split_point.as_ref().map_or(0, |sp| Arc::as_ptr(sp) as usize);
""" + anchor)
    anchor = "            sp_state.helpers_mask.clear();"
    s = replace_once(s, anchor, """
            println!("S {} begin {} {} {} {} {} {} {} {} {} {} {}", Arc::as_ptr(sp) as usize, len, req.move_count, self.idx, parent,
                (req.node_type != NodeTypeId::NonPv) as u8, (req.node_type == NodeTypeId::Root) as u8,
                req.cut_node as u8, req.depth, req.beta.value(), req.alpha.value(), req.best_score.value());
""" + anchor)
    anchor = "        sp_state.helpers_mask.set(self.idx);\n        *self.active_split_point_mut()"
    s = replace_once(s, anchor, """
        println!("S {} book {}", Arc::as_ptr(sp) as usize, self.idx);
""" + anchor)
    anchor = "                self.book_into(sp, sp_state);"
    s = replace_once(s, anchor, """
                println!("S {} late {}", Arc::as_ptr(sp) as usize, self.idx);
""" + anchor)
    anchor = "            sp_state.helpers_mask.reset(self.idx);"
    s = replace_once(s, anchor, """
            println!("S {} leave {}", Arc::as_ptr(&sp) as usize, self.idx);
""" + anchor)
    anchor = "                && sp.state().helpers_mask.none()\n            {\n                break;"
    s = replace_once(s, anchor, anchor.replace("                break;", """
                println!("S {} zero {}", Arc::as_ptr(sp) as usize, self.idx);
                break;"""))
    anchor = "        (sp_state.best_score(), sp_state.best_move(), counters)"
    s = replace_once(s, anchor, """
        let returned = sp_state.best_score();
        println!("S {} returned {} {}", Arc::as_ptr(sp) as usize, self.idx, returned.value());
        (returned, sp_state.best_move(), counters)""")
    thread.write_text(s)

    pvs = src / "search/pvs.rs"
    head, body = pvs.read_text().split("pub(super) fn search_split_point", 1)
    body, tail = body.split("/// Performs enhanced transposition cutoff", 1)
    anchor = "            if thread.should_stop_after_split() {"
    head = replace_once(head, anchor, """
            let stopped = thread.should_stop_after_split();
            println!("C {} {} {}", thread.idx(), s.value(), stopped as u8);
            if stopped {""")
    child = """
fn ticket_child<NT: NodeType>(
    id: usize, index: usize, ctx: &mut SearchContext, board: &Board, depth: Depth,
    alpha: Score, beta: Score, thread: &Arc<Thread>, cut: bool,
) -> Score {
    let score = search::<NT, EndGameStrategy>(ctx, board, depth, alpha, beta, thread, cut);
    println!("S {id} child {} {index} {} {depth} {} {} {} {}", thread.idx(),
        NT::PV_NODE as u8, alpha.value(), beta.value(), cut as u8, score.value());
    score
}
"""
    for node in ("NonPV", "PV"):
        body = replace_once(body, f"search::<{node}, EndGameStrategy>(", f"ticket_child::<{node}>(id, index, ")
    body = replace_once(body, "    while let Some((mv, move_count)) = move_iter.next() {", """
    let id = Arc::as_ptr(split_point) as usize;
    let worker = thread.idx();
    loop {
        let Some((mv, move_count)) = move_iter.next() else {
            println!("S {id} empty {worker}");
            break;
        };
        let index = move_count - 1;
        println!("S {id} take {worker} {index}");""")
    anchor = "        let alpha = split_point.state().alpha();"
    # Both scout and wide re-search load alpha; keep the last load for the selected score.
    require(body.count(anchor) == 2, "worker alpha reads")
    body = body.replace(anchor, anchor + "\n        let mut searched_alpha = alpha;\n"
                        '        println!("S {id} load {worker} {index} 0 {}", alpha.value());', 1)
    anchor = "            let alpha = split_point.state().alpha();"
    body = replace_once(body, anchor, anchor + "\n            searched_alpha = alpha;\n"
                        '            println!("S {id} load {worker} {index} 1 {}", alpha.value());')
    anchor = "            split_point.state().set_all_helpers_searching_locked(false);"
    body = replace_once(body, anchor, """
            let locked = split_point.state();
            println!("S {id} discard {worker} {index} {} {} {} {} 0 {} {} {} {}",
                locked.cutoff() as u8, thread.is_search_aborted() as u8, searched_alpha.value(), score.value(),
                locked.alpha().value(), locked.best_score().value(), locked.alpha().value(), locked.best_score().value());
""" + anchor)
    anchor = "        let sp = split_point.state();"
    body = replace_once(body, anchor, anchor + "\n        let before = (sp.alpha().value(), sp.best_score().value());")
    report = ('println!("S {id} report {worker} {index} {} {} {} {} {} {} {}", searched_alpha.value(), '
              'score.value(), cutoff as u8, before.0, before.1, sp.alpha().value(), sp.best_score().value());')
    anchor = "                    thread.mark_split_point_cutoff(sp);"
    body = replace_once(body, anchor, "                    " + report + "\n" + anchor)
    anchor = "        if cutoff {\n            break;\n        }"
    body = replace_once(body, anchor, anchor + " else { " + report + " }")
    pvs.write_text(head + child + "pub(super) fn search_split_point" + body + "/// Performs enhanced transposition cutoff" + tail)


@dataclass
class Generation:
    len: int
    start: int
    owner: int
    parent: object
    task: list
    children: dict = field(default_factory=dict)
    zero: bool = False
    checked: list = field(default_factory=list)
    events: list = field(default_factory=list)
    cut: bool = False


def generations(log):
    open_slots, all_splits = {}, []
    stats = Counter()
    returned = {}
    for row in log.splitlines():
        if row.startswith("C "):
            _, owner, score, stop = row.split()
            split, value = returned.pop(owner)
            require(score == value, "parent consumed a different raw score")
            split.checked = [score, stop]
            stats["stopped owner returns"] += int(stop)
            continue
        tag, addr, op, *args = row.split()
        require(tag == "S", "unexpected native output")
        if op == "begin":
            length, start, owner, parent, *task = args
            require(addr not in open_slots, "live slot reused")
            split = Generation(int(length), int(start), int(owner), open_slots[parent] if parent != "0" else None, task)
            open_slots[addr] = split
            all_splits.append(split)
            continue
        split = open_slots[addr]
        if op == "zero":
            require(int(args[0]) == split.owner, "non-owner waited for the mask")
            split.zero = True
            continue
        if op == "returned":
            require(split.zero and int(args[0]) == split.owner, "owner read result before zero")
            require(args[0] not in returned, "unconsumed earlier split result")
            returned[args[0]] = (split, args[1])
            del open_slots[addr]
            continue
        if op == "late":
            stats["late bookings"] += 1
            continue
        worker = int(args[0])
        if op == "take":
            split.children[(worker,args[1])] = []
        if op in ("load", "child"):
            split.children[(worker,args[1])].append((op, args[2:]))
            continue
        if op == "report" and args[4] == "1":
            split.cut = True
            stats["cutoff reports"] += 1
        if op == "discard":
            local, global_stop = args[2:4]
            require((local == "1") == split.cut, "local flag differs from locked cutoff history")
            if local == "0":
                ancestor = split.parent
                while ancestor is not None and not ancestor.cut:
                    ancestor = ancestor.parent
                require(ancestor is not None or global_stop == "1", "orphan stop observation")
                split.events.append(("external", worker, ["external", str(worker)]))
            args = args[:2] + args[4:]
            stats["discarded tickets"] += 1
        if op in ("report", "discard"):
            observed = split.children.pop((worker,args[1]))
            require([k for k,_ in observed] in (["load","child"], ["load","child","load","child"]),
                    "unexpected child/load control path")
            loads = [data for k,data in observed if k == "load"]
            require([x[0] for x in loads] == ["0","1"][:len(loads)], "unexpected alpha load site")
            calls = [data for k,data in observed if k == "child"]
            first, fresh = loads[0][1], loads[-1][1]
            stats["stale scout alpha at lock"] += int(int(first) < int(args[5]))
            stats["alpha raised before re-search"] += int(int(first) < int(fresh))
            stats["wide re-searches"] += len(calls) - 1
            args.extend([first, fresh, *[v for call in calls for v in call]])
        split.events.append((op, worker, [op, *args]))
    require(all_splits and not open_slots and not returned and all(not s.children for s in all_splits), "every generation must begin and observe mask zero")
    return all_splits, stats


def ordered(split, stats):
    # The successful old index supplies RMW modification order. A worker may log
    # after a later CAS; the stdout order for two separate CAS returns needn't match it.
    graph = TopologicalSorter()
    last_worker, takes, empties = {}, {}, []
    last_lock, last_report = None, -1
    pending = set()
    for j, (op, worker, words) in enumerate(split.events):
        graph.add(j)
        if worker in last_worker:
            graph.add(j, last_worker[worker])
        last_worker[worker] = j
        if op == "take":
            index = int(words[2])
            require(index not in takes, "duplicate successful atomic index")
            takes[index] = j
            pending.add(worker)
            stats["max simultaneously outstanding"] = max(stats["max simultaneously outstanding"], len(pending))
        elif op == "empty":
            empties.append(j)
        else:
            if last_lock is not None:
                graph.add(j, last_lock)
            last_lock = j
            if op in ("report", "discard"):
                pending.remove(worker)
            if op == "report":
                index = int(words[2])
                stats["out-of-order publications"] += int(index < last_report)
                last_report = index
    claimed = [takes[i] for i in sorted(takes)]
    for first, second in zip(claimed, claimed[1:]):
        graph.add(second, first)
    if claimed:
        for empty in empties:
            graph.add(empty, claimed[-1])
    return [["begin", str(split.len), str(split.start), str(split.owner), *split.task],
            *[words for _, _, words in split.events],
            ["order", *map(str, graph.static_order())], ["zero", *split.checked]]


def verify(splits, stats, directory, lean_cmd, lean_cwd):
    events = [row for split in splits for row in ordered(split, stats)]
    require(stats["max simultaneously outstanding"] >= 2, "serial run")
    require(stats["out-of-order publications"] > 0, "all publications used claim order")
    require(stats["cutoff reports"] > 0, "no local cutoff exercised")
    require(stats["wide re-searches"] > 0, "no wide re-search exercised")
    trace = directory / "tickets.txt"

    def invoke():
        trace.write_text("".join(" ".join(row) + "\n" for row in events))
        return subprocess.run([*lean_cmd, str(trace)], cwd=lean_cwd, capture_output=True, text=True)

    ok = invoke()
    require(ok.returncode == 0, ok.stderr)
    print(ok.stdout.strip())
    permutation = next(row for row in events if row[0] == "order" and len(row) > 2)
    missing = permutation.pop()
    bad = invoke()
    require(bad.returncode != 0 and "invalid split event ordering" in bad.stderr, "omitted event escaped")
    permutation.append(missing)
    permutation[1:] = reversed(permutation[1:])
    bad = invoke()
    require(bad.returncode != 0 and "invalid split event ordering" in bad.stderr, "reversed events escaped")
    permutation[1:] = reversed(permutation[1:])
    # Each mutant changes one native receipt and preserves the final engine score.
    row = next(row for row in events if row[0] == "take")
    index = row[2]
    row[2] = str(int(index) + 1)
    bad = invoke()
    require(bad.returncode != 0 and any(msg in bad.stderr for msg in
            ("invalid split transition: take", "invalid split event ordering")), "wrong index escaped")
    row[2] = index
    pending = set()
    for row in events:
        if row[0] == "take":
            pending.add(row[1])
        elif row[0] in ("report", "discard"):
            pending.remove(row[1])
            if row[0] == "report" and pending:
                other = row[1]
                row[1] = next(iter(pending))
                break
    else:
        raise AssertionError("no overlapping claim available for the ownership mutant")
    bad = invoke()
    require(bad.returncode != 0 and any(msg in bad.stderr for msg in
            ("invalid split transition: report", "invalid split event ordering")), "wrong reporting worker escaped")
    row[1] = other
    row = next(row for row in events if row[0] == "report")
    scout_alpha = row[14]
    row[14] = str(int(scout_alpha) + 1)
    bad = invoke()
    require(bad.returncode != 0 and "invalid locked score or child query" in bad.stderr, "wrong scout operand escaped")
    row[14] = scout_alpha
    row = next(row for row in events if row[0] == "zero")
    row[1] = str(int(row[1]) + 1)
    bad = invoke()
    require(bad.returncode != 0 and "owner returned an inconsistent score" in bad.stderr, "wrong owner return escaped")
    print(", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print("Rejected missing and reversed events, wrong CAS index, reporting worker, scout window and owner score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    subprocess.run(["lake", "build", "NeuralReversi.PvsParallel.Interleaving.Drive",
                    "NeuralReversi.PvsParallel.Interleaving.EventOrder",
                    "NeuralReversi.PvsParallel.Interleaving.Schedule"], cwd=LEAN, check=True)
    with tempfile.TemporaryDirectory(prefix="reversi-split-tickets-") as tmp:
        directory = Path(tmp)
        binary = capture(directory, args.target_dir.resolve() if args.target_dir else directory / "target",
                         instrument, fixture=FIXTURES / "parallel_tickets.rs", serial_trace=False)
        native = subprocess.run([binary, REPO], capture_output=True, text=True, check=True)
        splits, stats = generations(native.stdout)
        verify(splits, stats, directory, ["lake", "env", "lean", "--run", FIXTURES / "ParallelTickets.lean.in"], LEAN)


if __name__ == "__main__":
    main()
