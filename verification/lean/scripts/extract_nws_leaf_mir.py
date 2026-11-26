"""Check full-depth NonPV entry through the NWS wrapper and its boundary table."""

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

from mir_common import checked, normal_target
from cache_owner_mir import Body
from extract_cache_owner import routing
from extract_fixed_leaf_mir import constant, path_result, read_body
from extract_nonpv_leaf_mir import follow_pvs, parameters, pvs_body
from extract_split_scheduler_mir import Body as RegisterBody


worker = ("worker", "thread")
alpha = ("discAlpha", "alpha")
caches = ("caches", worker)
moves = ("legalMoves", "board")
board_hash = ("hash", "board")
probe = ("probe", "tt", "board", board_hash)
data = ("data", probe)
entry = ("entry", data)
slot = ("slot", probe)


@dataclass
class Inputs:
    parallel: bool
    has_entry: bool
    permits_cut: bool
    active: bool
    enough_nodes: bool
    stopped: bool

    calls = {
        "threading::thread::Thread::is_parallel_pool": "parallel",
        "threading::thread::Thread::has_active_split_point": "active",
        "threading::thread::Thread::should_stop": "stop",
        "Board::hash": "hash", "TranspositionTable::probe": "probe",
        "TTProbeResult::<'_>::data": "data", "TTProbeResult::<'_>::slot": "slot",
        "TTEntryData::score": "score", "TTEntryData::can_cut": "canCut",
        "null_window_search::<IgnoreSplitPointCutoff>": "ignoreNws",
        "null_window_search::<threading::thread::Thread>": "activeNws",
        "transposition_table::Bound::classify::<NonPV>": "bound",
        "TranspositionTable::store": "store",
    }

    def operand(self, s, local, calls):
        if s == "copy ((*_1).2: u64)":
            return True, "nodesAfter" if any(c[0] in ("activeNws", "ignoreNws") for c in calls) else "nodesBefore"
        specific = {"IgnoreSplitPointCutoff": "ignore", "const true": "true", "const false": "false",
                    "&(*((*_1).5: &transposition_table::TranspositionTable))": "tt",
                    "copy ((*_1).7: probcut::Selectivity)": "sel",
                    "Option::<Square>::None": "noMove", "Selectivity::None": "exact"}
        if s in specific:
            return True, specific[s]
        if m := re.fullmatch(r"&(_\d+)", s):
            return True, local(m[1])
        if m := re.fullmatch(r"discriminant\((_\d+)\)", s):
            val = local(m[1])
            if val == data:
                return True, int(self.has_entry)
            if val == ("some", slot):
                return True, 1
        if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: transposition_table::(TTEntryData|TTSlot<'_>)\)", s):
            val = local(m[1])
            if m[2] == "TTEntryData":
                checked(val == data and self.has_entry, "invalid TT data projection")
                return True, entry
            checked(val == ("some", slot), "invalid TT slot projection")
            return True, slot
        if m := re.fullmatch(r"Option::<TTSlot<'_>>::Some\(move (_\d+)\)", s):
            return True, ("some", local(m[1]))
        if m := re.fullmatch(r"SubWithOverflow\(copy (_\d+), copy (_\d+)\)", s):
            checked((local(m[1]), local(m[2])) == ("nodesAfter", "nodesBefore"), "wrong subtree delta")
            return True, "difference"
        if m := re.fullmatch(r"move \((_\d+)\.0: u64\)", s):
            checked(local(m[1]) == "difference", "wrong delta projection")
            return True, "subtreeNodes"
        if m := re.fullmatch(r"Ge\(move (_\d+), const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES\)", s):
            checked(local(m[1]) == "subtreeNodes", "wrong store threshold comparison")
            return True, int(self.enough_nodes)
        return False, None

    def response(self, call):
        names = {"parallel": self.parallel, "active": self.active, "canCut": self.permits_cut, "stop": self.stopped}
        return int(names[call[0]]) if call[0] in names else call

    def checked_target(self, body, term, local):
        match = re.fullmatch(r'assert\(!move \((_\d+)\.1: bool\), "attempt to compute `\{\} - \{\}`, which would overflow", '
                             r'move (_\d+), move (_\d+)\) -> \[success: bb(\d+), unwind: bb(\d+)\];', term)
        if match:
            checked((local(match[1]), local(match[2]), local(match[3])) == ("difference", "nodesAfter", "nodesBefore"),
                    "wrong subtree delta assertion")
            return normal_target(body, match[4], match[5])
        return None

    def outcome(self, n, boundary):
        at_table = n == boundary and self.parallel
        calls = [alpha]
        if n == boundary:
            calls.extend([worker, ("parallel", worker)])
        if at_table:
            calls.extend([board_hash, probe, data])
            if self.has_entry:
                calls.append(("canCut", entry, "beta", n, "sel", "true"))
                if self.permits_cut:
                    answer = ("score", entry)
                    return answer, [*calls, answer]
            calls.append(slot)
        empties = ("emptiesOf", "board")
        calls.extend([worker, caches, empties, ("count", empties), moves, worker, ("active", worker)])
        if self.active:
            calls.append(worker)
        raw = ("activeNws" if self.active else "ignoreNws", "ctx", "board", alpha, caches,
               worker if self.active else "ignore", moves, empties)
        score = ("scaled", raw)
        calls.extend([raw, score])
        if at_table and self.enough_nodes:
            calls.extend([worker, ("stop", worker)])
            if not self.stopped:
                bound = ("bound", score, "alpha", "beta")
                calls.extend([bound, ("store", "tt", slot, "board", score, bound, n, "noMove", "exact", "true", "false")])
        return score, calls


def parallel_pool(dump):
    constructor = RegisterBody(dump, "threading-thread-{impl#1}-new",
                               ["idx", "abort", "helper", "epoch", "size", "inFlight", "pool"])
    record, = constructor.defs["_0"]
    fields = record.split(" { ", 1)[1].removesuffix(" }").split(", ")
    index, value = next((i, field.split(": ", 1)[1]) for i, field in enumerate(fields)
                        if field.startswith("pool_size: "))
    checked(constructor.val(value) == "size", "worker did not retain the pool size")
    body = RegisterBody(dump, "is_parallel_pool", ["worker"])
    checked(list(body.cfg) == [0] and body.cfg[0][-1] == "return;"
            and body.val("_0") == ("Gt", ("field", "worker", index), "const 1_usize"),
            "unexpected parallel pool predicate")


def verify(dump, repo):
    parallel_pool(dump)
    pvs, params = pvs_body(dump), parameters(repo, dump, False)
    stem = "search-pvs-search"
    call_pc = Body(dump, stem).named("<SS as SearchStrategy>::shallow_search")[0]
    sites = [site for site, (pc, target, _) in enumerate(routing(dump, stem)) if pc == call_pc and target == "boundary"]
    checked(len(sites) == 1, "endgame PVS boundary call site")
    body = read_body(dump, repo)
    shallow = constant(dump, "DEPTH_TO_SHALLOW_SEARCH")
    boundary = constant(dump, "DEPTH_TO_NWS")
    strategy = (repo / "crates/reversi-core/src/search/strategy.rs").read_text(encoding="utf-8")
    min_nodes = re.findall(r"^const BOUNDARY_STORE_MIN_SUBTREE_NODES: u64 = (\d+);$", strategy, re.MULTILINE)
    checked(len(min_nodes) == 1, "boundary store node threshold")
    checked(int(min_nodes[0]) >= 1, "BOUNDARY_STORE_MIN_SUBTREE_NODES is 0, but "
            "PvsParallel/NativeStopReturns/Regression.lean tests a subtree just below the threshold")
    for n in range(shallow + 1, boundary + 1):
        for cut in (False, True):
            follow_pvs(pvs, params, n, cut)
        for flags in product((False, True), repeat=6):
            cfg = Inputs(*flags)
            got = path_result(body, n, boundary, shallow, cfg)
            checked(got == cfg.outcome(n, boundary), f"wrong {n}-empty NWS return or effects at {cfg}: {got}")
    return ("namespace NeuralReversi.PvsParallel.DeepLeafMir\n\n"
            "def parallelPool (size : Nat) : Bool := decide (1 < size)\n\n"
            f"def boundaryStoreMinNodes : Nat := {int(min_nodes[0])}\n\n"
            f"def boundaryPvsSite : Nat := {sites[0]}\n\n"
            "inductive Arm where\n  | probe | activeNws | ignoreNws\n  deriving DecidableEq\n\n"
            "def dispatch (empties : Nat) (parallel tableCut active : Bool) : Option Arm :=\n"
            f"  if {shallow} < empties ∧ empties ≤ {boundary} then\n"
            f"    if empties = {boundary} ∧ parallel ∧ tableCut then some .probe\n"
            "    else some (if active then .activeNws else .ignoreNws)\n"
            "  else none\n\nend NeuralReversi.PvsParallel.DeepLeafMir\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--mir", required=True, type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = verify(args.mir, args.repo.resolve())
    if args.check:
        checked(result == args.check.read_text(encoding="utf-8"), "NWS program differs")
        print("NonPV NWS returns, boundary TT probe arguments and conditional stores verified for full depths 7–12")
    else:
        print(result, end="")
