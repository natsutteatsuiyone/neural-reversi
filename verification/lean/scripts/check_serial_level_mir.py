"""Check the selectivity guards on serial full-depth roots up to thirteen empties."""

import argparse
from pathlib import Path
import re

from check_serial_child_calls_mir import REPO, REDUCTION, specializations
from mir_common import checked, CALL
from extract_endgame_order_mir import code
from extract_fixed_leaf_mir import constant, expected, path_result, read_body
from extract_header import ENDGAME, lmr_returns_zero
from extract_nonpv_leaf_mir import follow_pvs, parameters, pvs_body
from extract_nws_leaf_mir import Inputs, parallel_pool
from extract_outer import ladder_constants
from extract_pvs_context_mir import Graph


FIELD = "copy ((*_1).7: probcut::Selectivity)"
SINKS = {"compute_lmr_reduction::<NT, SS>": 0, "TranspositionTable::store": 7}
FORBIDDEN = {"TTEntryData::can_cut", "TTEntryData::score", "TranspositionTable::penalize",
             "enhanced_transposition_cutoff::<SS>", "<SS as SearchStrategy>::try_probcut",
             "<SS as SearchStrategy>::shallow_search", "<EndGameStrategy as SearchStrategy>::shallow_search"}
REG = re.compile(r"\b_\d+\b")
SWITCH = re.compile(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];")


def owner(graph, traits):
    """Explore both edges unless a specialization constant immediately defined the guard."""
    seen, pending = set(), [0]
    while pending:
        pc = pending.pop()
        if pc in seen:
            continue
        seen.add(pc)
        *ops, end = graph.cfg[pc]
        edges = graph.normal(pc)
        if ops and (switch := SWITCH.fullmatch(end)):
            lhs, sep, rhs = ops[-1].removesuffix(";").partition(" = ")
            if sep and lhs == switch[1] and rhs in traits:
                edges = [int(switch[3] if traits[rhs] else switch[2])]
        pending.extend(edges)

    ops = [op for pc in seen for op in graph.cfg[pc]]
    uses = {}
    for op in ops:
        for reg in REG.findall(op):
            uses[reg] = uses.get(reg, 0) + 1
    checked(not any(op.startswith("_3 =") for op in ops), "PV/root depth changed in place")
    sinks = []
    for pc in seen:
        *body, end = graph.cfg[pc]
        call = re.fullmatch(CALL, end)
        checked(not call or call[2] not in FORBIDDEN, f"reachable score shortcut at bb{pc}: {end}")
        for op in [*body, end]:
            if "Selectivity" not in op:
                continue
            reg, _, expr = op.removesuffix(";").partition(" = ")
            checked(expr == FIELD and call and call[2] in SINKS and uses[reg] == 2
                    and call[3].split(", ")[SINKS[call[2]]] == "move " + reg,
                    f"unexpected ordinal consumer at bb{pc}: {op}")
            sinks.append(call[2])
    checked(sorted(sinks) == sorted(SINKS), f"unexpected ordinal sinks: {sinks}")


# All full-context callees in the disc-domain NWS and shallow loops. The
# context methods and the transitive solve4 call have separate MIR extractors.
LEAVES = ("null_window_search", "search_move_nws_ec", "shallow_search", "shallow_search_move",
          "shallow_search_moves", "small_move_list-{impl#1}-evaluate_moves")
CTX = "SearchContext::<'_>::"
RECIPIENTS = {name: 0 for name in ("null_window_search::<C>", "search_move_nws_ec::<C>",
                                    "shallow_search", "shallow_search_move", "shallow_search_moves", "solve4",
                                    CTX + "increment_nodes", CTX + "update_endgame", CTX + "undo_endgame")}
RECIPIENTS["SmallMoveList::evaluate_moves"] = 1
BORROW = re.compile(r"(_\d+) = &(?:mut )?\(\*(_\d+)\);")


def leaf(cfg, ctx):
    ops = [op for block in cfg.values() for op in block]
    borrows = [m.groups() for op in ops if (m := BORROW.fullmatch(op))]
    aliases = {ctx}
    while extra := {dst for dst, src in borrows if src in aliases} - aliases:
        aliases.update(extra)
    for op in ops:
        refs = [reg for reg in REG.findall(op) if reg in aliases]
        if not refs or BORROW.fullmatch(op):
            continue
        call = re.fullmatch(CALL, op)
        if call and call[2] in RECIPIENTS:
            checked(len(refs) == 1 and call[3].split(", ")[RECIPIENTS[call[2]]] == "move " + refs[0],
                    f"unverified shallow solver context recipient: {op}")
        else:
            permitted = {f"&((*{refs[0]}).10: empty_list::EmptyList)",
                         f"&mut ((*{refs[0]}).12: search::counters::SearchStats)"}
            lhs, _, rhs = op.removesuffix(";").partition(" = ")
            checked(REG.fullmatch(lhs) and len(refs) == 1 and rhs in permitted,
                    f"unverified shallow solver context use: {op}")


def check(dump, repo=REPO):
    checked(ladder_constants(dump, repo) == [1, 2, 3, 4, 5], "unexpected public ladder")
    specs = specializations(dump, repo)
    graph = Graph(dump, "search-pvs-search", 7)
    def owners():
        for node in ("rootID", "pvID"):
            owner(graph, {**specs[node], ENDGAME: True})
    owners()
    reduction = (dump / REDUCTION).read_text()
    lmr_returns_zero(reduction)

    body, params, solver = pvs_body(dump), parameters(repo, dump, False), read_body(dump, repo)
    boundary, shallow = (constant(dump, name) for name in ("DEPTH_TO_NWS", "DEPTH_TO_SHALLOW_SEARCH"))
    parallel_pool(dump)
    serial = Inputs(False, False, False, False, False, False)
    for n in range(boundary + 1):
        for cut in (False, True):
            follow_pvs(body, params, n, cut)
        if n <= shallow:
            checked(path_result(solver, n, boundary, shallow) == expected(n), f"wrong small solver at {n}")
        else:
            checked(path_result(solver, n, boundary, shallow, serial) == serial.outcome(n, boundary),
                    f"selective or unexpected serial NWS call at {n}")
    bodies = {name: code(dump, "search-endgame-" + name) for name in LEAVES}
    for name, cfg in bodies.items():
        leaf(cfg, "_2" if name == LEAVES[-1] else "_1")

    def rejected(test, description):
        try:
            test()
        except ValueError:
            return
        raise ValueError("accepted " + description)

    for pc, guard in ((105, "PV_NODE"), (146, "IS_ENDGAME"), (169, "PV_NODE")):
        old = graph.cfg[pc]
        checked(guard in old[0], f"missing {guard} mutation anchor at bb{pc}")
        graph.cfg[pc] = [old[0].split(" = ")[0] + " = const false;", *old[1:]]
        try:
            rejected(owners, f"disabled {guard} guard at bb{pc}")
        finally:
            graph.cfg[pc] = old
    rejected(lambda: lmr_returns_zero(reduction.replace("const 0_u32", "const 1_u32")), "endgame LMR")
    narrowed = {**params, "const <SS as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW": boundary - 1}
    rejected(lambda: follow_pvs(body, narrowed, boundary, True), "late NonPV dispatch")
    first = bodies["null_window_search"]
    early_read = {**first, 0: ["_1000 = " + FIELD + ";", *first[0]]}
    rejected(lambda: leaf(early_read, "_1"), "NWS ordinal read")
    return 6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    print(f"checked all five serial PVS ladder levels through 13 empties and rejected {check(args.mir, args.repo)} mutations")
