"""Check PVS/ETC store slots and the final PVS score, bound and root-line guard."""

import argparse
from pathlib import Path
import re
import tempfile

from cache_owner_mir import PASS, Body
from check_midgame_handoff_mir import compare
from check_result_guards_mir import normal_blocks, read
from mir_common import checked, CALL
from extract_endgame_order_mir import code
from extract_pvs_context_mir import Graph
from extract_header import binding
from extract_single import SIGNATURE


PVS = "search-pvs-search"
ETC = "search-pvs-enhanced_transposition_cutoff"
ACCESS = "transposition_table-{impl#2}-slot"
PROBE = "TranspositionTable::probe"
STORE = "TranspositionTable::store"
SLOT = "TTProbeResult::<'_>::slot"
ETC_CALL = "enhanced_transposition_cutoff::<SS>"


PV_IDX = "search-root_move-{impl#1}-pv_idx"
SET_PV_IDX = "search-root_move-{impl#1}-set_pv_idx"

# All normal successors from classify to return, including both correction-history arms.
FINAL_OWNER = """
415: _702 = copy _333; _703 = copy _11; _704 = copy _5; _701 = transposition_table::Bound::classify::<NT>(move _702, move _703, move _704) => 417;
417: _706 = discriminant(_260); switchInt(move _706) -> [1: bb418, otherwise: bb424];
418: goto 419;
419: _707 = copy ((_260 as Some).0: score::Score); _708 = &mut ((*_1).4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>); _709 = discriminant((*_708)); switchInt(move _709) -> [1: bb420, otherwise: bb423];
420: goto 421;
421: _710 = &mut (((*_708) as Some).0: std::boxed::Box<search::correction_history::CorrectionHistory>); _712 = &mut (*(*_710)); _713 = &(*_2); _714 = copy _707; _715 = copy _333; _716 = copy _701; _717 = copy _3; _711 = CorrectionHistory::update(move _712, move _713, move _714, move _715, move _716, move _717) => 422;
422: _705 = const (); goto 425;
423: goto 424;
424: _705 = const (); goto 425;
425: _719 = const <NT as search::node_type::NodeType>::ROOT_NODE; switchInt(move _719) -> [0: bb429, otherwise: bb426];
426: _722 = &(*((*_1).1: &search::root_move::RootMoves)); _721 = RootMoves::pv_idx(move _722) => 427;
427: _720 = Eq(move _721, const 0_usize); switchInt(move _720) -> [0: bb432, otherwise: bb428];
428: goto 429;
429: _724 = &(*((*_1).5: &transposition_table::TranspositionTable)); _726 = &_179; _725 = TTProbeResult::<'_>::slot(move _726) => 430;
430: _727 = &(*_2); _728 = copy _333; _729 = copy _701; _730 = copy _3; _731 = copy _332; _732 = copy ((*_1).7: probcut::Selectivity); _733 = copy _187; _723 = TranspositionTable::store(move _724, move _725, move _727, move _728, move _729, move _730, move _731, move _732, const <SS as search::strategy::SearchStrategy>::IS_ENDGAME, move _733) => 431;
431: _718 = const (); goto 433;
432: _718 = const (); goto 433;
433: _0 = copy _333; goto 447;
447: return;
"""

PV_ACCESS = {
    PV_IDX: """
0: _2 = &((*_1).1: std::sync::atomic::Atomic<usize>); _3 = std::sync::atomic::Ordering::Relaxed; _0 = Atomic::<usize>::load(move _2, move _3) => 1;
1: return;
""",
    SET_PV_IDX: """
0: _4 = &((*_1).1: std::sync::atomic::Atomic<usize>); _5 = copy _2; _6 = std::sync::atomic::Ordering::Relaxed; _3 = Atomic::<usize>::store(move _4, move _5, move _6) => 1;
1: _0 = const (); return;
""",
}


def final_owner(dump, g, body):
    checked("fn " + body.signature + " {" == SIGNATURE, "changed PVS inputs")
    expected = {"alpha": "_4", "beta": "_5", "org_alpha": "_11", "best_score": "_333",
                "best_move": "_332", "tt_pv": "_187", "tt_probe_result": "_179"}
    checked(all(binding(body.source, name) == reg for name, reg in expected.items()), "changed PVS locals")
    # Save alpha before any update. No path may revisit this assignment with a narrowed window.
    before = body.reachable(0, stop_at={3})
    checked(before == {0, 1, 2, 3} and g.cfg[3][0] == "_11 = copy _4;"
            and all("_4" not in re.findall(r"_\d+", op) for pc in before - {3} for op in g.cfg[pc])
            and all(3 not in body.reachable(n) for n in g.normal(3)), "original alpha was not saved on entry")
    for pc, ops in g.cfg.items():
        for index, op in enumerate(ops):
            if (pc, index) != (3, 0) and re.search(r"\b_(?:5|11)\b", op):
                use = re.fullmatch(r"(_\d+) = (?:copy |&)_(?:5|11);", op)
                checked(use and use[1] not in ("_5", "_11"), "changed saved alpha or entry beta")

    compare(PVS, normal_blocks(body.source, {}), FINAL_OWNER, prefix=True)
    checked(430 not in body.reachable(0, stop_at={415}) and 415 in body.reachable(0),
            "final store reachable without its bound classification")
    for name, expected_body in PV_ACCESS.items():
        compare(name, normal_blocks(read(dump, name), {}), expected_body)


def verify(dump):
    # A hit and a miss have different field numbers in the pinned enum.
    checked(code(dump, ACCESS) == {
        0: ["_2 = discriminant((*_1));", "switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];"],
        1: ["unreachable;"], 2: ["falseEdge -> [real: bb4, imaginary: bb3];"],
        3: ["_3 = &(((*_1) as Miss).0: transposition_table::TTSlot<'_>);", "goto -> bb5;"],
        4: ["_3 = &(((*_1) as Hit).1: transposition_table::TTSlot<'_>);", "goto -> bb5;"],
        5: ["_0 = copy (*_3);", "return;"]}, "slot() must extract the physical address held by the result")

    g, body = Graph(dump, PVS, 7), Body(dump, PVS)
    pc, _, args, retained = body.named(PROBE)
    table, board = ("field", "arg1", 5), "arg2"
    checked([g.val(a) for a in args] == [table, board, ("Board::hash", board)], "probe of different board/table")
    original = (PROBE, table, board, ("Board::hash", board))
    expected_slot = (SLOT, original)
    references = re.compile(rf"\b{retained}\b")
    borrowed = []
    for i, ops in g.cfg.items():
        for op in ops:
            if references.search(op) and (i != pc or op != ops[-1]):
                ref = re.fullmatch(r"(_\d+) = &" + retained + ";", op)
                checked(ref, "retained probe was moved, replaced or mutably borrowed")
                borrowed.append((ref[1], op))
    ops = [op for block in g.cfg.values() for op in block]
    for ref, defined in borrowed:
        uses = [op for op in ops if op != defined and re.search(rf"\b{ref}\b", op)]
        call = re.fullmatch(CALL, uses[0]) if len(uses) == 1 else None
        checked(call and call[2] in (SLOT, "TTProbeResult::<'_>::data", "TTProbeResult::<'_>::best_move")
                and call[3] == "move " + ref, "reference to retained probe escaped its accessor")
    consumers = set()
    for i, name, argv in g.calls():
        if name == STORE:
            checked([g.val(a) for a in argv[:3]] == [table, expected_slot, board],
                    "ordinary/presearch store lost the frame's original slot")
        elif name == ETC_CALL:
            checked(g.val(argv[5]) == expected_slot and g.val(argv[1]) == board,
                    "ETC received a different parent slot or board")
        elif name not in ("search::<NonPV, SS>", "search::<PV, SS>"):
            continue
        consumers.add(i)
    return_pc, = g.normal(pc)
    after = body.reachable(return_pc)
    checked(pc not in after, "a recursive continuation can re-probe and replace the selected slot")
    checked(len(consumers) == 7 and consumers <= after and not (consumers & body.reachable(0, stop_at={pc})),
            "original probe does not dominate both stores, ETC and all four ordinary child calls")

    final_owner(dump, g, body)

    etc = Graph(dump, ETC, 7)
    slot_uses = [op for ops in etc.cfg.values() for op in ops if re.search(r"\b_6\b", op)]
    checked(len(slot_uses) == 1 and re.fullmatch(r"_\d+ = copy _6;", slot_uses[0]),
            "ETC altered or forwarded its slot parameter")
    stores = [[etc.val(a) for a in args[:3]] for _, name, args in etc.calls() if name == STORE]
    checked(stores == [[table, "arg6", board]], "ETC did not write the physical slot passed by its caller")


def check(dump):
    verify(dump)
    text = {stem: (dump / ("reversi_core." + stem + PASS)).read_text() for stem in (PVS, ETC, ACCESS, PV_IDX, SET_PV_IDX)}
    mutations = [
        (PVS, r"(_726 = )&_179;", r"\g<1>&mut _179;"),
        (PVS, r"_726 = &_179;", "_991 = TranspositionTable::probe(copy _724, copy _2, copy _87) "
         "-> [return: bb992, unwind: bb450];\n    }\n\n    bb992: {\n        _726 = &_991;"),
        (PVS, r"(_591 = search::<NonPV, SS>[^\n]+?return: )bb\d+", r"\g<1>bb97"),
        (PVS, r"0: bb432, otherwise: bb428", "0: bb428, otherwise: bb432"),
        (PVS, r"_728 = copy _333;", "_728 = copy _4;"),
        (PVS, r"_703 = copy _11;", "_703 = copy _4;"),
        (PVS, r"Eq\(move _721, const 0_usize\)", "Eq(const 0_usize, const 0_usize)"),
        (PVS, r"_0 = copy _333;(?=\s+StorageDead\(_701\);)", "_0 = copy _4;"),
        (PV_IDX, r"Ordering::Relaxed;", "Ordering::Acquire;"),
        (SET_PV_IDX, r"_5 = copy _2;", "_5 = const 0_usize;"),
        (ETC, r"_73 = &\(\*_2\);", "_73 = &_24;"),
        (ACCESS, r"0: bb2, 1: bb3", "0: bb3, 1: bb2"),
    ]
    with tempfile.TemporaryDirectory(prefix="reversi-tt-slot-mutants-") as tmp:
        folder = Path(tmp)
        paths = {stem: folder / ("reversi_core." + stem + PASS) for stem in text}
        for stem, original in text.items():
            paths[stem].write_text(original)
        for stem, pattern, replacement in mutations:
            mutated, n = re.subn(pattern, replacement, text[stem])
            checked(n == 1, f"missing or ambiguous {stem} mutation anchor {pattern}")
            if "_991 = TranspositionTable::probe" in replacement:
                mutated = mutated.replace("    let mut _0: Score;", "    let mut _0: Score;\n"
                                          "    let _991: transposition_table::TTProbeResult<'_>;")
            paths[stem].write_text(mutated)
            try:
                verify(folder)
            except ValueError:
                pass
            else:
                raise ValueError(f"accepted PVS/TT mutation {pattern}")
            finally:
                paths[stem].write_text(text[stem])
    return len(mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"verified PVS/ETC store slots, final owner score and root-line guard; rejected {check(args.mir)} MIR mutations")
