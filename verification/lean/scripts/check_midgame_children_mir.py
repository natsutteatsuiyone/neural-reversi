"""Bind midgame recursive requests, retained TT payloads and shallow move order to rustc MIR."""

from pathlib import Path
import re
from tempfile import TemporaryDirectory

from cache_owner_mir import Body, PASS
from check_result_guards_mir import normal_blocks, read
from mir_common import checked, CALL
from extract_header import binding
from extract_owner_sites import PVS, role_blocks
from extract_pvs_context_mir import Graph, bracket


SHALLOW = "search-midgame-evaluate_shallow_with_tt"
ONE = "search-midgame-evaluate_depth1"
PROBCUT = "search-midgame-try_probcut"
STRATEGY = "search-strategy-{impl#1}-shallow_search"
NEG = "<Score as Neg>::neg"
PROBE = "TranspositionTable::probe"
STORE = "TranspositionTable::store"
PENALTY = "TranspositionTable::penalize"
DATA = "TTProbeResult::<'_>::data"
SLOT = "TTProbeResult::<'_>::slot"
FIELD = lambda obj, n: ("field", obj, n)
ROOT = "const <NT as search::node_type::NodeType>::ROOT_NODE"
PV = "const <NT as search::node_type::NodeType>::PV_NODE"
ENDING = "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"
CTX, BOARD, DEPTH, ALPHA, BETA, THREAD, CUT = (f"arg{i}" for i in range(1, 8))


def neg(x):
    return NEG, x


def minus(depth):
    return FIELD(("SubWithOverflow", depth, "const 1_u32"), 0)


def unique(items, name):
    found = [x for x in items if x[1] == name]
    checked(len(found) == 1, f"expected one {name}; found {len(found)}")
    return found[0]


def fields(g, argv, *idx):
    return tuple(g.val(argv[i]) for i in idx)


def guard(g, pc, expr, no, yes):
    op = g.cfg[pc][-1]
    m = re.fullmatch(r"switchInt\((.+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", op)
    checked(m and (g.val(m[1]), int(m[2]), int(m[3])) == (expr, no, yes), f"bb{pc}: wrong guard {op}")


def only_assignments(g, reg, pcs):
    assigned = {pc for pc, ops in g.cfg.items() if any(op.startswith(reg + " = ") for op in ops)}
    checked(assigned == set(pcs), f"{reg}: assignments at {assigned}, expected {pcs}")


def negated(g, pc, out):
    succ, = g.normal(pc)
    m = re.fullmatch(CALL, g.cfg[succ][-1])
    checked(m and m[2] == NEG and m[3] == "move " + out, f"bb{pc}: child returned with wrong sign")


def inspect_pvs(dump):
    body, roles = role_blocks(dump)
    g = Graph(dump, PVS, 7)
    calls = {pc: (fn, args, dst) for pc, fn, args, dst in body.calls}
    kinds = bracket(dump, PVS)
    hi, scoutlo = neg(BETA), neg(("<Score as Add<i32>>::add", ALPHA, "const 1_i32"))
    d = minus(DEPTH)
    reduced = FIELD(("SubWithOverflow", d, ("compute_lmr_reduction::<NT, SS>",)), 0)
    for role, pc in roles.items():
        fn, args, dest = calls[pc]
        expected = {
            "passChild": ("search::<NT, SS>", ".pass", (DEPTH, hi, neg(ALPHA))),
            "scout": ("search::<NonPV, SS>", "(.move false)", (reduced, scoutlo, neg(ALPHA))),
            "reducedRetry": ("search::<NonPV, SS>", "(.move false)", (d, scoutlo, neg(ALPHA))),
            "presearch": ("search::<NonPV, SS>", "(.move false)", (d, scoutlo, neg(ALPHA))),
            "wide": ("search::<PV, SS>", "(.move true)", (d, hi, neg(ALPHA))),
        }[role]
        checked(len(args) == 7 and (fn, kinds[pc], fields(g, args, 2, 3, 4)) == expected,
                f"{role}: request board, depth or window changed")
        checked(fields(g, args, 0, 5) == (CTX, THREAD), f"{role}: unpaired context or worker")
        if role == "wide":
            checked(g.val(args[6]) == "const false", "wide cut flag")
        elif role in ("presearch", "reducedRetry"):
            checked(g.val(args[6]) == ("Not", CUT), role + " cut flag")
        negated(g, pc, dest)

    # Graph treats function arguments as their value at this instruction. Check all
    # mutations to depth separately, including the branch selecting its sole decrement.
    only_assignments(g, "_3", [153])
    checked(g.cfg[153][0] == "_3 = move (_257.0: u32);" and g.val("_257") ==
            ("SubWithOverflow", DEPTH, "const 1_u32"), "unexpected IIR depth")
    all_node = binding(body.source, "all_node")
    _, _, _, best = body.named("TTProbeResult::<'_>::best_move")
    g.roots.update({int(all_node[1:]): "all", int(best[1:]): "hint"})
    for pc, test, no, yes in [
        (0, PV, 1, 2), (144, ENDING, 145, 157), (145, ROOT, 146, 157), (146, "all", 147, 157),
        (148, ("Selectivity::is_enabled", FIELD(CTX, 7)), 156, 149),
        (149, ("Ge", DEPTH, "const search::pvs::IIR_MIN_DEPTH"), 155, 150),
        (151, ("Option::<Square>::is_none", "hint"), 154, 152),
        (340, ("Gt", ("compute_lmr_reduction::<NT, SS>",), "const 0_u32"), 342, 341),
        (345, ("Gt", ("compute_lmr_reduction::<NT, SS>",), "const 0_u32"), 356, 346)]:
        guard(g, pc, test, no, yes)
    typed = normal_blocks(body.source, {})
    checked(typed[1].startswith("_10 = copy _7; _8 = Not(move _10);") and
            typed[2].startswith("_8 = const false;") and calls[roles["passChild"]][1][6] == "move _131" and
            g.val("_131") == "all", "pass does not invert a non-PV cut flag")
    scout_flag = calls[roles["scout"]][1][6]
    checked(scout_flag == "move _604" and typed[341] == "_604 = const true; goto 343;" and
            typed[342] == "_607 = copy _7; _604 = Not(move _607); goto 343;", "wrong reduced scout cut flag")
    checked("<Score as PartialOrd>::gt(move _611, move _612)" in typed[346] and
            "_611 = &_573; _612 = &_4;" in typed[346] and typed[347] == "switchInt(move _610) -> [0: bb355, otherwise: bb348];",
            "full-depth retry must be gated by the reduced score and alpha")

    captured = unique(g.calls(), PROBE)
    retained = (PROBE, FIELD(CTX, 5), BOARD, ("Board::hash", BOARD))
    checked(fields(g, captured[2], 0, 1, 2) == retained[1:], "changed entry probe")
    entry = ("unSome", (DATA, retained))
    _, _, penalty = unique(g.calls(), PENALTY)
    checked(fields(g, penalty, 0, 1, 2, 3) == (FIELD(CTX, 5), (SLOT, retained), BOARD, entry),
            "penalty changes a different probe")
    # Unlike a fresh probe, the TT-margin arm still holds the entry from before a
    # same-board ProbCut verification. That child can already have changed this slot.
    threshold = ("<Score as Add>::add", BETA, "const search::pvs::TT_PROBCUT_MARGIN")
    compared = ("<Score as PartialOrd>::ge", ("TTEntryData::score", entry), threshold)
    guard(g, 201, compared, 204, 202)
    checked(g.val("_299") == threshold and "_0 = copy _299;" in g.cfg[203], "TT margin returns different word")


def inspect_shallow(dump):
    body = Body(dump, SHALLOW)
    g = Graph(dump, SHALLOW, 5)
    calls = list(g.calls())
    probe = (PROBE, FIELD(CTX, 5), BOARD, ("Board::hash", BOARD))
    checked(fields(g, unique(calls, PROBE)[2], 0, 1, 2) == probe[1:], "wrong shallow probe")
    entry = ("unSome", (DATA, probe))
    checked(fields(g, unique(calls, "TTEntryData::can_cut")[2], 0, 1, 2, 3, 4) ==
            (entry, BETA, DEPTH, "Selectivity::None", "const false"), "shallow TT eligibility")
    checked(fields(g, unique(calls, "TTEntryData::score")[2], 0) == (entry,), "shallow TT return")
    guard(g, 31, ("TTEntryData::can_cut", entry, BETA, DEPTH, "Selectivity::None", "const false"), 34, 32)
    played = unique(calls, "Board::make_move_with_flipped")
    _, _, order = body.named("SearchContext::<'_>::update")[:3]
    flip, sq = fields(g, played[2], 1, 2)
    move = g.val(binding(body.source, "mv"))
    checked((flip, sq) == (FIELD(move, 0), FIELD(move, 2)) and fields(g, order, 0, 1, 2) == (CTX, sq, flip)
            and fields(g, unique(calls, "SearchContext::<'_>::undo")[2], 0) == (CTX,), "unpaired child board")
    next_board = ("Board::make_move_with_flipped", BOARD, flip, sq)
    passed = ("Board::switch_players", BOARD)
    depth_one, = [args for _, fn, args in calls if fn == "evaluate_depth1"]
    checked(fields(g, depth_one, 0, 1, 2, 3) == (CTX, next_board, neg(BETA), neg(ALPHA)), "shallow depth two child")
    rpcs = [(pc, fields(g, a, 0, 1, 2, 3, 4), dst) for pc, fn, a, dst in body.calls if fn == "evaluate_shallow_with_tt"]
    checked([a for _, a, _ in rpcs] == [(CTX, passed, DEPTH, neg(BETA), neg(ALPHA)),
            (CTX, next_board, minus(DEPTH), neg(BETA), neg(ALPHA))], "wrong shallow pass/move request")
    for pc, _, dst in rpcs:
        negated(g, pc, dst)
    pc, _, _, dest = body.named("evaluate_depth1")
    negated(g, pc, dest)
    guard(g, 57, ("Eq", DEPTH, "const 2_u32"), 63, 58)

    # Distinguish the saved alpha at node entry from the alpha raised on loop
    # backedges. The original probe slot and TT PV bit survive the recursion.
    original = binding(body.source, "org_alpha")
    best, hint, score = (binding(body.source, n, last=True) for n in ("best_score", "best_move", "score"))
    checked(g.val(original) == ALPHA and g.defs[original] == {"copy _4"}, "saved alpha mismatch")
    for reg, pcs in [("_4", [76]), (original, [23]), (best, [45, 72, 74]), (hint, [46, 72, 76]), (score, [61, 67])]:
        only_assignments(g, reg, pcs)
    g = Graph(dump, SHALLOW, 5, {int(r[1:]): v for r, v in [(original, "original"), (best, "best"), (hint, "hint"), (score, "word")]})
    guard(g, 71, ("<Score as PartialOrd>::ge", "word", BETA), 73, 72)
    guard(g, 75, ("<Score as PartialOrd>::gt", "word", ALPHA), 77, 76)
    _, _, store = unique(calls, STORE)
    tag = ("transposition_table::Bound::classify::<NonPV>", "best", "original", BETA)
    checked(fields(g, store, *range(9)) == (FIELD(CTX, 5), (SLOT, probe), BOARD, "best", tag, DEPTH, "hint",
            "Selectivity::None", "const false"), "shallow caller did not store current bound at original slot")
    pv_test, observed, _ = g.val(store[9])
    checked(pv_test.startswith("Option::<TTEntryData>::is_some_and::<") and observed == (DATA, probe),
            "shallow store lost its entry PV bit")
    checked(normal_blocks(read(dump, SHALLOW + "-{closure#0}"), {}) == {
        0: "_3 = &_2; _0 = TTEntryData::is_pv(move _3) => 1;", 1: "return;"}, "wrong shallow entry flag")
    for pc, updates in [(72, {best: "word", hint: ("Option::<Square>::Some", sq)}),
                        (74, {best: ("<Score as Ord>::max", "best", "word")}),
                        (76, {"_4": "word", hint: ("Option::<Square>::Some", sq)})]:
        for dest, value in updates.items():
            op, = [op for op in g.cfg[pc] if op.startswith(dest + " = ")]
            checked(g.val(op.split(" = ",1)[1][:-1]) == value, "shallow loop accumulated wrong word or hint")
    checked(g.cfg[84][0] == "_0 = copy " + best + ";", "shallow store and return disagree")


def inspect_probcut(dump):
    body = Body(dump, PROBCUT)
    g = Graph(dump, PROBCUT, 7)
    pc, _, args, dest = body.named("search::<NonPV, MidGameStrategy>")
    veri = ("probcut_depth", DEPTH)
    bounded = ("<Score as Ord>::max", ("compute_probcut_beta", ALPHA, ("Selectivity::t_value", FIELD(CTX, 7)),
               ("midgame_coefficients", ("SearchContext::<'_>::ply", CTX), DEPTH)),
               ("<Score as Add<i32>>::add", neg("const score::Score::INF"), "const 1_i32"))
    checked(fields(g, args, *range(7)) == (CTX, BOARD, veri, ("<Score as Sub<i32>>::sub", bounded, "const 1_i32"),
            bounded, "arg7", "const true"), "ProbCut recursive verifier consumed wrong position/window")
    sel = "((*_1).7: probcut::Selectivity) = "
    writes = [(k, g.val(op[len(sel):-1])) for k, ops in g.cfg.items() for op in ops if op.startswith(sel)]
    checked(writes == [(20, "Selectivity::None"), (24, FIELD(CTX, 7))] and pc == 23 and g.normal(pc) == [24],
            "ProbCut failed to restore parent selectivity after return")
    g.roots[int(dest[1:])] = "verification"
    guard(g, 25, ("<Score as PartialOrd>::ge", "verification", bounded), 27, 26)
    checked(g.val("_68") == ALPHA and g.cfg[26][1] == "_0 = Option::<Score>::Some(move _68);",
            "ProbCut must return the parent beta, not the verification score")


def inspect_masks(dump):
    stem = "search-midgame-DEPTH1_MOVE_ORDER_MASKS"
    cfg = normal_blocks((dump / f"reversi_core.{stem}.built.after.mir").read_text(encoding="utf-8"), {})
    checked(set(cfg) == set(range(11)), "depth-one mask initialization")
    values = {}
    for pc in range(10):
        match = re.fullmatch(r"(_\d+) = Bitboard::new\(const (\d+)_u64\) => (\d+);", cfg[pc])
        checked(match and int(match[3]) == pc+1 and match[1] not in values, "nonliteral mask")
        values[match[1]] = int(match[2])
    array = re.fullmatch(r"_0 = \[(move _\d+(?:, move _\d+)*)\]; return;", cfg[10])
    checked(array, "depth-one masks must be returned in array order")
    returned = [values[reg] for reg in re.findall(r"_\d+", array[1])]
    g = Graph(dump, ONE, 4)
    mask = ("unSome", ("<std::slice::Iter<'_, Bitboard> as Iterator>::next", ("<&[Bitboard; 10] as IntoIterator>::into_iter",
            "const search::midgame::DEPTH1_MOVE_ORDER_MASKS")))
    mask_moves = ("<LegalMoves as BitAnd<Bitboard>>::bitand", ("Board::legal_moves", BOARD), mask)
    square = ("unSome", ("<BitboardIterator as Iterator>::next", ("<BitboardIterator as IntoIterator>::into_iter",
              ("LegalMoves::iter", mask_moves))))
    flip = ("flip::flip", square, ("Board::player", BOARD), ("Board::opponent", BOARD))
    checked(fields(g, unique(g.calls(), "evaluate")[2], 0, 1) ==
            (CTX, ("Board::make_move_with_flipped", BOARD, flip, square)), "depth-one evaluation outside mask iterator")
    checked(fields(g, unique(g.calls(), "evaluate_depth1")[2], 0, 1, 2, 3) ==
            (CTX, ("Board::switch_players", BOARD), neg("arg4"), neg("arg3")), "depth-one pass changed window")
    # The square loop returns to its iterator; exhaustion advances the mask.
    checked(g.normal(50) == [28] and g.normal(34) == [18] and g.normal(28) == [29] and g.normal(18) == [19],
            "mask and move loops do not resume their own iterators")
    return returned


def render_masks(dump):
    """Render the depth-one visit tiers that MidgameShallow.lean iterates."""
    def hex_word(value):
        digits = f"{value:016x}"
        return "0x" + "_".join(digits[i:i + 4] for i in range(0, 16, 4))
    words = [hex_word(mask) for mask in inspect_masks(dump)]
    rows = ",\n   ".join(", ".join(words[i:i + 5]) for i in range(0, len(words), 5))
    return MASKS.replace("@MASKS@", rows)


MASKS = """module
public import NeuralReversi.Bitboard.Model

@[expose] public section

/-! Depth-one visit tiers extracted from the rustc MIR of `search::midgame::DEPTH1_MOVE_ORDER_MASKS` by
`verification/lean/scripts/check_midgame_children_mir.py`. Regenerate with `check_endgame_mir.py --write`
instead of editing. -/

namespace NeuralReversi.PvsParallel.SerialTable.MidgameShallow

def depthOneMasks : List Bitboard.Bits :=
  [@MASKS@]

end NeuralReversi.PvsParallel.SerialTable.MidgameShallow
"""


def inspect_boundary(dump):
    body = Body(dump, STRATEGY)
    g = Graph(dump, STRATEGY, 6)
    guard(g, 1, ("Eq", DEPTH, "const search::endgame::DEPTH_TO_NWS"), 18, 2)
    guard(g, 4, ("threading::thread::Thread::is_parallel_pool", THREAD), 17, 5)
    checked(normal_blocks(read(dump, "search-threading-thread-{impl#1}-is_parallel_pool"), {}) ==
            {0: "_2 = copy ((*_1).11: usize); _0 = Gt(move _2, const 1_usize); return;"}, "one-worker pool became parallel")
    typed = normal_blocks(body.source, {})
    checked(g.normal(17) == g.normal(18) == [19] and typed[19] == "_9 = Option::<TTSlot<'_>>::None; goto 20;",
            "serial boundary retained a TT slot")
    pc, _, _, _ = body.named(PROBE)
    checked(pc not in body.reachable(0, stop_at={4}), "unguarded parallel TT probe")
    pc, _, _, _ = body.named(STORE)
    checked(typed[66] == "_137 = discriminant(_9); switchInt(move _137) -> [1: bb67, otherwise: bb79];" and
            pc not in body.reachable(0, stop_at={67}) and pc not in body.reachable(79), "unguarded boundary store")
    only_assignments(g, "_9", [16, 19])
    # Each ordinary midgame move resumes through a stop poll. Cancellation before
    # the parent store still retains whatever its already-returned child wrote.
    b = Body(dump, PVS)
    g = Graph(dump, PVS, 7)
    typed = normal_blocks(b.source, {})
    for pc, passed, halted in [(249, 251, 250), (377, 379, 378)]:
        guard(g, pc, ("threading::thread::Thread::should_stop", THREAD), passed, halted)
        checked(typed[halted].startswith("_0 = const score::Score::ZERO;"), "abort returned a partial score")
        for n in b.reachable(halted):
            checked(not any("TranspositionTable::store" in op or "update_root_move" in op for op in g.cfg[n]),
                    "aborted parent published a partial value")


# The getters above check these operands against the compiled MIR for all five
# general PVS call sites, the same-board verifier and the two shallow move depths.
SITES = {
    "passed": "⟨PvsBoardTree.pass q.board, q.depth, q.pv, !q.pv && !q.cut, -q.beta, -q.alpha⟩",
    "scout": "⟨next, q.depth - 1 - reduction, false, reduction > 0 || !q.cut, -(q.alpha + 1), -q.alpha⟩",
    "retry": "⟨next, q.depth - 1, false, !q.cut, -(q.alpha + 1), -q.alpha⟩",
    "wide": "⟨next, q.depth - 1, true, false, -q.beta, -q.alpha⟩",
    "verified": "⟨q.board, pcDepth, false, true, threshold - 1, threshold⟩",
    "shallowMove": "⟨next, q.depth - 1, false, false, -q.beta, -q.alpha⟩",
}


def check(dump, repo):
    folder = repo / "verification/lean/NeuralReversi/PvsParallel/SerialTable"
    sites = (folder / "MidgameSites.lean").read_text(encoding="utf-8")
    for name, value in SITES.items():
        matches = re.findall(r"^def " + name + r"\b[^\n]+ : Request :=\n  (.+)$", sites, re.M)
        checked(matches == [value], f"{name}: Lean descriptor differs from the checked MIR operands")
    inspect_pvs(dump)
    inspect_shallow(dump)
    inspect_probcut(dump)
    inspect_masks(dump)
    inspect_boundary(dump)
    return reject_mutants(dump)


def reject_mutants(dump):
    mutations = [
        (PVS, inspect_pvs, "Gt(move _606, const 0_u32)", "Gt(move _606, const 1_u32)"),
        (PVS, inspect_pvs, "SubWithOverflow(copy _593, copy _596)", "SubWithOverflow(copy _593, const 0_u32)"),
        (PVS, inspect_pvs, "_3 = move (_257.0: u32);", "_3 = copy _257;"),
        (PVS, inspect_pvs, "_324 = &_299;", "_324 = &_5;"),
        (PVS, inspect_boundary, "0: bb379, otherwise: bb378", "0: bb378, otherwise: bb379"),
        (SHALLOW, inspect_shallow, "_96 = copy (_92.2: square::Square);", "_96 = copy (_92.1: square::Square);"),
        (SHALLOW, inspect_shallow, "_150 = copy _40;", "_150 = copy _4;"),
        (SHALLOW, inspect_shallow, "_144 = TTProbeResult::<'_>::slot(move _145)", "_144 = TTProbeResult::<'_>::other(move _145)"),
        (SHALLOW, inspect_shallow, "_4 = move _141;", "_4 = copy _78;"),
        (PROBCUT, inspect_probcut, "_57 = copy _14;", "_57 = copy _3;"),
        (PROBCUT, inspect_probcut, "_62 = copy _52;", "_62 = Selectivity::None;"),
        (STRATEGY, inspect_boundary, "0: bb17, otherwise: bb5", "0: bb5, otherwise: bb17"),
        (ONE, inspect_masks, "const search::midgame::DEPTH1_MOVE_ORDER_MASKS", "const OTHER_MASKS"),
    ]
    fixed = {PVS, SHALLOW, SHALLOW+"-{closure#0}", PROBCUT, STRATEGY, ONE,
             "search-threading-thread-{impl#1}-is_parallel_pool"}
    with TemporaryDirectory() as folder:
        case = Path(folder)
        for stem in fixed:
            (case / f"reversi_core.{stem}{PASS}").symlink_to((dump / f"reversi_core.{stem}{PASS}").resolve())
        for p in dump.glob("reversi_core.search-midgame-DEPTH1_MOVE_ORDER_MASKS.built.after.mir"):
            (case / p.name).symlink_to(p.resolve())
        for stem, inspect, old, new in mutations:
            path = case / f"reversi_core.{stem}{PASS}"
            source = read(dump, stem)
            checked(source.count(old) == 1, f"{stem}: ambiguous mutation {old}")
            path.unlink()
            path.write_text(source.replace(old, new), encoding="utf-8")
            try:
                inspect(case)
            except ValueError:
                pass
            else:
                raise ValueError(f"{stem}: accepted mutation {old}")
            path.unlink()
            path.symlink_to((dump / path.name).resolve())
    return len(mutations)
