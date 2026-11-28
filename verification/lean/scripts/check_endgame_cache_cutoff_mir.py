"""Verify compiled MIR for EndGameCaches::SELECTIVITY, fail_low_score, and endgame_cache_cutoff."""

import argparse
from pathlib import Path
import re
import tempfile

from cache_owner_mir import Body, PASS
from check_result_guards_mir import normal_blocks, read
from extract_pvs_context_mir import Graph
from mir_common import checked


SEL = "search-endgame-{impl#0}-SELECTIVITY"
FAIL_LOW = "search-endgame-{impl#0}-fail_low_score"
FAIL_LOW_CL = FAIL_LOW + "-{closure#0}"
CUTOFF = "search-pvs-endgame_cache_cutoff"

SEL_BODY = "0: _0 = Selectivity::None; return;"
FAIL_LOW_CL_BODY = (
    "0: _3 = copy (*_2); _4 = &_3; _5 = &(*(_1.0: &score::DiscScore)); "
    "_0 = <DiscScore as PartialOrd>::le(move _4, move _5) => 1;\n"
    "1: return;"
)


def compare_blocks(stem, actual, expected_text):
    expected = {}
    for line in expected_text.strip().splitlines():
        num, ops = line.split(": ", 1)
        expected[int(num)] = ops
    checked(actual == expected, f"{stem}: MIR changed:\nactual={actual}\nexpected={expected}")


def verify_selectivity(dump):
    src = (dump / f"reversi_core.{SEL}.built.after.mir").read_text(encoding="utf-8")
    compare_blocks(SEL, normal_blocks(src, {}), SEL_BODY)


def verify_fail_low(dump):
    cl_src = read(dump, FAIL_LOW_CL)
    sig, = re.findall(r"^fn .*", cl_src, re.M)
    span = re.findall(r"\{closure@[^}]+\}", sig)[0].replace("\\", "/")
    compare_blocks(FAIL_LOW_CL, normal_blocks(cl_src, {}), FAIL_LOW_CL_BODY)

    g = Graph(dump, FAIL_LOW, 3)
    calls = [(fn, [g.val(a) for a in args]) for _, fn, args in g.calls()]
    ec = ("field", "arg1", 0)
    hashed = ("Board::hash", "arg2")
    slot = ("EndGameCache::<Deep>::slot", ec, hashed)
    probed = ("EndGameCache::<Deep>::probe", ec, slot, "arg2", "arg3")
    filter_fn = f"Option::<DiscScore>::filter::<{span}>"
    checked(len(calls) == 4 and calls[:3] == [
        ("Board::hash", ["arg2"]),
        ("EndGameCache::<Deep>::slot", [ec, hashed]),
        ("EndGameCache::<Deep>::probe", [ec, slot, "arg2", "arg3"]),
    ], f"{FAIL_LOW}: unexpected calls {calls}")
    fn4, args4 = calls[3]
    checked(fn4 == filter_fn and len(args4) == 2 and args4[0] == probed,
            f"{FAIL_LOW}: unexpected filter call {calls[3]}")
    m = re.fullmatch(re.escape(span) + r" \{ alpha: move (_\d+) \}", str(args4[1]))
    checked(m and g.val(m[1]) == "arg3", f"{FAIL_LOW}: filter closure did not capture alpha: {args4[1]}")
    checked(
        g.val("_0") == (filter_fn, probed, args4[1]),
        f"{FAIL_LOW}: returned value does not match filter result",
    )


def verify_cutoff(dump):
    body = Body(dump, CUTOFF)
    g = Graph(dump, CUTOFF, 8)
    checked(
        body.signature == (
            "endgame_cache_cutoff(_1: &mut SearchContext<'_>, _2: &Board, _3: &MoveList, "
            "_4: u32, _5: Score, _6: TTSlot<'_>, _7: bool, _8: &Arc<threading::thread::Thread>) -> Option<Score>"
        ),
        f"{CUTOFF}: unexpected signature {body.signature}",
    )

    # Context may only be read for its TT reference (`(*_1).5`), never mutated or aliased elsewhere.
    for ops in g.cfg.values():
        for op in ops:
            if "_1" not in re.findall(r"_\d+", op):
                continue
            checked(
                re.fullmatch(r"_\d+ = &\(\*\(\(\*_1\)\.5: &transposition_table::TranspositionTable\)\);", op),
                f"{CUTOFF}: unexpected SearchContext access: {op}",
            )

    # Thread::endgame_caches() must be called once before the move loop and never inside a cycle.
    cache_pc, _, cache_args, _ = body.named("threading::thread::Thread::endgame_caches")
    checked(
        cache_pc not in body.reachable(g.normal(cache_pc)[0]),
        f"{CUTOFF}: endgame_caches borrowed inside a loop",
    )
    checked(
        g.val(cache_args[0]) == "arg8",
        f"{CUTOFF}: endgame_caches borrowed from foreign thread",
    )

    # Verify child_alpha, move iteration, fail_low_score probe, negation, scaling, and TT store.
    caches = ("threading::thread::Thread::endgame_caches", "arg8")
    child_alpha = ("Score::to_disc_diff", ("<Score as Neg>::neg", "arg5"))
    mv_iter = ("<std::slice::Iter<'_, move_list::Move> as IntoIterator>::into_iter",
               ("MoveList::iter", "arg3"))
    mv = ("unSome", ("<std::slice::Iter<'_, move_list::Move> as Iterator>::next", mv_iter))
    next_board = ("Board::make_move_with_flipped", "arg2", ("field", mv, 0), ("field", mv, 2))
    fail_low = ("EndGameCaches::fail_low_score", caches, next_board, child_alpha)
    child_score = ("unSome", fail_low)
    parent_score = ("Score::from_disc_diff", ("<DiscScore as Neg>::neg", child_score))

    calls = {fn: [g.val(a) for a in args] for _, fn, args in g.calls()}
    checked(
        calls.get("EndGameCaches::fail_low_score") == [caches, next_board, child_alpha],
        f"{CUTOFF}: unexpected fail_low_score operands {calls.get('EndGameCaches::fail_low_score')}",
    )
    expected_store = [
        ("field", "arg1", 5),
        "arg6",
        "arg2",
        parent_score,
        "transposition_table::Bound::Lower",
        "arg4",
        ("Option::<Square>::Some", ("field", mv, 2)),
        "const search::endgame::EndGameCaches::SELECTIVITY",
        "const true",
        "arg7",
    ]
    stores = [[g.val(a) for a in args] for _, fn, args in g.calls() if fn == "TranspositionTable::store"]
    checked(stores == [expected_store], f"{CUTOFF}: unexpected TT store operands {stores}")

    # Return values must be either None (loop exhausted) or Some(parent_score) (cutoff hit).
    ret_defs = g.defs.get("_0", set())
    checked(
        ret_defs == {"Option::<Score>::None", "Option::<Score>::Some(move _54)"}
        and g.val("_54") == parent_score,
        f"{CUTOFF}: unexpected return values {ret_defs}",
    )


def verify(dump):
    verify_selectivity(dump)
    verify_fail_low(dump)
    verify_cutoff(dump)


def check(dump):
    verify(dump)
    stems = {
        SEL: f"reversi_core.{SEL}.built.after.mir",
        FAIL_LOW: f"reversi_core.{FAIL_LOW}{PASS}",
        FAIL_LOW_CL: f"reversi_core.{FAIL_LOW_CL}{PASS}",
        CUTOFF: f"reversi_core.{CUTOFF}{PASS}",
    }
    originals = {stem: (dump / fname).read_text(encoding="utf-8") for stem, fname in stems.items()}
    mutants = [
        (SEL, "Selectivity::None", "Selectivity::Level1"),
        (FAIL_LOW_CL, "<DiscScore as PartialOrd>::le", "<DiscScore as PartialOrd>::ge"),
        (FAIL_LOW, "((*_1).0: search::endgame::cache::EndGameCache", "((*_1).1: search::endgame::cache::EndGameCache"),
        (CUTOFF, "transposition_table::Bound::Lower", "transposition_table::Bound::Upper"),
        (CUTOFF, "const search::endgame::EndGameCaches::SELECTIVITY", "Selectivity::Level1"),
        (CUTOFF, "const true, move _53)", "const false, move _53)"),
        (CUTOFF, "_47 = &(*_2);", "_47 = &_29;"),
        (CUTOFF, "_46 = copy _6;", "_46 = copy _4;"),
        (CUTOFF, "_42 = <DiscScore as Neg>::neg(move _43)", "_42 = copy _43"),
        (CUTOFF, "_14 = <Score as Neg>::neg(move _15)", "_14 = copy _15"),
        (CUTOFF, "((*_1).5: &transposition_table::TranspositionTable)", "((*_1).10: u32)"),
    ]
    with tempfile.TemporaryDirectory(prefix="reversi-ec-cutoff-mutants-") as tmp:
        folder = Path(tmp)
        for stem, fname in stems.items():
            (folder / fname).write_text(originals[stem], encoding="utf-8")
        for stem, old, new in mutants:
            src = originals[stem]
            checked(old in src, f"missing mutation anchor in {stem}: {old}")
            mutated = src.replace(old, new, 1)
            (folder / stems[stem]).write_text(mutated, encoding="utf-8")
            try:
                verify(folder)
            except (ValueError, KeyError):
                pass
            else:
                raise ValueError(f"accepted {stem} mutation: {old} -> {new}")
            finally:
                (folder / stems[stem]).write_text(src, encoding="utf-8")
    return len(mutants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"verified endgame_cache_cutoff MIR and rejected {check(args.mir)} mutants")
