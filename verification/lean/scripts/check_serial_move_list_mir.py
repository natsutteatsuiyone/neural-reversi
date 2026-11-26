"""Check the generation and initialized-array accessors used by serial PVS."""

from argparse import ArgumentParser
from pathlib import Path
import re

from mir_common import PASS, checked, blocks
from extract_header import ordinary
from mir_mutants import assert_mutants_rejected


LIST = "{impl#2}-"
ARRAY = "move_array-{impl#0}-"
INDEX = "{impl#1}-index"

# All normal paths. Calls at this boundary use Rust's MaybeUninit, slice, and
# pointer semantics. MoveList's raw writes must initialize the same storage.
ACCESSORS = {
    LIST + "new": [
        ["_2 = &(*_1);", "_4 = &(*_1);", "_3 = Board::legal_moves(move _4) -> [return: bb1, unwind: bb3];"],
        ["_0 = MoveList::with_moves(move _2, move _3) -> [return: bb2, unwind: bb3];"],
        ["return;"],
    ],
    LIST + "with_moves": [
        ["_3 = MaybeUninit::<MoveList>::uninit() -> [return: bb1, unwind: bb5];"],
        ["_5 = &(*_1);", "_6 = copy _2;", "_8 = &mut _3;",
         "_7 = MaybeUninit::<MoveList>::as_mut_ptr(move _8) -> [return: bb2, unwind: bb5];"],
        ["_4 = MoveList::fill_in_place::<false>(move _5, move _6, move _7) -> [return: bb3, unwind: bb5];"],
        ["_9 = move _3;", "_0 = MaybeUninit::<MoveList>::assume_init(move _9) -> [return: bb4, unwind: bb5];"],
        ["return;"],
    ],
    LIST + "count": [
        ["_2 = &((*_1).0: move_list::move_array::MoveArray);",
         "_0 = MoveArray::len(move _2) -> [return: bb1, unwind: bb2];"],
        ["return;"],
    ],
    LIST + "wipeout_move": [["_0 = copy ((*_1).1: std::option::Option<square::Square>);", "return;"]],
    INDEX: [
        ["_5 = &((*_1).0: move_list::move_array::MoveArray);",
         "_4 = MoveArray::as_slice(move _5) -> [return: bb1, unwind: bb3];"],
        ["_6 = copy _2;", "_7 = PtrMetadata(copy _4);", "_8 = Lt(copy _6, copy _7);",
         'assert(move _8, "index out of bounds: the length is {} but the index is {}", move _7, copy _6) '
         "-> [success: bb2, unwind: bb3];"],
        ["_3 = &(*_4)[_6];", "_0 = &(*_3);", "return;"],
    ],
    ARRAY + "as_slice": [
        ["_4 = &(*_1);", "_3 = MoveArray::as_ptr(move _4) -> [return: bb1, unwind: bb4];"],
        ["_6 = &(*_1);", "_5 = MoveArray::len(move _6) -> [return: bb2, unwind: bb4];"],
        ["_2 = std::slice::from_raw_parts::<'_, move_list::Move>(move _3, move _5) -> [return: bb3, unwind: bb4];"],
        ["_0 = &(*_2);", "return;"],
    ],
    ARRAY + "as_ptr": [
        ["_4 = &((*_1).0: [std::mem::MaybeUninit<move_list::Move>; 35]);",
         "_3 = move _4 as &[std::mem::MaybeUninit<move_list::Move>] (PointerCoercion(Unsize, Implicit));",
         "_2 = core::slice::<impl [MaybeUninit<move_list::Move>]>::as_ptr(move _3) -> [return: bb1, unwind: bb3];"],
        ["_0 = std::ptr::const_ptr::<impl *const MaybeUninit<move_list::Move>>::cast::<move_list::Move>(move _2) "
         "-> [return: bb2, unwind: bb3];"],
        ["return;"],
    ],
    ARRAY + "len": [["_0 = copy ((*_1).1: usize);", "return;"]],
}


def mir(dump, function):
    return dump / f"reversi_core.move_list-{function}.{PASS}.after.mir"


def check(dump):
    for function, expected in ACCESSORS.items():
        source = mir(dump, function).read_text(encoding="utf-8")
        cfg = blocks(source)
        checked(cfg.keys() == set(range(len(expected))), f"{function}: unexpected normal move-list blocks")
        for pc, wanted in enumerate(expected):
            actual = [line for line in ordinary(cfg[pc]) if not re.fullmatch(
                r"FakeRead\(ForLet\(None\), _\d+\);|"
                r"AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", line)]
            checked(actual == wanted, f"{function}: bb{pc} changes serial move-list data flow: {actual}")


def check_mutants(dump):
    mutations = (
        (INDEX, "_6 = copy _2;", "_6 = const 0_usize;"),
        (LIST + "with_moves", "MoveList::fill_in_place::<false>", "MoveList::fill_in_place::<true>"),
        (LIST + "count", "_0 = MoveArray::len(move _2) -> [return: bb1, unwind: bb2];",
         "_0 = const 1_usize;\n        goto -> bb1;"),
        (LIST + "wipeout_move", "copy ((*_1).1: std::option::Option<square::Square>)", "Option::<Square>::None"),
        (ARRAY + "as_slice", "(move _3, move _5)", "(move _3, const 1_usize)"),
    )
    return assert_mutants_rejected(
        dump,
        [(mir(dump, function).name, old, new, function) for function, old, new in mutations],
        check,
        error_check=lambda err, fn: checked(str(err).startswith(f"{fn}:"), f"wrong mutant failure: {err}"),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    check(args.mir)
    print(f"verified serial move-list accessors and rejected {check_mutants(args.mir)} MIR mutants")
