"""Check the normal MoveList initializer that implements MoveList.Generation.fill.

The marked entry blocks are Capacity.dispatch. A scalar back edge implements
one Initialization.scalar step. A paired back edge writes the two records of
Initialization.paired, and the residual mask controls its tail. In both cases
the cache changes only on equality with the opponent as in Wipeout.generationCache.
The flip kernels' value contracts and standard-library operations are inputs.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

from mir_common import PASS, checked
from check_move_iterator_mir import call, compare, graph
from mir_mutants import assert_mutants_rejected


FILL = "move_list-{impl#2}-fill_in_place"

# Rustc register and normal-edge tables, factored by the common entry path.
# Unit temporaries and a plain resume cleanup have no effects on these paths.
ENTRY = """

# Take all three output addresses from the same MoveList. Read the actual board and selected mask.
0: _4 = &raw mut ((*_3).0: move_list::move_array::MoveArray);
    _6 = copy _4;
    _5 = MoveArray::data_ptr_from(move _6) -> [return: bb1, unwind: cleanup];
1: _8 = copy _4;
    _7 = MoveArray::len_ptr_from(move _8) -> [return: bb2, unwind: cleanup];
2: _9 = &raw mut ((*_3).1: std::option::Option<square::Square>);
    _11 = &(*_1);
    _10 = Board::opponent(move _11) -> [return: bb3, unwind: cleanup];
3: _13 = copy _2;
    _12 = LegalMoves::bits(move _13) -> [return: bb4, unwind: cleanup];

# The empty path initializes length and cache without any record write.
4: _15 = const SKIP_EMPTY;
    switchInt(move _15) -> [0: bb5, otherwise: bb10];
5: _17 = copy _12;
    _16 = Eq(move _17, const 0_u64);
    switchInt(move _16) -> [0: bb9, otherwise: bb6];
6: _21 = copy _7;
    _20 = std::ptr::mut_ptr::<impl *mut usize>::write(move _21, const 0_usize) -> [return: bb7, unwind: cleanup];
7: _23 = copy _9;
    _24 = Option::<Square>::None;
    _22 = std::ptr::mut_ptr::<impl *mut Option<Square>>::write(move _23, move _24) -> [return: bb8, unwind: cleanup];
8: goto -> {early};
9: goto -> bb10;

# On a nonzero mask, fewer than two bits means a singleton. Use ctz of that mask.
10: _28 = copy _12;
    _30 = copy _12;
    _31 = SubWithOverflow(copy _30, const 1_u64);
    assert(!move (_31.1: bool), "attempt to compute `{} - {}`, which would overflow", move _30, const 1_u64) -> [success: bb11, unwind: cleanup];
11: _29 = move (_31.0: u64);
    _27 = BitAnd(move _28, move _29);
    _26 = Eq(move _27, const 0_u64);
    switchInt(move _26) -> [0: bb25, otherwise: bb12];
12: _35 = copy _12;
    _34 = core::num::<impl u64>::trailing_zeros(move _35) -> [return: bb13, unwind: cleanup];
13: _33 = move _34 as u8 (IntToInt);
    _37 = copy _33;
    _36 = Square::from_u8_unchecked(move _37) -> [return: bb14, unwind: cleanup];

# Generate the flip for that square. Store at base, publish length one, and classify exact wipeout.
14: _39 = copy _36;
    _41 = &(*_1);
    _40 = Board::player(move _41) -> [return: bb15, unwind: cleanup];
15: _42 = copy _10;
    _38 = flip::flip(move _39, move _40, move _42) -> [return: bb16, unwind: cleanup];
16: _45 = copy _5;
    _47 = copy _36;
    _48 = copy _38;
    _46 = move_list::Move::new(move _47, move _48) -> [return: bb17, unwind: cleanup];
17: _44 = std::ptr::mut_ptr::<impl *mut move_list::Move>::write(move _45, move _46) -> [return: bb18, unwind: cleanup];
18: _50 = copy _7;
    _49 = std::ptr::mut_ptr::<impl *mut usize>::write(move _50, const 1_usize) -> [return: bb19, unwind: cleanup];
19: _52 = copy _9;
    _55 = &_38;
    _56 = &_10;
    _54 = <Bitboard as PartialEq>::eq(move _55, move _56) -> [return: bb20, unwind: cleanup];
20: switchInt(move _54) -> [0: bb22, otherwise: bb21];
21: _57 = copy _36;
    _53 = Option::<Square>::Some(move _57);
    goto -> bb23;
22: _53 = Option::<Square>::None;
    goto -> bb23;
23: _51 = std::ptr::mut_ptr::<impl *mut Option<Square>>::write(move _52, move _53) -> [return: bb24, unwind: cleanup];
24: goto -> {early};

# Bulk path precondition, including the debug-only minimum population check.
25: _59 = const true;
    switchInt(move _59) -> [0: bb27, otherwise: bb26];
26: _63 = copy _12;
    _62 = core::num::<impl u64>::count_ones(move _63) -> [return: bb28, unwind: cleanup];
27: goto -> bb31;
28: _61 = Ge(move _62, const 2_u32);
    switchInt(move _61) -> [0: bb30, otherwise: bb29];
29: goto -> bb31;
30: _64 = panic(const "assertion failed: bb.count_ones() >= 2") -> cleanup;
"""


SCALAR = """

# Start at slot zero with no cached square.
31: _65 = Option::<Square>::None;
    _66 = const 0_usize;
    _68 = &(*_1);
    _67 = Board::player(move _68) -> [return: bb32, unwind: cleanup];
32: goto -> bb33;

# Each iteration pops the lowest remaining bit and forms that same square.
33: falseUnwind -> [real: bb34, unwind: cleanup];
34: _72 = copy _12;
    _71 = Ne(move _72, const 0_u64);
    switchInt(move _71) -> [0: bb53, otherwise: bb35];
35: _75 = copy _12;
    _74 = core::num::<impl u64>::trailing_zeros(move _75) -> [return: bb36, unwind: cleanup];
36: _73 = move _74 as u8 (IntToInt);
    _77 = copy _12;
    _78 = SubWithOverflow(copy _77, const 1_u64);
    assert(!move (_78.1: bool), "attempt to compute `{} - {}`, which would overflow", move _77, const 1_u64) -> [success: bb37, unwind: cleanup];
37: _76 = move (_78.0: u64);
    _12 = BitAnd(copy _12, move _76);
    _80 = copy _73;
    _79 = Square::from_u8_unchecked(move _80) -> [return: bb38, unwind: cleanup];

# Compute the flip against the original coordinates and write the record at len.
38: _82 = copy _79;
    _83 = copy _67;
    _84 = copy _10;
    _81 = flip::flip(move _82, move _83, move _84) -> [return: bb39, unwind: cleanup];
39: _86 = const true;
    switchInt(move _86) -> [0: bb43, otherwise: bb40];
40: _89 = copy _66;
    _88 = Lt(move _89, const legal_moves::LegalMoves::MAX_COUNT);
    switchInt(move _88) -> [0: bb42, otherwise: bb41];
41: goto -> bb44;
42: _90 = panic(const "assertion failed: len < LegalMoves::MAX_COUNT") -> cleanup;
43: goto -> bb44;
44: _93 = copy _5;
    _94 = copy _66;
    _92 = std::ptr::mut_ptr::<impl *mut move_list::Move>::add(move _93, move _94) -> [return: bb45, unwind: cleanup];
45: _96 = copy _79;
    _97 = copy _81;
    _95 = move_list::Move::new(move _96, move _97) -> [return: bb46, unwind: cleanup];
46: _91 = std::ptr::mut_ptr::<impl *mut move_list::Move>::write(move _92, move _95) -> [return: bb47, unwind: cleanup];

# Advance one slot. Replace the cached square iff the flip equals the opponent.
47: _98 = AddWithOverflow(copy _66, const 1_usize);
    assert(!move (_98.1: bool), "attempt to compute `{} + {}`, which would overflow", copy _66, const 1_usize) -> [success: bb48, unwind: cleanup];
48: _66 = move (_98.0: usize);
    _100 = &_81;
    _101 = &_10;
    _99 = <Bitboard as PartialEq>::eq(move _100, move _101) -> [return: bb49, unwind: cleanup];
49: switchInt(move _99) -> [0: bb51, otherwise: bb50];
50: _103 = copy _79;
    _102 = Option::<Square>::Some(move _103);
    _65 = move _102;
    goto -> bb52;
51: goto -> bb52;
52: goto -> bb33;

# Publish the actual final length and the accumulated last wipeout.
53: _108 = copy _7;
    _109 = copy _66;
    _107 = std::ptr::mut_ptr::<impl *mut usize>::write(move _108, move _109) -> [return: bb54, unwind: cleanup];
54: _111 = copy _9;
    _112 = copy _65;
    _110 = std::ptr::mut_ptr::<impl *mut Option<Square>>::write(move _111, move _112) -> [return: bb55, unwind: cleanup];
55: goto -> bb57;
56: goto -> bb57;
57: return;
"""


PAIRED = """

# Build one AVX512 context using the current player and opponent bits.
31: _65 = Option::<Square>::None;
    _66 = const 0_usize;
    _70 = &(*_1);
    _69 = Board::player(move _70) -> [return: bb32, unwind: cleanup];
32: _68 = Bitboard::bits(move _69) -> [return: bb33, unwind: cleanup];
33: _72 = copy _10;
    _71 = Bitboard::bits(move _72) -> [return: bb34, unwind: cleanup];
34: _67 = flip_avx512::BoardCtx::new(move _68, move _71) -> [return: bb35, unwind: cleanup];
35: _74 = copy _10;
    _73 = Bitboard::bits(move _74) -> [return: bb36, unwind: cleanup];

# The range has population / 2 visits, then control goes to the actual residual mask.
36: _78 = copy _12;
    _77 = core::num::<impl u64>::count_ones(move _78) -> [return: bb37, unwind: cleanup];
37: _76 = move _77 as usize (IntToInt);
    _79 = Eq(const 2_usize, const 0_usize);
    assert(!move _79, "attempt to divide `{}` by zero", copy _76) -> [success: bb38, unwind: cleanup];
38: _75 = Div(move _76, const 2_usize);
    _83 = copy _75;
    _82 = std::ops::Range::<usize> { start: const 0_usize, end: move _83 };
    _81 = <std::ops::Range<usize> as IntoIterator>::into_iter(move _82) -> [return: bb39, unwind: cleanup];
39: _84 = move _81;
    goto -> bb40;
40: falseUnwind -> [real: bb41, unwind: cleanup];
41: _89 = &mut _84;
    _88 = &mut (*_89);
    _87 = <std::ops::Range<usize> as Iterator>::next(move _88) -> [return: bb42, unwind: cleanup];
42: _90 = discriminant(_87);
    switchInt(move _90) -> [0: bb44, 1: bb45, otherwise: bb43];
43: unreachable;
44: falseEdge -> [real: bb46, imaginary: bb45];

# Either pop two bits in ascending order, or skip the tail if the mask is now empty.
45: _94 = copy _12;
    _93 = core::num::<impl u64>::trailing_zeros(move _94) -> [return: bb47, unwind: cleanup];
46: _159 = copy _12;
    _158 = Ne(move _159, const 0_u64);
    switchInt(move _158) -> [0: bb77, otherwise: bb76];
47: _92 = move _93 as u8 (IntToInt);
    _96 = copy _12;
    _97 = SubWithOverflow(copy _96, const 1_u64);
    assert(!move (_97.1: bool), "attempt to compute `{} - {}`, which would overflow", move _96, const 1_u64) -> [success: bb48, unwind: cleanup];
48: _95 = move (_97.0: u64);
    _12 = BitAnd(copy _12, move _95);
    _100 = copy _12;
    _99 = core::num::<impl u64>::trailing_zeros(move _100) -> [return: bb49, unwind: cleanup];
49: _98 = move _99 as u8 (IntToInt);
    _102 = copy _12;
    _103 = SubWithOverflow(copy _102, const 1_u64);
    assert(!move (_103.1: bool), "attempt to compute `{} - {}`, which would overflow", move _102, const 1_u64) -> [success: bb50, unwind: cleanup];

# Feed both popped squares to the paired callee and preserve lane identity.
50: _101 = move (_103.0: u64);
    _12 = BitAnd(copy _12, move _101);
    _107 = &_67;
    _109 = copy _92;
    _108 = move _109 as usize (IntToInt);
    _111 = copy _98;
    _110 = move _111 as usize (IntToInt);
    _106 = flip_avx512::BoardCtx::flip2_wide_load(move _107, move _108, move _110) -> [return: bb51, unwind: cleanup];
51: _104 = copy (_106.0: u64);
    _105 = copy (_106.1: u64);
    _113 = copy _104;
    _112 = Bitboard::new(move _113) -> [return: bb52, unwind: cleanup];
52: _115 = copy _105;
    _114 = Bitboard::new(move _115) -> [return: bb53, unwind: cleanup];
53: _117 = copy _92;
    _116 = Square::from_u8_unchecked(move _117) -> [return: bb54, unwind: cleanup];
54: _119 = copy _98;
    _118 = Square::from_u8_unchecked(move _119) -> [return: bb55, unwind: cleanup];

# Check space for both initialized records before writing them at len and len+1.
55: _121 = const true;
    switchInt(move _121) -> [0: bb60, otherwise: bb56];
56: _125 = copy _66;
    _126 = AddWithOverflow(copy _125, const 2_usize);
    assert(!move (_126.1: bool), "attempt to compute `{} + {}`, which would overflow", move _125, const 2_usize) -> [success: bb57, unwind: cleanup];
57: _124 = move (_126.0: usize);
    _123 = Le(move _124, const legal_moves::LegalMoves::MAX_COUNT);
    switchInt(move _123) -> [0: bb59, otherwise: bb58];
58: goto -> bb61;
59: _127 = panic(const "assertion failed: len + 2 <= LegalMoves::MAX_COUNT") -> cleanup;
60: goto -> bb61;
61: _131 = copy _5;
    _132 = copy _66;
    _130 = std::ptr::mut_ptr::<impl *mut move_list::Move>::add(move _131, move _132) -> [return: bb62, unwind: cleanup];
62: _134 = copy _116;
    _135 = copy _112;
    _133 = move_list::Move::new(move _134, move _135) -> [return: bb63, unwind: cleanup];
63: _129 = std::ptr::mut_ptr::<impl *mut move_list::Move>::write(move _130, move _133) -> [return: bb64, unwind: cleanup];
64: _138 = copy _5;
    _140 = copy _66;
    _141 = AddWithOverflow(copy _140, const 1_usize);
    assert(!move (_141.1: bool), "attempt to compute `{} + {}`, which would overflow", move _140, const 1_usize) -> [success: bb65, unwind: cleanup];
65: _139 = move (_141.0: usize);
    _137 = std::ptr::mut_ptr::<impl *mut move_list::Move>::add(move _138, move _139) -> [return: bb66, unwind: cleanup];
66: _143 = copy _118;
    _144 = copy _114;
    _142 = move_list::Move::new(move _143, move _144) -> [return: bb67, unwind: cleanup];
67: _136 = std::ptr::mut_ptr::<impl *mut move_list::Move>::write(move _137, move _142) -> [return: bb68, unwind: cleanup];

# Advance by two. Compare both flips with opponent bits in record order, so the last hit wins.
68: _145 = AddWithOverflow(copy _66, const 2_usize);
    assert(!move (_145.1: bool), "attempt to compute `{} + {}`, which would overflow", copy _66, const 2_usize) -> [success: bb69, unwind: cleanup];
69: _66 = move (_145.0: usize);
    _148 = copy _104;
    _149 = copy _73;
    _147 = Eq(move _148, move _149);
    switchInt(move _147) -> [0: bb71, otherwise: bb70];
70: _151 = copy _116;
    _150 = Option::<Square>::Some(move _151);
    _65 = move _150;
    goto -> bb72;
71: goto -> bb72;
72: _153 = copy _105;
    _154 = copy _73;
    _152 = Eq(move _153, move _154);
    switchInt(move _152) -> [0: bb74, otherwise: bb73];
73: _156 = copy _118;
    _155 = Option::<Square>::Some(move _156);
    _65 = move _155;
    goto -> bb75;
74: goto -> bb75;
75: goto -> bb40;

# If an odd square remains, write its scalar AVX512 flip and advance one slot.
76: _162 = copy _12;
    _161 = core::num::<impl u64>::trailing_zeros(move _162) -> [return: bb78, unwind: cleanup];
77: goto -> bb94;
78: _160 = move _161 as u8 (IntToInt);
    _164 = &_67;
    _166 = copy _160;
    _165 = move _166 as usize (IntToInt);
    _163 = flip_avx512::BoardCtx::flip1(move _164, move _165) -> [return: bb79, unwind: cleanup];
79: _168 = copy _163;
    _167 = Bitboard::new(move _168) -> [return: bb80, unwind: cleanup];
80: _170 = copy _160;
    _169 = Square::from_u8_unchecked(move _170) -> [return: bb81, unwind: cleanup];
81: _172 = const true;
    switchInt(move _172) -> [0: bb85, otherwise: bb82];
82: _175 = copy _66;
    _174 = Lt(move _175, const legal_moves::LegalMoves::MAX_COUNT);
    switchInt(move _174) -> [0: bb84, otherwise: bb83];
83: goto -> bb86;
84: _176 = panic(const "assertion failed: len < LegalMoves::MAX_COUNT") -> cleanup;
85: goto -> bb86;
86: _179 = copy _5;
    _180 = copy _66;
    _178 = std::ptr::mut_ptr::<impl *mut move_list::Move>::add(move _179, move _180) -> [return: bb87, unwind: cleanup];
87: _182 = copy _169;
    _183 = copy _167;
    _181 = move_list::Move::new(move _182, move _183) -> [return: bb88, unwind: cleanup];
88: _177 = std::ptr::mut_ptr::<impl *mut move_list::Move>::write(move _178, move _181) -> [return: bb89, unwind: cleanup];
89: _184 = AddWithOverflow(copy _66, const 1_usize);
    assert(!move (_184.1: bool), "attempt to compute `{} + {}`, which would overflow", copy _66, const 1_usize) -> [success: bb90, unwind: cleanup];
90: _66 = move (_184.0: usize);
    _186 = copy _163;
    _187 = copy _73;
    _185 = Eq(move _186, move _187);
    switchInt(move _185) -> [0: bb92, otherwise: bb91];
91: _189 = copy _169;
    _188 = Option::<Square>::Some(move _189);
    _65 = move _188;
    goto -> bb93;
92: goto -> bb93;
93: goto -> bb94;

# Publish the actual final length and the accumulated last wipeout.
94: _191 = copy _7;
    _192 = copy _66;
    _190 = std::ptr::mut_ptr::<impl *mut usize>::write(move _191, move _192) -> [return: bb95, unwind: cleanup];
95: _194 = copy _9;
    _195 = copy _65;
    _193 = std::ptr::mut_ptr::<impl *mut Option<Square>>::write(move _194, move _195) -> [return: bb96, unwind: cleanup];
96: goto -> bb98;
97: goto -> bb98;
98: return;
"""


def mir(dump, stem):
    return dump / f"reversi_core.{stem}.{PASS}.after.mir"


def instructions(source):
    return {pc: [op.replace("core::panicking::panic", "panic") for op in ops]
            for pc, ops in graph(source).items()}


def normal_fill(source):
    checked(re.fullmatch(r"// MIR for `move_list::<impl at .+>::fill_in_place` after " + PASS,
                         source.splitlines()[0]) and re.search(
        r"^fn move_list::<impl at .+>::fill_in_place\(_1: &Board, _2: LegalMoves, _3: \*mut MoveList\) -> \(\) \{$",
        source, re.M), "unexpected MoveList::fill_in_place signature")
    cfg = instructions(source)
    cleanup, = [pc for pc, ops in cfg.items() if ops == ["unwind"]]
    units = set(re.findall(r"^\s*let (?:mut )?(_\d+): \(\);$", source, re.M))
    def meaningful(op):
        discard = re.fullmatch(r"(_\d+) = const \(\);", op)
        if discard:
            checked(discard[1] in units, f"fill_in_place: unit write into {op}")
        return not discard
    return {pc: [op.replace(f"unwind: bb{cleanup}]", "unwind: cleanup]").replace(f"-> bb{cleanup};", "-> cleanup;")
                 for op in ops if meaningful(op)] for pc, ops in cfg.items() if pc != cleanup}


def table(text):
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"(\d+): (.+)", line):
            rows.append((int(m[1]), [m[2]]))
        else:
            rows[-1][1].append(line)
    checked([pc for pc, _ in rows] == list(range(len(rows))), "invalid MoveList initializer specification")
    return [ops for _, ops in rows]


def check_fill(dump):
    source = mir(dump, FILL).read_text(encoding="utf-8")
    cfg = normal_fill(source)
    paired = any("flip_avx512::BoardCtx::" in op for ops in cfg.values() for op in ops)
    entry = ENTRY.replace("{early}", "bb97" if paired else "bb56")
    compare("MoveList::fill_in_place", cfg, table(entry + (PAIRED if paired else SCALAR)))
    return paired


def check_support(dump):
    def load(stem):
        return instructions(mir(dump, stem).read_text(encoding="utf-8"))

    simple = {
        "board-{impl#1}-player": [["_0 = copy ((*_1).0: bitboard::Bitboard);", "return;"]],
        "board-{impl#1}-opponent": [["_0 = copy ((*_1).1: bitboard::Bitboard);", "return;"]],
        "bitboard-{impl#0}-bits": [["_0 = copy (_1.0: u64);", "return;"]],
        "bitboard-{impl#0}-new": [["_2 = copy _1;", "_0 = Bitboard(move _2);", "return;"]],
        "bitboard-{impl#0}-is_empty": [["_2 = copy (_1.0: u64);", "_0 = Eq(move _2, const 0_u64);", "return;"]],
        "bitboard-{impl#24}-eq": [["_3 = copy ((*_1).0: u64);", "_4 = copy ((*_2).0: u64);",
                                 "_0 = Eq(move _3, move _4);", "return;"]],
        "legal_moves-{impl#0}-bits": [["_2 = copy (_1.0: bitboard::Bitboard);",
                                     call("_0 = Bitboard::bits(move _2)", 1, 2)], ["return;"], ["unwind"]],
        "move_list-move_array-{impl#0}-data_ptr_from": [
            ["_2 = copy _1;", "_0 = move _2 as *mut move_list::Move (PtrToPtr);", "return;"]],
        "move_list-move_array-{impl#0}-len_ptr_from": [["_0 = &raw mut ((*_1).1: usize);", "return;"]],
        "move_list-move_array-_-{constant#0}": [
            [call("_0 = offset_of::<MoveArray>(const 0_u32, const 0_u32)", 1, 2)], ["return;"], ["unwind"]],
        "move_list-move_array-_": [
            ["_1 = Eq(const move_list::move_array::_::{constant#0}, const 0_usize);",
             "switchInt(move _1) -> [0: bb1, otherwise: bb2];"],
            ['_2 = panic(const "assertion failed: offset_of!(MoveArray, data) == 0") -> bb3;'],
            ["_0 = const ();", "return;"], ["unwind"]],
    }
    for stem, expected in simple.items():
        compare(stem, load(stem), expected)

    # These are all paths for values in the domains established by writeSafetyContract
    # and generationContract: idx < 64 and flips != 0.
    nonzero_move = {
        0: ["_4 = const true;", "switchInt(move _4) -> [0: bb6, otherwise: bb1];"],
        1: ["_7 = copy _2;", call("_6 = Bitboard::is_empty(move _7)", 2, 8)],
        2: ["switchInt(move _6) -> [0: bb4, otherwise: bb3];"],
        4: ["_5 = const ();", "_3 = const ();", "goto -> bb7;"],
        6: ["_3 = const ();", "goto -> bb7;"],
        7: ["_11 = copy _2;", "_12 = copy _1;",
            "_0 = move_list::Move { flipped: move _11, value: const 0_i32, sq: move _12 };", "return;"],
    }
    in_range_square = {
        0: ["_3 = const true;", "switchInt(move _3) -> [0: bb6, otherwise: bb1];"],
        1: ["_6 = copy _1;", "_5 = Lt(move _6, const 64_u8);", "switchInt(move _5) -> [0: bb3, otherwise: bb2];"],
        2: ["_4 = const ();", "_2 = const ();", "goto -> bb7;"],
        6: ["_2 = const ();", "goto -> bb7;"],
        7: ["_19 = copy _1;", call("_0 = std::intrinsics::transmute::<u8, Square>(move _19)", 8, 9)],
        8: ["return;"],
    }
    for stem, paths in (("move_list-{impl#0}-new", nonzero_move), ("square-{impl#0}-from_u8_unchecked", in_range_square)):
        cfg = load(stem)
        for pc, ops in paths.items():
            checked(cfg.get(pc) == ops, f"{stem} bb{pc}: changed normal constructor path")


def check_flip_forwarding(dump, paired):
    # Generation.entry uses the value of this target-specific flip. Validate
    # the Rust dispatch and wide context setup before applying kernel contracts.
    cfg = instructions(mir(dump, "flip-flip").read_text(encoding="utf-8"))
    targets = ("avx512",) if paired else ("scalar", "neon", "avx2")
    target, = [target for target in targets if cfg.get(2) == [
        call(f"_4 = flip_{target}::flip(move _5, move _6, move _8)", 3, 5)]]
    compare("flip::flip", cfg, [
        ["_5 = copy _1;", "_7 = copy _2;", call("_6 = Bitboard::bits(move _7)", 1, 5)],
        ["_9 = copy _3;", call("_8 = Bitboard::bits(move _9)", 2, 5)],
        [call(f"_4 = flip_{target}::flip(move _5, move _6, move _8)", 3, 5)],
        [call("_0 = Bitboard::new(move _4)", 4, 5)], ["return;"], ["unwind"],
    ])
    if paired:
        ctx = instructions(mir(dump, "flip-flip_avx512-{impl#0}-new").read_text(encoding="utf-8"))
        intr = "std::arch::x86_64::"
        compare("AVX512 BoardCtx::new", ctx, [
            ["_5 = copy _1;", "_4 = move _5 as i64 (IntToInt);",
             call(f"_3 = {intr}_mm512_set1_epi64(move _4)", 1, 6)],
            ["_9 = copy _2;", "_8 = Not(move _9);", "_7 = move _8 as i64 (IntToInt);",
             call(f"_6 = {intr}_mm512_set1_epi64(move _7)", 2, 6)],
            [call(f"_10 = {intr}_mm512_setzero_si512()", 3, 6)],
            ["_12 = const 9223372036854775808_u64 as i64 (IntToInt);",
             call(f"_11 = {intr}_mm512_set1_epi64(move _12)", 4, 6)],
            [call(f"_13 = {intr}_mm512_set1_epi64(const -1_i64)", 5, 6)],
            ["_0 = flip_avx512::BoardCtx { pp: move _3, no: move _6, zero: move _10, msb: move _11, all_ones: move _13 };",
             "return;"], ["unwind"],
        ])


def check(dump):
    paired = check_fill(dump)
    check_support(dump)
    check_flip_forwarding(dump, paired)
    return paired


def check_mutants(dump):
    paired = check_fill(dump)
    common = (
        ("move _21, const 0_usize", "move _21, const 1_usize"),
        ("_42 = copy _10;", "_42 = copy _40;"),
        ("_53 = Option::<Square>::Some(move _57);", "_53 = Option::<Square>::None;"),
    )
    scalar = (
        ("_12 = BitAnd(copy _12, move _76);", "_12 = copy _12;"),
        ("_94 = copy _66;", "_94 = const 0_usize;"),
        ("_101 = &_10;", "_101 = &_81;"),
        ("_109 = copy _66;", "_109 = const 1_usize;"),
    )
    pair = (
        ("Div(move _76, const 2_usize)", "Div(move _76, const 3_usize)"),
        ("_104 = copy (_106.0: u64);", "_104 = copy (_106.1: u64);"),
        ("_144 = copy _114;", "_144 = copy _112;"),
        ("_155 = Option::<Square>::Some(move _156);", "_155 = Option::<Square>::None;"),
        ("0: bb77, otherwise: bb76", "0: bb76, otherwise: bb77"),
        ("_192 = copy _66;", "_192 = const 1_usize;"),
    )
    filename = mir(dump, FILL).name
    return assert_mutants_rejected(
        dump,
        [(filename, old, new, FILL) for old, new in common + (pair if paired else scalar)],
        check_fill,
        error_check=lambda err, _: checked(
            str(err).startswith("MoveList::fill_in_place bb"), f"unexpected mutation error: {err}"
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    mode = "paired" if check(args.mir) else "scalar"
    print(f"verified {mode} move generation and rejected {check_mutants(args.mir)} MIR mutants")
