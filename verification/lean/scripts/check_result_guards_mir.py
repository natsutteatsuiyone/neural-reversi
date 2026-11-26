"""Check the normal MIR that selects and forwards the public endgame score.

The publication extractor checks the saved-iteration branch. These checks cover
the unsaved branch and both quick-move callers, including receive failure.
The initial endgame estimate takes the current table or static/depth-two fallback.
Standard-library Arc, Option, Result, bool::then, iterators and channel operations
have their usual semantics. Score and tags use the TT packing contracts.
"""

import argparse
from pathlib import Path
import re

from mir_common import PASS, checked, blocks
from extract_header import ordinary


RESULT = "search-result-"
SEARCH = "search-{impl#2}-"
FINISH = RESULT + "{impl#1}-into_result"
CACHED = RESULT + "tt_move"
QUICK = RESULT + "quick_move"
RECV = SEARCH + "execute_search"
PUBLIC = RESULT + "{impl#2}-"
ENTRY = "transposition_table-{impl#3}-"
ESTIMATE = "search-endgame-estimate_aspiration_base_score"
ADVANCE_GENERATION = "transposition_table-{impl#7}-increment_generation"
PROBE_DATA = "transposition_table-{impl#2}-data"

CALLBACKS = {
    CACHED + "-{closure#0}": "cached_result",
    FINISH + "-{closure#0}": "no_iteration",
    QUICK + "-{closure#0}": "evaluate",
    QUICK + "-{closure#1}": "choose_best",
    RECV + "-{closure#0}": "disconnected",
}


def read(dump, stem):
    return (dump / f"reversi_core.{stem}.{PASS}.after.mir").read_text(encoding="utf-8")


def callback_names(dump):
    names = {}
    for stem, label in CALLBACKS.items():
        source = read(dump, stem)
        signature, = re.findall(r"^fn .*", source, re.M)
        span, = re.findall(r"\{closure@[^}]+\}", signature)
        checked(span not in names, f"overlapping closure identity at {label}")
        names[span] = label
    return names


def normal_blocks(source, callbacks):
    """Retain all ordinary instructions, following normal successors on return."""
    cleanup = set(re.findall(r"^    (bb\d+) \(cleanup\): \{", source, re.M))
    def compact(line):
        line = line.replace(" = core::panicking::panic(", " = panic(")
        for span, name in callbacks.items():
            line = line.replace(span, name)
        if m := re.fullmatch(r"(.+) -> \[return: bb(\d+), unwind: (bb\d+)\];", line):
            checked(m[3] in cleanup, f"unexpected unwind target {m[3]}")
            return f"{m[1]} => {m[2]};"
        if m := re.fullmatch(r"falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];|goto -> bb(\d+);", line):
            return f"goto {m[1] or m[2]};"
        return line
    return {pc: " ".join(compact(op) for op in ordinary(ops) if not re.fullmatch(
        r"FakeRead\(ForLet\(None\), _\d+\);|PlaceMention\([^;]+\);|"
        r"AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", op))
        for pc, ops in blocks(source).items()}


# Pinned compiler. One line per normal block, in block-number order.
EXPECTED = {
    "search-midgame-compute_start_depth": """
_3 = copy _1; _2 = core::num::<impl u32>::is_multiple_of(move _3, const 2_u32) => 1;
switchInt(move _2) -> [0: bb3, otherwise: bb2];
_0 = const 2_u32; goto 4;
_0 = const 1_u32; goto 4;
return;
""",
    "search-strategy-{impl#0}-evaluate": """
_3 = &(*_1); _4 = &(*_2); _0 = evaluate(move _3, move _4) => 1;
return;
""",
    "search-strategy-{impl#0}-shallow_search": """
switchInt(copy _3) -> [0: bb2, 1: bb3, otherwise: bb1];
_13 = &mut (*_1); _14 = &(*_2); _15 = copy _3; _16 = copy _4; _17 = copy _5; _0 = evaluate_shallow_with_tt(move _13, move _14, move _15, move _16, move _17) => 8;
goto 5;
goto 4;
_9 = &mut (*_1); _10 = &(*_2); _11 = copy _4; _12 = copy _5; _0 = evaluate_depth1(move _9, move _10, move _11, move _12) => 7;
_7 = &(*_1); _8 = &(*_2); _0 = evaluate(move _7, move _8) => 6;
goto 9;
goto 9;
goto 9;
return;
""",
    PROBE_DATA: """
_2 = discriminant((*_1)); switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];
FakeRead(ForMatchedPlace(None), _1); unreachable;
goto 4;
_0 = Option::<TTEntryData>::None; goto 5;
_3 = &(((*_1) as Hit).0: transposition_table::TTEntryData); _4 = copy (*_3); _0 = Option::<TTEntryData>::Some(move _4); goto 5;
return;
""",
    ESTIMATE: """
_7 = &(*((*_1).5: &transposition_table::TranspositionTable)); _8 = &(*_2); _10 = &(*_2); _9 = Board::hash(move _10) => 1;
_6 = TranspositionTable::probe(move _7, move _8, move _9) => 2;
_5 = &_6; _4 = TTProbeResult::<'_>::data(move _5) => 3;
_11 = discriminant(_4); switchInt(move _11) -> [1: bb4, otherwise: bb11];
goto 5;
_12 = copy ((_4 as Some).0: transposition_table::TTEntryData); _16 = &_12; _15 = TTEntryData::bound(move _16) => 6;
_14 = &_15; _18 = transposition_table::Bound::Exact; _17 = &_18; _13 = <transposition_table::Bound as PartialEq>::eq(move _14, move _17) => 7;
switchInt(move _13) -> [0: bb10, otherwise: bb8];
_20 = &_12; _0 = TTEntryData::score(move _20) => 9;
goto 17;
goto 11;
_3 = const (); _21 = EvalMode::Main; ((*_1).9: eval::EvalMode) = move _21; _24 = copy ((*_1).10: u32); _23 = Ge(move _24, const 20_u32); switchInt(move _23) -> [0: bb13, otherwise: bb12];
_26 = &mut (*_1); _27 = &(*_2); _28 = <Score as Neg>::neg(const score::Score::INF) => 14;
_22 = const (); _29 = &(*_1); _30 = &(*_2); _0 = evaluate(move _29, move _30) => 16;
_0 = evaluate_depth2(move _26, move _27, move _28, const score::Score::INF) => 15;
goto 17;
goto 17;
return;
""",
    ADVANCE_GENERATION: """
_5 = &((*_1).2: std::sync::atomic::Atomic<u8>); _6 = std::sync::atomic::Ordering::Relaxed; _4 = Atomic::<u8>::load(move _5, move _6) => 1;
_3 = core::num::<impl u8>::wrapping_add(move _4, const 1_u8) => 2;
_2 = BitAnd(move _3, const transposition_table::TTEntry::GENERATION_MASK); _8 = &((*_1).2: std::sync::atomic::Atomic<u8>); _9 = copy _2; _10 = std::sync::atomic::Ordering::Relaxed; _7 = Atomic::<u8>::store(move _8, move _9, move _10) => 3;
_0 = copy _2; return;
""",
    PUBLIC + "new_random_move": """
_2 = copy _1; _3 = Vec::<Square>::new() => 1;
_4 = Selectivity::None; _5 = Vec::<PvMove>::new() => 2;
_6 = <SearchCounters as Default>::default() => 3;
_0 = SearchResult::BestMove { sq: move _2, score: const score::Score::ZERO, n_nodes: const 0_u64, pv_line: move _3, depth: const 0_u32, selectivity: move _4, is_endgame: const false, pv_moves: move _5, counters: move _6 }; drop(_5) => 4;
drop(_3) => 5;
return;
""",
    CACHED: """
_6 = &(*_2); _7 = &(*_1); _9 = &(*_1); _8 = Board::hash(move _9) => 1;
_5 = TranspositionTable::lookup(move _6, move _7, move _8) => 2;
_4 = <Option<TTEntryData> as Try>::branch(move _5) => 3;
_10 = discriminant(_4); switchInt(move _10) -> [0: bb6, 1: bb5, otherwise: bb4];
FakeRead(ForMatchedPlace(None), _4); unreachable;
goto 7;
_14 = copy ((_4 as Continue).0: transposition_table::TTEntryData); _3 = copy _14; _19 = &_3; _18 = TTEntryData::bound(move _19) => 9;
_11 = copy ((_4 as Break).0: std::option::Option<!>); _13 = copy _11; _0 = <Option<SearchResult> as FromResidual<Option<!>>>::from_residual(move _13) => 8;
goto 22;
_17 = &_18; _21 = transposition_table::Bound::Exact; _20 = &_21; _16 = <transposition_table::Bound as PartialEq>::ne(move _17, move _20) => 10;
switchInt(move _16) -> [0: bb12, otherwise: bb11];
_0 = Option::<SearchResult>::None; goto 22;
_15 = const (); _26 = &_3; _25 = TTEntryData::best_move(move _26) => 13;
_24 = <Option<Square> as Try>::branch(move _25) => 14;
_27 = discriminant(_24); switchInt(move _27) -> [0: bb17, 1: bb16, otherwise: bb15];
FakeRead(ForMatchedPlace(None), _24); unreachable;
goto 18;
_31 = copy ((_24 as Continue).0: square::Square); _23 = copy _31; _33 = &(*_1); _34 = copy _23; _32 = Board::is_legal_move(move _33, move _34) => 20;
_28 = copy ((_24 as Break).0: std::option::Option<!>); _30 = copy _28; _0 = <Option<SearchResult> as FromResidual<Option<!>>>::from_residual(move _30) => 19;
goto 22;
_36 = &_23; _37 = &_3; _35 = cached_result { sq: move _36, data: move _37 }; _0 = core::bool::<impl bool>::then::<SearchResult, cached_result>(move _32, move _35) => 21;
goto 23;
goto 23;
return;
""",
    CACHED + "-{closure#0}": """
_2 = copy (*(_1.0: &square::Square)); _4 = &(*(_1.1: &transposition_table::TTEntryData)); _3 = TTEntryData::score(move _4) => 1;
_7 = Box::<[Square; 1]>::new_uninit() => 2;
_8 = copy (*(_1.0: &square::Square)); (((*_7).1: std::mem::ManuallyDrop<[square::Square; 1]>).0: [square::Square; 1]) = [move _8]; _6 = move _7; drop(_7) => 3;
_5 = std::boxed::box_assume_init_into_vec_unsafe::<Square, 1>(move _6) => 4;
_10 = &(*(_1.1: &transposition_table::TTEntryData)); _9 = TTEntryData::depth(move _10) => 5;
_12 = &(*(_1.1: &transposition_table::TTEntryData)); _11 = TTEntryData::selectivity(move _12) => 6;
_14 = &(*(_1.1: &transposition_table::TTEntryData)); _13 = TTEntryData::is_endgame(move _14) => 7;
_15 = Vec::<PvMove>::new() => 8;
_16 = <SearchCounters as Default>::default() => 9;
_0 = SearchResult::BestMove { sq: move _2, score: move _3, n_nodes: const 0_u64, pv_line: move _5, depth: move _9, selectivity: move _11, is_endgame: move _13, pv_moves: move _15, counters: move _16 }; drop(_15) => 10;
drop(_5) => 11;
return;
""",
    FINISH: """
_4 = discriminant((_1.1: std::option::Option<search::result::Iteration>)); switchInt(move _4) -> [0: bb3, 1: bb2, otherwise: bb1];
FakeRead(ForMatchedPlace(None), (_1.1: std::option::Option<search::result::Iteration>)); unreachable;
goto 4;
_20 = &(*_3); _21 = &(*((*_2).5: &transposition_table::TranspositionTable)); _19 = tt_move(move _20, move _21) => 10;
_5 = move (((_1.1: std::option::Option<search::result::Iteration>) as Some).0: search::result::Iteration); _9 = &(_5.0: std::vec::Vec<search::root_move::RootMove>); _8 = &(*_9); _7 = <Vec<RootMove> as Deref>::deref(move _8) => 5;
_6 = &(*_7); _12 = &_5; _11 = Iteration::best_move(move _12) => 6;
_10 = &(*_11); _13 = copy (_5.1: u32); _14 = copy (_5.2: probcut::Selectivity); _15 = copy (_5.3: bool); _17 = &(*_2); _16 = SearchContext::<'_>::counters(move _17) => 7;
_0 = SearchResult::from_root_move_snapshot(move _6, move _10, move _13, move _14, move _15, move _16) => 8;
drop(_5) => 9;
goto 19;
_23 = &(*_3); _24 = &(*((*_2).6: &eval::Eval)); _22 = no_iteration { board: move _23, ctx__eval: move _24 }; _18 = Option::<SearchResult>::unwrap_or_else::<no_iteration>(move _19, move _22) => 11;
_26 = &mut _18; _27 = discriminant((*_26)); switchInt(move _27) -> [0: bb12, otherwise: bb16];
goto 13;
_28 = &mut (((*_26) as BestMove).2: u64); _29 = &mut (((*_26) as BestMove).8: search::counters::SearchCounters); _31 = &(*_2); _30 = SearchContext::<'_>::counters(move _31) => 14;
_32 = copy (*_28); _33 = AddWithOverflow(copy (_30.0: u64), copy _32); assert(!move (_33.1: bool), "attempt to compute `{} + {}`, which would overflow", copy (_30.0: u64), move _32) -> [success: bb15, unwind: bb21];
(_30.0: u64) = move (_33.0: u64); _34 = copy (_30.0: u64); (*_28) = move _34; _35 = move _30; (*_29) = move _35; _25 = const (); goto 17;
_25 = const (); goto 17;
_0 = move _18; drop(_18) => 18;
goto 19;
drop(_1) => 20;
return;
""",
    FINISH + "-{closure#0}": """
_2 = &(*(_1.0: &board::Board)); _3 = &(*(_1.1: &eval::Eval)); _0 = quick_move(move _2, move _3) => 1;
return;
""",
    QUICK: """
_4 = &(*_1); _3 = Board::legal_moves(move _4) => 1;
_11 = copy _3; _10 = LegalMoves::iter(move _11) => 2;
_13 = &(*_2); _14 = &(*_1); _12 = evaluate { eval: move _13, board: move _14 }; _9 = <BitboardIterator as Iterator>::map::<(Square, Score), evaluate>(move _10, move _12) => 3;
_15 = choose_best; _8 = <Map<BitboardIterator, evaluate> as Iterator>::reduce::<choose_best>(move _9, move _15) => 4;
_16 = discriminant(_8); switchInt(move _16) -> [1: bb5, otherwise: bb7];
goto 6;
_6 = copy (((_8 as Some).0: (square::Square, score::Score)).0: square::Square); _7 = copy (((_8 as Some).0: (square::Square, score::Score)).1: score::Score); _19 = copy _3; _18 = LegalMoves::count(move _19) => 8;
_0 = SearchResult::NoLegalMove; goto 17;
_17 = move _18 as u64 (IntToInt); _20 = copy _6; _21 = copy _7; _22 = copy _17; _25 = Box::<[Square; 1]>::new_uninit() => 9;
_26 = copy _6; (((*_25).1: std::mem::ManuallyDrop<[square::Square; 1]>).0: [square::Square; 1]) = [move _26]; _24 = move _25; drop(_25) => 10;
_23 = std::boxed::box_assume_init_into_vec_unsafe::<Square, 1>(move _24) => 11;
_27 = Selectivity::None; _28 = Vec::<PvMove>::new() => 12;
_31 = <SearchStats as Default>::default() => 13;
_30 = &_31; _32 = copy _17; _29 = SearchStats::with_nodes(move _30, move _32) => 14;
_0 = SearchResult::BestMove { sq: move _20, score: move _21, n_nodes: move _22, pv_line: move _23, depth: const 1_u32, selectivity: move _27, is_endgame: const false, pv_moves: move _28, counters: move _29 }; drop(_28) => 15;
drop(_23) => 16;
goto 17;
return;
""",
    QUICK + "-{closure#0}": """
_3 = copy _2; _6 = &(*((*_1).0: &eval::Eval)); _10 = &(*((*_1).1: &board::Board)); _11 = copy _2; _9 = Board::make_move(move _10, move _11) => 1;
_8 = &_9; _7 = &(*_8); _12 = EvalMode::Main; _5 = Eval::evaluate_simple(move _6, move _7, move _12) => 2;
_4 = <Score as Neg>::neg(move _5) => 3;
_0 = (move _3, move _4); return;
""",
    QUICK + "-{closure#1}": """
_5 = &(_3.1: score::Score); _6 = &(_2.1: score::Score); _4 = <Score as PartialOrd>::gt(move _5, move _6) => 1;
switchInt(move _4) -> [0: bb3, otherwise: bb2];
_0 = copy _3; goto 4;
_0 = copy _2; goto 4;
return;
""",
    RECV: """
_7 = &((*_1).0: std::sync::Arc<transposition_table::TranspositionTable>); _6 = <Arc<TranspositionTable> as Deref>::deref(move _7) => 1;
_5 = &(*_6); _4 = TranspositionTable::increment_generation(move _5) => 2;
_8 = copy (_2.0: board::Board); _12 = &((*_1).1: std::sync::Arc<search::threading::pool::ThreadPool>); _11 = <Arc<ThreadPool> as Deref>::deref(move _12) => 3;
_10 = &(*_11); _13 = move _2; _14 = copy _3; _9 = ThreadPool::start_thinking(move _10, move _13, move _14) => 4;
_17 = &_9; _16 = std::sync::mpsc::Receiver::<SearchResult>::recv(move _17) => 5;
_19 = &_8; _20 = &((*_1).2: std::sync::Arc<eval::Eval>); _18 = disconnected { board: move _19, self__eval: move _20 }; _15 = Result::<SearchResult, RecvError>::unwrap_or_else::<disconnected>(move _16, move _18) => 6;
_24 = &((*_1).1: std::sync::Arc<search::threading::pool::ThreadPool>); _23 = <Arc<ThreadPool> as Deref>::deref(move _24) => 7;
_22 = &(*_23); _21 = ThreadPool::stop_timer(move _22) => 8;
_0 = move _15; drop(_15) => 9;
drop(_9) => 10;
drop(_2) => 11;
return;
""",
    RECV + "-{closure#0}": """
_4 = &(*(_1.0: &board::Board)); _3 = &(*_4); _8 = &(*(_1.1: &std::sync::Arc<eval::Eval>)); _7 = &(*_8); _6 = <Arc<Eval> as Deref>::deref(move _7) => 1;
_5 = &(*_6); _0 = quick_move(move _3, move _5) => 2;
return;
""",
    PUBLIC + "selectivity": """
_2 = discriminant((*_1)); switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];
FakeRead(ForMatchedPlace(None), _1); unreachable;
goto 4;
_0 = Selectivity::None; goto 5;
_3 = &(((*_1) as BestMove).5: probcut::Selectivity); _0 = copy (*_3); goto 5;
return;
""",
    PUBLIC + "is_endgame": """
_2 = discriminant((*_1)); switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];
FakeRead(ForMatchedPlace(None), _1); unreachable;
goto 4;
_0 = const false; goto 5;
_3 = &(((*_1) as BestMove).6: bool); _0 = copy (*_3); goto 5;
return;
""",
    "score-{impl#5}-value": """
_0 = copy (_1.0: i32); return;
""",
    ENTRY + "bound": """
_2 = copy ((*_1).2: u8); _0 = transposition_table::Bound::from_u8_unchecked(move _2) => 1;
return;
""",
    ENTRY + "score": """
_3 = copy ((*_1).0: i16); _2 = move _3 as i32 (IntToInt); _0 = Score::from_raw(move _2) => 1;
return;
""",
    ENTRY + "depth": """
_2 = copy ((*_1).3: u8); _0 = move _2 as u32 (IntToInt); return;
""",
    ENTRY + "selectivity": """
_2 = copy ((*_1).4: u8); _0 = Selectivity::from_u8(move _2) => 1;
return;
""",
    ENTRY + "is_endgame": """
_3 = copy ((*_1).6: u8); _2 = BitAnd(move _3, const transposition_table::FLAG_ENDGAME); _0 = Ne(move _2, const 0_u8); return;
""",
    "transposition_table-{impl#13}-eq": """
_4 = &(*_1); _3 = discriminant_value::<transposition_table::Bound>(move _4) => 1;
_6 = &(*_2); _5 = discriminant_value::<transposition_table::Bound>(move _6) => 2;
_0 = Eq(move _3, move _5); return;
""",
    "probcut-{impl#0}-from_u8": """
switchInt(copy _1) -> [0: bb2, 1: bb3, 2: bb4, 3: bb5, 4: bb6, otherwise: bb1];
_0 = Selectivity::None; goto 12;
goto 11;
goto 10;
goto 9;
goto 8;
goto 7;
_0 = Selectivity::Level4; goto 12;
_0 = Selectivity::Level3; goto 12;
_0 = Selectivity::Level2; goto 12;
_0 = Selectivity::Level1; goto 12;
_0 = Selectivity::Mid; goto 12;
return;
""",
    "board-{impl#1}-is_legal_move": """
_4 = &(*_1); _3 = Board::legal_moves(move _4) => 1;
_5 = copy _2; _0 = LegalMoves::contains(move _3, move _5) => 2;
return;
""",
    "board-{impl#1}-legal_moves": """
_2 = &(*_1); _0 = LegalMoves::new(move _2) => 1;
return;
""",
    "legal_moves-{impl#0}-new": """
_4 = &(*_1); _3 = Board::player(move _4) => 1;
_6 = &(*_1); _5 = Board::opponent(move _6) => 2;
_2 = Bitboard::legal_moves(move _3, move _5) => 3;
_8 = const true; switchInt(move _8) -> [0: bb8, otherwise: bb4];
_13 = copy _2; _12 = Bitboard::count(move _13) => 5;
_11 = move _12 as usize (IntToInt); _10 = Le(move _11, const legal_moves::LegalMoves::MAX_COUNT); switchInt(move _10) -> [0: bb7, otherwise: bb6];
_9 = const (); _7 = const (); goto 9;
_14 = panic(const "assertion failed: moves.count() as usize <= Self::MAX_COUNT") -> bb10;
_7 = const (); goto 9;
_15 = copy _2; _0 = LegalMoves(move _15); return;
""",
    "legal_moves-{impl#0}-contains": """
_3 = copy (_1.0: bitboard::Bitboard); _4 = copy _2; _0 = Bitboard::contains(move _3, move _4) => 1;
return;
""",
    "bitboard-{impl#0}-contains": """
_4 = copy (_1.0: u64); _7 = copy _2; _6 = Square::bitboard(move _7) => 1;
_5 = copy (_6.0: u64); _3 = BitAnd(move _4, move _5); _0 = Ne(move _3, const 0_u64); return;
""",
    "square-{impl#0}-bitboard": """
_2 = copy _1; _0 = Bitboard::from_square(move _2) => 1;
return;
""",
    "bitboard-{impl#0}-from_square": """
_4 = copy _1; _5 = discriminant(_4); _3 = move _5 as u8 (IntToInt); _6 = Lt(copy _3, const 64_u8); assert(move _6, "attempt to shift left by `{}`, which would overflow", copy _3) -> [success: bb1, unwind: bb2];
_2 = Shl(const 1_u64, move _3); _0 = Bitboard(move _2); return;
""",
}


# Follow the search receiver, the clone of its resident table, the root board
# and the returned result. Both timed-phase helpers access only tracking data.
RUN_PLACES = r"_(?:0|1|59|75|76|79|80|93|94|95|110|111|115|117|118)"
RUN_SLICE = """
_54 = &((*_1).1: std::sync::Arc<search::threading::pool::ThreadPool>);
_59 = &(*_1);
_58 = Search::timed_root_phase(move _59, move _60) => 26;
_76 = copy (*_2);
_80 = &((*_1).0: std::sync::Arc<transposition_table::TranspositionTable>);
_79 = <Arc<TranspositionTable> as Clone>::clone(move _80) => 46;
_82 = &((*_1).1: std::sync::Arc<search::threading::pool::ThreadPool>);
_84 = &((*_1).2: std::sync::Arc<eval::Eval>);
_92 = &((*_1).3: std::sync::Arc<search::correction_history::CorrectionHistories>);
_75 = SearchTask { board: move _76, mid_selectivity: move _77, tt: move _79, pool: move _81, eval: move _83, phase: move _85, multi_pv: move _86, callback: move _87, time_manager: move _89, eval_mode: move _90, corrections: move _91 };
drop(_79) => 56;
_94 = &mut (*_1);
_95 = move _75;
_93 = Search::execute_search(move _94, move _95, move _96) => 58;
((*_1).5: bool) = move _98;
_111 = &_93;
_110 = &(*_111);
_109 = SearchProgress::from_result(move _110) => 63;
_115 = &mut (*_1);
_118 = &_93;
_117 = &(*_118);
_114 = Search::update_endgame_tracking(move _115, move _116, move _117) => 70;
_0 = move _93;
drop(_93) => 73;
drop(_75) => 74;
"""


# Each row lists every access to the result, channel or task registers named in its pattern.
RETURN_SLICES = {
    SEARCH + "run": (RUN_PLACES, RUN_SLICE),
    SEARCH + "timed_root_phase": (r"_1", """
_4 = copy ((*_1).4: std::option::Option<u32>);
"""),
    SEARCH + "update_endgame_tracking": (r"_1", """
_6 = copy ((*_1).4: std::option::Option<u32>);
((*_1).4: std::option::Option<u32>) = move _9;
_13 = &((*_1).4: std::option::Option<u32>);
((*_1).4: std::option::Option<u32>) = move _22;
"""),
    "search-search_root": (r"_(?:0|5|9|1)", """
_3 = discriminant((_1.5: search::RootPhase));
FakeRead(ForMatchedPlace(None), (_1.5: search::RootPhase));
_8 = copy (((_1.5: search::RootPhase) as Endgame).0: probcut::Selectivity);
_9 = move _1;
_0 = search::endgame::search_root(move _9, move _10, move _11) => 6;
_4 = copy (((_1.5: search::RootPhase) as Midgame).0: u32);
_5 = move _1;
_0 = midgame::search_root(move _5, move _6, move _7) => 5;
drop(_1) => 8;
"""),
    "search-endgame-search_root": (r"_0", """
_0 = SearchResult::NoLegalMove;
_0 = solve_root(move _37, move _39, move _41, move _42, move _43, move _44, move _45) => 12;
"""),
    "search-endgame-solve_root": (r"_0", """
_0 = CompletedState::into_result(move _146, move _147, move _148) => 86;
_0 = CompletedState::into_result(move _157, move _158, move _159) => 93;
"""),
    "search-midgame-search_root": (r"_0", """
_0 = SearchResult::NoLegalMove;
_0 = SearchResult::new_random_move(move _38) => 17;
_0 = solve_root(move _107, move _109, move _111, move _112, move _113, const search::ENDGAME_TRACKING_MIN_SELECTIVITY, move _114) => 63;
_0 = CompletedState::into_result(move _100, move _101, move _103) => 61;
"""),
    "search-threading-thread-{impl#1}-main_thread_loop": (r"_(?:15|17|31|32|53|54|55)", """
_15 = move ((((_11 as Ok).0: search::threading::pool::Message) as StartThinking).0: search::SearchTask);
_17 = move ((((_11 as Ok).0: search::threading::pool::Message) as StartThinking).2: std::sync::mpsc::Sender<search::result::SearchResult>);
_26 = &(_15.3: std::sync::Arc<search::threading::pool::ThreadPool>);
_32 = move _15;
_31 = search::search_root(move _32, move _33) => 21;
_54 = &_17;
_55 = move _31;
_53 = std::sync::mpsc::Sender::<SearchResult>::send(move _54, move _55) => 30;
drop(_53) => 31;
drop(_31) => 32;
drop(_17) => 34;
drop(_15) => 36;
"""),
    "search-threading-pool-{impl#0}-start_thinking": (r"_(?:0|12|13|14|45|46|48)", """
_14 = std::sync::mpsc::channel::<SearchResult>() => 8;
_12 = move (_14.0: std::sync::mpsc::Sender<search::result::SearchResult>);
_13 = move (_14.1: std::sync::mpsc::Receiver<search::result::SearchResult>);
drop(_14) => 9;
_46 = move _2;
_48 = move _12;
_45 = Message::StartThinking(move _46, move _47, move _48);
drop(_48) => 25;
drop(_46) => 27;
_43 = std::sync::mpsc::Sender::<Message>::send(move _44, move _45) => 28;
_0 = move _13;
drop(_13) => 31;
drop(_12) => 32;
"""),
}


def check_places(source, pattern, expected):
    # Each comparison is against whole instructions in the pinned compiler's
    # well-typed MIR. Every use, including address-taking and writes, is matched.
    places = re.compile(rf"\b{pattern}\b")
    got = [op for bb in normal_blocks(source, {}).values() for op in re.split(r"(?<=;) ", bb)
           if places.search(op)]
    checked(got == expected.strip().splitlines(), f"changed result data flow in {source.splitlines()[0]}: {got}")


def check_function(dump, names, stem):
    source = read(dump, stem)
    if stem in RETURN_SLICES:
        check_places(source, *RETURN_SLICES[stem])
    elif stem == "score-{impl#5}-from_raw":
        check_places(source, r"_(?:0|16)", "_16 = copy _1;\n_0 = Score(move _16);")
    elif stem == "transposition_table-{impl#0}-from_u8_unchecked":
        check_places(source, r"_(?:0|19)", "_19 = copy _1;\n"
                     "_0 = std::intrinsics::transmute::<u8, transposition_table::Bound>(move _19) => 8;")
    else:
        cfg = normal_blocks(source, names)
        expected = EXPECTED[stem].strip().splitlines()
        checked(cfg.keys() == set(range(len(expected))), f"{stem}: normal return paths changed")
        for pc, instructions in enumerate(expected):
            checked(cfg[pc] == instructions, f"{stem} bb{pc}: expected {instructions}, saw {cfg[pc]}")


def check(dump):
    names = callback_names(dump)
    for stem in [*EXPECTED, *RETURN_SLICES, "score-{impl#5}-from_raw",
                 "transposition_table-{impl#0}-from_u8_unchecked"]:
        check_function(dump, names, stem)
    for stem, byte in (("FLAG_ENDGAME", 1), ("Bound-Exact-{constant#0}", 3),
                       ("{impl#6}-GENERATION_MASK", 127)):
        source = (dump / f"reversi_core.transposition_table-{stem}.built.after.mir").read_text(encoding="utf-8")
        checked(normal_blocks(source, {}) == {0: f"_0 = const {byte}_u8; return;"}, f"changed {stem}")
    inf = (dump / "reversi_core.score-{impl#5}-INF.built.after.mir").read_text(encoding="utf-8")
    checked(normal_blocks(inf, {}) == {0: "_1 = const core::num::<impl i16>::MAX as i32 (IntToInt); "
                                       "_0 = Score(move _1); return;"}, "changed depth-two full window")
    for path in dump.glob(f"reversi_core.transposition_table-*-ne.{PASS}.after.mir"):
        checked("(_1: &transposition_table::Bound, _2: &transposition_table::Bound) -> bool"
                not in path.read_text(encoding="utf-8"), "Bound overrides PartialEq::ne")


def check_mutants(dump):
    from unittest.mock import patch

    changes = [
        ("search-midgame-compute_start_depth", "_0 = const 1_u32;", "_0 = const 2_u32;"),
        ("search-strategy-{impl#0}-shallow_search", "0: bb2, 1: bb3", "0: bb3, 1: bb2"),
        (PROBE_DATA, "_0 = Option::<TTEntryData>::Some(move _4);", "_0 = Option::<TTEntryData>::None;"),
        (ESTIMATE, "transposition_table::Bound::Exact;", "transposition_table::Bound::Lower;"),
        (ESTIMATE, "TTEntryData::bound(move _16)", "TTEntryData::selectivity(move _16)"),
        (ESTIMATE, "EvalMode::Main;", "EvalMode::Small;"),
        (ESTIMATE, "const 20_u32", "const 19_u32"),
        (ESTIMATE, "0: bb13, otherwise: bb12", "0: bb12, otherwise: bb13"),
        (ESTIMATE, "_0 = evaluate_depth2(move _26, move _27, move _28, const score::Score::INF)",
         "_0 = evaluate_depth2(move _26, move _27, move _28, move _28)"),
        (ESTIMATE, "_0 = evaluate(move _29, move _30)", "_0 = evaluate_depth1(move _29, move _30)"),
        (ADVANCE_GENERATION, "wrapping_add(move _4, const 1_u8)", "wrapping_add(move _4, const 2_u8)"),
        (ADVANCE_GENERATION, "BitAnd(move _3, const transposition_table::TTEntry::GENERATION_MASK)",
         "BitAnd(move _3, const 63_u8)"),
        (RECV, "TranspositionTable::increment_generation(move _5)",
         "TranspositionTable::reset_generation(move _5)"),
        (SEARCH + "run", "<Arc<TranspositionTable> as Clone>::clone(move _80)", "TranspositionTable::duplicate(move _80)"),
        (CACHED, "_21 = transposition_table::Bound::Exact;", "_21 = transposition_table::Bound::Lower;"),
        (CACHED, "0: bb12, otherwise: bb11", "0: bb11, otherwise: bb12"),
        (CACHED, "(move _32, move _35)", "(const true, move _35)"),
        (CACHED + "-{closure#0}", "is_endgame: move _13", "is_endgame: const true"),
        (ENTRY + "score", "as i32 (IntToInt)", "as u32 (IntToInt)"),
        ("probcut-{impl#0}-from_u8", "_0 = Selectivity::Level4;", "_0 = Selectivity::None;"),
        (ENTRY + "is_endgame", "Ne(move _2", "Eq(move _2"),
        (QUICK, "is_endgame: const false", "is_endgame: const true"),
        (PUBLIC + "new_random_move", "is_endgame: const false", "is_endgame: const true"),
        ("search-endgame-search_root", "_0 = SearchResult::NoLegalMove;", "_0 = SearchResult::NoLegalMove;\n        _0 = move _12;"),
        ("search-threading-thread-{impl#1}-main_thread_loop", "_55 = move _31;", "_55 = SearchResult::NoLegalMove;"),
        (PUBLIC + "is_endgame", "_0 = copy (*_3);", "_0 = const true;"),
        (PUBLIC + "selectivity", "_0 = copy (*_3);", "_0 = Selectivity::None;"),
        (FINISH, "_0 = move _18;", "_18 = SearchResult::NoLegalMove;\n        _0 = move _18;"),
        (RECV, "_0 = move _15;", "_0 = SearchResult::NoLegalMove;"),
        (RECV + "-{closure#0}", "quick_move(move _3, move _5)", "random_move(move _3, move _5)"),
        (SEARCH + "run", "_0 = move _93;", "_0 = SearchResult::NoLegalMove;"),
        ("score-{impl#5}-value", "_0 = copy (_1.0: i32);", "_0 = const 0_i32;"),
    ]
    names = callback_names(dump)
    for stem, old, new in changes:
        source = read(dump, stem)
        checked(source.count(old) == 1, f"ambiguous or missing {stem} mutant {old}")
        with patch.dict(check_function.__globals__, {"read": lambda *args: source.replace(old, new)}):
            try:
                check_function(dump, names, stem)
            except ValueError:
                continue
        raise ValueError(f"accepted {stem} mutation {old}")
    return len(changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    check(args.mir)
    print(f"verified public endgame guards and rejected {check_mutants(args.mir)} MIR mutants")
