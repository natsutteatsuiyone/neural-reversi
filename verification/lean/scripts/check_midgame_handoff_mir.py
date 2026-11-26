"""Match the normal MIR of the timed midgame loop to the single-PV handoff.

The admitted path captures the sorted rows, projects (square, score) from the
completed line, sorts the same live root storage and forwards the saved word
and completion to solve_root on the original context. The selector for a
non-PV midgame child may instead enter the checked endgame boundary.
"""

import re

from mir_common import checked
from check_result_guards_mir import normal_blocks, read


# Resolve each closure from its own MIR signature so editing source line numbers
# cannot change identity and substituting a different closure fails the check.
CLOSURES = {'search-midgame-iterate-{closure#0}': 'row_score',
 'search-midgame-iterate-{closure#1}': 'deadline',
 'search-midgame-aspiration_search-{closure#0}': 'root_search',
 'search-midgame-aspiration_window-{closure#0}': 'row_history',
 'search-aspiration-midgame_window-{closure#0}': 'use_history',
 'search-root_move-{impl#1}-sort_all-{closure#0}': 'descending_score'}

# One line per normal block. Cleanup targets are handled by normal_blocks.
# The complete loop includes the zero-line and aborted exits, callbacks and
# time-manager gates. No alternate predecessor can bypass the capture.
BODIES = {
    'search-aspiration-midgame_window': """
0: _5 = copy _2; _7 = &_1; _6 = use_history { depth: move _7 }; _4 = Option::<(Score, i64)>::filter::<use_history>(move _5, move _6) => 1;
1: _8 = discriminant(_4); switchInt(move _8) -> [1: bb2, otherwise: bb16];
2: goto 3;
3: _9 = copy (((_4 as Some).0: (score::Score, i64)).0: score::Score); _10 = copy (((_4 as Some).0: (score::Score, i64)).1: i64); _13 = copy _3; _14 = Eq(const search::aspiration::HELPER_WIDTH_CYCLE, const 0_usize); assert(!move _14, "attempt to calculate the remainder of `{}` with a divisor of zero", copy _13) -> [success: bb4, unwind: bb19];
4: _12 = Rem(move _13, const search::aspiration::HELPER_WIDTH_CYCLE); _11 = move _12 as i32 (IntToInt); _18 = copy _11; _19 = MulWithOverflow(const search::aspiration::HELPER_WIDTH_STEP, copy _18); assert(!move (_19.1: bool), "attempt to compute `{} * {}`, which would overflow", const search::aspiration::HELPER_WIDTH_STEP, move _18) -> [success: bb5, unwind: bb19];
5: _17 = move (_19.0: i32); _16 = <Score as Add<i32>>::add(const search::aspiration::MIDGAME_DELTA, move _17) => 6;
6: _23 = copy _10; _22 = core::num::<impl i64>::abs(move _23) => 7;
7: _24 = Eq(const search::aspiration::MSS_DIVISOR, const 0_i64); assert(!move _24, "attempt to divide `{}` by zero", copy _22) -> [success: bb8, unwind: bb19];
8: _25 = Eq(const search::aspiration::MSS_DIVISOR, const -1_i64); _26 = Eq(copy _22, const i64::MIN); _27 = BitAnd(move _25, move _26); assert(!move _27, "attempt to compute `{} / {}`, which would overflow", copy _22, const search::aspiration::MSS_DIVISOR) -> [success: bb9, unwind: bb19];
9: _21 = Div(move _22, const search::aspiration::MSS_DIVISOR); _20 = move _21 as i32 (IntToInt); _15 = <Score as Add<i32>>::add(move _16, move _20) => 10;
10: _30 = copy _9; _31 = copy _15; _29 = <Score as Sub>::sub(move _30, move _31) => 11;
11: _32 = <Score as Neg>::neg(const score::Score::INF) => 12;
12: _28 = <Score as Ord>::max(move _29, move _32) => 13;
13: _35 = copy _9; _36 = copy _15; _34 = <Score as Add>::add(move _35, move _36) => 14;
14: _33 = <Score as Ord>::min(move _34, const score::Score::INF) => 15;
15: _37 = copy _15; _0 = (move _28, move _33, move _37); goto 18;
16: _38 = <Score as Neg>::neg(const score::Score::INF) => 17;
17: _0 = (move _38, const score::Score::INF, const search::aspiration::MIDGAME_DELTA); goto 18;
18: return;
""",
    'search-aspiration-midgame_window-{closure#0}': """
0: _3 = copy (*(_1.0: &u32)); _0 = Ge(move _3, const search::aspiration::ASPIRATION_MIN_DEPTH); return;
""",
    'search-aspiration-retry_midgame': """
0: _5 = const true; goto 1;
1: falseUnwind -> [real: bb2, unwind: bb12];
2: _11 = &mut _4; _13 = copy (*_1); _14 = copy (*_2); _12 = (move _13, move _14); _10 = <impl FnMut(Score, Score) -> (Score, bool) as FnMut<(Score, Score)>>::call_mut(move _11, move _12) => 3;
3: _8 = copy (_10.0: score::Score); _9 = copy (_10.1: bool); _16 = copy _9; switchInt(move _16) -> [0: bb4, otherwise: bb8];
4: _18 = copy _8; _19 = &mut (*_1); _20 = &mut (*_2); _21 = copy _3; _17 = widen_aspiration_window(move _18, move _19, move _20, move _21) => 5;
5: switchInt(move _17) -> [0: bb7, otherwise: bb6];
6: _15 = const (); _5 = const false; _26 = &mut _3; _28 = copy _3; _27 = <Score as Div<i32>>::div(move _28, const 2_i32) => 9;
7: goto 8;
8: _23 = copy _8; _24 = copy _5; _0 = (move _23, move _24); drop(_4) => 11;
9: _25 = <Score as AddAssign>::add_assign(move _26, move _27) => 10;
10: _7 = const (); goto 1;
11: return;
""",
    'search-context-{impl#0}-merge_counters': """
0: _3 = copy ((*_2).0: u64); _4 = AddWithOverflow(copy ((*_1).2: u64), copy _3); assert(!move (_4.1: bool), "attempt to compute `{} + {}`, which would overflow", copy ((*_1).2: u64), move _3) -> [success: bb1, unwind: bb3];
1: ((*_1).2: u64) = move (_4.0: u64); _6 = &mut ((*_1).12: search::counters::SearchStats); _7 = &(*_2); _5 = SearchStats::merge(move _6, move _7) => 2;
2: _0 = const (); return;
""",
    'search-midgame-aspiration_search': """
0: _8 = &mut (*_4); _9 = &mut (*_5); _10 = copy _6; _12 = &mut (*_1); _13 = &(*_2); _14 = &_3; _15 = &(*_7); _11 = root_search { ctx: move _12, board: move _13, depth: move _14, thread: move _15 }; _0 = retry_midgame::<root_search>(move _8, move _9, move _10, move _11) => 1;
1: return;
""",
    'search-midgame-aspiration_search-{closure#0}': """
0: _5 = &mut (*((*_1).0: &mut search::context::SearchContext<'_>)); _6 = &(*((*_1).1: &board::Board)); _7 = copy (*((*_1).2: &u32)); _8 = copy _2; _9 = copy _3; _10 = &(*((*_1).3: &std::sync::Arc<search::threading::thread::Thread>)); _4 = search::<Root, MidGameStrategy>(move _5, move _6, move _7, move _8, move _9, move _10, const false) => 1;
1: _11 = copy _4; _15 = &(*((*_1).3: &std::sync::Arc<search::threading::thread::Thread>)); _14 = <Arc<threading::thread::Thread> as Deref>::deref(move _15) => 2;
2: _13 = &(*_14); _12 = threading::thread::Thread::is_search_aborted(move _13) => 3;
3: _0 = (move _11, move _12); return;
""",
    'search-midgame-aspiration_window': """
0: _5 = &(*((*_1).1: &search::root_move::RootMoves)); _4 = RootMoves::current_pv(move _5) => 1;
1: _6 = row_history; _3 = Option::<RootMove>::map::<(Score, i64), row_history>(move _4, move _6) => 2;
2: _7 = copy _2; _8 = copy _3; _10 = copy ((*_1).3: std::option::Option<search::midgame::lazy_smp::Helper>); _9 = Option::<Helper>::map_or::<usize, fn(Helper) -> usize {Helper::index}>(move _10, const 0_usize, Helper::index) => 3;
3: _0 = midgame_window(move _7, move _8, move _9) => 4;
4: return;
""",
    'search-midgame-aspiration_window-{closure#0}': """
0: _3 = copy (_2.2: score::Score); _4 = copy (_2.3: i64); _0 = (move _3, move _4); drop(_2) => 1;
1: return;
""",
    'search-midgame-iterate': """
0: _8 = &((*_3).8: std::option::Option<std::sync::Arc<search::time_control::TimeManager>>); _7 = Option::<Arc<TimeManager>>::as_deref(move _8) => 1;
1: _10 = &((*_1).10: empty_list::EmptyList); _9 = EmptyList::count(move _10) => 2;
2: _12 = &(*_3); _14 = &(*((*_1).1: &search::root_move::RootMoves)); _13 = RootMoves::count(move _14) => 3;
3: _11 = SearchTask::pv_count(move _12, move _13) => 4;
4: _16 = copy _4; _15 = compute_start_depth(move _16) => 5;
5: _18 = copy _11; _17 = CompletedState::new(move _18) => 6;
6: goto 7;
7: falseUnwind -> [real: bb8, unwind: bb116];
8: _22 = copy _15; _23 = copy _4; _21 = Le(move _22, move _23); switchInt(move _21) -> [0: bb104, otherwise: bb9];
9: _24 = const false; _25 = Option::<(Square, Score)>::None; _29 = copy _11; _28 = std::ops::Range::<usize> { start: const 0_usize, end: move _29 }; _27 = <std::ops::Range<usize> as IntoIterator>::into_iter(move _28) => 10;
10: _30 = move _27; goto 11;
11: falseUnwind -> [real: bb12, unwind: bb116];
12: _34 = &mut _30; _33 = &mut (*_34); _32 = <std::ops::Range<usize> as Iterator>::next(move _33) => 13;
13: _35 = discriminant(_32); switchInt(move _35) -> [0: bb15, 1: bb16, otherwise: bb14];
14: FakeRead(ForMatchedPlace(None), _32); unreachable;
15: goto 17;
16: _37 = copy ((_32 as Some).0: usize); _39 = &(*((*_1).1: &search::root_move::RootMoves)); _40 = copy _37; _38 = RootMoves::set_pv_idx(move _39, move _40) => 18;
17: _26 = const (); goto 50;
18: _45 = &(*_1); _46 = copy _15; _44 = aspiration_window(move _45, move _46) => 19;
19: _41 = copy (_44.0: score::Score); _42 = copy (_44.1: score::Score); _43 = copy (_44.2: score::Score); _50 = &mut (*_1); _51 = &(*_2); _52 = copy _15; _54 = &mut _41; _53 = &mut (*_54); _56 = &mut _42; _55 = &mut (*_56); _57 = copy _43; _58 = &(*_6); _49 = midgame::aspiration_search(move _50, move _51, move _52, move _53, move _55, move _57, move _58) => 20;
20: _47 = copy (_49.0: score::Score); _48 = copy (_49.1: bool); _60 = &(*((*_1).1: &search::root_move::RootMoves)); _59 = RootMoves::sort_from_pv_idx(move _60) => 21;
21: _65 = &(*_6); _64 = <Arc<threading::thread::Thread> as Deref>::deref(move _65) => 22;
22: _63 = &(*_64); _62 = threading::thread::Thread::is_search_aborted(move _63) => 23;
23: switchInt(move _62) -> [0: bb25, otherwise: bb24];
24: _26 = const (); goto 49;
25: _61 = const (); _68 = discriminant(((*_3).7: std::option::Option<std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>>)); switchInt(move _68) -> [1: bb26, otherwise: bb39];
26: goto 27;
27: _69 = &((((*_3).7: std::option::Option<std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>>) as Some).0: std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>); _71 = &(*((*_1).1: &search::root_move::RootMoves)); _70 = RootMoves::current_pv(move _71) => 28;
28: _72 = discriminant(_70); switchInt(move _72) -> [1: bb29, otherwise: bb37];
29: goto 30;
30: _73 = move ((_70 as Some).0: search::root_move::RootMove); _75 = copy ((*_1).2: u64); _77 = &(*_5); _76 = Helpers::<'_>::nodes(move _77) => 31;
31: _78 = AddWithOverflow(copy _75, copy _76); assert(!move (_78.1: bool), "attempt to compute `{} + {}`, which would overflow", move _75, move _76) -> [success: bb32, unwind: bb114];
32: _74 = move (_78.0: u64); _82 = &(*_69); _81 = <Arc<dyn Fn(SearchProgress) + Send + Sync> as Deref>::deref(move _82) => 33;
33: _80 = &(*_81); _85 = &(*_1); _87 = &_73; _86 = &(*_87); _88 = copy _15; _89 = copy _4; _90 = copy _47; _91 = copy _74; _84 = SearchProgress::from_iteration(move _85, move _86, move _88, move _89, move _90, move _91, const false) => 34;
34: _83 = (move _84,); _79 = <dyn Fn(SearchProgress) + Send + Sync as Fn<(SearchProgress,)>>::call(move _80, move _83) => 35;
35: drop(_84) => 36;
36: _67 = const (); drop(_73) => 40;
37: drop(_70) => 38;
38: goto 39;
39: _67 = const (); goto 42;
40: drop(_70) => 41;
41: goto 42;
42: _92 = copy _48; _24 = move _92; _95 = &mut _17; _96 = &(*_1); _97 = copy _15; _94 = CompletedState::line_completed(move _95, move _96, move _97, const false) => 43;
43: _98 = row_score; _93 = Option::<&RootMove>::map::<(Square, Score), row_score>(move _94, move _98) => 44;
44: _25 = move _93; _102 = &(*_6); _101 = <Arc<threading::thread::Thread> as Deref>::deref(move _102) => 45;
45: _100 = &(*_101); _99 = threading::thread::Thread::is_search_aborted(move _100) => 46;
46: switchInt(move _99) -> [0: bb48, otherwise: bb47];
47: _26 = const (); goto 49;
48: _31 = const (); _20 = const (); goto 11;
49: goto 50;
50: _110 = discriminant(_25); switchInt(move _110) -> [1: bb54, otherwise: bb53];
51: _107 = move _17; _0 = Outcome::Done(move _107); drop(_107) => 52;
52: goto 109;
53: _106 = copy _7; _105 = record_abort(move _106) => 51;
54: goto 55;
55: _108 = copy (((_25 as Some).0: (square::Square, score::Score)).0: square::Square); _109 = copy (((_25 as Some).0: (square::Square, score::Score)).1: score::Score); _112 = &(*((*_1).1: &search::root_move::RootMoves)); _111 = RootMoves::sort_all(move _112) => 56;
56: _114 = discriminant(_7); switchInt(move _114) -> [1: bb57, otherwise: bb66];
57: goto 58;
58: _115 = copy ((_7 as Some).0: &search::time_control::TimeManager); _117 = copy ((*_3).6: bool); switchInt(move _117) -> [0: bb59, otherwise: bb62];
59: _119 = copy _15; _118 = Ge(move _119, const search::aspiration::ASPIRATION_MIN_DEPTH); switchInt(move _118) -> [0: bb61, otherwise: bb60];
60: _116 = copy _24; goto 63;
61: goto 62;
62: _116 = const false; goto 63;
63: _121 = &(*_115); _122 = copy _108; _124 = copy _109; _123 = Score::to_disc_diff_f32(move _124) => 64;
64: _125 = copy _15; _126 = copy _116; _120 = TimeManager::report_iteration(move _121, move _122, move _123, move _125, move _126) => 65;
65: _113 = const (); goto 67;
66: _113 = const (); goto 67;
67: _130 = &(*((*_1).1: &search::root_move::RootMoves)); _129 = RootMoves::count(move _130) => 68;
68: _128 = Eq(move _129, const 1_usize); switchInt(move _128) -> [0: bb74, otherwise: bb69];
69: _132 = copy _7; _133 = deadline; _131 = Option::<&TimeManager>::is_some_and::<deadline>(move _132, move _133) => 70;
70: switchInt(move _131) -> [0: bb73, otherwise: bb71];
71: _135 = move _17; _0 = Outcome::Done(move _135); drop(_135) => 72;
72: goto 108;
73: goto 75;
74: goto 75;
75: _127 = const (); _140 = &(*_6); _139 = <Arc<threading::thread::Thread> as Deref>::deref(move _140) => 76;
76: _138 = &(*_139); _137 = threading::thread::Thread::is_search_aborted(move _138) => 77;
77: switchInt(move _137) -> [0: bb80, otherwise: bb78];
78: _142 = move _17; _0 = Outcome::Done(move _142); drop(_142) => 79;
79: goto 108;
80: _136 = const (); _144 = discriminant(_7); switchInt(move _144) -> [1: bb81, otherwise: bb98];
81: goto 82;
82: _145 = copy ((_7 as Some).0: &search::time_control::TimeManager); _147 = copy _15; _148 = copy _4; _146 = Ge(move _147, move _148); switchInt(move _146) -> [0: bb97, otherwise: bb83];
83: _151 = const true; switchInt(move _151) -> [0: bb88, otherwise: bb84];
84: _154 = &_4; _155 = &_9; _153 = (move _154, move _155); _156 = copy (_153.0: &u32); _157 = copy (_153.1: &u32); _159 = copy (*_156); _160 = copy (*_157); _158 = Eq(move _159, move _160); switchInt(move _158) -> [0: bb86, otherwise: bb85];
85: _152 = const (); _150 = const (); goto 89;
86: _162 = core::panicking::AssertKind::Eq; _164 = move _162; _166 = &(*_156); _165 = &(*_166); _168 = &(*_157); _167 = &(*_168); _170 = Arguments::<'_>::from_str(const "time-controlled midgame searches always target all empties") => 87;
87: _169 = Option::<Arguments<'_>>::Some(move _170); _163 = core::panicking::assert_failed::<u32, u32>(move _164, move _165, move _167, move _169) -> bb116;
88: _150 = const (); goto 89;
89: _173 = &(*_145); _175 = copy ((*_1).2: u64); _177 = &(*_5); _176 = Helpers::<'_>::nodes(move _177) => 90;
90: _178 = AddWithOverflow(copy _175, copy _176); assert(!move (_178.1: bool), "attempt to compute `{} + {}`, which would overflow", move _175, move _176) -> [success: bb91, unwind: bb116];
91: _174 = move (_178.0: u64); _172 = TimeManager::should_enter_endgame(move _173, move _174) => 92;
92: switchInt(move _172) -> [0: bb95, otherwise: bb93];
93: _180 = move _17; _181 = copy _109; _0 = Outcome::Endgame(move _180, move _181); drop(_180) => 94;
94: goto 107;
95: _171 = const (); _182 = move _17; _0 = Outcome::Done(move _182); drop(_182) => 96;
96: goto 107;
97: goto 98;
98: _143 = const (); _185 = copy _7; _184 = should_stop_iteration(move _185) => 99;
99: switchInt(move _184) -> [0: bb102, otherwise: bb100];
100: _187 = move _17; _0 = Outcome::Done(move _187); drop(_187) => 101;
101: goto 108;
102: _183 = const (); _189 = copy _15; _188 = next_iteration_depth(move _189) => 103;
103: _15 = move _188; _20 = const (); goto 7;
104: _19 = const (); _193 = move _17; _0 = Outcome::Done(move _193); drop(_193) => 105;
105: drop(_17) => 106;
106: goto 111;
107: goto 108;
108: goto 109;
109: drop(_17) => 110;
110: goto 111;
111: return;
""",
    'search-midgame-iterate-{closure#0}': """
0: _3 = copy ((*_2).0: square::Square); _4 = copy ((*_2).1: score::Score); _0 = (move _3, move _4); return;
""",
    'search-midgame-iterate-{closure#1}': """
0: _3 = &(*_2); _0 = TimeManager::has_deadline(move _3) => 1;
1: return;
""",
    'search-midgame-next_iteration_depth': """
0: _3 = copy _1; _2 = Le(move _3, const search::midgame::DEPTH_STEP_THRESHOLD); switchInt(move _2) -> [0: bb3, otherwise: bb1];
1: _4 = copy _1; _5 = AddWithOverflow(copy _4, const 2_u32); assert(!move (_5.1: bool), "attempt to compute `{} + {}`, which would overflow", move _4, const 2_u32) -> [success: bb2, unwind: bb6];
2: _0 = move (_5.0: u32); goto 5;
3: _6 = copy _1; _7 = AddWithOverflow(copy _6, const 1_u32); assert(!move (_7.1: bool), "attempt to compute `{} + {}`, which would overflow", move _6, const 1_u32) -> [success: bb4, unwind: bb6];
4: _0 = move (_7.0: u32); goto 5;
5: return;
""",
    'search-midgame-search_root': """
0: _4 = copy (_1.0: board::Board); _7 = &_4; _6 = &(*_7); _5 = RootMoves::new(move _6) => 1;
1: _10 = &_4; _9 = &(*_10); _11 = copy (_1.1: probcut::Selectivity); _13 = &_5; _12 = &(*_13); _17 = &(_1.2: std::sync::Arc<transposition_table::TranspositionTable>); _16 = &(*_17); _15 = <Arc<TranspositionTable> as Deref>::deref(move _16) => 2;
2: _14 = &(*_15); _21 = &(_1.4: std::sync::Arc<eval::Eval>); _20 = &(*_21); _19 = <Arc<Eval> as Deref>::deref(move _20) => 3;
3: _18 = &(*_19); _8 = SearchContext::<'_>::new(move _9, move _11, move _12, move _14, move _18) => 4;
4: _23 = discriminant((_1.9: std::option::Option<eval::EvalMode>)); switchInt(move _23) -> [1: bb5, otherwise: bb7];
5: goto 6;
6: _24 = copy (((_1.9: std::option::Option<eval::EvalMode>) as Some).0: eval::EvalMode); _25 = copy _24; (_8.9: eval::EvalMode) = move _25; _22 = const (); goto 8;
7: _22 = const (); goto 8;
8: _29 = &(*(_8.1: &search::root_move::RootMoves)); _28 = RootMoves::count(move _29) => 9;
9: _27 = Eq(move _28, const 0_usize); switchInt(move _27) -> [0: bb11, otherwise: bb10];
10: _0 = SearchResult::NoLegalMove; goto 72;
11: _26 = const (); _33 = &_4; _35 = Board::new() => 12;
12: _34 = &_35; _32 = <Board as PartialEq>::eq(move _33, move _34) => 13;
13: switchInt(move _32) -> [0: bb18, otherwise: bb14];
14: _36 = copy (_1.6: bool); switchInt(move _36) -> [0: bb15, otherwise: bb19];
15: _40 = &_4; _39 = &(*_40); _38 = random_move(move _39) => 16;
16: _0 = SearchResult::new_random_move(move _38) => 17;
17: goto 72;
18: goto 19;
19: _31 = const (); _42 = &(_8.10: empty_list::EmptyList); _41 = EmptyList::count(move _42) => 20;
20: _45 = copy _2; _44 = <u32 as Ord>::max(move _45, const 1_u32) => 21;
21: _46 = copy _41; _43 = <u32 as Ord>::min(move _44, move _46) => 22;
22: _51 = &(_1.10: std::sync::Arc<search::correction_history::CorrectionHistories>); _50 = <Arc<CorrectionHistories> as Deref>::deref(move _51) => 23;
23: _49 = &(*_50); _55 = &(*_3); _54 = <Arc<threading::thread::Thread> as Deref>::deref(move _55) => 24;
24: _53 = &(*_54); _52 = threading::thread::Thread::idx(move _53) => 25;
25: _48 = CorrectionHistories::take(move _49, move _52) => 26;
26: _47 = Option::<Box<CorrectionHistory>>::Some(move _48); drop(_48) => 27;
27: drop((_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>)) => 28;
28: (_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>) = move _47; drop(_47) => 30;
30: _58 = &_1; _57 = &(*_58); _59 = copy _43; _56 = Helpers::<'_>::start(move _57, move _59) => 31;
31: _62 = &mut _8; _61 = &mut (*_62); _64 = &_4; _63 = &(*_64); _66 = &_1; _65 = &(*_66); _67 = copy _43; _69 = &_56; _68 = &(*_69); _70 = &(*_3); _60 = iterate(move _61, move _63, move _65, move _67, move _68, move _70) => 32;
32: _72 = move _56; _71 = Helpers::<'_>::stop(move _72) => 33;
33: _75 = &mut (_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>); _74 = Option::<Box<CorrectionHistory>>::take(move _75) => 34;
34: _76 = discriminant(_74); switchInt(move _76) -> [1: bb35, otherwise: bb41];
35: goto 36;
36: _77 = move ((_74 as Some).0: std::boxed::Box<search::correction_history::CorrectionHistory>); _81 = &(_1.10: std::sync::Arc<search::correction_history::CorrectionHistories>); _80 = <Arc<CorrectionHistories> as Deref>::deref(move _81) => 37;
37: _79 = &(*_80); _85 = &(*_3); _84 = <Arc<threading::thread::Thread> as Deref>::deref(move _85) => 38;
38: _83 = &(*_84); _82 = threading::thread::Thread::idx(move _83) => 39;
39: _86 = move _77; _78 = CorrectionHistories::put(move _79, move _82, move _86) => 40;
40: _73 = const (); drop(_77) => 43;
41: drop(_74) => 42;
42: _73 = const (); goto 45;
43: drop(_74) => 44;
44: goto 45;
45: _88 = &mut _8; _90 = &(_71.0: search::counters::SearchCounters); _89 = &(*_90); _87 = SearchContext::<'_>::merge_counters(move _88, move _89) => 46;
46: _91 = discriminant(_60); switchInt(move _91) -> [0: bb48, 1: bb49, otherwise: bb47];
47: FakeRead(ForMatchedPlace(None), _60); unreachable;
48: goto 50;
49: _106 = move ((_60 as Endgame).0: search::result::CompletedState); _107 = copy ((_60 as Endgame).1: score::Score); _109 = &mut _8; _108 = &mut (*_109); _111 = &_1; _110 = &(*_111); _112 = &(*_3); _113 = copy _107; _114 = move _106; _115 = Selectivity::None; _0 = solve_root(move _108, move _110, move _112, move _113, move _114, const search::ENDGAME_TRACKING_MIN_SELECTIVITY, move _115) => 64;
50: _92 = move ((_60 as Done).0: search::result::CompletedState); _94 = copy (_1.6: bool); switchInt(move _94) -> [0: bb51, otherwise: bb60];
51: _96 = &(_1.8: std::option::Option<std::sync::Arc<search::time_control::TimeManager>>); _95 = Option::<Arc<TimeManager>>::is_some(move _96) => 52;
52: switchInt(move _95) -> [0: bb59, otherwise: bb53];
53: _99 = &mut (_92.1: std::option::Option<search::result::Iteration>); _98 = Option::<Iteration>::take(move _99) => 54;
54: _100 = move (_71.1: std::vec::Vec<search::result::Iteration>); _97 = vote_best(move _98, move _100) => 55;
55: drop((_92.1: std::option::Option<search::result::Iteration>)) => 56;
56: (_92.1: std::option::Option<search::result::Iteration>) = move _97; drop(_97) => 58;
58: _93 = const (); goto 61;
59: goto 60;
60: _93 = const (); goto 61;
61: _101 = move _92; _103 = &_8; _102 = &(*_103); _105 = &_4; _104 = &(*_105); _0 = CompletedState::into_result(move _101, move _102, move _104) => 62;
62: drop(_92) => 63;
63: goto 66;
64: drop(_106) => 65;
65: goto 66;
66: drop(_71) => 67;
67: drop(_60) => 68;
68: drop(_56) => 69;
69: drop(_8) => 70;
70: drop(_5) => 71;
71: drop(_1) => 75;
72: drop(_8) => 73;
73: drop(_5) => 74;
74: drop(_1) => 75;
75: return;
""",
    'search-root_move-{impl#1}-sort_all': """
0: _4 = &((*_1).0: std::sync::Mutex<std::vec::Vec<search::root_move::RootMove>>); _3 = std::sync::Mutex::<Vec<RootMove>>::lock(move _4) => 1;
1: _2 = Result::<std::sync::MutexGuard<'_, Vec<RootMove>>, PoisonError<std::sync::MutexGuard<'_, Vec<RootMove>>>>::unwrap(move _3) => 2;
2: _10 = &mut _2; _9 = <std::sync::MutexGuard<'_, Vec<RootMove>> as DerefMut>::deref_mut(move _10) => 3;
3: _8 = &mut (*_9); _7 = <Vec<RootMove> as DerefMut>::deref_mut(move _8) => 4;
4: _6 = &mut (*_7); _11 = descending_score; _5 = std::slice::<impl [RootMove]>::sort_by_key::<Reverse<Score>, descending_score>(move _6, move _11) => 5;
5: _0 = const (); drop(_2) => 6;
6: return;
""",
    'search-root_move-{impl#1}-sort_all-{closure#0}': """
0: _3 = copy ((*_2).1: score::Score); _0 = Reverse::<Score>(move _3); return;
""",
}

# The midgame's own stores carry a false endgame tag. Only the special
# non-PV boundary may borrow the endgame caches and store an endgame result.
PVS = "search-pvs-search"
BOUNDARY = "search-strategy-{impl#1}-shallow_search"
EFFECTS = {
    PVS: (2, "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"),
    "search-pvs-enhanced_transposition_cutoff": (1, "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"),
    "search-midgame-evaluate_shallow_with_tt": (1, "const false"),
    BOUNDARY: (1, "const true"),
}
# These are the PVS entry blocks up to the ordinary shallow-search dispatch.
# The remaining PVS blocks are checked in the frame and cache-owner extractors.
BOUNDARY_PREFIX = """
0: _9 = const <NT as search::node_type::NodeType>::PV_NODE; switchInt(move _9) -> [0: bb1, otherwise: bb2];
1: _10 = copy _7; _8 = Not(move _10); goto 3;
2: _8 = const false; goto 3;
3: _11 = copy _4; _13 = const <NT as search::node_type::NodeType>::PV_NODE; switchInt(move _13) -> [0: bb10, otherwise: bb4];
4: _16 = copy _3; _15 = Eq(move _16, const 0_u32); switchInt(move _15) -> [0: bb8, otherwise: bb5];
5: _19 = &mut (*_1); _18 = SearchContext::<'_>::clear_pv(move _19) => 6;
6: _20 = &(*_1); _21 = &(*_2); _0 = <SS as SearchStrategy>::evaluate(move _20, move _21) => 7;
7: goto 445;
8: _14 = const (); _23 = &mut (*_1); _22 = SearchContext::<'_>::prepare_pv(move _23) => 9;
9: _12 = const (); goto 37;
10: _25 = &((*_1).10: empty_list::EmptyList); _24 = EmptyList::count(move _25) => 11;
11: _27 = const <SS as search::strategy::SearchStrategy>::IS_ENDGAME; switchInt(move _27) -> [0: bb12, otherwise: bb26];
12: _29 = copy _3; _30 = copy _24; _28 = Ge(move _29, move _30); switchInt(move _28) -> [0: bb25, otherwise: bb13];
13: _32 = copy _24; _31 = Le(move _32, const <search::strategy::EndGameStrategy as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW); switchInt(move _31) -> [0: bb24, otherwise: bb14];
14: _35 = const true; switchInt(move _35) -> [0: bb21, otherwise: bb15];
15: _40 = copy _5; _39 = Score::value(move _40) => 16;
16: _38 = &_39; _44 = copy _4; _43 = Score::value(move _44) => 17;
17: _45 = AddWithOverflow(copy _43, const 1_i32); assert(!move (_45.1: bool), "attempt to compute `{} + {}`, which would overflow", move _43, const 1_i32) -> [success: bb18, unwind: bb450];
18: _42 = move (_45.0: i32); _41 = &_42; _37 = (move _38, move _41); _46 = copy (_37.0: &i32); _47 = copy (_37.1: &i32); _49 = copy (*_46); _50 = copy (*_47); _48 = Eq(move _49, move _50); switchInt(move _48) -> [0: bb20, otherwise: bb19];
19: _36 = const (); _34 = const (); goto 22;
20: _52 = core::panicking::AssertKind::Eq; _54 = move _52; _56 = &(*_46); _55 = &(*_56); _58 = &(*_47); _57 = &(*_58); _59 = Option::<Arguments<'_>>::None; _53 = core::panicking::assert_failed::<i32, i32>(move _54, move _55, move _57, move _59) -> bb450;
21: _34 = const (); goto 22;
22: _60 = &mut (*_1); _61 = &(*_2); _62 = copy _24; _63 = copy _4; _64 = copy _5; _65 = &(*_6); _0 = <EndGameStrategy as SearchStrategy>::shallow_search(move _60, move _61, move _62, move _63, move _64, move _65) => 23;
23: goto 444;
24: goto 26;
25: goto 26;
26: _26 = const (); _68 = copy _3; _67 = Le(move _68, const <SS as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW); switchInt(move _67) -> [0: bb28, otherwise: bb27];
27: _70 = &mut (*_1); _71 = &(*_2); _72 = copy _3; _73 = copy _4; _74 = copy _5; _75 = &(*_6); _0 = <SS as SearchStrategy>::shallow_search(move _70, move _71, move _72, move _73, move _74, move _75) => 29;
"""

# MidgameAspiration/Model.lean hard-codes the aspiration values, and the others are semantic.
# DEPTH_STEP_THRESHOLD is a free tuning value: no model depends on it.
CONSTANTS = {
    "search-strategy-{impl#0}-IS_ENDGAME": "0: _0 = const false; return;",
    "search-strategy-{impl#1}-DEPTH_TO_SHALLOW": "0: _0 = const search::endgame::DEPTH_TO_NWS; return;",
    "search-aspiration-ASPIRATION_MIN_DEPTH": "0: _0 = const 5_u32; return;",
    "search-aspiration-MSS_DIVISOR": "0: _0 = const 10193_i64; return;",
    "search-aspiration-HELPER_WIDTH_CYCLE": "0: _0 = const 8_usize; return;",
    "search-aspiration-HELPER_WIDTH_STEP": """
0: _1 = Eq(const 4_i32, const 0_i32); assert(!move _1, "attempt to divide `{}` by zero", const score::Score::SCALE) -> [success: bb1, unwind: bb3];
1: _2 = Eq(const 4_i32, const -1_i32); _3 = Eq(const score::Score::SCALE, const i32::MIN); _4 = BitAnd(move _2, move _3); assert(!move _4, "attempt to compute `{} / {}`, which would overflow", const score::Score::SCALE, const 4_i32) -> [success: bb2, unwind: bb3];
2: _0 = Div(const score::Score::SCALE, const 4_i32); return;
""",
    "search-aspiration-MIDGAME_DELTA": """
0: _1 = DiscScore::new(const 2_i32) => 1;
1: _0 = Score::from_disc_diff(move _1) => 2;
2: return;
""",
    "search-ENDGAME_TRACKING_MIN_SELECTIVITY": "0: _0 = Selectivity::Level2; return;",
}


def compare(stem, cfg, text, *, prefix=False):
    expected = {}
    for line in text.strip().splitlines():
        number, ops = line.split(": ", 1)
        expected[int(number)] = ops
    if not prefix:
        checked(cfg.keys() == expected.keys(), f"{stem}: changed normal blocks")
    for pc, ops in expected.items():
        checked(cfg.get(pc) == ops, f"{stem} bb{pc}: expected {ops}; found {cfg.get(pc)}")


def verify(stem, source, closures):
    cfg = normal_blocks(source, closures)
    if stem in BODIES:
        compare(stem, cfg, BODIES[stem])
    if stem == PVS:
        compare(stem, cfg, BOUNDARY_PREFIX, prefix=True)
    if stem in EFFECTS:
        count, tag = EFFECTS[stem]
        sites = [op.split("TranspositionTable::store(", 1)[1].split(") => ", 1)[0].split(", ")
                 for op in cfg.values() if "TranspositionTable::store(" in op]
        checked(len(sites) == count and all(len(args) == 10 and args[8] == tag for args in sites),
                f"{stem}: expected {count} table writes with endgame tag {tag}")


def check(dump):
    closures = {}
    for stem, name in CLOSURES.items():
        signature, = re.findall(r"^fn .*", read(dump, stem), re.M)
        span, = re.findall(r"\{closure@[^}]+\}", signature)
        checked(span not in closures, f"{stem}: duplicate callback")
        closures[span] = name
    for stem in (*BODIES, *EFFECTS):
        verify(stem, read(dump, stem), closures)
    for stem, expected in CONSTANTS.items():
        body = (dump / f"reversi_core.{stem}.built.after.mir").read_text(encoding="utf-8")
        try:
            compare(stem, normal_blocks(body, {}), expected)
        except ValueError as error:
            raise ValueError(f"{error}\n{stem} changed, but the Lean models hard-code its value") from None
    return reject_mutants(dump, closures)


def reject_mutants(dump, closures):
    iterate = "search-midgame-iterate"
    caller = "search-midgame-search_root"
    retry = "search-aspiration-retry_midgame"
    mutants = [
        (iterate, "_25 = Option::<(Square, Score)>::None;", "_25 = copy _93;"),
        (iterate, "_40 = copy _37;", "_40 = const 1_usize;"),
        (iterate, "RootMoves::sort_from_pv_idx(move _60)", "RootMoves::count(move _60)"),
        (iterate, "move _97, const false)", "move _97, const true)"),
        (iterate, "0: bb80, otherwise: bb78", "0: bb78, otherwise: bb80"),
        (iterate, "_146 = Ge(move _147, move _148)", "_146 = Le(move _147, move _148)"),
        (iterate, "0: bb95, otherwise: bb93", "0: bb93, otherwise: bb95"),
        (iterate, "_181 = copy _109;", "_181 = copy _47;"),
        (iterate + "-{closure#0}", "((*_2).1: score::Score)", "((*_2).2: score::Score)"),
        (caller, "_113 = copy _107;", "_113 = const score::Score::ZERO;"),
        (caller, "_109 = &mut _8;", "_109 = &mut _99;"),
        ("search-root_move-{impl#1}-sort_all-{closure#0}", "((*_2).1: score::Score)", "((*_2).2: score::Score)"),
        ("search-midgame-next_iteration_depth", "AddWithOverflow(copy _4, const 2_u32)", "AddWithOverflow(copy _4, const 1_u32)"),
        ("search-midgame-aspiration_window-{closure#0}", "(_2.2: score::Score)", "(_2.1: score::Score)"),
        ("search-midgame-aspiration_search-{closure#0}", "search::<Root, MidGameStrategy>", "search::<PV, MidGameStrategy>"),
        (retry, "0: bb7, otherwise: bb6", "0: bb6, otherwise: bb7"),
        (retry, "div(move _28, const 2_i32)", "div(move _28, const 3_i32)"),
        (PVS, "Ge(move _29, move _30)", "Gt(move _29, move _30)"),
        (PVS, "Le(move _32, const <search::strategy::EndGameStrategy", "Lt(move _32, const <search::strategy::EndGameStrategy"),
        (PVS, "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME, move _444)", "const true, move _444)"),
        ("search-midgame-evaluate_shallow_with_tt", "const false, move _157)", "const true, move _157)"),
        (BOUNDARY, "const true, const false)", "const false, const false)"),
    ]
    for stem, old, new in mutants:
        source = read(dump, stem)
        checked(source.count(old) == 1, f"{stem}: missing or ambiguous mutation: {old}")
        try:
            verify(stem, source.replace(old, new), closures)
        except ValueError:
            continue
        raise ValueError(f"{stem}: accepted mutation {old}")
    return len(mutants)
