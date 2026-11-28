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
1: _9 = copy ((*_1).10: u32); _11 = &(*_3); _13 = &(*((*_1).1: &search::root_move::RootMoves)); _12 = RootMoves::count(move _13) => 2;
2: _10 = SearchTask::pv_count(move _11, move _12) => 3;
3: _15 = copy _4; _14 = compute_start_depth(move _15) => 4;
4: _17 = copy _10; _16 = CompletedState::new(move _17) => 5;
5: goto 6;
6: falseUnwind -> [real: bb7, unwind: bb115];
7: _21 = copy _14; _22 = copy _4; _20 = Le(move _21, move _22); switchInt(move _20) -> [0: bb103, otherwise: bb8];
8: _23 = const false; _24 = Option::<(Square, Score)>::None; _28 = copy _10; _27 = std::ops::Range::<usize> { start: const 0_usize, end: move _28 }; _26 = <std::ops::Range<usize> as IntoIterator>::into_iter(move _27) => 9;
9: _29 = move _26; goto 10;
10: falseUnwind -> [real: bb11, unwind: bb115];
11: _33 = &mut _29; _32 = &mut (*_33); _31 = <std::ops::Range<usize> as Iterator>::next(move _32) => 12;
12: _34 = discriminant(_31); switchInt(move _34) -> [0: bb14, 1: bb15, otherwise: bb13];
13: FakeRead(ForMatchedPlace(None), _31); unreachable;
14: goto 16;
15: _36 = copy ((_31 as Some).0: usize); _38 = &(*((*_1).1: &search::root_move::RootMoves)); _39 = copy _36; _37 = RootMoves::set_pv_idx(move _38, move _39) => 17;
16: _25 = const (); goto 49;
17: _44 = &(*_1); _45 = copy _14; _43 = aspiration_window(move _44, move _45) => 18;
18: _40 = copy (_43.0: score::Score); _41 = copy (_43.1: score::Score); _42 = copy (_43.2: score::Score); _49 = &mut (*_1); _50 = &(*_2); _51 = copy _14; _53 = &mut _40; _52 = &mut (*_53); _55 = &mut _41; _54 = &mut (*_55); _56 = copy _42; _57 = &(*_6); _48 = midgame::aspiration_search(move _49, move _50, move _51, move _52, move _54, move _56, move _57) => 19;
19: _46 = copy (_48.0: score::Score); _47 = copy (_48.1: bool); _59 = &(*((*_1).1: &search::root_move::RootMoves)); _58 = RootMoves::sort_from_pv_idx(move _59) => 20;
20: _64 = &(*_6); _63 = <Arc<threading::thread::Thread> as Deref>::deref(move _64) => 21;
21: _62 = &(*_63); _61 = threading::thread::Thread::is_search_aborted(move _62) => 22;
22: switchInt(move _61) -> [0: bb24, otherwise: bb23];
23: _25 = const (); goto 48;
24: _60 = const (); _67 = discriminant(((*_3).7: std::option::Option<std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>>)); switchInt(move _67) -> [1: bb25, otherwise: bb38];
25: goto 26;
26: _68 = &((((*_3).7: std::option::Option<std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>>) as Some).0: std::sync::Arc<dyn std::ops::Fn(search::progress::SearchProgress) + std::marker::Send + std::marker::Sync>); _70 = &(*((*_1).1: &search::root_move::RootMoves)); _69 = RootMoves::current_pv(move _70) => 27;
27: _71 = discriminant(_69); switchInt(move _71) -> [1: bb28, otherwise: bb36];
28: goto 29;
29: _72 = move ((_69 as Some).0: search::root_move::RootMove); _74 = copy ((*_1).2: u64); _76 = &(*_5); _75 = Helpers::<'_>::nodes(move _76) => 30;
30: _77 = AddWithOverflow(copy _74, copy _75); assert(!move (_77.1: bool), "attempt to compute `{} + {}`, which would overflow", move _74, move _75) -> [success: bb31, unwind: bb113];
31: _73 = move (_77.0: u64); _81 = &(*_68); _80 = <Arc<dyn Fn(SearchProgress) + Send + Sync> as Deref>::deref(move _81) => 32;
32: _79 = &(*_80); _84 = &(*_1); _86 = &_72; _85 = &(*_86); _87 = copy _14; _88 = copy _4; _89 = copy _46; _90 = copy _73; _83 = SearchProgress::from_iteration(move _84, move _85, move _87, move _88, move _89, move _90, const false) => 33;
33: _82 = (move _83,); _78 = <dyn Fn(SearchProgress) + Send + Sync as Fn<(SearchProgress,)>>::call(move _79, move _82) => 34;
34: drop(_83) => 35;
35: _66 = const (); drop(_72) => 39;
36: drop(_69) => 37;
37: goto 38;
38: _66 = const (); goto 41;
39: drop(_69) => 40;
40: goto 41;
41: _91 = copy _47; _23 = move _91; _94 = &mut _16; _95 = &(*_1); _96 = copy _14; _93 = CompletedState::line_completed(move _94, move _95, move _96, const false) => 42;
42: _97 = row_score; _92 = Option::<&RootMove>::map::<(Square, Score), row_score>(move _93, move _97) => 43;
43: _24 = move _92; _101 = &(*_6); _100 = <Arc<threading::thread::Thread> as Deref>::deref(move _101) => 44;
44: _99 = &(*_100); _98 = threading::thread::Thread::is_search_aborted(move _99) => 45;
45: switchInt(move _98) -> [0: bb47, otherwise: bb46];
46: _25 = const (); goto 48;
47: _30 = const (); _19 = const (); goto 10;
48: goto 49;
49: _109 = discriminant(_24); switchInt(move _109) -> [1: bb53, otherwise: bb52];
50: _106 = move _16; _0 = Outcome::Done(move _106); drop(_106) => 51;
51: goto 108;
52: _105 = copy _7; _104 = record_abort(move _105) => 50;
53: goto 54;
54: _107 = copy (((_24 as Some).0: (square::Square, score::Score)).0: square::Square); _108 = copy (((_24 as Some).0: (square::Square, score::Score)).1: score::Score); _111 = &(*((*_1).1: &search::root_move::RootMoves)); _110 = RootMoves::sort_all(move _111) => 55;
55: _113 = discriminant(_7); switchInt(move _113) -> [1: bb56, otherwise: bb65];
56: goto 57;
57: _114 = copy ((_7 as Some).0: &search::time_control::TimeManager); _116 = copy ((*_3).6: bool); switchInt(move _116) -> [0: bb58, otherwise: bb61];
58: _118 = copy _14; _117 = Ge(move _118, const search::aspiration::ASPIRATION_MIN_DEPTH); switchInt(move _117) -> [0: bb60, otherwise: bb59];
59: _115 = copy _23; goto 62;
60: goto 61;
61: _115 = const false; goto 62;
62: _120 = &(*_114); _121 = copy _107; _123 = copy _108; _122 = Score::to_disc_diff_f32(move _123) => 63;
63: _124 = copy _14; _125 = copy _115; _119 = TimeManager::report_iteration(move _120, move _121, move _122, move _124, move _125) => 64;
64: _112 = const (); goto 66;
65: _112 = const (); goto 66;
66: _129 = &(*((*_1).1: &search::root_move::RootMoves)); _128 = RootMoves::count(move _129) => 67;
67: _127 = Eq(move _128, const 1_usize); switchInt(move _127) -> [0: bb73, otherwise: bb68];
68: _131 = copy _7; _132 = deadline; _130 = Option::<&TimeManager>::is_some_and::<deadline>(move _131, move _132) => 69;
69: switchInt(move _130) -> [0: bb72, otherwise: bb70];
70: _134 = move _16; _0 = Outcome::Done(move _134); drop(_134) => 71;
71: goto 107;
72: goto 74;
73: goto 74;
74: _126 = const (); _139 = &(*_6); _138 = <Arc<threading::thread::Thread> as Deref>::deref(move _139) => 75;
75: _137 = &(*_138); _136 = threading::thread::Thread::is_search_aborted(move _137) => 76;
76: switchInt(move _136) -> [0: bb79, otherwise: bb77];
77: _141 = move _16; _0 = Outcome::Done(move _141); drop(_141) => 78;
78: goto 107;
79: _135 = const (); _143 = discriminant(_7); switchInt(move _143) -> [1: bb80, otherwise: bb97];
80: goto 81;
81: _144 = copy ((_7 as Some).0: &search::time_control::TimeManager); _146 = copy _14; _147 = copy _4; _145 = Ge(move _146, move _147); switchInt(move _145) -> [0: bb96, otherwise: bb82];
82: _150 = const true; switchInt(move _150) -> [0: bb87, otherwise: bb83];
83: _153 = &_4; _154 = &_9; _152 = (move _153, move _154); _155 = copy (_152.0: &u32); _156 = copy (_152.1: &u32); _158 = copy (*_155); _159 = copy (*_156); _157 = Eq(move _158, move _159); switchInt(move _157) -> [0: bb85, otherwise: bb84];
84: _151 = const (); _149 = const (); goto 88;
85: _161 = core::panicking::AssertKind::Eq; _163 = move _161; _165 = &(*_155); _164 = &(*_165); _167 = &(*_156); _166 = &(*_167); _169 = Arguments::<'_>::from_str(const "time-controlled midgame searches always target all empties") => 86;
86: _168 = Option::<Arguments<'_>>::Some(move _169); _162 = core::panicking::assert_failed::<u32, u32>(move _163, move _164, move _166, move _168) -> bb115;
87: _149 = const (); goto 88;
88: _172 = &(*_144); _174 = copy ((*_1).2: u64); _176 = &(*_5); _175 = Helpers::<'_>::nodes(move _176) => 89;
89: _177 = AddWithOverflow(copy _174, copy _175); assert(!move (_177.1: bool), "attempt to compute `{} + {}`, which would overflow", move _174, move _175) -> [success: bb90, unwind: bb115];
90: _173 = move (_177.0: u64); _171 = TimeManager::should_enter_endgame(move _172, move _173) => 91;
91: switchInt(move _171) -> [0: bb94, otherwise: bb92];
92: _179 = move _16; _180 = copy _108; _0 = Outcome::Endgame(move _179, move _180); drop(_179) => 93;
93: goto 106;
94: _170 = const (); _181 = move _16; _0 = Outcome::Done(move _181); drop(_181) => 95;
95: goto 106;
96: goto 97;
97: _142 = const (); _184 = copy _7; _183 = should_stop_iteration(move _184) => 98;
98: switchInt(move _183) -> [0: bb101, otherwise: bb99];
99: _186 = move _16; _0 = Outcome::Done(move _186); drop(_186) => 100;
100: goto 107;
101: _182 = const (); _188 = copy _14; _187 = next_iteration_depth(move _188) => 102;
102: _14 = move _187; _19 = const (); goto 6;
103: _18 = const (); _192 = move _16; _0 = Outcome::Done(move _192); drop(_192) => 104;
104: drop(_16) => 105;
105: goto 110;
106: goto 107;
107: goto 108;
108: drop(_16) => 109;
109: goto 110;
110: return;
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
10: _0 = SearchResult::NoLegalMove; goto 71;
11: _26 = const (); _33 = &_4; _35 = Board::new() => 12;
12: _34 = &_35; _32 = <Board as PartialEq>::eq(move _33, move _34) => 13;
13: switchInt(move _32) -> [0: bb18, otherwise: bb14];
14: _36 = copy (_1.6: bool); switchInt(move _36) -> [0: bb15, otherwise: bb19];
15: _40 = &_4; _39 = &(*_40); _38 = random_move(move _39) => 16;
16: _0 = SearchResult::new_random_move(move _38) => 17;
17: goto 71;
18: goto 19;
19: _31 = const (); _41 = copy (_8.10: u32); _44 = copy _2; _43 = <u32 as Ord>::max(move _44, const 1_u32) => 20;
20: _45 = copy _41; _42 = <u32 as Ord>::min(move _43, move _45) => 21;
21: _50 = &(_1.10: std::sync::Arc<search::correction_history::CorrectionHistories>); _49 = <Arc<CorrectionHistories> as Deref>::deref(move _50) => 22;
22: _48 = &(*_49); _54 = &(*_3); _53 = <Arc<threading::thread::Thread> as Deref>::deref(move _54) => 23;
23: _52 = &(*_53); _51 = threading::thread::Thread::idx(move _52) => 24;
24: _47 = CorrectionHistories::take(move _48, move _51) => 25;
25: _46 = Option::<Box<CorrectionHistory>>::Some(move _47); drop(_47) => 26;
26: drop((_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>)) => 27;
27: (_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>) = move _46; drop(_46) => 29;
29: _57 = &_1; _56 = &(*_57); _58 = copy _42; _55 = Helpers::<'_>::start(move _56, move _58) => 30;
30: _61 = &mut _8; _60 = &mut (*_61); _63 = &_4; _62 = &(*_63); _65 = &_1; _64 = &(*_65); _66 = copy _42; _68 = &_55; _67 = &(*_68); _69 = &(*_3); _59 = iterate(move _60, move _62, move _64, move _66, move _67, move _69) => 31;
31: _71 = move _55; _70 = Helpers::<'_>::stop(move _71) => 32;
32: _74 = &mut (_8.4: std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>); _73 = Option::<Box<CorrectionHistory>>::take(move _74) => 33;
33: _75 = discriminant(_73); switchInt(move _75) -> [1: bb34, otherwise: bb40];
34: goto 35;
35: _76 = move ((_73 as Some).0: std::boxed::Box<search::correction_history::CorrectionHistory>); _80 = &(_1.10: std::sync::Arc<search::correction_history::CorrectionHistories>); _79 = <Arc<CorrectionHistories> as Deref>::deref(move _80) => 36;
36: _78 = &(*_79); _84 = &(*_3); _83 = <Arc<threading::thread::Thread> as Deref>::deref(move _84) => 37;
37: _82 = &(*_83); _81 = threading::thread::Thread::idx(move _82) => 38;
38: _85 = move _76; _77 = CorrectionHistories::put(move _78, move _81, move _85) => 39;
39: _72 = const (); drop(_76) => 42;
40: drop(_73) => 41;
41: _72 = const (); goto 44;
42: drop(_73) => 43;
43: goto 44;
44: _87 = &mut _8; _89 = &(_70.0: search::counters::SearchCounters); _88 = &(*_89); _86 = SearchContext::<'_>::merge_counters(move _87, move _88) => 45;
45: _90 = discriminant(_59); switchInt(move _90) -> [0: bb47, 1: bb48, otherwise: bb46];
46: FakeRead(ForMatchedPlace(None), _59); unreachable;
47: goto 49;
48: _105 = move ((_59 as Endgame).0: search::result::CompletedState); _106 = copy ((_59 as Endgame).1: score::Score); _108 = &mut _8; _107 = &mut (*_108); _110 = &_1; _109 = &(*_110); _111 = &(*_3); _112 = copy _106; _113 = move _105; _114 = Selectivity::None; _0 = solve_root(move _107, move _109, move _111, move _112, move _113, const search::ENDGAME_TRACKING_MIN_SELECTIVITY, move _114) => 63;
49: _91 = move ((_59 as Done).0: search::result::CompletedState); _93 = copy (_1.6: bool); switchInt(move _93) -> [0: bb50, otherwise: bb59];
50: _95 = &(_1.8: std::option::Option<std::sync::Arc<search::time_control::TimeManager>>); _94 = Option::<Arc<TimeManager>>::is_some(move _95) => 51;
51: switchInt(move _94) -> [0: bb58, otherwise: bb52];
52: _98 = &mut (_91.1: std::option::Option<search::result::Iteration>); _97 = Option::<Iteration>::take(move _98) => 53;
53: _99 = move (_70.1: std::vec::Vec<search::result::Iteration>); _96 = vote_best(move _97, move _99) => 54;
54: drop((_91.1: std::option::Option<search::result::Iteration>)) => 55;
55: (_91.1: std::option::Option<search::result::Iteration>) = move _96; drop(_96) => 57;
57: _92 = const (); goto 60;
58: goto 59;
59: _92 = const (); goto 60;
60: _100 = move _91; _102 = &_8; _101 = &(*_102); _104 = &_4; _103 = &(*_104); _0 = CompletedState::into_result(move _100, move _101, move _103) => 61;
61: drop(_91) => 62;
62: goto 65;
63: drop(_105) => 64;
64: goto 65;
65: drop(_70) => 66;
66: drop(_59) => 67;
67: drop(_55) => 68;
68: drop(_8) => 69;
69: drop(_5) => 70;
70: drop(_1) => 74;
71: drop(_8) => 72;
72: drop(_5) => 73;
73: drop(_1) => 74;
74: return;
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
    "search-pvs-endgame_cache_cutoff": (1, "const true"),
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
7: goto 457;
8: _14 = const (); _23 = &mut (*_1); _22 = SearchContext::<'_>::prepare_pv(move _23) => 9;
9: _12 = const (); goto 36;
10: _24 = copy ((*_1).10: u32); _26 = const <SS as search::strategy::SearchStrategy>::IS_ENDGAME; switchInt(move _26) -> [0: bb11, otherwise: bb25];
11: _28 = copy _3; _29 = copy _24; _27 = Ge(move _28, move _29); switchInt(move _27) -> [0: bb24, otherwise: bb12];
12: _31 = copy _24; _30 = Le(move _31, const <search::strategy::EndGameStrategy as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW); switchInt(move _30) -> [0: bb23, otherwise: bb13];
13: _34 = const true; switchInt(move _34) -> [0: bb20, otherwise: bb14];
14: _39 = copy _5; _38 = Score::value(move _39) => 15;
15: _37 = &_38; _43 = copy _4; _42 = Score::value(move _43) => 16;
16: _44 = AddWithOverflow(copy _42, const 1_i32); assert(!move (_44.1: bool), "attempt to compute `{} + {}`, which would overflow", move _42, const 1_i32) -> [success: bb17, unwind: bb462];
17: _41 = move (_44.0: i32); _40 = &_41; _36 = (move _37, move _40); _45 = copy (_36.0: &i32); _46 = copy (_36.1: &i32); _48 = copy (*_45); _49 = copy (*_46); _47 = Eq(move _48, move _49); switchInt(move _47) -> [0: bb19, otherwise: bb18];
18: _35 = const (); _33 = const (); goto 21;
19: _51 = core::panicking::AssertKind::Eq; _53 = move _51; _55 = &(*_45); _54 = &(*_55); _57 = &(*_46); _56 = &(*_57); _58 = Option::<Arguments<'_>>::None; _52 = core::panicking::assert_failed::<i32, i32>(move _53, move _54, move _56, move _58) -> bb462;
20: _33 = const (); goto 21;
21: _59 = &mut (*_1); _60 = &(*_2); _61 = copy _24; _62 = copy _4; _63 = copy _5; _64 = &(*_6); _0 = <EndGameStrategy as SearchStrategy>::shallow_search(move _59, move _60, move _61, move _62, move _63, move _64) => 22;
22: goto 456;
23: goto 25;
24: goto 25;
25: _25 = const (); _67 = copy _3; _66 = Le(move _67, const <SS as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW); switchInt(move _66) -> [0: bb27, otherwise: bb26];
26: _69 = &mut (*_1); _70 = &(*_2); _71 = copy _3; _72 = copy _4; _73 = copy _5; _74 = &(*_6); _0 = <SS as SearchStrategy>::shallow_search(move _69, move _70, move _71, move _72, move _73, move _74) => 28;
27: _65 = const (); _76 = &(*_2); _77 = copy _24; _79 = copy _4; _78 = Score::to_disc_diff(move _79) => 29;
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
        (iterate, "_24 = Option::<(Square, Score)>::None;", "_24 = copy _92;"),
        (iterate, "_39 = copy _36;", "_39 = const 1_usize;"),
        (iterate, "RootMoves::sort_from_pv_idx(move _59)", "RootMoves::count(move _59)"),
        (iterate, "move _96, const false)", "move _96, const true)"),
        (iterate, "0: bb79, otherwise: bb77", "0: bb77, otherwise: bb79"),
        (iterate, "_145 = Ge(move _146, move _147)", "_145 = Le(move _146, move _147)"),
        (iterate, "0: bb94, otherwise: bb92", "0: bb92, otherwise: bb94"),
        (iterate, "_180 = copy _108;", "_180 = copy _46;"),
        (iterate + "-{closure#0}", "((*_2).1: score::Score)", "((*_2).2: score::Score)"),
        (caller, "_112 = copy _106;", "_112 = const score::Score::ZERO;"),
        (caller, "_108 = &mut _8;", "_108 = &mut _98;"),
        ("search-root_move-{impl#1}-sort_all-{closure#0}", "((*_2).1: score::Score)", "((*_2).2: score::Score)"),
        ("search-midgame-next_iteration_depth", "AddWithOverflow(copy _4, const 2_u32)", "AddWithOverflow(copy _4, const 1_u32)"),
        ("search-midgame-aspiration_window-{closure#0}", "(_2.2: score::Score)", "(_2.1: score::Score)"),
        ("search-midgame-aspiration_search-{closure#0}", "search::<Root, MidGameStrategy>", "search::<PV, MidGameStrategy>"),
        (retry, "0: bb7, otherwise: bb6", "0: bb6, otherwise: bb7"),
        (retry, "div(move _28, const 2_i32)", "div(move _28, const 3_i32)"),
        (PVS, "Ge(move _28, move _29)", "Gt(move _28, move _29)"),
        (PVS, "Le(move _31, const <search::strategy::EndGameStrategy", "Lt(move _31, const <search::strategy::EndGameStrategy"),
        (PVS, "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME, move _464)", "const true, move _464)"),
        ("search-pvs-endgame_cache_cutoff", "const true, move _53)", "const false, move _53)"),
        ("search-midgame-evaluate_shallow_with_tt", "const false, move _155)", "const true, move _155)"),
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
