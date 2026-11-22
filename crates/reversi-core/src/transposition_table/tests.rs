use super::*;
use crate::search::node_type::{NonPV, PV};

const START_PLAYER: u64 = 0x0000_0008_1000_0000;
const START_OPPONENT: u64 = 0x0000_0010_0800_0000;
const CLUSTER_FIXTURE_SEARCH_LIMIT: u64 = 1_000_000;

fn raw_score(raw_value: i32) -> Score {
    Score::from_raw(raw_value)
}

fn sq(idx: usize) -> Option<Square> {
    Some(Square::from_u8(idx as u8).unwrap())
}

fn make_board(player: u64, opponent: u64) -> Board {
    Board::from_bitboards(player, opponent)
}

fn raw_entry_data(
    raw_value: i32,
    best_move: Option<Square>,
    bound: Bound,
    depth: Depth,
    selectivity: Selectivity,
    generation: u8,
    is_endgame: bool,
) -> TTEntryData {
    TTEntryData::new(
        raw_score(raw_value),
        best_move,
        bound,
        depth,
        selectivity,
        generation,
        is_endgame,
        false,
    )
}

fn read(entry: &TTEntry, board: &Board) -> Option<TTEntryData> {
    entry.read_for_lookup(board.player().bits(), board.opponent().bits())
}

fn stable_data(entry: &TTEntry, board: &Board) -> TTEntryData {
    read(entry, board).expect("entry should match board")
}

fn is_hit(probe: &TTProbeResult<'_>) -> bool {
    matches!(probe, TTProbeResult::Hit { .. })
}

fn slot_at(tt: &TranspositionTable, entry_index: usize) -> TTSlot<'_> {
    TTSlot(&tt.entries[entry_index])
}

fn slot_index(tt: &TranspositionTable, slot: TTSlot<'_>) -> usize {
    tt.entries
        .element_offset(slot.0)
        .expect("slot should belong to the table")
}

#[test]
fn store_rejects_sentinels_and_scores_that_do_not_fit() {
    let tt = TranspositionTable::new(0);
    let board = Board::new();
    let key = board.hash();
    let slot = tt.probe(&board, key).slot();
    let write = |score, depth| {
        tt.store(
            slot,
            &board,
            score,
            Bound::Exact,
            depth,
            Some(Square::D3),
            Selectivity::None,
            false,
            false,
        )
    };
    for score in [
        -Score::INF,
        Score::INF,
        Score::INF + 1,
        -Score::INF - 1,
        Score::ZERO + i32::MIN,
    ] {
        write(score, 12);
        assert!(
            tt.lookup(&board, key).is_none(),
            "stored invalid score {score:?}"
        );
    }

    for valid in [-Score::INF + 1, Score::INF - 1] {
        write(valid, 12);
        write(Score::INF, 13);
        let entry = tt.lookup(&board, key).unwrap();
        assert_eq!(entry.score(), valid);
        assert_eq!(entry.depth(), 12);
    }
}

struct ClusterFixture {
    cluster_idx: usize,
    stored: Board,
    colliding: Board,
    miss: Board,
}

impl ClusterFixture {
    fn new(tt: &TranspositionTable, stored: Board) -> Self {
        let cluster_idx = tt.get_cluster_idx(stored.hash());
        let colliding = find_collision_board(tt, cluster_idx, 1, |board| *board == stored);
        let miss = find_collision_board(tt, cluster_idx, 10_000, |board| {
            *board == stored || *board == colliding
        });

        Self {
            cluster_idx,
            stored,
            colliding,
            miss,
        }
    }
}

fn find_collision_board(
    tt: &TranspositionTable,
    cluster_idx: usize,
    seed_start: u64,
    reject: impl Fn(&Board) -> bool,
) -> Board {
    (seed_start..seed_start.saturating_add(CLUSTER_FIXTURE_SEARCH_LIMIT))
        .map(|seed| make_board(seed, seed << 32))
        .find(|board| tt.get_cluster_idx(board.hash()) == cluster_idx && !reject(board))
        .expect("cluster fixture search should find a matching board")
}

fn mark_slot_in_flight(tt: &TranspositionTable, entry_index: usize) {
    // SAFETY: tests pass indices produced from `get_cluster_idx` plus a
    // `0..CLUSTER_SIZE` offset, so the slot is inside the table allocation.
    let entry = unsafe { tt.entries.get_unchecked(entry_index) };

    assert!(
        entry
            .seq
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    );
}

mod bound {
    use super::*;

    #[test]
    fn classify_returns_lower_when_score_reaches_beta_for_any_node_type() {
        assert_eq!(
            Bound::classify::<PV>(raw_score(50), raw_score(30), raw_score(50)),
            Bound::Lower
        );
        assert_eq!(
            Bound::classify::<NonPV>(raw_score(51), raw_score(30), raw_score(50)),
            Bound::Lower
        );
    }

    #[test]
    fn classify_returns_exact_only_for_pv_scores_between_alpha_and_beta() {
        assert_eq!(
            Bound::classify::<PV>(raw_score(31), raw_score(30), raw_score(50)),
            Bound::Exact
        );
        assert_eq!(
            Bound::classify::<NonPV>(raw_score(31), raw_score(30), raw_score(50)),
            Bound::Upper
        );
    }

    #[test]
    fn classify_returns_upper_at_or_below_alpha() {
        assert_eq!(
            Bound::classify::<PV>(raw_score(30), raw_score(30), raw_score(50)),
            Bound::Upper
        );
        assert_eq!(
            Bound::classify::<PV>(raw_score(29), raw_score(30), raw_score(50)),
            Bound::Upper
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out-of-range")]
    fn from_u8_unchecked_rejects_invalid_discriminants_in_debug_builds() {
        // SAFETY: this debug-only test intentionally passes an invalid
        // discriminant and expects the debug assertion to panic before the
        // transmute executes.
        let _ = unsafe { Bound::from_u8_unchecked(4) };
    }
}

mod tt_entry_data {
    use super::*;

    #[test]
    fn packing_roundtrip_preserves_every_field() {
        let original = TTEntryData::new(
            Score::MIN,
            sq(63),
            Bound::Exact,
            60,
            Selectivity::None,
            127,
            true,
            true,
        );

        let unpacked = TTEntryData::from_u64(original.to_u64());

        assert_eq!(unpacked.score(), Score::MIN);
        assert_eq!(unpacked.best_move(), sq(63));
        assert_eq!(unpacked.bound(), Bound::Exact);
        assert_eq!(unpacked.depth(), 60);
        assert_eq!(unpacked.selectivity(), Selectivity::None);
        assert_eq!(unpacked.generation(), 127);
        assert!(unpacked.is_endgame());
        assert!(unpacked.is_pv());

        let non_pv = TTEntryData::new(
            raw_score(0),
            None,
            Bound::Lower,
            1,
            Selectivity::Level1,
            0,
            false,
            false,
        );
        assert!(!non_pv.is_endgame());
        assert!(!non_pv.is_pv());

        let endgame_only = TTEntryData::new(
            raw_score(0),
            None,
            Bound::Lower,
            1,
            Selectivity::Level1,
            0,
            true,
            false,
        );
        assert!(endgame_only.is_endgame());
        assert!(!endgame_only.is_pv());
    }

    #[test]
    fn can_cut_uses_lower_or_exact_when_score_reaches_beta() {
        let lower = raw_entry_data(100, None, Bound::Lower, 6, Selectivity::Level2, 0, false);
        let exact = raw_entry_data(100, None, Bound::Exact, 6, Selectivity::Level2, 0, false);
        let upper = raw_entry_data(100, None, Bound::Upper, 6, Selectivity::Level2, 0, false);

        let lower_at_beta =
            raw_entry_data(50, None, Bound::Lower, 6, Selectivity::Level2, 0, false);

        assert!(lower_at_beta.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(lower.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(exact.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(!upper.can_cut(raw_score(50), 6, Selectivity::Level2, false));
    }

    #[test]
    fn can_cut_uses_upper_or_exact_when_score_is_below_beta() {
        let lower = raw_entry_data(30, None, Bound::Lower, 6, Selectivity::Level2, 0, false);
        let exact = raw_entry_data(30, None, Bound::Exact, 6, Selectivity::Level2, 0, false);
        let upper = raw_entry_data(30, None, Bound::Upper, 6, Selectivity::Level2, 0, false);

        assert!(!lower.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(exact.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(upper.can_cut(raw_score(50), 6, Selectivity::Level2, false));
    }

    #[test]
    fn can_cut_rejects_entries_that_are_too_shallow_too_selective_or_not_endgame() {
        let midgame = raw_entry_data(100, None, Bound::Lower, 6, Selectivity::Level2, 0, false);
        let endgame = raw_entry_data(100, None, Bound::Lower, 6, Selectivity::Level2, 0, true);

        assert!(!midgame.can_cut(raw_score(50), 7, Selectivity::Level2, false));
        assert!(!midgame.can_cut(raw_score(50), 6, Selectivity::Level3, false));
        assert!(!midgame.can_cut(raw_score(50), 6, Selectivity::Level2, true));
        assert!(endgame.can_cut(raw_score(50), 6, Selectivity::Level2, true));
    }

    #[test]
    fn relative_age_uses_the_seven_bit_generation_ring() {
        let data = raw_entry_data(0, None, Bound::Exact, 0, Selectivity::Level1, 126, false);

        assert_eq!(data.relative_age(126), 0);
        assert_eq!(data.relative_age(127), 1);
        assert_eq!(data.relative_age(2), 4);
    }

    #[test]
    fn replacement_score_penalizes_old_entries_by_age_weight() {
        let fresh = raw_entry_data(0, None, Bound::Lower, 8, Selectivity::Level1, 5, false);
        let old = raw_entry_data(0, None, Bound::Lower, 16, Selectivity::Level1, 1, false);

        assert_eq!(fresh.replacement_score(5), 8);
        assert_eq!(old.replacement_score(5), 16 - 4 * TTEntry::AGE_WEIGHT);
    }
}

mod tt_entry {
    use super::*;

    #[test]
    fn same_board_non_exact_update_too_shallow_keeps_existing_entry() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);

        entry.save(
            &board,
            raw_entry_data(50, sq(5), Bound::Lower, 10, Selectivity::Level2, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(60, sq(6), Bound::Lower, 7, Selectivity::Level2, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 50);
        assert_eq!(data.depth(), 10);
        assert_eq!(data.best_move(), sq(5));
    }

    #[test]
    fn same_board_non_exact_update_replaces_within_two_plies() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);

        entry.save(
            &board,
            raw_entry_data(50, sq(5), Bound::Lower, 10, Selectivity::Level2, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(60, sq(6), Bound::Lower, 8, Selectivity::Level2, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 60);
        assert_eq!(data.depth(), 8);
        assert_eq!(data.best_move(), sq(6));
    }

    #[test]
    fn same_board_non_exact_update_replaces_when_selectivity_increases() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);

        entry.save(
            &board,
            raw_entry_data(50, sq(5), Bound::Upper, 12, Selectivity::Level1, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(60, sq(6), Bound::Upper, 8, Selectivity::Level3, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 60);
        assert_eq!(data.selectivity(), Selectivity::Level3);
    }

    #[test]
    fn same_board_non_exact_update_more_selective_but_too_shallow_keeps_existing_entry() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);

        entry.save(
            &board,
            raw_entry_data(50, sq(5), Bound::Upper, 12, Selectivity::Level1, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(60, sq(6), Bound::Lower, 7, Selectivity::None, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 50);
        assert_eq!(data.depth(), 12);
        assert_eq!(data.best_move(), sq(5));
    }

    #[test]
    fn penalize_lowers_depth_once_per_observed_entry() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);
        let observed = raw_entry_data(50, sq(5), Bound::Upper, 12, Selectivity::Level1, 1, false);
        entry.save(&board, observed);

        entry.penalize(board.player().bits(), board.opponent().bits(), observed);
        entry.penalize(board.player().bits(), board.opponent().bits(), observed);

        let data = stable_data(&entry, &board);
        assert_eq!(data.depth(), 11);
        assert_eq!(data.score().value(), 50);
        assert_eq!(data.bound(), Bound::Upper);
        assert_eq!(data.best_move(), sq(5));
        assert_eq!(data.selectivity(), Selectivity::Level1);
    }

    #[test]
    fn penalize_leaves_a_different_board_untouched() {
        let entry = TTEntry::default();
        let stored = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);
        let other = make_board(0x0000_0000_0000_01ff, 0x0000_0000_0000_fe00);
        let observed = raw_entry_data(50, sq(5), Bound::Upper, 12, Selectivity::Level1, 1, false);
        entry.save(&stored, observed);

        entry.penalize(other.player().bits(), other.opponent().bits(), observed);

        assert_eq!(stable_data(&entry, &stored).depth(), 12);
    }

    #[test]
    fn same_board_non_exact_update_replaces_when_generation_changes() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);

        entry.save(
            &board,
            raw_entry_data(50, sq(5), Bound::Lower, 12, Selectivity::Level2, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(60, sq(6), Bound::Lower, 2, Selectivity::Level2, 2, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 60);
        assert_eq!(data.generation(), 2);
    }

    #[test]
    fn exact_update_always_replaces_and_does_not_preserve_old_best_move() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_0f0f, 0x0000_0000_0000_f0f0);

        entry.save(
            &board,
            raw_entry_data(30, sq(11), Bound::Lower, 12, Selectivity::Level3, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(40, None, Bound::Exact, 1, Selectivity::Level1, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 40);
        assert_eq!(data.bound(), Bound::Exact);
        assert_eq!(data.depth(), 1);
        assert_eq!(data.best_move(), None);
    }

    #[test]
    fn same_board_non_exact_update_preserves_best_move_when_new_move_is_none() {
        let entry = TTEntry::default();
        let board = make_board(0x0000_0000_0000_0f0f, 0x0000_0000_0000_f0f0);

        entry.save(
            &board,
            raw_entry_data(30, sq(11), Bound::Lower, 12, Selectivity::Level3, 1, false),
        );
        entry.save(
            &board,
            raw_entry_data(40, None, Bound::Lower, 11, Selectivity::Level3, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 40);
        assert_eq!(data.best_move(), sq(11));
    }

    #[test]
    fn different_board_replaces_the_slot_even_with_weaker_metadata() {
        let old_board = Board::from_bitboards(Square::D4.bitboard(), Square::C4.bitboard());
        let old_move = Square::B4;
        assert!(old_board.is_legal_move(old_move));
        let old = raw_entry_data(
            80,
            Some(old_move),
            Bound::Exact,
            20,
            Selectivity::None,
            10,
            true,
        );
        let new = raw_entry_data(90, None, Bound::Upper, 1, Selectivity::Level1, 10, false);

        for board in [
            Board::from_bitboards(Square::D4.bitboard(), Square::E4.bitboard()),
            Board::from_bitboards(Square::B4.bitboard(), Square::C4.bitboard()),
        ] {
            assert!(!board.is_legal_move(old_move));
            let entry = TTEntry::default();
            entry.save(&old_board, old);
            entry.save(&board, new);

            assert!(read(&entry, &old_board).is_none());
            assert_eq!(stable_data(&entry, &board), new);
        }
    }

    #[test]
    fn snapshot_reads_return_none_while_a_write_is_in_flight() {
        let entry = TTEntry::default();

        assert!(
            entry
                .seq
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        );

        assert!(entry.try_load_snapshot().is_none());
        assert!(entry.read_for_lookup(1, 2).is_none());
    }

    #[test]
    fn seqlock_sequence_is_even_and_monotonic_after_successful_saves() {
        let entry = TTEntry::default();
        let board = make_board(1, 2);

        entry.save(
            &board,
            raw_entry_data(10, sq(1), Bound::Lower, 5, Selectivity::Level1, 0, false),
        );
        let first = entry.snapshot_seq();

        entry.save(
            &board,
            raw_entry_data(20, sq(2), Bound::Exact, 6, Selectivity::Level1, 0, false),
        );
        let second = entry.snapshot_seq();

        assert_eq!(first & 1, 0);
        assert_eq!(second & 1, 0);
        assert!(second > first);
    }

    #[test]
    fn seqlock_write_rejects_a_stale_snapshot() {
        let entry = TTEntry::default();
        let first_board = make_board(1, 2);
        let second_board = make_board(3, 4);

        entry.save(
            &first_board,
            raw_entry_data(10, sq(1), Bound::Exact, 5, Selectivity::Level1, 0, false),
        );
        let stale_seq = entry.snapshot_seq();
        entry.save(
            &second_board,
            raw_entry_data(20, sq(2), Bound::Exact, 6, Selectivity::Level3, 1, false),
        );

        assert!(!entry.seqlock_write(
            stale_seq,
            first_board.player().bits(),
            first_board.opponent().bits(),
            raw_entry_data(30, sq(3), Bound::Exact, 7, Selectivity::Level3, 2, false),
        ));

        assert!(read(&entry, &first_board).is_none());
        let data = stable_data(&entry, &second_board);
        assert_eq!(data.score().value(), 20);
        assert_eq!(data.depth(), 6);
        assert_eq!(data.best_move(), sq(2));
        assert_eq!(data.generation(), 1);
    }
}

mod transposition_table {
    use super::*;

    #[test]
    fn new_nonzero_table_allocates_the_requested_whole_mebibytes() {
        let tt = TranspositionTable::new(1);
        let expected_clusters = (1024 * 1024) / CLUSTER_BYTE_SIZE;

        assert_eq!(tt.cluster_count, expected_clusters as u64);
        assert_eq!(tt.entries.len(), expected_clusters * CLUSTER_SIZE);
        assert_eq!(tt.mb_size(), 1);
        assert_eq!(TranspositionTable::new(0).mb_size(), 0);
        assert_eq!(TranspositionTable::new(2).mb_size(), 2);
    }

    #[test]
    fn get_cluster_idx_maps_keys_across_aligned_clusters() {
        let tt = TranspositionTable::new(1);
        let clusters = tt.cluster_count as usize;

        assert_eq!(tt.get_cluster_idx(0), 0);
        assert_eq!(
            tt.get_cluster_idx(1u64 << 63),
            (clusters / 2) * CLUSTER_SIZE
        );
        assert_eq!(tt.get_cluster_idx(u64::MAX), (clusters - 1) * CLUSTER_SIZE);
    }

    #[test]
    fn generation_wraps_at_seven_bits_and_can_be_reset() {
        let tt = TranspositionTable::new(0);

        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);
        assert_eq!(tt.increment_generation(), 0);
        assert_eq!(tt.increment_generation(), 1);

        tt.reset_generation();
        assert_eq!(tt.generation(), 0);
    }

    #[test]
    fn probe_store_lookup_and_probe_result_accessors_report_the_stored_entry() {
        let tt = TranspositionTable::new(1);
        let board = make_board(START_PLAYER, START_OPPONENT);
        let key = board.hash();
        let generation = tt.increment_generation();

        let miss = tt.probe(&board, key);
        assert!(!is_hit(&miss));
        assert_eq!(miss.data(), None);
        assert_eq!(miss.best_move(), None);

        tt.store(
            miss.slot(),
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            true,
            false,
        );

        let lookup = tt
            .lookup(&board, key)
            .expect("lookup should hit stored board");
        assert_eq!(lookup.score().value(), 100);
        assert_eq!(lookup.bound(), Bound::Exact);
        assert_eq!(lookup.depth(), 20);
        assert_eq!(lookup.best_move(), sq(10));
        assert_eq!(lookup.selectivity(), Selectivity::Level3);
        assert_eq!(lookup.generation(), generation);
        assert!(lookup.is_endgame());

        let hit = tt.probe(&board, key);
        assert!(is_hit(&hit));
        assert_eq!(slot_index(&tt, hit.slot()), slot_index(&tt, miss.slot()));
        assert_eq!(hit.data(), Some(lookup));
        assert_eq!(hit.best_move(), sq(10));
    }

    #[test]
    fn lookup_misses_for_a_different_board_in_the_same_cluster() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            slot_at(&tt, fixture.cluster_idx),
            &fixture.stored,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            false,
            false,
        );

        assert!(tt.lookup(&fixture.stored, fixture.stored.hash()).is_some());
        assert!(
            tt.lookup(&fixture.colliding, fixture.colliding.hash())
                .is_none()
        );

        tt.store(
            slot_at(&tt, fixture.cluster_idx + 1),
            &fixture.colliding,
            raw_score(50),
            Bound::Lower,
            12,
            sq(11),
            Selectivity::Level2,
            false,
            false,
        );

        let slot1 = tt
            .lookup(&fixture.colliding, fixture.colliding.hash())
            .expect("lookup should also hit the second slot in the cluster");
        assert_eq!(slot1.score().value(), 50);
        assert_eq!(slot1.depth(), 12);
    }

    #[test]
    fn probe_selects_the_lowest_replacement_score_when_both_slots_are_occupied_misses() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            slot_at(&tt, fixture.cluster_idx),
            &fixture.stored,
            raw_score(10),
            Bound::Lower,
            4,
            sq(1),
            Selectivity::Level1,
            false,
            false,
        );
        tt.store(
            slot_at(&tt, fixture.cluster_idx + 1),
            &fixture.colliding,
            raw_score(20),
            Bound::Lower,
            20,
            sq(2),
            Selectivity::Level1,
            false,
            false,
        );
        for _ in 0..10 {
            tt.increment_generation();
        }

        let probe = tt.probe(&fixture.miss, fixture.miss.hash());

        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx);

        // When slot 1 has a strictly lower replacement score, slot 1 is chosen.
        tt.store(
            slot_at(&tt, fixture.cluster_idx),
            &fixture.stored,
            raw_score(10),
            Bound::Exact,
            20,
            sq(1),
            Selectivity::Level1,
            false,
            false,
        );
        tt.store(
            slot_at(&tt, fixture.cluster_idx + 1),
            &fixture.colliding,
            raw_score(20),
            Bound::Exact,
            4,
            sq(2),
            Selectivity::Level1,
            false,
            false,
        );
        let probe = tt.probe(&fixture.miss, fixture.miss.hash());
        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx + 1);

        // When both slots tie on replacement score, slot 0 is chosen.
        tt.store(
            slot_at(&tt, fixture.cluster_idx + 1),
            &fixture.colliding,
            raw_score(20),
            Bound::Exact,
            20,
            sq(2),
            Selectivity::Level1,
            false,
            false,
        );
        let probe = tt.probe(&fixture.miss, fixture.miss.hash());
        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx);
    }

    #[test]
    fn probe_ranks_occupied_slots_across_generation_wrap() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));
        let store = |offset, board: &Board, depth| {
            tt.store(
                slot_at(&tt, fixture.cluster_idx + offset),
                board,
                Score::ZERO,
                Bound::Exact,
                depth,
                None,
                Selectivity::None,
                false,
                false,
            );
        };
        for _ in 0..127 {
            tt.increment_generation();
        }
        store(1, &fixture.colliding, 24);
        assert_eq!(tt.increment_generation(), 0);

        for (fresh_depth, expected_slot) in [(15, 0), (16, 0), (17, 1)] {
            store(0, &fixture.stored, fresh_depth);
            let probe = tt.probe(&fixture.miss, fixture.miss.hash());
            assert!(!is_hit(&probe));
            assert_eq!(
                slot_index(&tt, probe.slot()),
                fixture.cluster_idx + expected_slot
            );
        }
    }

    #[test]
    fn probe_prefers_unused_slot_after_generation_wrap() {
        let tt = TranspositionTable::new(0);

        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);

        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            slot_at(&tt, fixture.cluster_idx),
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
            false,
        );
        assert_eq!(
            stable_data(&tt.entries[fixture.cluster_idx], &fixture.stored).generation(),
            127
        );
        assert!(
            !tt.entries[fixture.cluster_idx + 1]
                .snapshot_data()
                .is_occupied()
        );

        tt.increment_generation();

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx + 1);
    }

    #[test]
    fn probe_does_not_treat_an_in_flight_slot_as_unused_after_generation_wrap() {
        let tt = TranspositionTable::new(0);

        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);

        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));
        tt.store(
            slot_at(&tt, fixture.cluster_idx),
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
            false,
        );

        mark_slot_in_flight(&tt, fixture.cluster_idx + 1);
        tt.increment_generation();

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx);
    }

    #[test]
    fn probe_uses_the_second_slot_for_hits_and_misses_when_first_slot_stays_busy() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            slot_at(&tt, fixture.cluster_idx + 1),
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
            false,
        );

        mark_slot_in_flight(&tt, fixture.cluster_idx);

        let hit = tt.probe(&fixture.stored, fixture.stored.hash());
        let expected = stable_data(&tt.entries[fixture.cluster_idx + 1], &fixture.stored);
        assert_eq!(hit.data(), Some(expected));
        assert_eq!(slot_index(&tt, hit.slot()), fixture.cluster_idx + 1);

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx + 1);
    }

    #[test]
    fn probe_falls_back_to_cluster_head_when_all_slots_are_in_flight() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        for i in 0..CLUSTER_SIZE {
            mark_slot_in_flight(&tt, fixture.cluster_idx + i);
        }

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(slot_index(&tt, probe.slot()), fixture.cluster_idx);
    }

    #[test]
    fn clear_removes_entries_without_resetting_generation() {
        let clears: [fn(&mut TranspositionTable); 2] =
            [TranspositionTable::clear, |tt| tt.clear_shared()];
        for clear in clears {
            let mut tt = TranspositionTable::new(0);
            let generation = tt.increment_generation();
            let board = make_board(START_PLAYER, START_OPPONENT);
            let key = board.hash();
            let slot = tt.probe(&board, key).slot();

            tt.store(
                slot,
                &board,
                raw_score(100),
                Bound::Exact,
                20,
                sq(10),
                Selectivity::Level1,
                false,
                false,
            );
            assert!(tt.lookup(&board, key).is_some());
            assert!(tt.usage_rate() > 0.0);

            clear(&mut tt);

            assert!(tt.lookup(&board, key).is_none());
            assert_eq!(tt.usage_rate(), 0.0);
            assert_eq!(tt.generation(), generation);
        }
    }

    #[test]
    fn usage_rate_counts_occupied_slots_in_the_sampled_clusters() {
        let tt = TranspositionTable::new(0);
        let board = make_board(START_PLAYER, START_OPPONENT);
        let slot = tt.probe(&board, board.hash()).slot();

        tt.store(
            slot,
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level1,
            false,
            false,
        );

        assert_eq!(tt.usage_rate(), 1.0 / (16.0 * CLUSTER_SIZE as f64));

        // For tables larger than 1024 clusters (step > 1), verify non-zero sample stride.
        let large_tt = TranspositionTable::new(1);
        let step = large_tt.cluster_count as usize / 1024;
        assert!(step > 1);
        let sampled_slot = slot_at(&large_tt, step * CLUSTER_SIZE);
        large_tt.store(
            sampled_slot,
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level1,
            false,
            false,
        );
        assert_eq!(large_tt.usage_rate(), 1.0 / (1024.0 * CLUSTER_SIZE as f64));
    }

    #[test]
    fn penalize_lowers_depth_through_table_slot() {
        let tt = TranspositionTable::new(0);
        let board = make_board(START_PLAYER, START_OPPONENT);
        let key = board.hash();
        let slot = tt.probe(&board, key).slot();

        tt.store(
            slot,
            &board,
            raw_score(40),
            Bound::Lower,
            10,
            sq(3),
            Selectivity::Level2,
            false,
            false,
        );
        let stored = tt.lookup(&board, key).expect("stored entry should exist");
        tt.penalize(slot, &board, stored);

        let penalized = tt
            .lookup(&board, key)
            .expect("penalized entry should exist");
        assert_eq!(penalized.depth(), 9);
    }

    #[test]
    fn duplicate_copies_entries_and_generation_then_stays_independent() {
        let tt = TranspositionTable::new(0);
        let generation = tt.increment_generation();
        let board = make_board(START_PLAYER, START_OPPONENT);
        let slot = tt.probe(&board, board.hash()).slot();
        tt.store(
            slot,
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::None,
            true,
            false,
        );

        // SAFETY: no other thread touches `tt`.
        let copy = unsafe { tt.duplicate() };
        assert_eq!(copy.generation(), generation);
        assert_eq!(
            copy.lookup(&board, board.hash()),
            tt.lookup(&board, board.hash())
        );

        let other = make_board(START_OPPONENT, START_PLAYER);
        let slot = copy.probe(&other, other.hash()).slot();
        copy.store(
            slot,
            &other,
            raw_score(5),
            Bound::Exact,
            1,
            sq(1),
            Selectivity::None,
            false,
            false,
        );
        assert!(copy.lookup(&other, other.hash()).is_some());
        assert!(tt.lookup(&other, other.hash()).is_none());
    }
}
