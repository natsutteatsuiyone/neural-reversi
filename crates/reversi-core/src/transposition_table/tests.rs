use super::*;
use crate::search::node_type::{NonPV, PV};

const START_PLAYER: u64 = 0x0000_0008_1000_0000;
const START_OPPONENT: u64 = 0x0000_0010_0800_0000;
const CLUSTER_FIXTURE_SEARCH_LIMIT: u64 = 1_000_000;

fn raw_score(raw_value: i32) -> ScaledScore {
    ScaledScore::from_raw(raw_value)
}

fn sq(idx: usize) -> Square {
    Square::from_usize(idx).unwrap()
}

fn make_board(player: u64, opponent: u64) -> Board {
    Board::from_bitboards(player, opponent)
}

fn raw_entry_data(
    raw_value: i32,
    best_move: Square,
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
    )
}

fn read(entry: &TTEntry, board: &Board) -> Option<TTEntryData> {
    entry.read_for_lookup(board.player().bits(), board.opponent().bits())
}

fn stable_data(entry: &TTEntry, board: &Board) -> TTEntryData {
    read(entry, board).expect("entry should match board")
}

fn is_hit(probe: &TTProbeResult) -> bool {
    matches!(probe, TTProbeResult::Hit { .. })
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
    fn discriminants_encode_cutoff_bit_masks() {
        assert_eq!(Bound::None as u8, 0b00);
        assert_eq!(Bound::Lower as u8, 0b01);
        assert_eq!(Bound::Upper as u8, 0b10);
        assert_eq!(Bound::Exact as u8, 0b11);

        assert_ne!((Bound::Lower as u8) & (Bound::Lower as u8), 0);
        assert_ne!((Bound::Exact as u8) & (Bound::Lower as u8), 0);
        assert_eq!((Bound::Upper as u8) & (Bound::Lower as u8), 0);

        assert_ne!((Bound::Upper as u8) & (Bound::Upper as u8), 0);
        assert_ne!((Bound::Exact as u8) & (Bound::Upper as u8), 0);
        assert_eq!((Bound::Lower as u8) & (Bound::Upper as u8), 0);
    }

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
    fn layout_stays_exactly_one_atomic_word() {
        assert_eq!(mem::size_of::<TTEntryData>(), mem::size_of::<u64>());
        assert_eq!(mem::size_of::<TTEntry>(), 32);
        assert_eq!(mem::align_of::<TTEntry>(), 32);
        assert_eq!(mem::size_of::<TTEntry>() * CLUSTER_SIZE, CACHE_LINE_SIZE);
    }

    #[test]
    fn packing_roundtrip_preserves_every_field() {
        let original = raw_entry_data(
            ScaledScore::MIN.value(),
            sq(63),
            Bound::Exact,
            60,
            Selectivity::None,
            127,
            true,
        );

        let unpacked = TTEntryData::from_u64(original.to_u64());

        assert_eq!(unpacked.score(), ScaledScore::MIN);
        assert_eq!(unpacked.best_move(), sq(63));
        assert_eq!(unpacked.bound(), Bound::Exact);
        assert_eq!(unpacked.depth(), 60);
        assert_eq!(unpacked.selectivity(), Selectivity::None);
        assert_eq!(unpacked.generation(), 127);
        assert!(unpacked.is_endgame());
    }

    #[test]
    fn generation_is_masked_without_corrupting_endgame_flag() {
        let midgame = raw_entry_data(50, sq(5), Bound::Lower, 10, Selectivity::Level1, 255, false);
        let endgame = raw_entry_data(50, sq(5), Bound::Lower, 10, Selectivity::Level1, 255, true);

        assert_eq!(midgame.generation(), 127);
        assert!(!midgame.is_endgame());
        assert_eq!(endgame.generation(), 127);
        assert!(endgame.is_endgame());
    }

    #[test]
    fn zero_word_decodes_to_an_unoccupied_empty_entry() {
        let data = TTEntryData::from_u64(0);

        assert!(!data.is_occupied());
        assert_eq!(data.bound(), Bound::None);
        assert_eq!(data.score(), ScaledScore::ZERO);
        assert_eq!(data.best_move(), Square::A1);
        assert_eq!(data.depth(), 0);
        assert_eq!(data.selectivity(), Selectivity::Level1);
        assert_eq!(data.generation(), 0);
        assert!(!data.is_endgame());
    }

    #[test]
    fn with_best_move_replaces_only_the_move_field() {
        let original = raw_entry_data(25, sq(3), Bound::Upper, 9, Selectivity::Level2, 17, true);

        let updated = original.with_best_move(sq(44));

        assert_eq!(updated.score(), original.score());
        assert_eq!(updated.bound(), original.bound());
        assert_eq!(updated.depth(), original.depth());
        assert_eq!(updated.selectivity(), original.selectivity());
        assert_eq!(updated.generation(), original.generation());
        assert_eq!(updated.is_endgame(), original.is_endgame());
        assert_eq!(updated.best_move(), sq(44));
    }

    #[test]
    fn can_cut_uses_lower_or_exact_when_score_reaches_beta() {
        let lower = raw_entry_data(
            100,
            Square::None,
            Bound::Lower,
            6,
            Selectivity::Level2,
            0,
            false,
        );
        let exact = raw_entry_data(
            100,
            Square::None,
            Bound::Exact,
            6,
            Selectivity::Level2,
            0,
            false,
        );
        let upper = raw_entry_data(
            100,
            Square::None,
            Bound::Upper,
            6,
            Selectivity::Level2,
            0,
            false,
        );

        let lower_at_beta = raw_entry_data(
            50,
            Square::None,
            Bound::Lower,
            6,
            Selectivity::Level2,
            0,
            false,
        );

        assert!(lower_at_beta.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(lower.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(exact.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(!upper.can_cut(raw_score(50), 6, Selectivity::Level2, false));
    }

    #[test]
    fn can_cut_uses_upper_or_exact_when_score_is_below_beta() {
        let lower = raw_entry_data(
            30,
            Square::None,
            Bound::Lower,
            6,
            Selectivity::Level2,
            0,
            false,
        );
        let exact = raw_entry_data(
            30,
            Square::None,
            Bound::Exact,
            6,
            Selectivity::Level2,
            0,
            false,
        );
        let upper = raw_entry_data(
            30,
            Square::None,
            Bound::Upper,
            6,
            Selectivity::Level2,
            0,
            false,
        );

        assert!(!lower.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(exact.can_cut(raw_score(50), 6, Selectivity::Level2, false));
        assert!(upper.can_cut(raw_score(50), 6, Selectivity::Level2, false));
    }

    #[test]
    fn can_cut_rejects_entries_that_are_too_shallow_too_selective_or_not_endgame() {
        let midgame = raw_entry_data(
            100,
            Square::None,
            Bound::Lower,
            6,
            Selectivity::Level2,
            0,
            false,
        );
        let endgame = raw_entry_data(
            100,
            Square::None,
            Bound::Lower,
            6,
            Selectivity::Level2,
            0,
            true,
        );

        assert!(!midgame.can_cut(raw_score(50), 7, Selectivity::Level2, false));
        assert!(!midgame.can_cut(raw_score(50), 6, Selectivity::Level3, false));
        assert!(!midgame.can_cut(raw_score(50), 6, Selectivity::Level2, true));
        assert!(endgame.can_cut(raw_score(50), 6, Selectivity::Level2, true));
    }

    #[test]
    fn relative_age_uses_the_seven_bit_generation_ring() {
        let data = raw_entry_data(
            0,
            Square::None,
            Bound::Exact,
            0,
            Selectivity::Level1,
            126,
            false,
        );

        assert_eq!(data.relative_age(126), 0);
        assert_eq!(data.relative_age(127), 1);
        assert_eq!(data.relative_age(2), 4);
    }

    #[test]
    fn replacement_score_penalizes_old_entries_by_age_weight() {
        let fresh = raw_entry_data(
            0,
            Square::None,
            Bound::Lower,
            8,
            Selectivity::Level1,
            5,
            false,
        );
        let old = raw_entry_data(
            0,
            Square::None,
            Bound::Lower,
            16,
            Selectivity::Level1,
            1,
            false,
        );

        assert_eq!(fresh.replacement_score(5), 8);
        assert_eq!(old.replacement_score(5), 16 - 4 * TTEntry::AGE_WEIGHT);
        assert!(old.replacement_score(5) < fresh.replacement_score(5));
    }
}

mod tt_entry {
    use super::*;

    #[test]
    fn save_and_read_preserves_board_and_metadata() {
        let entry = TTEntry::default();
        let board = make_board(START_PLAYER, START_OPPONENT);

        entry.save(
            &board,
            raw_entry_data(
                ScaledScore::MIN.value(),
                sq(3),
                Bound::Exact,
                60,
                Selectivity::Level3,
                127,
                true,
            ),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score(), ScaledScore::MIN);
        assert_eq!(data.bound(), Bound::Exact);
        assert_eq!(data.depth(), 60);
        assert_eq!(data.best_move(), sq(3));
        assert_eq!(data.selectivity(), Selectivity::Level3);
        assert_eq!(data.generation(), 127);
        assert!(data.is_endgame());
    }

    #[test]
    fn read_misses_when_board_bits_do_not_match() {
        let entry = TTEntry::default();
        let stored = make_board(1, 2);
        let different = make_board(3, 4);

        entry.save(
            &stored,
            raw_entry_data(50, sq(5), Bound::Exact, 10, Selectivity::Level3, 1, false),
        );

        assert!(read(&entry, &stored).is_some());
        assert!(read(&entry, &different).is_none());
    }

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
            raw_entry_data(60, sq(6), Bound::Upper, 2, Selectivity::Level3, 1, false),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 60);
        assert_eq!(data.selectivity(), Selectivity::Level3);
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
            raw_entry_data(
                40,
                Square::None,
                Bound::Exact,
                1,
                Selectivity::Level1,
                1,
                false,
            ),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 40);
        assert_eq!(data.bound(), Bound::Exact);
        assert_eq!(data.depth(), 1);
        assert_eq!(data.best_move(), Square::None);
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
            raw_entry_data(
                40,
                Square::None,
                Bound::Lower,
                11,
                Selectivity::Level3,
                1,
                false,
            ),
        );

        let data = stable_data(&entry, &board);
        assert_eq!(data.score().value(), 40);
        assert_eq!(data.best_move(), sq(11));
    }

    #[test]
    fn different_board_replaces_the_slot_even_with_weaker_metadata() {
        let entry = TTEntry::default();
        let old_board = make_board(0x0000_0000_0000_00ff, 0x0000_0000_0000_ff00);
        let new_board = make_board(0x0000_0000_00ff_0000, 0x0000_0000_ff00_0000);

        entry.save(
            &old_board,
            raw_entry_data(80, sq(8), Bound::Exact, 20, Selectivity::None, 10, true),
        );
        entry.save(
            &new_board,
            raw_entry_data(
                90,
                Square::None,
                Bound::Upper,
                1,
                Selectivity::Level1,
                0,
                false,
            ),
        );

        assert!(read(&entry, &old_board).is_none());
        let data = stable_data(&entry, &new_board);
        assert_eq!(data.score().value(), 90);
        assert_eq!(data.best_move(), Square::None);
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
    fn new_zero_mebibyte_table_uses_minimal_test_allocation() {
        let tt = TranspositionTable::new(0);

        assert_eq!(tt.cluster_count, 16);
        assert_eq!(tt.entries.len(), 16 * CLUSTER_SIZE);
        assert_eq!(tt.mb_size(), 0);
        assert_eq!(tt.usage_rate(), 0.0);
    }

    #[test]
    fn new_nonzero_table_allocates_the_requested_whole_mebibytes() {
        let tt = TranspositionTable::new(1);
        let expected_clusters = (1024 * 1024) / (mem::size_of::<TTEntry>() * CLUSTER_SIZE);

        assert_eq!(tt.cluster_count, expected_clusters as u64);
        assert_eq!(tt.entries.len(), expected_clusters * CLUSTER_SIZE);
        assert_eq!(tt.mb_size(), 1);
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
        assert_eq!(miss.best_move(), Square::None);

        tt.store(
            miss.index(),
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            true,
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
        assert_eq!(hit.index(), miss.index());
        assert_eq!(hit.data(), Some(lookup));
        assert_eq!(hit.best_move(), sq(10));
    }

    #[test]
    fn lookup_misses_for_a_different_board_in_the_same_cluster() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            fixture.cluster_idx,
            &fixture.stored,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            false,
        );

        assert!(tt.lookup(&fixture.stored, fixture.stored.hash()).is_some());
        assert!(
            tt.lookup(&fixture.colliding, fixture.colliding.hash())
                .is_none()
        );
    }

    #[test]
    fn probe_prefers_the_first_unused_slot_in_a_cluster() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            fixture.cluster_idx,
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(probe.index(), fixture.cluster_idx + 1);
    }

    #[test]
    fn probe_selects_the_lowest_replacement_score_when_both_slots_are_occupied_misses() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            fixture.cluster_idx,
            &fixture.stored,
            raw_score(10),
            Bound::Lower,
            4,
            sq(1),
            Selectivity::Level1,
            false,
        );
        tt.store(
            fixture.cluster_idx + 1,
            &fixture.colliding,
            raw_score(20),
            Bound::Lower,
            20,
            sq(2),
            Selectivity::Level1,
            false,
        );
        for _ in 0..10 {
            tt.increment_generation();
        }

        let probe = tt.probe(&fixture.miss, fixture.miss.hash());

        assert!(!is_hit(&probe));
        assert_eq!(probe.index(), fixture.cluster_idx);
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
            fixture.cluster_idx,
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
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
        assert_eq!(probe.index(), fixture.cluster_idx + 1);
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
            fixture.cluster_idx,
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        mark_slot_in_flight(&tt, fixture.cluster_idx + 1);
        tt.increment_generation();

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(probe.index(), fixture.cluster_idx);
    }

    #[test]
    fn probe_keeps_the_last_stable_victim_when_first_slot_stays_busy() {
        let tt = TranspositionTable::new(0);
        let fixture = ClusterFixture::new(&tt, make_board(START_PLAYER, START_OPPONENT));

        tt.store(
            fixture.cluster_idx + 1,
            &fixture.stored,
            raw_score(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        mark_slot_in_flight(&tt, fixture.cluster_idx);

        let probe = tt.probe(&fixture.colliding, fixture.colliding.hash());

        assert!(!is_hit(&probe));
        assert_eq!(probe.index(), fixture.cluster_idx + 1);
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
        assert_eq!(probe.index(), fixture.cluster_idx);
    }

    #[test]
    fn clear_removes_entries_without_resetting_generation() {
        let tt = TranspositionTable::new(0);
        let generation = tt.increment_generation();
        let board = make_board(START_PLAYER, START_OPPONENT);
        let key = board.hash();
        let idx = tt.probe(&board, key).index();

        tt.store(
            idx,
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level1,
            false,
        );
        assert!(tt.lookup(&board, key).is_some());
        assert!(tt.usage_rate() > 0.0);

        tt.clear();

        assert!(tt.lookup(&board, key).is_none());
        assert_eq!(tt.usage_rate(), 0.0);
        assert_eq!(tt.generation(), generation);
    }

    #[test]
    fn usage_rate_counts_occupied_slots_in_the_sampled_clusters() {
        let tt = TranspositionTable::new(0);
        let board = make_board(START_PLAYER, START_OPPONENT);
        let idx = tt.probe(&board, board.hash()).index();

        tt.store(
            idx,
            &board,
            raw_score(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level1,
            false,
        );

        assert_eq!(tt.usage_rate(), 1.0 / (16.0 * CLUSTER_SIZE as f64));
    }

    #[test]
    fn prefetch_accepts_any_key_without_changing_observable_state() {
        let tt = TranspositionTable::new(0);

        tt.prefetch(0);
        tt.prefetch(u64::MAX);

        assert_eq!(tt.usage_rate(), 0.0);
        assert_eq!(tt.generation(), 0);
    }
}
