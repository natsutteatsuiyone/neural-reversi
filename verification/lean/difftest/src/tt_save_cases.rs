use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::{fmt, mem};

use flip_scalar_difftest::opt_or_dash;

use crate::board::Board;
use crate::constants::CACHE_LINE_SIZE;
use crate::engine::node_type::NodeType;
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::square::Square;
use crate::types::Depth;

include!(concat!(env!("OUT_DIR"), "/tt_slot_source.rs"));

fn read_metadata(data: TTEntryData) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}",
        data.score().value(),
        data.bound() as u8,
        data.depth(),
        data.selectivity().as_u8(),
        data.generation(),
        u8::from(data.is_endgame()),
        u8::from(data.is_pv()),
        data.best_move().map_or(64, Square::index),
    )
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let endian = u8::from(cfg!(target_endian = "big"));
    let initial = Board::new();
    let boards = [
        initial,
        initial.switch_players(),
        Board::from_bitboards(0, 0),
    ];
    let old_metadata = [
        TTEntryData::from_u64(0),
        TTEntryData::new(
            Score::ZERO,
            Some(Square::H8),
            Bound::None,
            10,
            Selectivity::Level2,
            8,
            false,
            true,
        ),
        TTEntryData::new(
            Score::from_raw(-64),
            Some(Square::A1),
            Bound::Exact,
            10,
            Selectivity::Level2,
            8,
            true,
            true,
        ),
        TTEntryData::new(
            Score::from_raw(64),
            Some(Square::H8),
            Bound::Lower,
            255,
            Selectivity::None,
            8,
            false,
            false,
        ),
    ];
    let updates = [
        (Bound::Exact, 1, Selectivity::Level1, 8, None),
        (Bound::Upper, 7, Selectivity::Level2, 8, None),
        (Bound::Lower, 8, Selectivity::Level2, 8, None),
        (Bound::Upper, 7, Selectivity::Level3, 8, None),
        (Bound::Upper, 5, Selectivity::Level3, 8, None),
        (Bound::Lower, 1, Selectivity::Level1, 9, None),
        (Bound::Lower, 255, Selectivity::None, 8, Some(Square::H8)),
    ];
    for stored in boards {
        for query in boards {
            let old_player = stored.player().bits();
            let old_opponent = stored.opponent().bits();
            let new_player = query.player().bits();
            let new_opponent = query.opponent().bits();
            for old in old_metadata {
                let old_raw = old.to_u64();
                for (bound, depth, selectivity, generation, best_move) in updates {
                    let new = TTEntryData::new(
                        Score::from_raw(128),
                        best_move,
                        bound,
                        depth,
                        selectivity,
                        generation,
                        true,
                        false,
                    );
                    let new_raw = new.to_u64();
                    let access = format!("{}|{}", read_metadata(old), read_metadata(new));
                    for seq in [0, 2, 3, u64::MAX - 1] {
                        let entry = TTEntry {
                            seq: AtomicU64::new(seq),
                            player: AtomicU64::new(old_player),
                            opponent: AtomicU64::new(old_opponent),
                            data: AtomicU64::new(old_raw),
                        };
                        let snapshot = opt_or_dash(entry.try_load_snapshot().map(
                            |(player, opponent, sequence, data)| {
                                format!("{player}:{opponent}:{sequence}:{}", data.to_u64())
                            },
                        ));
                        let lookup = opt_or_dash(
                            entry
                                .read_for_lookup(new_player, new_opponent)
                                .map(TTEntryData::to_u64),
                        );
                        entry.save(&query, new);
                        writeln!(
                            out,
                            "{endian} {seq} {old_player} {old_opponent} {old_raw} {new_player} {new_opponent} {new_raw} {access} {snapshot} {lookup} {} {} {} {}",
                            entry.seq.load(Ordering::Relaxed),
                            entry.player.load(Ordering::Relaxed),
                            entry.opponent.load(Ordering::Relaxed),
                            entry.data.load(Ordering::Relaxed),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}
