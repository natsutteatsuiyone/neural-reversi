use std::io::{self, Write};

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::move_list::MoveList;

fn write_case(
    out: &mut impl Write,
    board: &Board,
    mask: u64,
    mode: &str,
    moves: MoveList,
) -> io::Result<()> {
    let square = |sq: Option<crate::square::Square>| sq.map_or(64, |sq| sq.index());
    write!(
        out,
        "{} {} {mask} {mode} {} {} {}",
        board.player().bits(),
        board.opponent().bits(),
        moves.count(),
        square(moves.first().map(|mv| mv.sq)),
        square(moves.wipeout_move()),
    )?;
    for mv in moves.iter() {
        write!(out, " {}:{}:{}", mv.sq.index(), mv.flipped.bits(), mv.value)?;
    }
    writeln!(out)
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let backend = cfg_select! {
        all(
            target_arch = "x86_64",
            target_feature = "avx512cd",
            target_feature = "avx512vl"
        ) => "avx512",
        all(target_arch = "x86_64", target_feature = "avx2") => "avx2",
        all(target_arch = "aarch64", target_feature = "neon") => "neon",
        _ => "portable",
    };
    writeln!(out, "backend {backend}")?;

    let boards = [
        Board::from_bitboards(0, 0),
        Board::from_bitboards(2, 1),
        Board::from_bitboards(1, 2),
        Board::from_bitboards((1 << 10) | (1 << 17) | (1 << 18), 1 << 9),
        Board::new(),
        Board::from_bitboards(0x0004_3e24_8068_0404, 0x006a_400a_4400_7a02),
        Board::from_bitboards(0x0081_6600_041c_1020, 0x0066_804a_1a22_6a00),
    ];
    for board in boards.into_iter().chain((0u64..256).map(|seed| {
        let player = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let opponent = seed.wrapping_mul(0xd1b5_4a32_d192_ed03) & !player;
        Board::from_bitboards(player, opponent)
    })) {
        write_case(out, &board, u64::MAX, "new", MoveList::new(&board))?;
        let legal = board.legal_moves();
        for mask in [
            0,
            u64::MAX,
            0x55aa_55aa_55aa_55aa,
            legal.bits().wrapping_neg() & legal.bits(),
        ] {
            let selected = legal & Bitboard::new(mask);
            write_case(
                out,
                &board,
                mask,
                "subset",
                MoveList::with_moves(&board, selected),
            )?;
            if let Some(nonempty) = selected.non_empty() {
                write_case(
                    out,
                    &board,
                    mask,
                    "nonempty",
                    MoveList::with_at_least_one_move(&board, nonempty),
                )?;
            }
        }
    }
    Ok(())
}
