use rand::{RngExt, seq::SliceRandom};
use reversi_core::{board::Board, disc::Disc, game_state::GameState, square::Square};

pub(crate) enum Opening {
    Sequence(String),
    Position(GameState),
}

impl Opening {
    pub(crate) fn description(&self) -> String {
        match self {
            Self::Sequence(moves) => moves.clone(),
            Self::Position(state) => format!("{} {}", board_string(state), color(state)),
        }
    }
}

pub(crate) fn color(state: &GameState) -> &'static str {
    if state.side_to_move() == Disc::Black {
        "B"
    } else {
        "W"
    }
}

pub(crate) fn board_string(state: &GameState) -> String {
    Square::iter()
        .map(|sq| {
            state
                .board()
                .get_disc_at(sq, state.side_to_move())
                .to_char()
        })
        .collect()
}

fn allowed(sq: Square) -> bool {
    let index = sq.index();
    !((index % 8 < 2 || index % 8 >= 6) && (index / 8 < 2 || index / 8 >= 6))
}

fn scattered(discs: u8, rng: &mut impl RngExt) -> GameState {
    let mut squares = Vec::new();
    for ring in 0..4 {
        let mut members: Vec<_> = Square::iter()
            .filter(|&sq| {
                let index = sq.index() as i32;
                allowed(sq)
                    && ((2 * (index % 8) - 7).abs().max((2 * (index / 8) - 7).abs()) - 1) / 2
                        == ring
            })
            .collect();
        members.shuffle(rng);
        squares.extend(members);
    }
    let squares = &mut squares[..discs as usize];
    squares.shuffle(rng);
    let mut whites = discs / 2;
    if discs % 2 == 1 && rng.random_bool(0.5) {
        whites += 1;
    }
    let imbalance = whites / 3;
    whites = (i16::from(whites) + rng.random_range(0..=2 * imbalance) as i16 - i16::from(imbalance))
        as u8;
    let mut colors: Vec<_> = (0..discs)
        .map(|n| if n < whites { Disc::White } else { Disc::Black })
        .collect();
    colors.shuffle(rng);
    let mut text = ['-'; 64];
    for (&sq, disc) in squares.iter().zip(colors) {
        text[sq.index()] = disc.to_char();
    }
    let side = if discs.is_multiple_of(2) {
        Disc::Black
    } else {
        Disc::White
    };
    let board = Board::from_string(&text.iter().collect::<String>(), side).unwrap();
    GameState::from_board(board, side)
}

pub(crate) fn random_position(discs: u8, rng: &mut impl RngExt) -> GameState {
    if discs >= 10 && rng.random_bool(0.5) {
        for _ in 0..5 {
            let initial = scattered(5, rng);
            let mut board = *initial.board();
            let mut side = initial.side_to_move();
            let mut completed = true;
            for _ in 5..discs {
                let moves: Vec<_> = Square::iter()
                    .filter(|&sq| allowed(sq) && board.is_legal_move(sq))
                    .collect();
                if moves.is_empty() {
                    completed = false;
                    break;
                }
                board = board.make_move(moves[rng.random_range(0..moves.len())]);
                side = side.opposite();
            }
            let minimum = u32::from(discs / 4).max(3);
            if completed
                && board.get_player_count() > minimum
                && board.get_opponent_count() > minimum
            {
                return GameState::from_board(board, side);
            }
        }
    }
    scattered(discs, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn random_positions_preserve_server_invariants_and_seed() {
        let mut one = StdRng::seed_from_u64(42);
        let mut two = StdRng::seed_from_u64(42);
        for discs in 4..=48 {
            for _ in 0..20 {
                let state = random_position(discs, &mut one);
                assert_eq!(
                    board_string(&state),
                    board_string(&random_position(discs, &mut two))
                );
                assert_eq!(
                    state.board().get_player_count() + state.board().get_opponent_count(),
                    u32::from(discs)
                );
                assert_eq!(
                    state.side_to_move(),
                    if discs.is_multiple_of(2) {
                        Disc::Black
                    } else {
                        Disc::White
                    }
                );
                for sq in Square::iter().filter(|&sq| !allowed(sq)) {
                    assert_eq!(
                        state.board().get_disc_at(sq, state.side_to_move()),
                        Disc::Empty
                    );
                }
            }
        }
    }
}
