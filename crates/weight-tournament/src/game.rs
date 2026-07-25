//! One-ply games between two main-network weight files.

use std::path::Path;

use anyhow::{Context, Result, bail};
use reversi_core::board::Board;
use reversi_core::constants::INITIAL_EMPTY_COUNT;
use reversi_core::disc::Disc;
use reversi_core::eval::Network;
use reversi_core::eval::pattern_feature::PatternFeatures;
use reversi_core::move_list::MoveList;
use reversi_core::square::Square;
use reversi_core::types::ScaledScore;

use crate::tournament::Weight;

struct OnePlyEngine {
    network: Network,
}

impl OnePlyEngine {
    fn from_path(path: &Path) -> Result<Self> {
        let network = Network::new(path)
            .with_context(|| format!("failed to load main-network weight {}", path.display()))?;
        Ok(Self { network })
    }

    fn select_move(&self, board: &Board) -> Option<Square> {
        let moves = MoveList::new(board);
        let mut best_move = None;
        let mut best_score = -ScaledScore::INF;

        for mv in moves.iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            let score = -self.evaluate_position(&next);

            if score > best_score {
                best_score = score;
                best_move = Some(mv.sq);
            }
        }

        best_move
    }

    fn evaluate_position(&self, board: &Board) -> ScaledScore {
        if board.is_game_over() {
            return ScaledScore::from_disc_diff(board.solve(board.get_empty_count()));
        }

        if !board.has_legal_moves() {
            return -self.evaluate_position(&board.switch_players());
        }

        let ply = INITIAL_EMPTY_COUNT - board.get_empty_count() as usize;
        let pattern_features = PatternFeatures::new(board, ply);
        self.network
            .evaluate(board, pattern_features.p_feature(ply), ply)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MatchResult {
    pub(crate) games: usize,
    pub(crate) engine1_wins: usize,
    pub(crate) engine2_wins: usize,
    pub(crate) draws: usize,
    pub(crate) engine1_score: i64,
}

impl MatchResult {
    pub(crate) fn winner(self) -> MatchWinner {
        match self.engine1_wins.cmp(&self.engine2_wins) {
            std::cmp::Ordering::Greater => MatchWinner::Engine1,
            std::cmp::Ordering::Less => MatchWinner::Engine2,
            std::cmp::Ordering::Equal => match self.engine1_score.cmp(&0) {
                std::cmp::Ordering::Greater => MatchWinner::Engine1,
                std::cmp::Ordering::Less => MatchWinner::Engine2,
                std::cmp::Ordering::Equal => MatchWinner::Draw,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatchWinner {
    Engine1,
    Engine2,
    Draw,
}

pub(crate) fn play_match(
    engine1: &Weight,
    engine2: &Weight,
    openings: &[String],
    mut on_game_completed: impl FnMut(),
) -> Result<MatchResult> {
    let engine1_player = OnePlyEngine::from_path(&engine1.path)?;
    let engine2_player = OnePlyEngine::from_path(&engine2.path)?;
    let mut result = MatchResult::default();

    for opening in openings {
        for engine1_is_black in [true, false] {
            let engine1_score =
                play_game(&engine1_player, &engine2_player, engine1_is_black, opening)
                    .with_context(|| {
                        format!(
                            "failed to play {} vs {} from opening '{opening}'",
                            engine1.name, engine2.name
                        )
                    })?;
            on_game_completed();

            result.games += 1;
            result.engine1_score += i64::from(engine1_score);
            match engine1_score.cmp(&0) {
                std::cmp::Ordering::Greater => result.engine1_wins += 1,
                std::cmp::Ordering::Less => result.engine2_wins += 1,
                std::cmp::Ordering::Equal => result.draws += 1,
            }
        }
    }

    Ok(result)
}

fn play_game(
    engine1: &OnePlyEngine,
    engine2: &OnePlyEngine,
    engine1_is_black: bool,
    opening: &str,
) -> Result<i32> {
    let (mut board, mut side_to_move) = apply_opening(opening)?;

    while !board.is_game_over() {
        if !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
            continue;
        }

        let engine = if (side_to_move == Disc::Black) == engine1_is_black {
            engine1
        } else {
            engine2
        };
        let sq = engine
            .select_move(&board)
            .context("one-ply evaluator found no move in a position with legal moves")?;
        board = board.make_move(sq);
        side_to_move = side_to_move.opposite();
    }

    let black_score = score_from_black_perspective(&board, side_to_move);
    Ok(if engine1_is_black {
        black_score
    } else {
        -black_score
    })
}

fn apply_opening(opening: &str) -> Result<(Board, Disc)> {
    let opening = opening.trim();
    if !opening.len().is_multiple_of(2) {
        bail!("opening sequence has odd length: '{opening}'");
    }

    let mut board = Board::new();
    let mut side_to_move = Disc::Black;

    let moves = Square::parse_sequence(opening)
        .map_err(|e| anyhow::anyhow!("invalid move in opening sequence: {e}"))?;
    for sq in moves {
        if !board.is_legal_move(sq) {
            bail!("illegal opening move {sq} for {side_to_move:?}");
        }

        board = board.make_move(sq);
        side_to_move = side_to_move.opposite();

        if !board.is_game_over() && !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
        }
    }

    Ok((board, side_to_move))
}

fn score_from_black_perspective(board: &Board, side_to_move: Disc) -> i32 {
    let score = board.solve(board.get_empty_count());
    if side_to_move == Disc::Black {
        score
    } else {
        -score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_parser_accepts_compact_match_runner_format() {
        let (board, side_to_move) = apply_opening("f5d6c4d3").expect("opening should be legal");

        assert_eq!(board.get_player_count() + board.get_opponent_count(), 8);
        assert_eq!(side_to_move, Disc::Black);
    }

    #[test]
    fn opening_parser_rejects_odd_length() {
        let err = apply_opening("f5d").expect_err("opening should be rejected");

        assert!(err.to_string().contains("odd length"));
    }

    #[test]
    fn opening_parser_rejects_illegal_move() {
        let err = apply_opening("a1").expect_err("opening should be rejected");

        assert!(err.to_string().contains("illegal opening move"));
    }

    #[test]
    fn match_winner_uses_disc_score_as_tiebreak() {
        let result = MatchResult {
            games: 2,
            engine1_wins: 1,
            engine2_wins: 1,
            draws: 0,
            engine1_score: 4,
        };

        assert_eq!(result.winner(), MatchWinner::Engine1);
    }
}
