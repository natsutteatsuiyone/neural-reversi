//! Simple in-process weight-file matches for the WebAssembly build.

use reversi_core::{
    board::Board, constants::INITIAL_EMPTY_COUNT, disc::Disc,
    eval::pattern_feature::PatternFeatures, move_list::MoveList, square::Square,
    types::ScaledScore,
};
use wasm_bindgen::prelude::*;

use crate::eval::Eval;

struct OnePlyEngine {
    eval: Eval,
}

impl OnePlyEngine {
    fn from_weights(weights: &[u8]) -> Result<Self, JsValue> {
        let eval = Eval::from_bytes(weights)
            .map_err(|err| JsValue::from_str(&format!("failed to load weights: {err}")))?;
        Ok(Self { eval })
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
        self.eval
            .evaluate_network(pattern_features.p_feature(ply), ply)
    }
}

/// Result of one completed game between the two loaded weight files.
#[wasm_bindgen]
pub struct WeightMatchGameResult {
    engine1_is_black: bool,
    winner: String,
    black_score: i32,
    engine1_score: i32,
    black_count: u32,
    white_count: u32,
    moves: String,
}

#[wasm_bindgen]
impl WeightMatchGameResult {
    /// Returns whether engine 1 played black in this game.
    #[wasm_bindgen(getter)]
    pub fn engine1_is_black(&self) -> bool {
        self.engine1_is_black
    }

    /// Returns `engine1`, `engine2`, or `draw`.
    #[wasm_bindgen(getter)]
    pub fn winner(&self) -> String {
        self.winner.clone()
    }

    /// Returns the final score from Black's perspective.
    #[wasm_bindgen(getter)]
    pub fn black_score(&self) -> i32 {
        self.black_score
    }

    /// Returns the final score from engine 1's perspective.
    #[wasm_bindgen(getter)]
    pub fn engine1_score(&self) -> i32 {
        self.engine1_score
    }

    /// Returns Black's final disc count before assigning remaining empties.
    #[wasm_bindgen(getter)]
    pub fn black_count(&self) -> u32 {
        self.black_count
    }

    /// Returns White's final disc count before assigning remaining empties.
    #[wasm_bindgen(getter)]
    pub fn white_count(&self) -> u32 {
        self.white_count
    }

    /// Returns the played moves separated by spaces. Passes are written as `pass`.
    #[wasm_bindgen(getter)]
    pub fn moves(&self) -> String {
        self.moves.clone()
    }
}

/// Runs simple one-ply games between two zstd-compressed wasm weight files.
#[wasm_bindgen]
pub struct WeightMatchRunner {
    engine1: OnePlyEngine,
    engine2: OnePlyEngine,
}

#[wasm_bindgen]
impl WeightMatchRunner {
    /// Creates a runner from two zstd-compressed wasm weight files.
    ///
    /// # Errors
    ///
    /// Returns an error if either weight file cannot be decompressed or parsed.
    #[wasm_bindgen(constructor)]
    pub fn new(engine1_weights: &[u8], engine2_weights: &[u8]) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        Ok(Self {
            engine1: OnePlyEngine::from_weights(engine1_weights)?,
            engine2: OnePlyEngine::from_weights(engine2_weights)?,
        })
    }

    /// Plays one game from an optional opening string.
    ///
    /// `opening` uses the same compact coordinate format as `match-runner`,
    /// for example `f5d6c4d3`. If `engine1_is_black` is false, the two engines
    /// are color-swapped.
    ///
    /// # Errors
    ///
    /// Returns an error if the opening string is malformed or contains an
    /// illegal move.
    pub fn play_game(
        &self,
        engine1_is_black: bool,
        opening: &str,
    ) -> Result<WeightMatchGameResult, JsValue> {
        let mut board = Board::new();
        let mut side_to_move = Disc::Black;
        let mut moves = Vec::new();

        apply_opening(opening, &mut board, &mut side_to_move, &mut moves)
            .map_err(|err| JsValue::from_str(&err))?;

        while !board.is_game_over() {
            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                moves.push("pass".to_string());
                continue;
            }

            let engine = self.engine_for_side(side_to_move, engine1_is_black);
            let sq = engine
                .select_move(&board)
                .ok_or_else(|| JsValue::from_str("no legal move found"))?;

            board = board.make_move(sq);
            side_to_move = side_to_move.opposite();
            moves.push(sq.to_string());
        }

        Ok(build_result(
            board,
            side_to_move,
            engine1_is_black,
            moves.join(" "),
        ))
    }
}

impl WeightMatchRunner {
    fn engine_for_side(&self, side_to_move: Disc, engine1_is_black: bool) -> &OnePlyEngine {
        if (side_to_move == Disc::Black) == engine1_is_black {
            &self.engine1
        } else {
            &self.engine2
        }
    }
}

fn apply_opening(
    opening: &str,
    board: &mut Board,
    side_to_move: &mut Disc,
    moves: &mut Vec<String>,
) -> Result<(), String> {
    let opening = opening.trim();
    if opening.is_empty() {
        return Ok(());
    }

    if !opening.len().is_multiple_of(2) {
        return Err(format!("opening sequence has odd length: '{opening}'"));
    }

    let bytes = opening.as_bytes();
    for chunk in bytes.chunks_exact(2) {
        let move_str = std::str::from_utf8(chunk)
            .map_err(|_| format!("opening sequence is not valid UTF-8: '{opening}'"))?;
        let sq = move_str
            .parse::<Square>()
            .map_err(|_| format!("invalid move in opening sequence: {move_str}"))?;

        if !board.is_legal_move(sq) {
            return Err(format!(
                "illegal opening move {move_str} for {:?}",
                side_to_move
            ));
        }

        *board = board.make_move(sq);
        *side_to_move = side_to_move.opposite();
        moves.push(sq.to_string());

        if !board.is_game_over() && !board.has_legal_moves() {
            *board = board.switch_players();
            *side_to_move = side_to_move.opposite();
            moves.push("pass".to_string());
        }
    }

    Ok(())
}

fn build_result(
    board: Board,
    side_to_move: Disc,
    engine1_is_black: bool,
    moves: String,
) -> WeightMatchGameResult {
    let black_score = score_from_black_perspective(&board, side_to_move);
    let (black_count, white_count) = absolute_counts(&board, side_to_move);
    let engine1_score = if engine1_is_black {
        black_score
    } else {
        -black_score
    };
    let winner = match engine1_score.cmp(&0) {
        std::cmp::Ordering::Greater => "engine1",
        std::cmp::Ordering::Less => "engine2",
        std::cmp::Ordering::Equal => "draw",
    }
    .to_string();

    WeightMatchGameResult {
        engine1_is_black,
        winner,
        black_score,
        engine1_score,
        black_count,
        white_count,
        moves,
    }
}

fn score_from_black_perspective(board: &Board, side_to_move: Disc) -> i32 {
    let score = board.solve(board.get_empty_count());
    if side_to_move == Disc::Black {
        score
    } else {
        -score
    }
}

fn absolute_counts(board: &Board, side_to_move: Disc) -> (u32, u32) {
    if side_to_move == Disc::Black {
        (board.get_player_count(), board.get_opponent_count())
    } else {
        (board.get_opponent_count(), board.get_player_count())
    }
}
