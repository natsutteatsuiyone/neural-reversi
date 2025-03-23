use reversi_core::board::Board;
use reversi_core::piece::Piece;
use reversi_core::square::Square;

pub struct GameState {
    board: Board,
    side_to_move: Piece,
    last_move_was_pass: bool,
    move_history: Vec<(Option<Square>, Piece)>,
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: Board::new(),
            side_to_move: Piece::Black,
            last_move_was_pass: false,
            move_history: Vec::new(),
        }
    }

    pub fn side_to_move(&self) -> Piece {
        self.side_to_move
    }

    pub fn make_move(&mut self, sq: Option<Square>) -> Result<(), String> {
        match sq {
            Some(square) => {
                if self.board.get_moves() & square.bitboard() == 0 {
                    return Err(format!("Illegal move: {:?}", square));
                }

                self.board = self.board.make_move(square);
                self.move_history.push((Some(square), self.side_to_move));
                self.last_move_was_pass = false;

                self.side_to_move = self.side_to_move.opposite();

                if !self.board.has_legal_moves() {
                    self.handle_pass();
                }
            },
            None => {
                // Pass move
                if self.board.has_legal_moves() {
                    return Err("Cannot pass when legal moves are available".to_string());
                }

                self.handle_pass();
            }
        }

        Ok(())
    }

    fn handle_pass(&mut self) {
        self.move_history.push((None, self.side_to_move));
        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
        self.last_move_was_pass = true;
    }

    pub fn is_game_over(&self) -> bool {
        if self.last_move_was_pass && !self.board.has_legal_moves() {
            return true;
        }

        self.board.get_empty_count() == 0
    }

    pub fn get_score(&self) -> (u32, u32) {
        let black_count;
        let white_count;

        if self.side_to_move == Piece::Black {
            black_count = self.board.get_player_count();
            white_count = self.board.get_opponent_count();
        } else {
            white_count = self.board.get_player_count();
            black_count = self.board.get_opponent_count();
        }

        (black_count, white_count)
    }
}
