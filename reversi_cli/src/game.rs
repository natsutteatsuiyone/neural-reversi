use colored::Colorize;
use reversi_core::{board::Board, piece::Piece, square::Square};

pub struct GameState {
    pub board: Board,
    side_to_move: Piece,
    history: Vec<(Square, Board)>,
    last_move: Square,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            side_to_move: Piece::Black,
            history: Vec::new(),
            last_move: Square::None,
        }
    }

    pub fn make_move(&mut self, sq: Square) {
        self.history.push((sq, self.board));
        self.board = self.board.make_move(sq);
        self.last_move = sq;
        self.side_to_move = self.side_to_move.opposite();

        if !self.board.has_legal_moves() {
            self.make_pass();
            if !self.board.has_legal_moves() {
                self.make_pass();
            }
        }
    }

    pub fn make_pass(&mut self) {
        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
    }

    pub fn get_side_to_move(&self) -> Piece {
        self.side_to_move
    }

    pub fn get_board_string(&self) -> String {
        let mut result = String::new();

        result.push_str("   a b c d e f g h\n");
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        for y in 0..8 {
            result.push_str(&format!("{} |", y + 1));
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = self.board.get_piece_at(sq, self.side_to_move);

                let symbol = match piece {
                    Piece::Black => "X",
                    Piece::White => "O",
                    Piece::Empty => if self.board.is_legal_move(sq) { "." } else { " " },
                };
                result.push_str(&format!("{}|", symbol));
            }

            match y {
                0 => result.push_str(&format!(" {}'s turn",
                    if self.side_to_move == Piece::Black { "Black(X)" } else { "White(O)" })),
                1 => result.push_str(&format!(" Black: {}", self.get_black_count())),
                2 => result.push_str(&format!(" White: {}", self.get_white_count())),
                _ => {}
            }

            result.push('\n');
            if y < 7 {
                result.push_str("  +-+-+-+-+-+-+-+-+\n");
            }
        }
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        result
    }

    pub fn undo(&mut self) -> bool {
        if let Some((_, prev_board)) = self.history.pop() {
            self.board = prev_board;
            self.last_move = self.history.last().map_or(Square::None, |(sq, _)| *sq);
            true
        } else {
            false
        }
    }

    pub fn print(&self) {
        println!("      a   b   c   d   e   f   g   h");
        println!("    ┌───┬───┬───┬───┬───┬───┬───┬───┐");

        for y in 0..8 {
            print!("  {} │", y + 1);
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = self.board.get_piece_at(sq, self.side_to_move);
                let is_legal = self.board.is_legal_move(sq);
                let is_last_move = sq == self.last_move;

                let symbol = match piece {
                    Piece::Black if is_last_move => " X ".on_bright_black().bright_green(),
                    Piece::White if is_last_move => " O ".on_bright_black().bright_yellow(),
                    Piece::Black => " X ".bright_green(),
                    Piece::White => " O ".bright_yellow(),
                    Piece::Empty if is_legal => " · ".bright_cyan(),
                    Piece::Empty => "   ".black(),
                };
                print!("{}│", symbol);
            }

            match y {
                2 => println!(
                    "   {}",
                    match self.side_to_move {
                        Piece::Black => "Black's turn (X)".bright_green(),
                        Piece::White => "White's turn (O)".bright_yellow(),
                        _ => unreachable!(),
                    }
                ),
                3 => println!(
                    "   Black: {}",
                    format!("{:2}", self.get_black_count()).bright_green()
                ),
                4 => println!(
                    "   White: {}",
                    format!("{:2}", self.get_white_count()).bright_yellow()
                ),
                6 => {
                    if self.board.is_game_over() {
                        let black_count = self.get_black_count();
                        let white_count = self.get_white_count();
                        match black_count.cmp(&white_count) {
                            std::cmp::Ordering::Greater => {
                                println!("   {}", "Black wins!".bright_green())
                            }
                            std::cmp::Ordering::Less => {
                                println!("   {}", "White wins!".bright_yellow())
                            }
                            std::cmp::Ordering::Equal => println!("  {}", "Draw".bright_cyan()),
                        }
                    } else {
                        println!();
                    }
                }
                7 => {
                    if self.board.is_game_over() {
                        println!("   {}", "*** Game Over ***".bright_red());
                    } else {
                        println!();
                    }
                }
                _ => println!(),
            }

            if y < 7 {
                println!("    ├───┼───┼───┼───┼───┼───┼───┼───┤");
            }
        }

        println!("    └───┴───┴───┴───┴───┴───┴───┴───┘");
    }

    fn get_black_count(&self) -> u32 {
        if self.side_to_move == Piece::Black {
            self.board.get_player_count()
        } else {
            self.board.get_opponent_count()
        }
    }

    fn get_white_count(&self) -> u32 {
        if self.side_to_move == Piece::White {
            self.board.get_player_count()
        } else {
            self.board.get_opponent_count()
        }
    }
}
