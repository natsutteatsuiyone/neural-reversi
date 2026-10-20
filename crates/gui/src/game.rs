//! Native game timeline. Boards are always relative to `Position::turn`.
use reversi_core::{analysis::GameAnalysisMove, board::Board, disc::Disc, square::Square};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupError {
    TranscriptLength,
    InvalidNotation,
    IllegalMove(Square),
    GameOver,
    BoardLength,
    BoardCharacter,
    MissingColors,
    TooFewDiscs,
    NoMoves,
    TurnHasNoMoves,
    MissingTurn,
}

// Match JavaScript trim() and /\s/g, including BOM but excluding U+0085.
fn setup_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\u{0009}'..='\u{000d}'
            | '\u{0020}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Position {
    pub board: Board,
    pub turn: Disc,
}

impl Position {
    pub fn initial() -> Self {
        Self {
            board: Board::new(),
            turn: Disc::Black,
        }
    }

    pub fn cell(&self, index: usize) -> Disc {
        assert!(index < 64, "Board cell index must be below 64");
        self.board
            .get_disc_at(Square::from_u8(index as u8).unwrap(), self.turn)
    }

    pub fn scores(&self) -> (u32, u32) {
        let player = self.board.get_player_count();
        let opponent = self.board.get_opponent_count();
        if self.turn == Disc::Black {
            (player, opponent)
        } else {
            (opponent, player)
        }
    }

    pub fn game_over(&self) -> bool {
        self.board.is_game_over()
    }

    /// Parses an editable board without requiring that it is playable yet.
    /// Whitespace and the original editor's B/W/*/. aliases are supported.
    pub fn from_board_string(text: &str, turn: Disc) -> Result<Self, SetupError> {
        if turn == Disc::Empty {
            return Err(SetupError::MissingTurn);
        }
        let cells = text.chars().filter(|&ch| !setup_whitespace(ch));
        let length: usize = cells.clone().map(char::len_utf16).sum();
        if length != 0 && length != 64 {
            return Err(SetupError::BoardLength);
        }
        let mut black = 0u64;
        let mut white = 0u64;
        for (index, ch) in cells.enumerate() {
            match ch.to_ascii_uppercase() {
                'X' | 'B' | '*' => black |= 1u64 << index,
                'O' | 'W' => white |= 1u64 << index,
                '-' | '.' => {}
                _ => return Err(SetupError::BoardCharacter),
            }
        }
        let board = if turn == Disc::Black {
            Board::from_bitboards(black, white)
        } else {
            Board::from_bitboards(white, black)
        };
        Ok(Self { board, turn })
    }

    /// Imports the legacy coordinate-only format; forced passes are implicit.
    pub fn from_transcript(text: &str) -> Result<Self, SetupError> {
        let text = text.trim_matches(setup_whitespace);
        if !text.encode_utf16().count().is_multiple_of(2) {
            return Err(SetupError::TranscriptLength);
        }
        let mut position = Self::initial();
        let mut notation = text.encode_utf16();
        while let (Some(column), Some(row)) = (notation.next(), notation.next()) {
            let column = column | 0x20;
            if !(u16::from(b'a')..=u16::from(b'h')).contains(&column)
                || !(u16::from(b'1')..=u16::from(b'8')).contains(&row)
            {
                return Err(SetupError::InvalidNotation);
            }
            let square = Square::from_file_rank(
                (column - u16::from(b'a')) as u8,
                (row - u16::from(b'1')) as u8,
            );
            let mut moves = position.board.get_moves();
            if moves.is_empty() {
                let passed = position.passed();
                moves = passed.board.get_moves();
                if moves.is_empty() {
                    return Err(SetupError::GameOver);
                }
                position = passed;
            }
            if !moves.contains(square) {
                return Err(SetupError::IllegalMove(square));
            }
            position = Self {
                board: position.board.make_move(square),
                turn: position.turn.opposite(),
            };
        }
        if position.must_pass() {
            position = position.passed();
        }
        Ok(position)
    }

    pub fn validate_setup(&self) -> Result<(), SetupError> {
        if self.turn == Disc::Empty {
            return Err(SetupError::MissingTurn);
        }
        let (black, white) = self.scores();
        if black == 0 || white == 0 {
            return Err(SetupError::MissingColors);
        }
        if black + white < 4 {
            return Err(SetupError::TooFewDiscs);
        }
        if !self.board.has_legal_moves() {
            return Err(if self.board.switch_players().has_legal_moves() {
                SetupError::TurnHasNoMoves
            } else {
                SetupError::NoMoves
            });
        }
        Ok(())
    }

    fn passed(self) -> Self {
        Self {
            board: self.board.switch_players(),
            turn: self.turn.opposite(),
        }
    }

    fn must_pass(self) -> bool {
        !self.board.has_legal_moves() && self.board.switch_players().has_legal_moves()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoveRecord {
    /// Position after this individual action, before any following pass.
    pub position: Position,
    pub player: Disc,
    pub square: Option<Square>,
    pub score: Option<f32>,
    pub remaining_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub initial: Position,
    pub history: Vec<MoveRecord>,
    /// Number of visible actions, including passes.
    pub cursor: usize,
}

impl Game {
    pub fn new(initial: Position) -> Self {
        let mut game = Self {
            initial,
            history: Vec::new(),
            cursor: 0,
        };
        if initial.must_pass() {
            game.push_pass(initial);
        }
        game
    }

    pub fn position(&self) -> Position {
        self.cursor
            .checked_sub(1)
            .and_then(|index| self.history.get(index))
            .map_or(self.initial, |record| record.position)
    }

    pub fn play(
        &mut self,
        square: Square,
        score: Option<f32>,
        remaining_ms: Option<u64>,
    ) -> Result<(), String> {
        let current = self.position();
        let must_pass = current.must_pass();
        let position = if must_pass { current.passed() } else { current };
        if position.turn == Disc::Empty
            || square == Square::None
            || !position.board.is_legal_move(square)
        {
            return Err(if position.game_over() {
                "The game is over".into()
            } else {
                format!("Illegal move: {square}")
            });
        }
        // An invalid click or history navigation must never discard a variation.
        self.history.truncate(self.cursor);
        if must_pass {
            self.push_pass(current);
        }
        let next = Position {
            board: position.board.make_move(square),
            turn: position.turn.opposite(),
        };
        self.history.push(MoveRecord {
            position: next,
            player: position.turn,
            square: Some(square),
            score,
            remaining_ms,
        });
        self.cursor = self.history.len();
        // No fictitious pass is recorded after a terminal move.
        if next.must_pass() {
            self.push_pass(next);
        }
        Ok(())
    }

    fn push_pass(&mut self, position: Position) {
        self.history.push(MoveRecord {
            position: position.passed(),
            player: position.turn,
            square: None,
            score: None,
            remaining_ms: None,
        });
        self.cursor = self.history.len();
    }

    pub fn navigate(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.history.len());
    }

    /// Exports the visible prefix, omitting passes exactly as the old UI did.
    pub fn transcript(&self) -> String {
        self.history[..self.cursor]
            .iter()
            .filter_map(|record| record.square)
            .map(|square| square.to_string())
            .collect()
    }

    /// Full retained timeline, including passes; independent of history browsing.
    pub fn analysis_moves(&self) -> Vec<GameAnalysisMove> {
        self.history
            .iter()
            .map(|record| match record.square {
                Some(square) => GameAnalysisMove::Play(square),
                None => GameAnalysisMove::Pass,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_string(position: &Position) -> String {
        (0..64)
            .map(|index| position.cell(index).to_char())
            .collect()
    }

    #[test]
    fn forced_pass_navigation_branch_and_terminal_move() {
        let initial = Position::from_board_string(
            "XO------\n--------\n--------\n--------\n--------\n--------\n--------\nXO------",
            Disc::Black,
        )
        .unwrap();
        let mut game = Game::new(initial);
        game.play(Square::C1, Some(4.0), Some(500)).unwrap();
        assert_eq!(game.history.len(), 2);
        assert_eq!(game.history[1].square, None);
        assert_eq!(game.position().turn, Disc::Black);
        assert_eq!(game.position().scores(), (4, 1));
        game.navigate(1);
        assert_eq!(game.position().turn, Disc::White);
        assert!(game.play(Square::A1, None, None).is_err());
        assert_eq!(game.history.len(), 2);
        // Playing from immediately before a forced pass retains correct colors.
        game.play(Square::C8, None, None).unwrap();
        assert_eq!(game.history.len(), 3);
        assert_eq!(game.history[1].square, None);
        assert_eq!(game.history[2].player, Disc::Black);
        assert_eq!(game.position().scores(), (6, 0));
        assert!(game.position().game_over());
        assert_eq!(game.transcript(), "c1c8");
        assert!(game.play(Square::D8, None, None).is_err());
        game.navigate(0);
        assert_eq!(game.history.len(), 3);
        game.play(Square::C8, None, None).unwrap();
        assert_eq!(game.history.len(), 2);
        assert_eq!(game.history[0].square, Some(Square::C8));
    }

    #[test]
    fn transcript_replay_preserves_passes_colors_and_terminal_score() {
        let text = "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8a7b8c7h1b2b1";
        let mut game = Game::new(Position::initial());
        for square in Square::parse_sequence(text).unwrap() {
            game.play(square, None, None).unwrap();
        }
        assert_eq!(game.position().scores(), (57, 7));
        assert!(game.position().game_over());
        assert_eq!(
            Position::from_transcript(&game.transcript()).unwrap(),
            game.position()
        );
        assert!(Position::from_transcript(&format!("{text}a1")).is_err());
        assert!(Position::from_transcript("--").is_err());
    }

    #[test]
    fn editor_preserves_absolute_colors_and_requires_playable_setup() {
        let text = "bw......\n........\n........\n........\n........\n........\n........\n*w......";
        let black = Position::from_board_string(text, Disc::Black).unwrap();
        let white = Position::from_board_string(text, Disc::White).unwrap();
        assert_eq!(board_string(&black), board_string(&white));
        assert_eq!(white.cell(0), Disc::Black);
        assert_eq!(white.scores(), (2, 2));
        assert!(black.validate_setup().is_ok());
        assert!(white.validate_setup().is_err());
        assert!(
            Position::from_board_string("", Disc::Black)
                .unwrap()
                .validate_setup()
                .is_err()
        );
        assert!(Position::from_board_string(text, Disc::Empty).is_err());
    }

    #[test]
    fn board_import_preserves_legacy_whitespace_and_error_precedence() {
        let initial = Position::initial();
        let board = board_string(&initial);
        assert_eq!(
            Position::from_board_string(&format!("\u{feff}{board}\n"), Disc::Black),
            Ok(initial)
        );
        assert_eq!(
            Position::from_board_string(&format!("\u{0085}{board}"), Disc::Black),
            Err(SetupError::BoardLength)
        );
        assert_eq!(
            Position::from_board_string(&"\u{10000}".repeat(32), Disc::Black),
            Err(SetupError::BoardCharacter)
        );
        assert_eq!(
            Position::from_board_string("?", Disc::Black),
            Err(SetupError::BoardLength)
        );
    }

    #[test]
    fn transcript_import_checks_length_then_each_move_in_order() {
        assert_eq!(
            Position::from_transcript("\u{feff}D3\u{feff}"),
            Position::from_transcript("d3")
        );
        assert_eq!(
            Position::from_transcript("d3 c3"),
            Err(SetupError::TranscriptLength)
        );
        assert_eq!(
            Position::from_transcript("d3  c3"),
            Err(SetupError::InvalidNotation)
        );
        assert_eq!(
            Position::from_transcript("\u{0085}d3\u{0085}"),
            Err(SetupError::InvalidNotation)
        );
        assert_eq!(
            Position::from_transcript("\u{10000}"),
            Err(SetupError::InvalidNotation)
        );
        assert_eq!(
            Position::from_transcript("a1zz"),
            Err(SetupError::IllegalMove(Square::A1))
        );
    }
}
