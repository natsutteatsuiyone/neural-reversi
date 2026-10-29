//! Application state and main loop for the TUI.

use std::time::Duration;

use ratatui::DefaultTerminal;
use reversi_core::disc::Disc;
use reversi_core::game_state::GameState;
use reversi_core::level;
use reversi_core::search::Search;
use reversi_core::search::result::{PvMove, SearchResult};
use reversi_core::square::Square;

use crate::config::EngineConfig;

use super::event::{self, Event};
use super::parse;
use super::render;
use super::search_worker::{SearchKind, SearchWorker};

/// Game mode configuration determining which players are controlled by AI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    /// Human plays Black, AI plays White
    HumanVsAi,
    /// AI plays Black, Human plays White
    AiVsHuman,
    /// AI plays both sides
    AiVsAi,
    /// Human plays both sides
    HumanVsHuman,
}

impl GameMode {
    /// Returns whether the AI should play for the given side.
    pub fn is_ai_turn(&self, side: Disc) -> bool {
        matches!(
            (self, side),
            (GameMode::HumanVsAi, Disc::White)
                | (GameMode::AiVsHuman, Disc::Black)
                | (GameMode::AiVsAi, _)
        )
    }

    /// Returns a display string for the game mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            GameMode::HumanVsAi => "Human vs AI",
            GameMode::AiVsHuman => "AI vs Human",
            GameMode::AiVsAi => "AI vs AI",
            GameMode::HumanVsHuman => "Human vs Human",
        }
    }

    /// Returns all game modes in order.
    pub const fn all() -> [GameMode; 4] {
        [
            GameMode::HumanVsAi,
            GameMode::AiVsHuman,
            GameMode::AiVsAi,
            GameMode::HumanVsHuman,
        ]
    }

    /// Converts the game mode to an index.
    pub fn to_index(self) -> usize {
        match self {
            GameMode::HumanVsAi => 0,
            GameMode::AiVsHuman => 1,
            GameMode::AiVsAi => 2,
            GameMode::HumanVsHuman => 3,
        }
    }

    /// Creates a game mode from an index.
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => GameMode::HumanVsAi,
            1 => GameMode::AiVsHuman,
            2 => GameMode::AiVsAi,
            _ => GameMode::HumanVsHuman,
        }
    }
}

/// Tab selection for the board editor dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardEditTab {
    /// Move sequence input (e.g. "f5d6c3")
    Moves,
    /// Board string input (64 chars of X/O/-)
    BoardString,
    /// Bitboard hex value input
    Bitboard,
}

/// Focused field in the bitboard tab of the board editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitboardField {
    Player,
    Opponent,
    Side,
}

impl BitboardField {
    fn next(self) -> Option<Self> {
        match self {
            BitboardField::Player => Some(BitboardField::Opponent),
            BitboardField::Opponent => Some(BitboardField::Side),
            BitboardField::Side => None,
        }
    }

    fn prev(self) -> Option<Self> {
        match self {
            BitboardField::Player => None,
            BitboardField::Opponent => Some(BitboardField::Player),
            BitboardField::Side => Some(BitboardField::Opponent),
        }
    }
}

impl BoardEditTab {
    /// Returns the next tab in order, wrapping around.
    pub fn next(self) -> Self {
        match self {
            BoardEditTab::Moves => BoardEditTab::BoardString,
            BoardEditTab::BoardString => BoardEditTab::Bitboard,
            BoardEditTab::Bitboard => BoardEditTab::Moves,
        }
    }

    /// Returns the previous tab in order, wrapping around.
    pub fn prev(self) -> Self {
        match self {
            BoardEditTab::Moves => BoardEditTab::Bitboard,
            BoardEditTab::BoardString => BoardEditTab::Moves,
            BoardEditTab::Bitboard => BoardEditTab::BoardString,
        }
    }

    /// Returns the display name.
    pub fn as_str(&self) -> &'static str {
        match self {
            BoardEditTab::Moves => "Moves",
            BoardEditTab::BoardString => "Board String",
            BoardEditTab::Bitboard => "Bitboard",
        }
    }

    /// Returns all tabs in order.
    pub const fn all() -> [BoardEditTab; 3] {
        [
            BoardEditTab::Moves,
            BoardEditTab::BoardString,
            BoardEditTab::Bitboard,
        ]
    }
}

/// UI mode for handling different interaction states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    /// Normal game play mode
    Normal,
    /// Computing hints in background
    HintsLoading,
    /// Showing hints/analysis
    Hints,
    /// Level selection dialog
    LevelSelect,
    /// Mode selection dialog
    ModeSelect,
    /// Confirming quit
    ConfirmQuit,
    /// Board editor dialog
    BoardEdit,
}

/// Parses a level string from the level-select dialog, returning the level
/// only when it is within the valid `1..=MAX_LEVEL` range.
fn parse_level_input(input: &str) -> Option<usize> {
    match input.parse::<usize>() {
        Ok(n) if (1..=level::MAX_LEVEL).contains(&n) => Some(n),
        _ => None,
    }
}

/// Main application state.
pub struct App {
    /// Current game state
    pub game: GameState,
    search: SearchWorker,
    /// Current AI level
    pub level: usize,
    /// Current game mode
    pub mode: GameMode,
    /// Current UI mode
    pub ui_mode: UiMode,
    /// Cursor position on the board (0-7 for both row and col)
    pub cursor: (usize, usize),
    /// Whether the application should quit
    pub should_quit: bool,
    /// Last AI search result for display
    pub last_ai_result: Option<SearchResult>,
    /// Current hint results
    pub hints: Vec<PvMove>,
    /// Status message to display
    pub status_message: Option<String>,
    /// Input buffer for level selection
    pub level_input: String,
    /// Currently selected mode index in mode selection dialog
    pub mode_selection: usize,
    /// Current board editor tab
    pub board_edit_tab: BoardEditTab,
    /// Board editor main input buffer
    pub board_edit_input: String,
    /// Board editor second input buffer (bitboard opponent)
    pub board_edit_input2: String,
    /// Board editor side to move selection
    pub board_edit_side: Disc,
    /// Board editor active field for bitboard tab
    pub board_edit_focus: BitboardField,
    /// Board editor validation error message
    pub board_edit_error: Option<String>,
}

impl App {
    /// Creates a new App instance.
    pub fn new(config: &EngineConfig) -> Self {
        let search = Search::new(&config.search_options());

        Self {
            game: GameState::new(),
            search: SearchWorker::new(search),
            level: config.level,
            mode: GameMode::HumanVsAi,
            ui_mode: UiMode::Normal,
            cursor: (3, 3), // Start at center
            should_quit: false,
            last_ai_result: None,
            hints: Vec::new(),
            status_message: None,
            level_input: String::new(),
            mode_selection: 0,
            board_edit_tab: BoardEditTab::Moves,
            board_edit_input: String::new(),
            board_edit_input2: String::new(),
            board_edit_side: Disc::Black,
            board_edit_focus: BitboardField::Player,
            board_edit_error: None,
        }
    }

    /// Returns whether an AI move is waiting for completion.
    #[must_use]
    pub(super) fn ai_thinking(&self) -> bool {
        self.search.activity() == Some(SearchKind::Move)
    }

    /// Runs the main TUI loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> std::io::Result<()> {
        // Enable mouse capture
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;

        loop {
            self.check_search_result();

            // Trigger AI move if it's AI's turn
            if self.search.is_available()
                && !self.game.board().is_game_over()
                && self.mode.is_ai_turn(self.game.side_to_move())
                && self.ui_mode == UiMode::Normal
            {
                self.search.start(
                    *self.game.board(),
                    level::get_level(self.level),
                    SearchKind::Move,
                );
            }

            // Draw the UI
            terminal.draw(|frame| render::render(frame, &self))?;

            // Handle events with timeout for responsive updates
            let timeout = if self.search.activity().is_some() {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(100)
            };

            // ConfirmQuit needs raw `n`/`N`; normal mapping turns `n` into NewGame.
            let text_input = matches!(
                self.ui_mode,
                UiMode::BoardEdit | UiMode::LevelSelect | UiMode::ConfirmQuit
            );
            if let Some(event) = event::poll_event(timeout, text_input)? {
                self.handle_event(event);
            }

            if self.should_quit {
                break;
            }
        }

        // Disable mouse capture on exit
        crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture)?;

        Ok(())
    }

    /// Handles an input event.
    fn handle_event(&mut self, event: Event) {
        match self.ui_mode {
            UiMode::Normal => self.handle_normal_event(event),
            UiMode::HintsLoading => self.handle_hints_loading_event(event),
            UiMode::Hints => self.handle_hints_event(event),
            UiMode::LevelSelect => self.handle_level_select_event(event),
            UiMode::ModeSelect => self.handle_mode_select_event(event),
            UiMode::ConfirmQuit => self.handle_confirm_quit_event(event),
            UiMode::BoardEdit => self.handle_board_edit_event(event),
        }
    }

    /// Handles events in normal game mode.
    fn handle_normal_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit => {
                self.ui_mode = UiMode::ConfirmQuit;
            }
            Event::CursorUp if self.cursor.0 > 0 => {
                self.cursor.0 -= 1;
            }
            Event::CursorDown if self.cursor.0 < 7 => {
                self.cursor.0 += 1;
            }
            Event::CursorLeft if self.cursor.1 > 0 => {
                self.cursor.1 -= 1;
            }
            Event::CursorRight if self.cursor.1 < 7 => {
                self.cursor.1 += 1;
            }
            Event::Select => {
                self.try_make_move_at_cursor();
            }
            Event::Click(row, col) if row < 8 && col < 8 => {
                self.cursor = (row, col);
                self.try_make_move_at_cursor();
            }
            Event::Undo => {
                self.undo_move();
            }
            Event::NewGame => {
                self.new_game();
            }
            Event::Hint => {
                self.show_hints();
            }
            Event::Go => {
                self.force_ai_move();
            }
            Event::ChangeMode => {
                self.mode_selection = self.mode.to_index();
                self.ui_mode = UiMode::ModeSelect;
            }
            Event::ChangeLevel => {
                self.level_input = self.level.to_string();
                self.ui_mode = UiMode::LevelSelect;
            }
            Event::EditBoard if self.search.is_available() => {
                self.open_board_editor();
            }
            _ => {}
        }
    }

    /// Handles events while hints are loading.
    fn handle_hints_loading_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit | Event::Hint => {
                self.search.cancel_hint();
                self.ui_mode = UiMode::Normal;
            }
            _ => {}
        }
    }

    /// Handles events in hints mode.
    fn handle_hints_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit | Event::Hint | Event::Select => {
                self.ui_mode = UiMode::Normal;
                self.hints.clear();
            }
            _ => {}
        }
    }

    /// Handles events in level selection mode.
    fn handle_level_select_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit => {
                self.ui_mode = UiMode::Normal;
                self.level_input.clear();
            }
            Event::Select => {
                match parse_level_input(&self.level_input) {
                    Some(new_level) => {
                        self.level = new_level;
                        self.status_message = Some(format!("Level set to {new_level}"));
                    }
                    None => {
                        self.status_message =
                            Some(format!("Level must be between 1 and {}", level::MAX_LEVEL));
                    }
                }
                self.ui_mode = UiMode::Normal;
                self.level_input.clear();
            }
            Event::Char(c) if c.is_ascii_digit() && self.level_input.len() < 3 => {
                self.level_input.push(c);
            }
            Event::Backspace => {
                self.level_input.pop();
            }
            _ => {}
        }
    }

    /// Handles events in mode selection mode.
    fn handle_mode_select_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit => {
                self.ui_mode = UiMode::Normal;
            }
            Event::CursorUp if self.mode_selection > 0 => {
                self.mode_selection -= 1;
            }
            Event::CursorDown if self.mode_selection < 3 => {
                self.mode_selection += 1;
            }
            Event::Select => {
                self.mode = GameMode::from_index(self.mode_selection);
                self.status_message = Some(format!("Mode: {}", self.mode.as_str()));
                self.ui_mode = UiMode::Normal;
            }
            Event::Char('1') => {
                self.mode = GameMode::HumanVsAi;
                self.status_message = Some(format!("Mode: {}", self.mode.as_str()));
                self.ui_mode = UiMode::Normal;
            }
            Event::Char('2') => {
                self.mode = GameMode::AiVsHuman;
                self.status_message = Some(format!("Mode: {}", self.mode.as_str()));
                self.ui_mode = UiMode::Normal;
            }
            Event::Char('3') => {
                self.mode = GameMode::AiVsAi;
                self.status_message = Some(format!("Mode: {}", self.mode.as_str()));
                self.ui_mode = UiMode::Normal;
            }
            Event::Char('4') => {
                self.mode = GameMode::HumanVsHuman;
                self.status_message = Some(format!("Mode: {}", self.mode.as_str()));
                self.ui_mode = UiMode::Normal;
            }
            _ => {}
        }
    }

    /// Handles events in quit confirmation mode.
    fn handle_confirm_quit_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit | Event::Char('y') | Event::Char('Y') => {
                self.should_quit = true;
            }
            Event::Char('n') | Event::Char('N') | Event::Char('q') | Event::Quit => {
                self.ui_mode = UiMode::Normal;
            }
            _ => {}
        }
    }

    /// Explains why a game action is blocked by the current search.
    fn engine_busy_message(&self) -> String {
        if self.ai_thinking() {
            "AI is thinking...".to_string()
        } else {
            "Engine is busy; try again shortly".to_string()
        }
    }

    /// Tries to make a move at the current cursor position.
    fn try_make_move_at_cursor(&mut self) {
        if !self.search.is_available() {
            self.status_message = Some(self.engine_busy_message());
            return;
        }

        if self.game.board().is_game_over() {
            self.status_message = Some("Game is over!".to_string());
            return;
        }

        let sq = Square::from_file_rank(self.cursor.1 as u8, self.cursor.0 as u8);
        if self.game.make_move(sq).is_ok() {
            self.last_ai_result = None;
            self.status_message = None;
        } else {
            self.status_message = Some("Illegal move".to_string());
        }
    }

    /// Undoes the last move.
    fn undo_move(&mut self) {
        if !self.search.is_available() {
            self.status_message = Some(self.engine_busy_message());
            return;
        }

        let vs_ai = matches!(self.mode, GameMode::HumanVsAi | GameMode::AiVsHuman);
        let mut undone = false;
        while self.game.undo() {
            undone = true;
            if !vs_ai || !self.mode.is_ai_turn(self.game.side_to_move()) {
                break;
            }
        }

        if undone {
            self.last_ai_result = None;
            self.status_message = Some("Move undone".to_string());
        } else {
            self.status_message = Some("Nothing to undo".to_string());
        }
    }

    /// Starts a new game.
    fn new_game(&mut self) {
        if !self.search.reset() {
            self.status_message = Some(self.engine_busy_message());
            return;
        }
        self.game = GameState::new();
        self.last_ai_result = None;
        self.hints.clear();
        self.cursor = (3, 3);
        self.status_message = Some("New game started".to_string());
    }

    /// Shows move hints (starts background search).
    fn show_hints(&mut self) {
        if !self.search.is_available() {
            self.status_message = Some(self.engine_busy_message());
            return;
        }

        if self.game.board().is_game_over() {
            self.status_message = Some("Game is over!".to_string());
            return;
        }

        if !self.game.board().has_legal_moves() {
            self.status_message = Some("No legal moves".to_string());
            return;
        }

        if self.search.start(
            *self.game.board(),
            level::get_level(self.level),
            SearchKind::Hint,
        ) {
            self.ui_mode = UiMode::HintsLoading;
        }
    }

    /// Forces AI to make a move.
    fn force_ai_move(&mut self) {
        if !self.search.is_available() {
            return;
        }

        if !self.game.board().is_game_over() && self.game.board().has_legal_moves() {
            self.search.start(
                *self.game.board(),
                level::get_level(self.level),
                SearchKind::Move,
            );
        }
    }

    /// Opens the board editor dialog.
    fn open_board_editor(&mut self) {
        self.board_edit_tab = BoardEditTab::Moves;
        self.board_edit_input.clear();
        self.board_edit_input2.clear();
        self.board_edit_side = self.game.side_to_move();
        self.board_edit_focus = BitboardField::Player;
        self.board_edit_error = None;
        self.ui_mode = UiMode::BoardEdit;
    }

    /// Handles events in board editor mode.
    fn handle_board_edit_event(&mut self, event: Event) {
        match event {
            Event::ForceQuit => {
                self.should_quit = true;
            }
            Event::Quit => {
                self.ui_mode = UiMode::Normal;
            }
            Event::Tab => {
                if self.board_edit_tab == BoardEditTab::Bitboard
                    && let Some(next) = self.board_edit_focus.next()
                {
                    self.board_edit_focus = next;
                } else {
                    self.board_edit_tab = self.board_edit_tab.next();
                    self.board_edit_input.clear();
                    self.board_edit_input2.clear();
                    self.board_edit_focus = BitboardField::Player;
                    self.board_edit_error = None;
                }
            }
            Event::BackTab => {
                if self.board_edit_tab == BoardEditTab::Bitboard
                    && let Some(prev) = self.board_edit_focus.prev()
                {
                    self.board_edit_focus = prev;
                } else {
                    self.board_edit_tab = self.board_edit_tab.prev();
                    self.board_edit_input.clear();
                    self.board_edit_input2.clear();
                    self.board_edit_focus = BitboardField::Player;
                    self.board_edit_error = None;
                }
            }
            Event::CursorUp if self.board_edit_tab == BoardEditTab::Bitboard => {
                if let Some(prev) = self.board_edit_focus.prev() {
                    self.board_edit_focus = prev;
                }
            }
            Event::CursorDown if self.board_edit_tab == BoardEditTab::Bitboard => {
                if let Some(next) = self.board_edit_focus.next() {
                    self.board_edit_focus = next;
                }
            }
            Event::CursorLeft | Event::CursorRight => {
                let on_side_field = match self.board_edit_tab {
                    BoardEditTab::Moves => false,
                    BoardEditTab::BoardString => true,
                    BoardEditTab::Bitboard => self.board_edit_focus == BitboardField::Side,
                };
                if on_side_field {
                    self.board_edit_side = if self.board_edit_side == Disc::Black {
                        Disc::White
                    } else {
                        Disc::Black
                    };
                }
            }
            Event::Select => {
                self.apply_board_edit();
            }
            Event::Char(c) if !c.is_control() => {
                self.board_edit_error = None;
                if let Some(target) = self.active_board_edit_input()
                    && target.len() < 128
                {
                    target.push(c);
                }
            }
            Event::Backspace => {
                self.board_edit_error = None;
                if let Some(target) = self.active_board_edit_input() {
                    target.pop();
                }
            }
            _ => {}
        }
    }

    /// Returns the focused input buffer, or `None` when the side selector has focus.
    fn active_board_edit_input(&mut self) -> Option<&mut String> {
        match (self.board_edit_tab, self.board_edit_focus) {
            (BoardEditTab::Bitboard, BitboardField::Opponent) => Some(&mut self.board_edit_input2),
            (BoardEditTab::Bitboard, BitboardField::Side) => None,
            _ => Some(&mut self.board_edit_input),
        }
    }

    /// Validates the board editor input and applies the new board state.
    fn apply_board_edit(&mut self) {
        use reversi_core::bitboard::Bitboard;
        use reversi_core::board::Board;

        let result: Result<GameState, String> = match self.board_edit_tab {
            BoardEditTab::Moves => parse::parse_move_string(&self.board_edit_input)
                .and_then(|moves| GameState::from_moves(&moves)),
            BoardEditTab::BoardString => {
                Board::from_string(&self.board_edit_input, self.board_edit_side)
                    .map(|board| GameState::from_board(board, self.board_edit_side))
                    .map_err(|e| format!("{e:?}"))
            }
            BoardEditTab::Bitboard => {
                let player = parse::parse_hex_u64(&self.board_edit_input);
                let opponent = parse::parse_hex_u64(&self.board_edit_input2);
                match (player, opponent) {
                    (Ok(p), Ok(o)) => {
                        if p & o != 0 {
                            Err("Player and opponent bitboards overlap".to_string())
                        } else {
                            let board = Board::from_bitboards(Bitboard::from(p), Bitboard::from(o));
                            Ok(GameState::from_board(board, self.board_edit_side))
                        }
                    }
                    (Err(e), _) => Err(format!("Player: {e}")),
                    (_, Err(e)) => Err(format!("Opponent: {e}")),
                }
            }
        };

        // For BoardString/Bitboard tabs, require at least 4 discs on the board
        let result = result.and_then(|game| {
            if matches!(
                self.board_edit_tab,
                BoardEditTab::BoardString | BoardEditTab::Bitboard
            ) && game.board().player_count() + game.board().opponent_count() < 4
            {
                Err("At least 4 discs required".to_string())
            } else {
                Ok(game)
            }
        });

        // For BoardString/Bitboard tabs, verify the specified side has legal moves
        let result = result.and_then(|game| {
            if matches!(
                self.board_edit_tab,
                BoardEditTab::BoardString | BoardEditTab::Bitboard
            ) && !game.board().has_legal_moves()
                && !game.board().is_game_over()
            {
                Err(format!(
                    "{} has no legal moves",
                    if self.board_edit_side == Disc::Black {
                        "Black"
                    } else {
                        "White"
                    }
                ))
            } else {
                Ok(game)
            }
        });

        match result {
            Ok(game) => {
                self.game = game;
                self.search.reset();
                self.last_ai_result = None;
                self.hints.clear();
                self.ui_mode = UiMode::Normal;
                self.status_message = Some("Board position set".to_string());
            }
            Err(e) => {
                self.board_edit_error = Some(e);
            }
        }
    }

    /// Applies the completed search to the game or hint display.
    fn check_search_result(&mut self) {
        let Some((kind, result)) = self.search.poll() else {
            return;
        };
        match kind {
            SearchKind::Move => {
                let best_move = result.best_move();
                self.last_ai_result = Some(result);
                if let Some(mv) = best_move {
                    self.game
                        .make_move(mv)
                        .expect("search returned an illegal move");
                }
            }
            SearchKind::Hint => {
                self.hints = result.pv_moves().to_vec();
                self.ui_mode = UiMode::Hints;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_input_accepts_valid_range() {
        assert_eq!(parse_level_input("1"), Some(1));
        assert_eq!(parse_level_input("21"), Some(21));
        let max = level::MAX_LEVEL;
        assert_eq!(parse_level_input(&max.to_string()), Some(max));
    }

    #[test]
    fn parse_level_input_rejects_out_of_range() {
        assert_eq!(parse_level_input("0"), None);
        let over = level::MAX_LEVEL + 1;
        assert_eq!(parse_level_input(&over.to_string()), None);
        assert_eq!(parse_level_input("999"), None);
    }

    #[test]
    fn parse_level_input_rejects_garbage() {
        assert_eq!(parse_level_input(""), None);
        assert_eq!(parse_level_input("abc"), None);
        assert_eq!(parse_level_input("-1"), None);
    }

    #[test]
    fn undo_against_ai_returns_to_human_turn_across_forced_pass() {
        use reversi_core::board::Board;

        let start = Board::from_string(
            "------------X---X-----O--O--O-O-O------------O----X--------X----",
            Disc::Black,
        )
        .unwrap();
        let mut app = App::new(&EngineConfig {
            hash_size: 1,
            level: 1,
            threads: Some(1),
            eval_file: None,
            eval_sm_file: None,
        });
        app.game = GameState::from_board(start, Disc::Black);
        app.game.make_move(Square::C5).unwrap();
        app.game.make_move(Square::C3).unwrap();
        app.game.make_move(Square::D6).unwrap();

        app.undo_move();

        assert_eq!(*app.game.board(), start);
        assert_eq!(app.game.side_to_move(), Disc::Black);
        assert!(app.game.move_history().is_empty());
    }
}
