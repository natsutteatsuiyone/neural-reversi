//! Application state and main loop for the TUI.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use ratatui::DefaultTerminal;
use reversi_core::disc::Disc;
use reversi_core::level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_result::{PvMove, SearchResult};
use reversi_core::search::{self, SearchRunOptions};
use reversi_core::square::Square;

use crate::game::GameState;

use super::event::{self, Event};
use super::parse;
use super::render;

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

/// Result from AI search thread.
pub struct AiResult {
    pub search_result: SearchResult,
    pub best_move: Option<Square>,
}

/// Result from hint search thread.
pub struct HintResult {
    pub pv_moves: Vec<PvMove>,
}

/// Main application state.
pub struct App {
    /// Current game state
    pub game: GameState,
    /// Search engine instance
    search: search::Search,
    /// Hash size for creating new search instances
    hash_size: usize,
    /// Number of threads for search
    threads: Option<usize>,
    /// Path to evaluation weights
    eval_path: Option<std::path::PathBuf>,
    /// Path to small evaluation weights
    eval_sm_path: Option<std::path::PathBuf>,
    /// Current AI level
    pub level: usize,
    /// Search selectivity
    selectivity: Selectivity,
    /// Current game mode
    pub mode: GameMode,
    /// Current UI mode
    pub ui_mode: UiMode,
    /// Cursor position on the board (0-7 for both row and col)
    pub cursor: (usize, usize),
    /// Whether the application should quit
    pub should_quit: bool,
    /// AI search result receiver
    ai_receiver: Option<Receiver<AiResult>>,
    /// Whether AI is currently thinking
    pub ai_thinking: bool,
    /// Last AI search result for display
    pub last_ai_result: Option<SearchResult>,
    /// Hint search result receiver
    hint_receiver: Option<Receiver<HintResult>>,
    /// Whether hint search is in progress
    pub hint_thinking: bool,
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
    /// Board editor active field index for bitboard tab (0=player, 1=opponent, 2=side)
    pub board_edit_focus: u8,
    /// Board editor validation error message
    pub board_edit_error: Option<String>,
}

impl App {
    /// Creates a new App instance.
    pub fn new(
        hash_size: usize,
        initial_level: usize,
        selectivity: Selectivity,
        threads: Option<usize>,
        eval_path: Option<&Path>,
        eval_sm_path: Option<&Path>,
    ) -> Result<Self, String> {
        let search_options = SearchOptions::new(hash_size)
            .with_threads(threads)
            .with_eval_paths(eval_path, eval_sm_path);
        let search = search::Search::new(&search_options);

        Ok(Self {
            game: GameState::new(),
            search,
            hash_size,
            threads,
            eval_path: eval_path.map(|p| p.to_path_buf()),
            eval_sm_path: eval_sm_path.map(|p| p.to_path_buf()),
            level: initial_level,
            selectivity,
            mode: GameMode::HumanVsAi,
            ui_mode: UiMode::Normal,
            cursor: (3, 3), // Start at center
            should_quit: false,
            ai_receiver: None,
            ai_thinking: false,
            last_ai_result: None,
            hint_receiver: None,
            hint_thinking: false,
            hints: Vec::new(),
            status_message: None,
            level_input: String::new(),
            mode_selection: 0,
            board_edit_tab: BoardEditTab::Moves,
            board_edit_input: String::new(),
            board_edit_input2: String::new(),
            board_edit_side: Disc::Black,
            board_edit_focus: 0,
            board_edit_error: None,
        })
    }

    /// Runs the main TUI loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> std::io::Result<()> {
        // Enable mouse capture
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;

        loop {
            // Check for AI results
            self.check_ai_result();
            // Check for hint results
            self.check_hint_result();

            // Trigger AI move if it's AI's turn
            if !self.ai_thinking
                && !self.game.board().is_game_over()
                && self.mode.is_ai_turn(self.game.side_to_move())
                && self.ui_mode == UiMode::Normal
            {
                self.start_ai_search();
            }

            // Draw the UI
            terminal.draw(|frame| render::render(frame, &self))?;

            // Handle events with timeout for responsive updates
            let timeout = if self.ai_thinking || self.hint_thinking {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(100)
            };

            let text_input = matches!(
                self.ui_mode,
                UiMode::BoardEdit | UiMode::LevelSelect
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
            Event::CursorUp => {
                if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                }
            }
            Event::CursorDown => {
                if self.cursor.0 < 7 {
                    self.cursor.0 += 1;
                }
            }
            Event::CursorLeft => {
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                }
            }
            Event::CursorRight => {
                if self.cursor.1 < 7 {
                    self.cursor.1 += 1;
                }
            }
            Event::Select => {
                self.try_make_move_at_cursor();
            }
            Event::Click(row, col) => {
                if row < 8 && col < 8 {
                    self.cursor = (row, col);
                    self.try_make_move_at_cursor();
                }
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
            Event::EditBoard => {
                if !self.ai_thinking {
                    self.open_board_editor();
                }
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
                // Cancel hint search and return to normal mode
                self.hint_receiver = None;
                self.hint_thinking = false;
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
                if let Ok(new_level) = self.level_input.parse::<usize>() {
                    self.level = new_level;
                    self.status_message = Some(format!("Level set to {new_level}"));
                }
                self.ui_mode = UiMode::Normal;
                self.level_input.clear();
            }
            Event::Char(c) if c.is_ascii_digit() => {
                if self.level_input.len() < 3 {
                    self.level_input.push(c);
                }
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
            Event::CursorUp => {
                if self.mode_selection > 0 {
                    self.mode_selection -= 1;
                }
            }
            Event::CursorDown => {
                if self.mode_selection < 3 {
                    self.mode_selection += 1;
                }
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
            Event::Char('n') | Event::Char('N') | Event::Quit => {
                self.ui_mode = UiMode::Normal;
            }
            _ => {}
        }
    }

    /// Tries to make a move at the current cursor position.
    fn try_make_move_at_cursor(&mut self) {
        if self.ai_thinking {
            self.status_message = Some("AI is thinking...".to_string());
            return;
        }

        if self.game.board().is_game_over() {
            self.status_message = Some("Game is over!".to_string());
            return;
        }

        let sq = Square::from_usize_unchecked(self.cursor.0 * 8 + self.cursor.1);
        if self.game.board().is_legal_move(sq) {
            self.game.make_move(sq);
            self.last_ai_result = None;
            self.status_message = None;
        } else {
            self.status_message = Some("Illegal move".to_string());
        }
    }

    /// Undoes the last move.
    fn undo_move(&mut self) {
        if self.ai_thinking {
            self.status_message = Some("Cannot undo while AI is thinking".to_string());
            return;
        }

        // Undo twice if playing against AI (undo AI's move and player's move)
        let undo_count = if matches!(self.mode, GameMode::HumanVsAi | GameMode::AiVsHuman) {
            2
        } else {
            1
        };

        let mut undone = false;
        for _ in 0..undo_count {
            if self.game.undo() {
                undone = true;
            } else {
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
        self.game = GameState::new();
        self.search.init();
        self.last_ai_result = None;
        self.hints.clear();
        self.cursor = (3, 3);
        self.status_message = Some("New game started".to_string());
    }

    /// Shows move hints (starts background search).
    fn show_hints(&mut self) {
        if self.ai_thinking || self.hint_thinking {
            self.status_message = Some("Search in progress...".to_string());
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

        self.start_hint_search();
    }

    /// Starts hint search in a background thread.
    fn start_hint_search(&mut self) {
        let board = *self.game.board();
        let level = self.level;
        let selectivity = self.selectivity;

        let hash_size = self.hash_size;
        let threads = self.threads;
        let eval_path = self.eval_path.clone();
        let eval_sm_path = self.eval_sm_path.clone();

        let (tx, rx): (Sender<HintResult>, Receiver<HintResult>) = mpsc::channel();
        self.hint_receiver = Some(rx);
        self.hint_thinking = true;
        self.ui_mode = UiMode::HintsLoading;

        thread::spawn(move || {
            let search_options = SearchOptions::new(hash_size)
                .with_threads(threads)
                .with_eval_paths(eval_path.as_deref(), eval_sm_path.as_deref());
            let mut search = search::Search::new(&search_options);
            let options =
                SearchRunOptions::with_level(level::get_level(level), selectivity).multi_pv(true);
            let result = search.run(&board, &options);

            let _ = tx.send(HintResult {
                pv_moves: result.pv_moves,
            });
        });
    }

    /// Forces AI to make a move.
    fn force_ai_move(&mut self) {
        if self.ai_thinking {
            return;
        }

        if !self.game.board().is_game_over() && self.game.board().has_legal_moves() {
            self.start_ai_search();
        }
    }

    /// Starts AI search in a background thread.
    fn start_ai_search(&mut self) {
        let board = *self.game.board();
        let level = self.level;
        let selectivity = self.selectivity;

        // Clone parameters for the thread
        let hash_size = self.hash_size;
        let threads = self.threads;
        let eval_path = self.eval_path.clone();
        let eval_sm_path = self.eval_sm_path.clone();

        let (tx, rx): (Sender<AiResult>, Receiver<AiResult>) = mpsc::channel();
        self.ai_receiver = Some(rx);
        self.ai_thinking = true;

        thread::spawn(move || {
            let search_options = SearchOptions::new(hash_size)
                .with_threads(threads)
                .with_eval_paths(eval_path.as_deref(), eval_sm_path.as_deref());
            let mut search = search::Search::new(&search_options);
            let options = SearchRunOptions::with_level(level::get_level(level), selectivity);
            let result = search.run(&board, &options);

            let best_move = result.best_move;
            let _ = tx.send(AiResult {
                search_result: result,
                best_move,
            });
        });
    }

    /// Opens the board editor dialog.
    fn open_board_editor(&mut self) {
        self.board_edit_tab = BoardEditTab::Moves;
        self.board_edit_input.clear();
        self.board_edit_input2.clear();
        self.board_edit_side = self.game.side_to_move();
        self.board_edit_focus = 0;
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
                if self.board_edit_tab == BoardEditTab::Bitboard && self.board_edit_focus < 2 {
                    self.board_edit_focus += 1;
                } else {
                    self.board_edit_tab = self.board_edit_tab.next();
                    self.board_edit_input.clear();
                    self.board_edit_input2.clear();
                    self.board_edit_focus = 0;
                    self.board_edit_error = None;
                }
            }
            Event::BackTab => {
                if self.board_edit_tab == BoardEditTab::Bitboard && self.board_edit_focus > 0 {
                    self.board_edit_focus -= 1;
                } else {
                    self.board_edit_tab = self.board_edit_tab.prev();
                    self.board_edit_input.clear();
                    self.board_edit_input2.clear();
                    self.board_edit_focus = 0;
                    self.board_edit_error = None;
                }
            }
            Event::CursorUp => {
                if self.board_edit_tab == BoardEditTab::Bitboard && self.board_edit_focus > 0 {
                    self.board_edit_focus -= 1;
                }
            }
            Event::CursorDown => {
                if self.board_edit_tab == BoardEditTab::Bitboard && self.board_edit_focus < 2 {
                    self.board_edit_focus += 1;
                }
            }
            Event::CursorLeft | Event::CursorRight => {
                let on_side_field = match self.board_edit_tab {
                    BoardEditTab::Moves => false,
                    BoardEditTab::BoardString => true,
                    BoardEditTab::Bitboard => self.board_edit_focus == 2,
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
            Event::Char(c) => {
                self.board_edit_error = None;
                let target = self.active_board_edit_input();
                if target.len() < 128 {
                    target.push(c);
                }
            }
            Event::Backspace => {
                self.board_edit_error = None;
                self.active_board_edit_input().pop();
            }
            _ => {}
        }
    }

    /// Returns a mutable reference to the currently active input buffer.
    fn active_board_edit_input(&mut self) -> &mut String {
        if self.board_edit_tab == BoardEditTab::Bitboard && self.board_edit_focus == 1 {
            &mut self.board_edit_input2
        } else {
            &mut self.board_edit_input
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
                self.search.init();
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

    /// Checks for AI search results.
    fn check_ai_result(&mut self) {
        if let Some(ref rx) = self.ai_receiver
            && let Ok(result) = rx.try_recv()
        {
            self.ai_thinking = false;
            self.ai_receiver = None;
            self.last_ai_result = Some(result.search_result);

            if let Some(mv) = result.best_move {
                self.game.make_move(mv);
            }
        }
    }

    /// Checks for hint search results.
    fn check_hint_result(&mut self) {
        if let Some(ref rx) = self.hint_receiver
            && let Ok(result) = rx.try_recv()
        {
            self.hint_thinking = false;
            self.hint_receiver = None;
            self.hints = result.pv_moves;
            self.ui_mode = UiMode::Hints;
        }
    }
}
