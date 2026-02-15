//! Event handling for the TUI.

use std::time::Duration;

use crossterm::event::{
    self, Event as CrosstermEvent, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
};

/// Application events.
#[derive(Debug, Clone)]
pub enum Event {
    /// Quit the application
    Quit,
    /// Force quit the application (Ctrl+C)
    ForceQuit,
    /// Move cursor up
    CursorUp,
    /// Move cursor down
    CursorDown,
    /// Move cursor left
    CursorLeft,
    /// Move cursor right
    CursorRight,
    /// Select/confirm action (Enter or Space)
    Select,
    /// Mouse click at board position (row, col)
    Click(usize, usize),
    /// Undo last move
    Undo,
    /// Start a new game
    NewGame,
    /// Show hints
    Hint,
    /// Force AI to move
    Go,
    /// Change game mode
    ChangeMode,
    /// Change AI level
    ChangeLevel,
    /// Open board editor
    EditBoard,
    /// Tab key (for switching tabs/fields)
    Tab,
    /// Shift+Tab (for switching tabs/fields backward)
    BackTab,
    /// Character input (for dialogs)
    Char(char),
    /// Backspace key
    Backspace,
}

/// Board area configuration for mouse click detection.
/// These values should match the render layout.
pub struct BoardArea {
    pub start_row: u16,
    pub start_col: u16,
    pub cell_width: u16,
    pub cell_height: u16,
}

impl Default for BoardArea {
    fn default() -> Self {
        Self {
            // Layout calculation:
            // - Title block: 3 rows (y=0-2)
            // - Content starts at y=3
            // - Board block border: +1 row
            // - Board inner area starts at y=4
            // - Column header row: y=4
            // - Top border row: y=5
            // - First cell row (row=0): y=6
            start_row: 6,
            // - Board block border: +1 col
            // - Row number + separator: 3 chars ("1 │")
            // - Cell content starts at x=4 (1 + 3)
            start_col: 4,
            cell_width: 4,  // Each cell is 4 chars wide (" X │")
            cell_height: 2, // Each cell is 2 rows tall (content + separator)
        }
    }
}

/// Polls for an event with a timeout.
///
/// When `text_input` is true, letter and digit keys are passed through as
/// `Char` events instead of being mapped to game commands. This is used by
/// text input dialogs like the board editor.
pub fn poll_event(timeout: Duration, text_input: bool) -> std::io::Result<Option<Event>> {
    if !event::poll(timeout)? {
        return Ok(None);
    }

    match event::read()? {
        CrosstermEvent::Key(key) if key.kind == KeyEventKind::Press => {
            // Check for Shift+Tab (BackTab key code)
            if matches!(key.code, KeyCode::BackTab) {
                return Ok(Some(Event::BackTab));
            }
            // Check for Ctrl+C
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c'))
            {
                return Ok(Some(Event::ForceQuit));
            }
            if text_input {
                Ok(Some(map_key_event_text_input(key.code)))
            } else {
                Ok(Some(map_key_event(key.code)))
            }
        }
        CrosstermEvent::Mouse(mouse) => Ok(map_mouse_event(mouse)),
        _ => Ok(None),
    }
}

/// Maps a key code to an application event.
fn map_key_event(code: KeyCode) -> Event {
    match code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => Event::Quit,

        // Cursor movement - Arrow keys
        KeyCode::Up => Event::CursorUp,
        KeyCode::Down => Event::CursorDown,
        KeyCode::Left => Event::CursorLeft,
        KeyCode::Right => Event::CursorRight,

        // Cursor movement - WASD
        KeyCode::Char('w') => Event::CursorUp,
        KeyCode::Char('s') => Event::CursorDown,
        KeyCode::Char('a') => Event::CursorLeft,
        KeyCode::Char('d') => Event::CursorRight,

        // Cursor movement - Vim style
        KeyCode::Char('k') => Event::CursorUp,
        KeyCode::Char('j') => Event::CursorDown,
        KeyCode::Char('h') => Event::CursorLeft,
        KeyCode::Char('l') => Event::CursorRight,

        // Selection
        KeyCode::Enter | KeyCode::Char(' ') => Event::Select,

        // Game commands
        KeyCode::Char('u') => Event::Undo,
        KeyCode::Char('n') => Event::NewGame,
        KeyCode::Char('i') => Event::Hint,
        KeyCode::Char('g') => Event::Go,
        KeyCode::Char('m') => Event::ChangeMode,
        KeyCode::Char('v') => Event::ChangeLevel,

        // Tab key
        KeyCode::Tab => Event::Tab,

        // Board editor
        KeyCode::Char('e') => Event::EditBoard,

        // Backspace
        KeyCode::Backspace => Event::Backspace,

        // Other characters
        KeyCode::Char(c) => Event::Char(c),

        // Default
        _ => Event::Char('\0'),
    }
}

/// Maps a key code to an application event in text input mode.
///
/// Only special keys (Esc, Enter, Tab, arrows, Backspace) are mapped to
/// their respective events. All printable characters are passed through as
/// `Char` events, allowing text input dialogs to receive raw characters.
fn map_key_event_text_input(code: KeyCode) -> Event {
    match code {
        KeyCode::Esc => Event::Quit,
        KeyCode::Enter => Event::Select,
        KeyCode::Tab => Event::Tab,
        KeyCode::Up => Event::CursorUp,
        KeyCode::Down => Event::CursorDown,
        KeyCode::Left => Event::CursorLeft,
        KeyCode::Right => Event::CursorRight,
        KeyCode::Backspace => Event::Backspace,
        KeyCode::Char(c) => Event::Char(c),
        _ => Event::Char('\0'),
    }
}

/// Maps a mouse event to an application event.
fn map_mouse_event(mouse: crossterm::event::MouseEvent) -> Option<Event> {
    match mouse.kind {
        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            let board_area = BoardArea::default();

            // Check if click is within board area
            if mouse.row >= board_area.start_row && mouse.column >= board_area.start_col {
                let row = (mouse.row - board_area.start_row) / board_area.cell_height;
                let col = (mouse.column - board_area.start_col) / board_area.cell_width;

                if row < 8 && col < 8 {
                    return Some(Event::Click(row as usize, col as usize));
                }
            }
            None
        }
        _ => None,
    }
}
