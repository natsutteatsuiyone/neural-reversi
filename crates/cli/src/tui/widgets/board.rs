//! Board widget for rendering the Reversi game board.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Widget,
};
use reversi_core::{board::Board, disc::Disc, square::Square};

/// Widget for rendering the Reversi game board.
pub struct BoardWidget<'a> {
    /// The game board to render
    board: &'a Board,
    /// Current side to move
    side_to_move: Disc,
    /// Cursor position (row, col)
    cursor: (usize, usize),
    /// Last move played
    last_move: Option<Square>,
    /// Whether to show legal moves
    show_legal_moves: bool,
}

impl<'a> BoardWidget<'a> {
    /// Creates a new board widget.
    pub fn new(board: &'a Board, side_to_move: Disc) -> Self {
        Self {
            board,
            side_to_move,
            cursor: (0, 0),
            last_move: None,
            show_legal_moves: true,
        }
    }

    /// Sets the cursor position.
    pub fn cursor(mut self, row: usize, col: usize) -> Self {
        self.cursor = (row, col);
        self
    }

    /// Sets the last move.
    pub fn last_move(mut self, sq: Option<Square>) -> Self {
        self.last_move = sq;
        self
    }

    /// Sets whether to show legal moves.
    #[allow(dead_code)]
    pub fn show_legal_moves(mut self, show: bool) -> Self {
        self.show_legal_moves = show;
        self
    }
}

impl Widget for BoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Minimum size check
        if area.width < 36 || area.height < 18 {
            return;
        }

        let legal_moves = self.board.get_moves();

        // Column headers
        let header = Line::from(vec![
            Span::raw("    "),
            Span::styled("a", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("b", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("c", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("d", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("e", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("f", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("g", Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled("h", Style::default().fg(Color::Cyan)),
        ]);
        buf.set_line(area.x, area.y, &header, area.width);

        // Top border
        let top_border = "  ┌───┬───┬───┬───┬───┬───┬───┬───┐";
        buf.set_string(area.x, area.y + 1, top_border, Style::default());

        // Board rows
        for row in 0..8 {
            let y = area.y + 2 + (row as u16) * 2;

            // Row number and cells
            let row_num = format!("{} │", row + 1);
            buf.set_string(area.x, y, &row_num, Style::default().fg(Color::Cyan));

            for col in 0..8 {
                let sq = Square::from_usize_unchecked(row * 8 + col);
                let piece = self.board.get_disc_at(sq, self.side_to_move);
                let is_legal = legal_moves.contains(sq);
                let is_cursor = self.cursor == (row, col);
                let is_last_move = self.last_move == Some(sq);

                // Determine cell content and style
                let (content, mut style) = match piece {
                    Disc::Black => (" ● ", Style::default().fg(Color::Green)),
                    Disc::White => (" ○ ", Style::default().fg(Color::Yellow)),
                    Disc::Empty if is_legal && self.show_legal_moves => {
                        (" · ", Style::default().fg(Color::DarkGray))
                    }
                    Disc::Empty => ("   ", Style::default()),
                };

                // Apply cursor highlight
                if is_cursor {
                    style = style.bg(Color::DarkGray).add_modifier(Modifier::BOLD);
                }

                // Apply last move highlight
                if is_last_move {
                    style = style.bg(Color::Rgb(50, 50, 80));
                }

                let x = area.x + 3 + (col as u16) * 4;
                buf.set_string(x, y, content, style);

                // Cell separator
                if col < 7 {
                    buf.set_string(x + 3, y, "│", Style::default());
                }
            }

            // Right border
            buf.set_string(area.x + 34, y, "│", Style::default());

            // Row separator
            if row < 7 {
                let separator = "  ├───┼───┼───┼───┼───┼───┼───┼───┤";
                buf.set_string(area.x, y + 1, separator, Style::default());
            }
        }

        // Bottom border
        let bottom_border = "  └───┴───┴───┴───┴───┴───┴───┴───┘";
        buf.set_string(area.x, area.y + 17, bottom_border, Style::default());

        // Cursor position indicator
        let cursor_sq = Square::from_usize_unchecked(self.cursor.0 * 8 + self.cursor.1);
        let cursor_info = format!("  Cursor: {}", cursor_sq);
        buf.set_string(
            area.x,
            area.y + 18,
            &cursor_info,
            Style::default().fg(Color::Cyan),
        );
    }
}
