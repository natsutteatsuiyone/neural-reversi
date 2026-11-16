//! Rendering logic for the TUI.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use reversi_core::player_color::PlayerColor;
use reversi_core::score::Score;

use super::app::{App, BitboardField, BoardEditTab, GameMode, UiMode};
use super::widgets::BoardWidget;

/// Main render function.
pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Main layout: title, content, help bar
    let main_layout = Layout::vertical([
        Constraint::Length(3), // Title
        Constraint::Min(20),   // Content
        Constraint::Length(3), // Help bar
    ])
    .split(area);

    render_title(frame, main_layout[0]);
    render_content(frame, main_layout[1], app);
    render_help_bar(frame, main_layout[2], app);

    // Render overlays based on UI mode
    match app.ui_mode {
        UiMode::HintsLoading => render_hints_loading_popup(frame),
        UiMode::Hints => render_hints_popup(frame, app),
        UiMode::LevelSelect => render_level_dialog(frame, app),
        UiMode::ModeSelect => render_mode_dialog(frame, app),
        UiMode::ConfirmQuit => render_quit_dialog(frame),
        UiMode::BoardEdit => render_board_edit_dialog(frame, app),
        UiMode::Normal => {}
    }
}

/// Renders the title bar.
fn render_title(frame: &mut Frame, area: Rect) {
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " Neural Reversi ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            concat!("v", env!("CARGO_PKG_VERSION")),
            Style::default().fg(Color::DarkGray),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(title, area);
}

/// Renders the main content area (board + info panel).
fn render_content(frame: &mut Frame, area: Rect, app: &App) {
    let content_layout = Layout::horizontal([
        Constraint::Length(42), // Board area
        Constraint::Min(20),    // Info panel
    ])
    .split(area);

    render_board(frame, content_layout[0], app);
    render_info_panel(frame, content_layout[1], app);
}

/// Renders the game board.
fn render_board(frame: &mut Frame, area: Rect, app: &App) {
    let board_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Board ");

    let inner_area = board_block.inner(area);
    frame.render_widget(board_block, area);

    let board_widget = BoardWidget::new(app.game.board(), app.game.side_to_move())
        .cursor(app.cursor.0, app.cursor.1)
        .last_move(app.game.last_move());

    frame.render_widget(board_widget, inner_area);
}

/// Renders the information panel.
fn render_info_panel(frame: &mut Frame, area: Rect, app: &App) {
    let info_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Info ");

    let inner_area = info_block.inner(area);
    frame.render_widget(info_block, area);

    let mut head = Vec::new();
    let lines = &mut head;

    // Game status
    let (black_count, white_count) = app.game.score();
    let side_to_move = app.game.side_to_move();

    lines.push(Line::from(""));

    // Turn indicator
    let turn_text = match side_to_move {
        PlayerColor::Black => Span::styled("Black's turn (●)", Style::default().fg(Color::Green)),
        PlayerColor::White => Span::styled("White's turn (○)", Style::default().fg(Color::Yellow)),
    };
    lines.push(Line::from(turn_text));
    lines.push(Line::from(""));

    // Score
    lines.push(Line::from(vec![
        Span::raw("Black: "),
        Span::styled(
            format!("{:2}", black_count),
            Style::default().fg(Color::Green),
        ),
        Span::raw("  "),
        Span::raw("White: "),
        Span::styled(
            format!("{:2}", white_count),
            Style::default().fg(Color::Yellow),
        ),
    ]));
    lines.push(Line::from(""));

    // Game info
    lines.push(Line::from(vec![
        Span::raw("Level: "),
        Span::styled(format!("{}", app.level), Style::default().fg(Color::Cyan)),
    ]));
    lines.push(Line::from(vec![
        Span::raw("Mode:  "),
        Span::styled(app.mode.as_str(), Style::default().fg(Color::Cyan)),
    ]));

    // Last move
    if let Some(last_sq) = app.game.last_move() {
        lines.push(Line::from(vec![
            Span::raw("Last:  "),
            Span::styled(format!("{}", last_sq), Style::default().fg(Color::Magenta)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::raw("Last:  "),
            Span::styled("--", Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from("─".repeat(inner_area.width as usize)));

    // Move history rows, each with the number of moves it holds.
    // Moves are green for Black, yellow for White.
    let mut history: Vec<(Vec<Span>, usize)> = Vec::new();
    let max_width = inner_area.width.saturating_sub(2) as usize;
    let mut current_len = 1usize;
    for entry in app.game.move_history() {
        let Some(sq) = entry.action.square() else {
            continue;
        };
        let move_str = format!("{} ", sq);
        let move_len = move_str.len();

        if history.is_empty() || (current_len + move_len > max_width && current_len > 1) {
            history.push((vec![Span::raw(" ")], 0));
            current_len = 1;
        }

        let color = if entry.side_to_move == PlayerColor::Black {
            Color::Green
        } else {
            Color::Yellow
        };
        let (spans, n_moves) = history.last_mut().expect("a row was just pushed");
        spans.push(Span::styled(move_str, Style::default().fg(color)));
        *n_moves += 1;
        current_len += move_len;
    }

    let mut tail = Vec::new();
    let lines = &mut tail;
    lines.push(Line::from(""));
    lines.push(Line::from("─".repeat(inner_area.width as usize)));
    lines.push(Line::from(""));

    // AI status
    if app.ai_thinking() {
        lines.push(Line::from(Span::styled(
            "AI Thinking...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::SLOW_BLINK),
        )));
    } else if let Some(ref result) = app.last_ai_result {
        lines.push(Line::from(Span::styled(
            "Last AI Search:",
            Style::default().fg(Color::Cyan),
        )));
        lines.push(Line::from(vec![
            Span::raw("  Depth: "),
            Span::styled(
                if result.probability() == 100 {
                    format!("{}", result.depth())
                } else {
                    format!("{}@{}%", result.depth(), result.probability())
                },
                Style::default().fg(Color::White),
            ),
        ]));
        let score = result.score().unwrap_or(Score::ZERO).to_disc_diff_f32();
        lines.push(Line::from(vec![
            Span::raw("  Eval:  "),
            Span::styled(
                format!("{score:+.2}"),
                Style::default().fg(if score >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Nodes: "),
            Span::styled(
                format!("{}", result.n_nodes()),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    // Game over status
    if app.game.board().is_game_over() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "*** Game Over ***",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));

        let winner = if black_count > white_count {
            Span::styled("Black wins!", Style::default().fg(Color::Green))
        } else if white_count > black_count {
            Span::styled("White wins!", Style::default().fg(Color::Yellow))
        } else {
            Span::styled("Draw!", Style::default().fg(Color::Cyan))
        };
        lines.push(Line::from(winner));
    }

    // Status message
    if let Some(ref msg) = app.status_message {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            msg.as_str(),
            Style::default().fg(Color::Gray),
        )));
    }

    // When the panel is too short, give up the spacing first and then the
    // oldest moves, so that the result and the status message stay visible.
    let height = inner_area.height as usize;
    let history_rows = |history: &[(Vec<Span>, usize)]| match history.len() {
        0 => 0,
        rows => rows + 1,
    };
    if head.len() + history_rows(&history) + tail.len() > height {
        head.retain(|line| line.width() > 0);
        tail.retain(|line| line.width() > 0);
    }
    let mut hidden_moves = 0;
    while history.len() > 1 && head.len() + history_rows(&history) + tail.len() > height {
        hidden_moves += history.remove(0).1;
    }

    let mut lines = head;
    if !history.is_empty() {
        let label = if hidden_moves > 0 {
            format!("History (first {hidden_moves} moves hidden):")
        } else {
            "History:".to_string()
        };
        lines.push(Line::from(Span::styled(
            label,
            Style::default().fg(Color::Cyan),
        )));
        lines.extend(history.into_iter().map(|(spans, _)| Line::from(spans)));
    }
    lines.extend(tail);

    let info = Paragraph::new(lines);
    frame.render_widget(info, inner_area);
}

/// Renders the help bar at the bottom.
fn render_help_bar(frame: &mut Frame, area: Rect, app: &App) {
    let help_items = if app.ai_thinking() {
        vec![("", "AI is thinking...")]
    } else {
        vec![
            ("Enter", "Move"),
            ("U", "Undo"),
            ("N", "New"),
            ("I", "Hint"),
            ("G", "Go"),
            ("M", "Mode"),
            ("V", "Level"),
            ("E", "Edit"),
            ("Q", "Quit"),
        ]
    };

    let spans: Vec<Span> = help_items
        .iter()
        .flat_map(|(key, desc)| {
            vec![
                Span::styled(
                    format!(" [{key}] "),
                    Style::default().fg(Color::Black).bg(Color::Cyan),
                ),
                Span::raw(format!("{desc} ")),
            ]
        })
        .collect();

    let help = Paragraph::new(Line::from(spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(help, area);
}

/// Renders the hints loading popup.
fn render_hints_loading_popup(frame: &mut Frame) {
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Computing Hints...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::SLOW_BLINK),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Press Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let area = popup_area(40, 20, &lines, frame.area());
    frame.render_widget(Clear, area);

    let popup = Paragraph::new(lines)
        .alignment(ratatui::layout::Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Hints "),
        );
    frame.render_widget(popup, area);
}

const HINTS_WIDTH_PERCENT: u16 = 60;

/// Renders the hints popup.
fn render_hints_popup(frame: &mut Frame, app: &App) {
    let bounds = frame.area();
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Move Hints",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(" No. ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(" Move ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(" Score ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(" PV Line", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from("─".repeat(50)),
    ];

    // One row per move: the popup cannot scroll, so when the terminal is too
    // short the lowest-ranked moves are left out, and a long PV is cut.
    const FRAME_SIZE: usize = 2;
    const FOOTER_ROWS: usize = 2;
    const ROW_PREFIX_WIDTH: usize = 20;
    let rows = (bounds.height as usize).saturating_sub(lines.len() + FOOTER_ROWS + FRAME_SIZE);
    let shown = if app.hints.len() > rows {
        rows.saturating_sub(1)
    } else {
        app.hints.len()
    };
    let header_width = lines.iter().map(Line::width).max().unwrap_or_default();
    let pv_width = (bounds.width as usize * HINTS_WIDTH_PERCENT as usize / 100)
        .max(header_width + FRAME_SIZE)
        .min(bounds.width as usize)
        .saturating_sub(FRAME_SIZE + ROW_PREFIX_WIDTH);

    for (i, hint) in app.hints.iter().take(shown).enumerate() {
        let mut pv_str: String = hint
            .pv_line
            .iter()
            .map(|sq| format!("{sq}"))
            .collect::<Vec<_>>()
            .join(" ");
        if pv_str.len() > pv_width {
            pv_str.truncate(pv_width.saturating_sub(1));
            pv_str.push('…');
        }

        lines.push(Line::from(vec![
            Span::raw(format!("  {:2}  ", i + 1)),
            Span::styled(
                format!("  {}   ", hint.sq),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!("{:+6.2} ", hint.score.to_disc_diff_f32()),
                Style::default().fg(if hint.score >= Score::ZERO {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
            Span::raw(pv_str),
        ]));
    }

    if shown < app.hints.len() {
        lines.push(Line::from(Span::styled(
            format!("  … {} more moves", app.hints.len() - shown),
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Press Enter or Esc to close",
        Style::default().fg(Color::DarkGray),
    )));

    let area = popup_area(HINTS_WIDTH_PERCENT, 50, &lines, bounds);
    frame.render_widget(Clear, area);
    let popup = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Hints "),
    );
    frame.render_widget(popup, area);
}

/// Renders the level selection dialog.
fn render_level_dialog(frame: &mut Frame, app: &App) {
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Set AI Level",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("Current: "),
            Span::styled(format!("{}", app.level), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("New level: "),
            Span::styled(
                if app.level_input.is_empty() {
                    "_".to_string()
                } else {
                    format!("{}_", app.level_input)
                },
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::RAPID_BLINK),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Enter: Confirm  Esc: Cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let area = popup_area(40, 20, &lines, frame.area());
    frame.render_widget(Clear, area);

    let dialog = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Level "),
    );
    frame.render_widget(dialog, area);
}

/// Renders the mode selection dialog.
fn render_mode_dialog(frame: &mut Frame, app: &App) {
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Select Game Mode",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    for (i, mode) in GameMode::all().iter().enumerate() {
        let is_selected = i == app.mode_selection;
        let is_current = *mode == app.mode;

        let prefix = if is_selected { "▶ " } else { "  " };
        let suffix = if is_current { " (current)" } else { "" };

        let style = if is_selected {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{prefix}{}. ", i + 1), style),
            Span::styled(mode.as_str(), style),
            Span::styled(suffix, Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "↑↓: Select  Enter: Confirm  Esc: Cancel",
        Style::default().fg(Color::DarkGray),
    )));
    let area = popup_area(45, 30, &lines, frame.area());
    frame.render_widget(Clear, area);

    let dialog = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Mode "),
    );
    frame.render_widget(dialog, area);
}

/// Renders the quit confirmation dialog.
fn render_quit_dialog(frame: &mut Frame) {
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Quit Neural Reversi?",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Press Y to quit, N to cancel"),
    ];
    let area = popup_area(40, 15, &lines, frame.area());
    frame.render_widget(Clear, area);

    let dialog = Paragraph::new(lines)
        .alignment(ratatui::layout::Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" Confirm "),
        );
    frame.render_widget(dialog, area);
}

/// Returns a centered popup area in `bounds` for `lines`: the given
/// percentage of `bounds`, grown as far as `bounds` allows to show every line.
fn popup_area(percent_x: u16, percent_y: u16, lines: &[Line], bounds: Rect) -> Rect {
    let percent_of = |size: u16, percent: u16| size as usize * percent as usize / 100;
    let content_width = lines.iter().map(Line::width).max().unwrap_or_default();
    let width = percent_of(bounds.width, percent_x)
        .max(content_width + 2)
        .min(bounds.width as usize);
    // Lines wider than the popup wrap onto further rows.
    let inner_width = width.saturating_sub(2).max(1);
    let content_height: usize = lines
        .iter()
        .map(|line| line.width().div_ceil(inner_width).max(1))
        .sum();
    let height = percent_of(bounds.height, percent_y)
        .max(content_height + 2)
        .min(bounds.height as usize);
    let (width, height) = (width as u16, height as u16);
    Rect::new(
        bounds.x + (bounds.width - width) / 2,
        bounds.y + (bounds.height - height) / 2,
        width,
        height,
    )
}

/// Renders the board editor dialog.
fn render_board_edit_dialog(frame: &mut Frame, app: &App) {
    let mut lines = vec![Line::from("")];

    // Tab bar
    let tab_spans: Vec<Span> = BoardEditTab::all()
        .iter()
        .flat_map(|tab| {
            let is_active = *tab == app.board_edit_tab;
            let style = if is_active {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            vec![
                Span::styled(format!(" {} ", tab.as_str()), style),
                Span::raw(" "),
            ]
        })
        .collect();
    lines.push(Line::from(tab_spans));
    lines.push(Line::from(""));

    // Tab-specific content
    match app.board_edit_tab {
        BoardEditTab::Moves => {
            lines.push(Line::from(Span::styled(
                "Enter move sequence (e.g. f5d6c3d3c4):",
                Style::default().fg(Color::Cyan),
            )));
            lines.push(Line::from(vec![
                Span::raw("  > "),
                Span::styled(
                    format!("{}_", app.board_edit_input),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::RAPID_BLINK),
                ),
            ]));
        }
        BoardEditTab::BoardString => {
            lines.push(Line::from(Span::styled(
                "Enter 64-char board string (X/O/-):",
                Style::default().fg(Color::Cyan),
            )));
            lines.push(Line::from(vec![
                Span::raw("  > "),
                Span::styled(
                    format!("{}_", app.board_edit_input),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::RAPID_BLINK),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  Length: "),
                Span::styled(
                    format!("{}/64", app.board_edit_input.len()),
                    Style::default().fg(if app.board_edit_input.len() == 64 {
                        Color::Green
                    } else {
                        Color::Yellow
                    }),
                ),
            ]));
            lines.push(Line::from(""));
            render_side_selector(&mut lines, app.board_edit_side);
        }
        BoardEditTab::Bitboard => {
            let player_style = if app.board_edit_focus == BitboardField::Player {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::RAPID_BLINK)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            let opponent_style = if app.board_edit_focus == BitboardField::Opponent {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::RAPID_BLINK)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            lines.push(Line::from(Span::styled(
                "Enter bitboard hex values:",
                Style::default().fg(Color::Cyan),
            )));
            lines.push(Line::from(vec![
                Span::raw("  Player:   > "),
                Span::styled(format!("{}_", app.board_edit_input), player_style),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  Opponent: > "),
                Span::styled(format!("{}_", app.board_edit_input2), opponent_style),
            ]));
            lines.push(Line::from(""));
            render_side_selector(&mut lines, app.board_edit_side);
        }
    }

    // Error message
    if let Some(ref err) = app.board_edit_error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            err.as_str(),
            Style::default().fg(Color::Red),
        )));
    }

    // Help
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Tab: Switch  ←→: Side  Enter: Apply  Esc: Cancel",
        Style::default().fg(Color::DarkGray),
    )));
    let area = popup_area(70, 55, &lines, frame.area());
    frame.render_widget(Clear, area);

    let dialog = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Board Editor "),
    );
    frame.render_widget(dialog, area);
}

/// Renders the side-to-move selector line.
fn render_side_selector(lines: &mut Vec<Line<'_>>, side: PlayerColor) {
    let black_style = if side == PlayerColor::Black {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let white_style = if side == PlayerColor::White {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    lines.push(Line::from(vec![
        Span::raw("  Side: "),
        Span::styled("● Black", black_style),
        Span::raw("  "),
        Span::styled("○ White", white_style),
        Span::raw("  (←→ to toggle)"),
    ]));
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend};
    use reversi_core::search::result::{PvMove, SearchResult};
    use reversi_core::square::Square;

    use super::*;
    use crate::config::EngineConfig;

    fn app() -> App {
        App::new(&EngineConfig {
            hash_size: 1,
            level: 1,
            threads: Some(1),
            eval_file: None,
            eval_sm_file: None,
        })
    }

    /// Draws `app` on a `width` x `height` terminal and returns the screen text.
    fn screen(app: &App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    + "\n"
            })
            .collect()
    }

    #[test]
    fn every_dialog_shows_its_key_help_on_a_small_terminal() {
        let mut app = app();
        app.board_edit_tab = BoardEditTab::BoardString;
        app.board_edit_error = Some("Board string too short".into());
        for (mode, last_line) in [
            (UiMode::ConfirmQuit, "Press Y to quit, N to cancel"),
            (UiMode::ModeSelect, "Enter: Confirm  Esc: Cancel"),
            (UiMode::LevelSelect, "Enter: Confirm  Esc: Cancel"),
            (UiMode::HintsLoading, "Press Esc to cancel"),
            (UiMode::BoardEdit, "Enter: Apply  Esc: Cancel"),
        ] {
            app.ui_mode = mode;
            let screen = screen(&app, 80, 24);
            assert!(screen.contains(last_line), "{last_line:?} in\n{screen}");
        }
    }

    #[test]
    fn hints_that_do_not_fit_are_counted_and_the_key_help_stays() {
        let mut app = app();
        app.ui_mode = UiMode::Hints;
        app.hints = Square::iter()
            .take(20)
            .map(|sq| PvMove {
                sq,
                score: Score::ZERO,
                pv_line: vec![Square::A1; 40],
            })
            .collect();

        // 24 rows hold the 7 fixed lines, the frame, 14 hints and the count.
        let screen = screen(&app, 80, 24);
        assert!(screen.contains("… 6 more moves"), "{screen}");
        assert!(screen.contains("Press Enter or Esc to close"), "{screen}");
        assert!(screen.contains("  14  "), "{screen}");
        assert!(!screen.contains("  15  "), "{screen}");
    }

    #[test]
    fn a_finished_game_shows_the_result_and_status_on_a_small_terminal() {
        let mut app = app();
        while !app.game.is_game_over() {
            match app.game.board().legal_moves().iter().next() {
                Some(sq) => app.game.make_move(sq).unwrap(),
                None => app.game.make_pass().unwrap(),
            }
        }
        app.last_ai_result = Some(SearchResult::new_random_move(Square::A1));
        app.status_message = Some("Game is over!".into());

        let screen = screen(&app, 80, 24);
        assert!(screen.contains("Nodes:"), "{screen}");
        assert!(screen.contains("*** Game Over ***"), "{screen}");
        assert!(screen.contains("Game is over!"), "{screen}");
        assert!(screen.contains("moves hidden"), "{screen}");
    }
}
