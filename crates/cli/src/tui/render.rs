//! Rendering logic for the TUI.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use reversi_core::piece::Piece;

use super::app::{App, GameMode, UiMode};
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

    let board_widget = BoardWidget::new(app.game.board(), app.game.get_side_to_move())
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

    let mut lines = Vec::new();

    // Game status
    let (black_count, white_count) = app.game.get_score();
    let side_to_move = app.game.get_side_to_move();

    lines.push(Line::from(""));

    // Turn indicator
    let turn_text = match side_to_move {
        Piece::Black => Span::styled("Black's turn (●)", Style::default().fg(Color::Green)),
        Piece::White => Span::styled("White's turn (○)", Style::default().fg(Color::Yellow)),
        _ => Span::raw(""),
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

    // Move history
    let history = app.game.get_move_history();
    if !history.is_empty() {
        lines.push(Line::from(Span::styled(
            "History:",
            Style::default().fg(Color::Cyan),
        )));

        // Format moves with colors: green for Black, yellow for White
        let max_width = inner_area.width.saturating_sub(2) as usize;
        let mut current_spans: Vec<Span> = vec![Span::raw(" ")];
        let mut current_len = 1usize;

        for (i, sq) in history.iter().enumerate() {
            let is_black = i % 2 == 0;
            let move_str = format!("{} ", sq);
            let move_len = move_str.len();

            if current_len + move_len > max_width && current_len > 1 {
                lines.push(Line::from(current_spans));
                current_spans = vec![Span::raw(" ")];
                current_len = 1;
            }

            let color = if is_black {
                Color::Green
            } else {
                Color::Yellow
            };
            current_spans.push(Span::styled(move_str, Style::default().fg(color)));
            current_len += move_len;
        }

        if current_len > 1 {
            lines.push(Line::from(current_spans));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from("─".repeat(inner_area.width as usize)));
    lines.push(Line::from(""));

    // AI status
    if app.ai_thinking {
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
                if result.get_probability() == 100 {
                    format!("{}", result.depth)
                } else {
                    format!("{}@{}%", result.depth, result.get_probability())
                },
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Eval:  "),
            Span::styled(
                format!("{:+.2}", result.score),
                Style::default().fg(if result.score >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Nodes: "),
            Span::styled(
                format!("{}", result.n_nodes),
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

    let info = Paragraph::new(lines);
    frame.render_widget(info, inner_area);
}

/// Renders the help bar at the bottom.
fn render_help_bar(frame: &mut Frame, area: Rect, app: &App) {
    let help_items = if app.ai_thinking {
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
    let area = centered_rect(40, 20, frame.area());
    frame.render_widget(Clear, area);

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

/// Renders the hints popup.
fn render_hints_popup(frame: &mut Frame, app: &App) {
    let area = centered_rect(60, 50, frame.area());
    frame.render_widget(Clear, area);

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

    for (i, hint) in app.hints.iter().enumerate() {
        let pv_str: String = hint
            .pv_line
            .iter()
            .map(|sq| format!("{sq}"))
            .collect::<Vec<_>>()
            .join(" ");

        lines.push(Line::from(vec![
            Span::raw(format!("  {:2}  ", i + 1)),
            Span::styled(
                format!("  {}   ", hint.sq),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!("{:+6.2} ", hint.score),
                Style::default().fg(if hint.score >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
            Span::raw(pv_str),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Press Enter or Esc to close",
        Style::default().fg(Color::DarkGray),
    )));

    let popup = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Hints "),
    );
    frame.render_widget(popup, area);
}

/// Renders the level selection dialog.
fn render_level_dialog(frame: &mut Frame, app: &App) {
    let area = centered_rect(40, 20, frame.area());
    frame.render_widget(Clear, area);

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
    let area = centered_rect(45, 30, frame.area());
    frame.render_widget(Clear, area);

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
    let area = centered_rect(40, 15, frame.area());
    frame.render_widget(Clear, area);

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

/// Creates a centered rectangle with the given percentage of the parent area.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}
