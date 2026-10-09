use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, BufRead, IsTerminal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::{DefaultTerminal, Frame};

use super::connection::Event;
use reversi_core::disc::Disc;
use reversi_core::square::Square;
use reversi_core::types::Scoref;

use super::protocol::{DisplayClock, DisplayState, MatchState, OsEvent};
use super::session::{self, SessionAction};

pub(super) enum Frontend {
    Plain,
    Tui(Box<Tui>),
}

pub(super) enum Input {
    None,
    Commands(Vec<String>),
    Quit,
}

impl Frontend {
    pub(super) fn start(plain: bool, tx: &Sender<Event>, title: String) -> Result<Self, String> {
        if plain || !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            let tx = tx.clone();
            thread::Builder::new()
                .name("ggs-stdin".into())
                .spawn(move || {
                    for line in io::stdin().lock().lines() {
                        match line {
                            Ok(line) => {
                                if tx.send(Event::Stdin(line)).is_err() {
                                    break;
                                }
                            }
                            Err(error) => {
                                let _ = tx.send(Event::InputFailed(error.to_string()));
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| format!("failed to start stdin reader: {e}"))?;
            return Ok(Self::Plain);
        }
        Ok(Self::Tui(Box::new(Tui::start(tx.clone(), title)?)))
    }

    pub(super) fn server_line(&mut self, line: &str) {
        match self {
            Self::Plain => println!("{line}"),
            // READY / ALERT only delimit responses.
            Self::Tui(_) if matches!(line.trim(), "READY" | "ALERT") => {}
            Self::Tui(tui) => tui.model.push_line(line.to_string()),
        }
    }

    pub(super) fn log(&mut self, message: &str) {
        match self {
            Self::Plain => eprintln!("[ggs] {message}"),
            Self::Tui(tui) => tui.model.push_line(format!("[ggs] {message}")),
        }
    }

    pub(super) fn command(&mut self, line: &str) {
        if let Self::Tui(tui) = self {
            tui.model.push_line(format!("> {line}"));
        }
    }

    pub(super) fn snapshot(&mut self, snapshot: DisplayState) {
        if let Self::Tui(tui) = self {
            tui.model.snapshot(snapshot);
        }
    }

    pub(super) fn on_os_event(&mut self, event: &OsEvent) {
        if let Self::Tui(tui) = self {
            tui.model.on_os_event(event);
        }
    }

    pub(super) fn on_action(&mut self, action: &SessionAction) {
        if let Self::Tui(tui) = self {
            tui.model.on_action(action);
        }
    }

    pub(super) fn worker_finished(&mut self, id: &str, search_id: u64) {
        if let Self::Tui(tui) = self {
            tui.model.worker_finished(id, search_id);
        }
    }

    pub(super) fn worker_started(&mut self, id: &str, search_id: u64, ponder: bool) {
        if let Self::Tui(tui) = self {
            tui.model.worker_started(id, search_id, ponder);
        }
    }

    pub(super) fn eval(&mut self, id: &str, search_id: u64, score: Scoref) {
        if let Self::Tui(tui) = self
            && let Some(game) = tui.model.current_worker_game(id, search_id)
        {
            game.eval = Some(score);
        }
    }

    pub(super) fn search_report(&mut self, id: &str, search_id: u64, message: &str) {
        match self {
            Self::Plain => eprintln!("[ggs] {message}"),
            Self::Tui(tui) => {
                tui.model.push_line(format!("[ggs] {message}"));
                if let Some(game) = tui.model.current_worker_game(id, search_id) {
                    game.last_search = Some(message.to_string());
                }
            }
        }
    }

    pub(super) fn input(&mut self, event: event::Event) -> Input {
        match self {
            Self::Plain => Input::None,
            Self::Tui(tui) => tui.model.input(event),
        }
    }

    pub(super) fn draw(&mut self) -> Result<(), String> {
        if let Self::Tui(tui) = self {
            tui.terminal
                .draw(|frame| tui.model.draw(frame))
                .map_err(|e| format!("terminal draw failed: {e}"))?;
        }
        Ok(())
    }
}

pub(super) struct Tui {
    terminal: DefaultTerminal,
    model: Model,
    stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
}

impl Tui {
    fn start(tx: Sender<Event>, title: String) -> Result<Self, String> {
        let terminal = match ratatui::try_init() {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = ratatui::try_restore();
                return Err(format!("terminal setup failed: {error}"));
            }
        };
        let mut tui = Self {
            terminal,
            model: Model::new(title),
            stop: Arc::new(AtomicBool::new(false)),
            reader: None,
        };
        if let Err(error) = execute!(io::stdout(), EnableBracketedPaste)
            && error.kind() != io::ErrorKind::Unsupported
        {
            return Err(format!("paste setup failed: {error}"));
        }
        execute!(io::stdout(), EnableMouseCapture)
            .map_err(|e| format!("mouse setup failed: {e}"))?;
        let stop = Arc::clone(&tui.stop);
        tui.reader = Some(
            thread::Builder::new()
                .name("ggs-keyboard".into())
                .spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        let next = event::poll(Duration::from_millis(100)).and_then(|ready| {
                            if ready {
                                event::read().map(Some)
                            } else {
                                Ok(None)
                            }
                        });
                        match next {
                            Ok(Some(event)) => {
                                if tx.send(Event::Terminal(event)).is_err() {
                                    break;
                                }
                            }
                            Ok(None) => {
                                if tx.send(Event::Tick).is_err() {
                                    break;
                                }
                            }
                            Err(error) => {
                                let _ = tx.send(Event::InputFailed(error.to_string()));
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| format!("failed to start keyboard reader: {e}"))?,
        );
        Ok(tui)
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        let _ = execute!(io::stdout(), DisableMouseCapture, DisableBracketedPaste);
        let _ = ratatui::try_restore();
    }
}

#[derive(Default)]
struct DisplayGame {
    state: Option<MatchState>,
    clocks: Vec<DisplayClock>,
    /// When `clocks` arrived; shown as elapsed time since the snapshot.
    clocks_at: Option<Instant>,
    last_move: Option<String>,
    opponent: Option<String>,
    variant: Option<String>,
    result: Option<String>,
    /// Final disc difference from our side.
    score: Option<f64>,
    activity: Activity,
    /// Search id of the game's latest worker. Events from older workers,
    /// including those of an ended game whose id was recycled, are ignored.
    worker: Option<u64>,
    /// Score of the last completed search, from our side.
    eval: Option<Scoref>,
    last_search: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Activity {
    #[default]
    Waiting,
    WaitingForBoard,
    BoardUnavailable,
    ServerSnapshot,
    SearchQueued,
    PonderQueued,
    PonderStopping,
    Searching,
    Pondering,
    WaitingForServer,
}

impl Activity {
    /// What the engine is doing for the game; `None` while it is idle.
    fn label(self) -> Option<&'static str> {
        match self {
            Self::SearchQueued => Some("Search queued"),
            Self::PonderQueued => Some("Ponder queued"),
            Self::Searching => Some("Searching"),
            Self::Pondering => Some("Pondering"),
            Self::Waiting
            | Self::WaitingForBoard
            | Self::BoardUnavailable
            | Self::ServerSnapshot
            | Self::PonderStopping
            | Self::WaitingForServer => None,
        }
    }
}

/// Transcript lines kept for scrollback; older lines are dropped.
const MAX_TRANSCRIPT_LINES: usize = 10_000;

enum Scroll {
    Follow,
    Row(usize),
}

struct Model {
    title: String,
    /// The newest [`MAX_TRANSCRIPT_LINES`] transcript lines.
    lines: VecDeque<String>,
    scroll: Scroll,
    horizontal: usize,
    page_height: usize,
    games: BTreeMap<String, DisplayGame>,
    selected: Option<String>,
    editor: Editor,
    previous_enter: bool,
    help_open: bool,
    /// Screen areas from the last draw, for mouse hit testing.
    games_area: Cell<Rect>,
    games_first: Cell<usize>,
    transcript_area: Rect,
    dragging_scrollbar: bool,
}

impl Model {
    fn snapshot(&mut self, snapshot: DisplayState) {
        let DisplayState {
            state,
            clocks,
            moves,
        } = snapshot;
        let id = state.id.clone();
        let select_board = self.selected.as_ref().is_none_or(|selected| {
            selected != &id
                && self
                    .games
                    .get(selected)
                    .is_some_and(|game| game.state.is_none() && session::is_game_of(selected, &id))
        });
        let game = self.games.entry(id.clone()).or_default();
        if state.opponent.is_some() {
            game.opponent.clone_from(&state.opponent);
        }
        game.state = Some(state);
        game.clocks = clocks;
        game.clocks_at = Some(Instant::now());
        if let Some((_, mv)) = moves.into_iter().last() {
            game.last_move = Some(mv);
        }
        if !matches!(
            game.activity,
            Activity::SearchQueued
                | Activity::PonderQueued
                | Activity::PonderStopping
                | Activity::Searching
                | Activity::Pondering
        ) {
            game.activity = Activity::ServerSnapshot;
        }
        if select_board {
            self.selected = Some(id);
        }
    }

    /// The game `id`, if `search_id` is its latest worker.
    fn current_worker_game(&mut self, id: &str, search_id: u64) -> Option<&mut DisplayGame> {
        self.games
            .get_mut(id)
            .filter(|game| game.worker == Some(search_id))
    }

    fn worker_started(&mut self, id: &str, search_id: u64, ponder: bool) {
        if let Some(game) = self.current_worker_game(id, search_id)
            && game.result.is_none()
        {
            game.activity = if ponder {
                Activity::Pondering
            } else {
                Activity::Searching
            };
        }
    }

    fn worker_finished(&mut self, id: &str, search_id: u64) {
        if let Some(game) = self.current_worker_game(id, search_id)
            && game.result.is_none()
        {
            game.activity = Activity::WaitingForServer;
        }
    }

    fn on_action(&mut self, action: &SessionAction) {
        let (id, worker, activity) = match action {
            SessionAction::StartSearch {
                game_id, search_id, ..
            } => (game_id, Some(*search_id), Activity::SearchQueued),
            SessionAction::StartPonder {
                game_id, search_id, ..
            } => (game_id, Some(*search_id), Activity::PonderQueued),
            SessionAction::StopPonder { game_id } => (game_id, None, Activity::PonderStopping),
            _ => return,
        };
        if let Some(game) = self.games.get_mut(id)
            && game.result.is_none()
        {
            game.activity = activity;
            if worker.is_some() {
                game.worker = worker;
            }
        }
    }

    fn visible_game_ids(&self) -> Vec<&String> {
        self.games
            .iter()
            .filter_map(|(id, game)| {
                let child_has_board = game.state.is_none()
                    && self.games.iter().any(|(other_id, other)| {
                        other_id != id && other.state.is_some() && session::is_game_of(id, other_id)
                    });
                (!child_has_board).then_some(id)
            })
            .collect()
    }

    /// Listed games that have a board, in list order.
    fn board_ids(&self) -> Vec<&String> {
        self.visible_game_ids()
            .into_iter()
            .filter(|id| self.games[*id].state.is_some())
            .collect()
    }

    fn selected_index(&self) -> usize {
        let games = self.visible_game_ids();
        self.selected
            .as_ref()
            .and_then(|id| games.iter().position(|key| *key == id))
            .unwrap_or(0)
    }

    fn selected_game(&self) -> Option<(&String, &DisplayGame)> {
        let games = self.visible_game_ids();
        let id = games.get(self.selected_index())?;
        self.games.get_key_value(*id)
    }

    fn select_relative(&mut self, offset: usize) {
        let next = {
            let games = self.visible_game_ids();
            if games.is_empty() {
                None
            } else {
                games
                    .get((self.selected_index() + offset) % games.len())
                    .map(|id| (*id).clone())
            }
        };
        if next.is_some() {
            self.selected = next;
        }
    }

    fn on_os_event(&mut self, event: &OsEvent) {
        match event {
            OsEvent::MatchCreated {
                id,
                opponent,
                variant,
            } => {
                self.games.retain(|game_id, game| {
                    game.result.is_none() && !session::is_game_of(id, game_id)
                });
                self.games.insert(
                    id.clone(),
                    DisplayGame {
                        opponent: opponent.clone(),
                        variant: variant.clone(),
                        activity: Activity::WaitingForBoard,
                        ..DisplayGame::default()
                    },
                );
            }
            OsEvent::MatchStateUnavailable { id, variant, .. }
            | OsEvent::SpectatedMatchState { id, variant } => {
                let game = self.games.entry(id.clone()).or_default();
                game.state = None;
                game.clocks.clear();
                game.clocks_at = None;
                if variant.is_some() {
                    game.variant.clone_from(variant);
                }
                game.activity = Activity::BoardUnavailable;
            }
            OsEvent::MatchEnd { id, result, score } => {
                // A synchro match ends after its games; keep their own results.
                for (game_id, game) in &mut self.games {
                    if session::is_game_of(id, game_id) && game.result.is_none() {
                        game.result = Some(result.clone());
                        game.score = *score;
                    }
                }
            }
            _ => {}
        }
        let (selection_valid, replacement) = {
            let visible = self.visible_game_ids();
            let selection_valid = self
                .selected
                .as_ref()
                .is_some_and(|id| visible.contains(&id));
            (selection_valid, visible.first().map(|id| (*id).clone()))
        };
        if !selection_valid {
            self.selected = replacement;
        }
    }

    fn new(title: String) -> Self {
        Self {
            title,
            lines: VecDeque::new(),
            scroll: Scroll::Follow,
            horizontal: 0,
            page_height: 1,
            games: BTreeMap::new(),
            selected: None,
            editor: Editor::default(),
            previous_enter: false,
            help_open: false,
            games_area: Cell::new(Rect::default()),
            games_first: Cell::new(0),
            transcript_area: Rect::default(),
            dragging_scrollbar: false,
        }
    }

    /// Appends a transcript line, dropping the oldest beyond the cap while a
    /// scrolled view stays on the same line.
    fn push_line(&mut self, line: String) {
        self.lines.push_back(line);
        if self.lines.len() > MAX_TRANSCRIPT_LINES {
            self.lines.pop_front();
            if let Scroll::Row(row) = &mut self.scroll {
                *row = row.saturating_sub(1);
            }
        }
    }

    /// Scrolls the transcript to `row`, following new output once the end is visible.
    fn scroll_to(&mut self, row: usize) {
        self.scroll = if row >= self.lines.len().saturating_sub(self.page_height) {
            Scroll::Follow
        } else {
            Scroll::Row(row)
        };
    }

    fn mouse(&mut self, mouse: MouseEvent) {
        let at = Position::new(mouse.column, mouse.row);
        let shift = mouse.modifiers.contains(KeyModifiers::SHIFT);
        let scrollbar = Rect {
            x: self.transcript_area.right(),
            width: 1,
            ..self.transcript_area
        };
        match mouse.kind {
            MouseEventKind::ScrollLeft => self.horizontal = self.horizontal.saturating_sub(8),
            MouseEventKind::ScrollRight => self.horizontal = self.horizontal.saturating_add(8),
            MouseEventKind::ScrollUp if shift => {
                self.horizontal = self.horizontal.saturating_sub(8)
            }
            MouseEventKind::ScrollDown if shift => {
                self.horizontal = self.horizontal.saturating_add(8)
            }
            MouseEventKind::ScrollUp if self.games_area.get().contains(at) => {
                let count = self.visible_game_ids().len();
                if count > 0 {
                    self.select_relative(count - 1);
                }
            }
            MouseEventKind::ScrollDown if self.games_area.get().contains(at) => {
                self.select_relative(1)
            }
            MouseEventKind::ScrollUp => self.scroll = Scroll::Row(self.top().saturating_sub(3)),
            MouseEventKind::ScrollDown => self.scroll_to(self.top() + 3),
            MouseEventKind::Down(MouseButton::Left) if scrollbar.contains(at) => {
                self.dragging_scrollbar = true;
                self.drag_scrollbar(mouse.row);
            }
            MouseEventKind::Drag(MouseButton::Left) if self.dragging_scrollbar => {
                self.drag_scrollbar(mouse.row)
            }
            MouseEventKind::Down(MouseButton::Left) if self.games_area.get().contains(at) => {
                let inner = Block::bordered().inner(self.games_area.get());
                if inner.contains(at) {
                    let index = self.games_first.get() + usize::from(mouse.row - inner.y);
                    if let Some(id) = self.visible_game_ids().get(index) {
                        self.selected = Some((*id).clone());
                    }
                }
            }
            MouseEventKind::Up(_) => self.dragging_scrollbar = false,
            _ => {}
        }
    }

    fn drag_scrollbar(&mut self, row: u16) {
        let area = self.transcript_area;
        let offset = usize::from(
            row.saturating_sub(area.y)
                .min(area.height.saturating_sub(1)),
        );
        let last = self.lines.len().saturating_sub(self.page_height);
        self.scroll_to(offset * last / usize::from(area.height.saturating_sub(1)).max(1));
    }

    fn top(&self) -> usize {
        let last = self.lines.len().saturating_sub(self.page_height);
        match self.scroll {
            Scroll::Follow => last,
            Scroll::Row(row) => row.min(last),
        }
    }

    fn input(&mut self, event: event::Event) -> Input {
        let event::Event::Key(key) = event else {
            if let event::Event::Mouse(mouse) = event {
                self.mouse(mouse);
            } else if let event::Event::Paste(text) = event {
                self.previous_enter = false;
                self.help_open = false;
                self.editor
                    .insert(&text.replace("\r\n", "\n").replace('\r', "\n"));
            }
            return Input::None;
        };
        if key.kind == KeyEventKind::Release {
            return Input::None;
        }
        if key.code == KeyCode::F(1) {
            self.previous_enter = false;
            self.help_open = !self.help_open;
            return Input::None;
        }
        self.help_open = false;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let is_lf =
            ((key.code == KeyCode::Char('j') || key.code == KeyCode::Enter) && ctrl && !alt)
                || key.code == KeyCode::Char('\n');
        let previous_enter = std::mem::replace(
            &mut self.previous_enter,
            key.code == KeyCode::Enter && !is_lf,
        );
        if is_lf {
            return if previous_enter {
                Input::None
            } else {
                Input::Commands(self.editor.submit())
            };
        }
        match key.code {
            KeyCode::Char('c') if ctrl && !alt => return Input::Quit,
            KeyCode::PageUp => {
                self.scroll = Scroll::Row(self.top().saturating_sub(self.page_height))
            }
            KeyCode::PageDown => self.scroll_to(self.top().saturating_add(self.page_height)),
            KeyCode::Home if ctrl => self.scroll = Scroll::Row(0),
            KeyCode::End if ctrl => self.scroll = Scroll::Follow,
            KeyCode::Left if alt => self.horizontal = self.horizontal.saturating_sub(8),
            KeyCode::Right if alt => self.horizontal = self.horizontal.saturating_add(8),
            KeyCode::Tab | KeyCode::F(6) => self.select_relative(1),
            KeyCode::BackTab => {
                let game_count = self.visible_game_ids().len();
                if game_count > 0 {
                    self.select_relative(game_count - 1);
                }
            }
            KeyCode::Enter => return Input::Commands(self.editor.submit()),
            _ => self.editor.key(key),
        }
        Input::None
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        self.games_area.set(Rect::default());
        let has_game = self.selected_game().is_some();
        let has_board = self
            .selected_game()
            .is_some_and(|(_, game)| game.state.is_some());
        let list = self.game_list_height();
        let full_game = has_board && area.width >= 70 && area.height >= 19 + list;
        let boards = self.board_ids().len().max(1) as u16;
        let desired_dashboard = if area.width < 36 || area.height < 14 {
            3
        } else if full_game {
            // Further boards are shown while the transcript keeps at least eight rows.
            let spare = area.height.saturating_sub(19 + list) / 11;
            list + 11 * boards.min(1 + spare)
        } else if has_game {
            list + if has_board { 6 } else { 5 }
        } else {
            5
        };
        let header = u16::from(area.height >= 6);
        let footer = u16::from(area.height >= 8);
        let editor = area.height.saturating_sub(header + footer).min(3);
        let available = area.height.saturating_sub(header + footer + editor);
        let mut dashboard =
            desired_dashboard.min(available.saturating_sub(u16::from(available > 0)));
        let side_by_side = full_game && area.width >= 130;
        if side_by_side {
            dashboard = available;
        }
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(header),
                Constraint::Length(if side_by_side { 0 } else { dashboard }),
                Constraint::Min(0),
                Constraint::Length(editor),
                Constraint::Length(footer),
            ])
            .split(area);
        if areas[0].height > 0 {
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        " GGS ",
                        Style::default()
                            .fg(Color::Black)
                            .bg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {}", visible(&self.title)),
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                ])),
                areas[0],
            );
        }
        let transcript = if side_by_side {
            let columns =
                Layout::horizontal([Constraint::Length(80), Constraint::Min(1)]).split(areas[2]);
            self.draw_dashboard(frame, columns[0]);
            columns[1]
        } else {
            self.draw_dashboard(frame, areas[1]);
            areas[2]
        };
        self.draw_transcript(frame, transcript);
        self.draw_editor(frame, areas[3]);
        let hints: Vec<Span<'_>> = [("Enter", "send"), ("Ctrl+C", "exit"), ("F1", "help")]
            .into_iter()
            .flat_map(|(key, action)| {
                [
                    Span::styled(
                        format!(" {key} "),
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!("{action} "), Style::default().fg(Color::DarkGray)),
                ]
            })
            .collect();
        frame.render_widget(Paragraph::new(Line::from(hints)), areas[4]);
        if self.help_open {
            self.draw_help(frame, area);
        }
    }

    fn draw_transcript(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let (mode, mode_color) = if matches!(self.scroll, Scroll::Follow) {
            ("LIVE", Color::Green)
        } else {
            ("SCROLLED", Color::Yellow)
        };
        let mut title = vec![
            Span::styled(
                " Transcript ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {mode} "),
                Style::default().fg(Color::Black).bg(mode_color),
            ),
            Span::styled(
                format!(" {} lines ", self.lines.len()),
                Style::default().fg(Color::DarkGray),
            ),
        ];
        if self.horizontal > 0 {
            title.push(Span::styled(
                format!("x={} ", self.horizontal),
                Style::default().fg(Color::DarkGray),
            ));
        }
        let block = panel(Line::from(title), false);
        let inner = block.inner(area);
        self.transcript_area = inner;
        self.page_height = usize::from(inner.height).max(1);
        let lines: Vec<Line<'_>> = self
            .lines
            .iter()
            .skip(self.top())
            .take(usize::from(inner.height))
            .map(|line| {
                Line::styled(
                    clip(&visible(line), self.horizontal, usize::from(inner.width)),
                    transcript_style(line),
                )
            })
            .collect();
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new(lines), inner);
        let mut scrollbar = ScrollbarState::new(self.lines.len().saturating_sub(self.page_height))
            .position(self.top())
            .viewport_content_length(self.page_height);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar,
        );
    }

    fn draw_dashboard(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let Some((id, game)) = self.selected_game() else {
            self.draw_welcome(frame, area);
            return;
        };
        if area.height < 7 {
            self.draw_game_list(frame, area);
            return;
        }
        let areas = Layout::vertical([
            Constraint::Length(self.game_list_height()),
            Constraint::Min(1),
        ])
        .split(area);
        self.draw_game_list(frame, areas[0]);
        match &game.state {
            Some(_) if areas[1].height >= 11 && area.width >= 70 => {
                let boards = self.board_ids();
                let fit = usize::from(areas[1].height / 11).clamp(1, boards.len());
                let selected = boards.iter().position(|board| *board == id).unwrap_or(0);
                let first = (selected + 1).saturating_sub(fit);
                for (slot, board_id) in boards[first..first + fit].iter().enumerate() {
                    let game = &self.games[*board_id];
                    let Some(state) = &game.state else { continue };
                    let panel = Rect {
                        y: areas[1].y + 11 * slot as u16,
                        height: 11,
                        ..areas[1]
                    };
                    let details = Layout::horizontal([Constraint::Length(30), Constraint::Min(1)])
                        .split(panel);
                    let last = game.last_move.as_deref().and_then(|mv| mv.parse().ok());
                    self.draw_board(frame, details[0], state, last);
                    self.draw_facts(frame, details[1], board_id, game, state, *board_id == id);
                }
            }
            Some(state) => self.draw_compact_game(frame, areas[1], id, game, state),
            None => self.draw_waiting_game(frame, areas[1], id, game),
        }
    }

    fn draw_welcome(&self, frame: &mut Frame<'_>, area: Rect) {
        let lines = if area.height >= 5 {
            vec![
                Line::from("No active board. Enter any GGS command below."),
                Line::from("Try: help    tell /os who"),
                Line::from("Chat: tell <user> <text>"),
            ]
        } else {
            vec![Line::from("No board yet. Try: help")]
        };
        frame.render_widget(
            Paragraph::new(lines).block(panel(" GGS console ", true)),
            area,
        );
    }

    /// Rows for up to four games plus the border.
    fn game_list_height(&self) -> u16 {
        self.visible_game_ids().len().clamp(1, 4) as u16 + 2
    }

    fn draw_game_list(&self, frame: &mut Frame<'_>, area: Rect) {
        let game_ids = self.visible_game_ids();
        let selected_index = self.selected_index();
        let block = panel(
            format!(
                " Games {}/{} · Tab/F6 select ",
                selected_index.saturating_add(1).min(game_ids.len()),
                game_ids.len()
            ),
            false,
        );
        let rows = usize::from(block.inner(area).height).max(1);
        let first = (selected_index + 1).saturating_sub(rows);
        self.games_area.set(area);
        self.games_first.set(first);
        let lines: Vec<Line<'_>> = game_ids
            .iter()
            .enumerate()
            .skip(first)
            .take(rows)
            .map(|(index, id)| {
                let game = &self.games[*id];
                let opponent = game
                    .state
                    .as_ref()
                    .and_then(|state| state.opponent.as_deref())
                    .or(game.opponent.as_deref())
                    .unwrap_or("?");
                let selected = index == selected_index;
                let marker = if selected { '▶' } else { ' ' };
                let mut label = format!("{marker} {id:<6} vs {opponent:<8}");
                if let Some(state) = &game.state {
                    let (black, white) = absolute_counts(state);
                    label += &format!("  ● {black:>2}-{white:<2} ○");
                    if let Some(color) = state.my_color {
                        label += &format!("  you {}", disc_mark(color));
                    }
                }
                if let Some(eval) = game.eval {
                    label += &format!("  eval {:>+3}", eval.round() as i32);
                }
                let status = game_status(game);
                if !status.is_empty() {
                    label += &format!("  {status}");
                }
                let mut style = Style::default().fg(result_color(game));
                if selected {
                    style = style
                        .add_modifier(Modifier::BOLD)
                        .bg(Color::Rgb(30, 40, 50));
                }
                Line::styled(visible(&label), style)
            })
            .collect();
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_board(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &MatchState,
        last: Option<Square>,
    ) {
        let block = panel(
            Line::from(vec![
                Span::raw(" Board  "),
                Span::styled("●", Style::default().fg(BLACK_DISC).bg(BOARD_DARK)),
                Span::raw(" Black  "),
                Span::styled("●", Style::default().fg(WHITE_DISC).bg(BOARD_DARK)),
                Span::raw(" White "),
            ]),
            false,
        );
        let inner = block.inner(area);
        let board = state.board.to_string_as_board(state.side_to_move);
        let label = Style::default().fg(Color::DarkGray);
        let mut lines = vec![Line::from(vec![
            Span::raw("  "),
            Span::styled(" a  b  c  d  e  f  g  h ", label),
        ])];
        for (row, cells) in board.lines().enumerate() {
            let mut spans = vec![Span::styled(format!("{} ", row + 1), label)];
            for (column, cell) in cells.chars().enumerate() {
                let background = if last.is_some_and(|square| square.index() == row * 8 + column) {
                    LAST_MOVE
                } else if (row + column) % 2 == 0 {
                    BOARD_DARK
                } else {
                    Color::Rgb(18, 112, 72)
                };
                let (symbol, foreground) = match cell {
                    'X' => ('●', BLACK_DISC),
                    'O' => ('●', WHITE_DISC),
                    _ => ('·', Color::Rgb(70, 150, 110)),
                };
                spans.push(Span::styled(
                    format!(" {symbol} "),
                    Style::default()
                        .fg(foreground)
                        .bg(background)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            lines.push(Line::from(spans));
        }
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_facts(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        id: &str,
        game: &DisplayGame,
        state: &MatchState,
        selected: bool,
    ) {
        let (black, white) = absolute_counts(state);
        let block = panel(
            format!(
                " Game {id} · vs {} · {} ",
                opponent_name(state, game),
                variant_name(state, game)
            ),
            selected,
        );
        let inner = block.inner(area);
        let status = game_status(game);
        let status_color = if game.result.is_some() {
            result_color(game)
        } else {
            ACCENT
        };
        let mut lines = vec![
            fact_line(
                "Status",
                if status.is_empty() { "Idle" } else { &status },
                status_color,
            ),
            fact_line("Turn", turn_label(state), Color::Yellow),
            fact_line("Discs", format!("● {black}  ○ {white}"), Color::White),
        ];
        lines.extend(clock_lines(state, game));
        // `<id> <move> <score> | <search stats> | ...`
        let report = game
            .last_search
            .as_deref()
            .map(|report| report.strip_prefix(id).unwrap_or(report).trim_start());
        let mut parts = report.map_or(vec!["None yet"], |report| report.split(" | ").collect());
        let eval = parts.remove(0);
        let eval_color = match game.eval {
            Some(eval) if eval > 0.0 => Color::Green,
            Some(eval) if eval < 0.0 => Color::Red,
            _ => Color::White,
        };
        lines.push(fact_line("Eval", eval, eval_color));
        // depth · nodes, then time · speed
        for (label, chunk) in ["Search", "Time"].into_iter().zip(parts.chunks(2)) {
            lines.push(fact_line(label, chunk.join(" · "), Color::DarkGray));
        }
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_compact_game(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        id: &str,
        game: &DisplayGame,
        state: &MatchState,
    ) {
        let (black, white) = absolute_counts(state);
        let clocks: Vec<String> = clock_lines(state, game)
            .iter()
            .map(ToString::to_string)
            .collect();
        let lines = vec![
            Line::from(format!(
                "Opponent: {}   {}",
                visible(opponent_name(state, game)),
                visible(&game_status(game))
            )),
            Line::from(format!(
                "You: {}   Turn: {}   Discs: X {black} O {white}",
                disc_name(state.my_color),
                turn_label(state)
            )),
            Line::from(clocks.join("   ")),
            Line::from(format!(
                "Eval: {}   Result: {}",
                game.eval
                    .map_or("-".into(), |eval| format!("{:+}", eval.round() as i32)),
                visible(&result_text(game))
            )),
        ];
        frame.render_widget(
            Paragraph::new(lines).block(panel(
                format!(" Game {id} · board hidden at this size "),
                false,
            )),
            area,
        );
    }

    fn draw_waiting_game(&self, frame: &mut Frame<'_>, area: Rect, id: &str, game: &DisplayGame) {
        let lines = vec![
            Line::from("No board snapshot is available yet."),
            Line::from(format!(
                "Opponent: {}   Variant: {}",
                visible(game.opponent.as_deref().unwrap_or("Unknown")),
                visible(game.variant.as_deref().unwrap_or("Unknown"))
            )),
            Line::from(visible(&game_status(game))),
        ];
        frame.render_widget(
            Paragraph::new(lines).block(panel(format!(" Game {id} "), true)),
            area,
        );
    }

    fn draw_help(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.width < 20 || area.height < 5 {
            return;
        }
        let width = area.width.saturating_sub(4).min(66);
        let height = area.height.saturating_sub(2).min(15);
        let popup = Rect::new(
            area.x + (area.width - width) / 2,
            area.y + (area.height - height) / 2,
            width,
            height,
        );
        let lines = vec![
            Line::from("Enter             Send the editor contents"),
            Line::from("Up / Down         Command history"),
            Line::from("Tab / F6          Next game; Shift+Tab previous"),
            Line::from("PgUp / PgDn       Scroll transcript"),
            Line::from("Alt+Left / Right  Scroll transcript horizontally"),
            Line::from("Ctrl+Home / End   Transcript start / live output"),
            Line::from("Ctrl+U            Clear the editor"),
            Line::from("Ctrl+C            Exit"),
            Line::from("Mouse wheel       Scroll transcript; select game over the list"),
            Line::from("Click             Select a game; click or drag the scrollbar"),
            Line::from("Shift+drag        Select text to copy"),
            Line::from("Paste             Pasted newlines may send commands"),
        ];
        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(panel(" Keyboard help · F1 close ", true)),
            popup,
        );
    }

    fn draw_editor(&self, frame: &mut Frame<'_>, area: Rect) {
        let block = panel(" GGS command ", true);
        let inner = block.inner(area);
        let text = visible(&self.editor.text);
        let cursor_width = width(&visible(&self.editor.text[..self.editor.cursor]));
        let offset = cursor_width.saturating_sub(usize::from(inner.width).saturating_sub(1));
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(clip(&text, offset, usize::from(inner.width))),
            inner,
        );
        if inner.width > 0 && inner.height > 0 {
            frame.set_cursor_position((inner.x + (cursor_width - offset) as u16, inner.y));
        }
    }
}

/// The result, or what the engine is doing; empty while idle.
fn game_status(game: &DisplayGame) -> String {
    if game.result.is_some() {
        result_text(game)
    } else {
        game.activity.label().unwrap_or_default().to_string()
    }
}

const ACCENT: Color = Color::Cyan;
const BOARD_DARK: Color = Color::Rgb(10, 96, 62);
const LAST_MOVE: Color = Color::Rgb(176, 132, 20);
const BLACK_DISC: Color = Color::Rgb(16, 16, 16);
const WHITE_DISC: Color = Color::Rgb(240, 240, 240);

/// A rounded panel, outlined in the accent color when `focused`.
fn panel<'a>(title: impl Into<Line<'a>>, focused: bool) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if focused { ACCENT } else { Color::DarkGray }))
        .title(title)
}

fn fact_line(label: &str, value: impl AsRef<str>, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(" {label:<8} "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(visible(value.as_ref()), Style::default().fg(value_color)),
    ])
}

fn disc_name(disc: Option<Disc>) -> &'static str {
    match disc {
        Some(Disc::Black) => "● Black",
        Some(Disc::White) => "○ White",
        Some(Disc::Empty) => "Empty",
        None => "Spectator / unknown",
    }
}

fn disc_mark(disc: Disc) -> &'static str {
    match disc {
        Disc::Black => "●",
        Disc::White => "○",
        Disc::Empty => "·",
    }
}

fn absolute_counts(state: &MatchState) -> (u32, u32) {
    let player = state.board.get_player_count();
    let opponent = state.board.get_opponent_count();
    match state.side_to_move {
        reversi_core::disc::Disc::Black => (player, opponent),
        reversi_core::disc::Disc::White => (opponent, player),
        reversi_core::disc::Disc::Empty => {
            let board = state.board.to_string_as_board(state.side_to_move);
            (
                board.chars().filter(|cell| *cell == 'X').count() as u32,
                board.chars().filter(|cell| *cell == 'O').count() as u32,
            )
        }
    }
}

fn opponent_name<'a>(state: &'a MatchState, game: &'a DisplayGame) -> &'a str {
    state
        .opponent
        .as_deref()
        .or(game.opponent.as_deref())
        .unwrap_or("Unknown")
}

fn variant_name<'a>(state: &'a MatchState, game: &'a DisplayGame) -> &'a str {
    state
        .variant
        .as_deref()
        .or(game.variant.as_deref())
        .unwrap_or("Unknown")
}

/// Both players' clocks as of the last server snapshot, then the time since it.
/// Clocks are not counted down locally: a synchro game's clock stays paused
/// until both of its pair's moves arrive (`Match::play`).
fn clock_lines(state: &MatchState, game: &DisplayGame) -> Vec<Line<'static>> {
    if game.clocks.is_empty() {
        return vec![fact_line("Clock", "Unavailable", Color::White)];
    }
    let mut lines: Vec<_> = game
        .clocks
        .iter()
        .map(|clock| {
            let running = game.result.is_none() && clock.color == state.side_to_move;
            let mut value = clock.main_ms.map_or_else(|| "??:??".into(), format_clock);
            if Some(clock.color) == state.my_color
                && let Some(mine) = state.my_clock
                && mine.increment_ms > 0
            {
                value += &format!(" +{}s/move", mine.increment_ms as f64 / 1_000.0);
            }
            if running {
                value += "  ◀";
            }
            fact_line(
                &format!("{} {}", disc_mark(clock.color), clock.name),
                value,
                if running { Color::Yellow } else { Color::White },
            )
        })
        .collect();
    if game.result.is_none()
        && let Some(at) = game.clocks_at
    {
        let elapsed = format_clock(at.elapsed().as_secs() * 1_000);
        lines.push(fact_line("Elapsed", elapsed, Color::Yellow));
    }
    lines
}

fn result_text(game: &DisplayGame) -> String {
    match (game.score, &game.result) {
        (Some(score), _) if score > 0.0 => format!("Win {score:+}"),
        (Some(score), _) if score < 0.0 => format!("Loss {score:+}"),
        (Some(_), _) => "Draw".into(),
        (None, Some(result)) => result.clone(),
        (None, None) => "In progress".into(),
    }
}

fn result_color(game: &DisplayGame) -> Color {
    match game.score {
        Some(score) if score > 0.0 => Color::Green,
        Some(score) if score < 0.0 => Color::Red,
        Some(_) => Color::Yellow,
        None => Color::Gray,
    }
}

fn format_clock(milliseconds: u64) -> String {
    let seconds = milliseconds.div_ceil(1_000);
    if seconds >= 3_600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3_600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else {
        format!("{:02}:{:02}", seconds / 60, seconds % 60)
    }
}

fn turn_label(state: &MatchState) -> String {
    let suffix = match state.my_color {
        Some(color) if color == state.side_to_move => " (you)",
        Some(_) => " (opponent)",
        None => " (watching)",
    };
    format!("{}{suffix}", disc_name(Some(state.side_to_move)))
}

fn transcript_style(line: &str) -> Style {
    let lower = line.to_ascii_lowercase();
    if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if line.starts_with("> ") {
        Style::default().fg(Color::Cyan)
    } else if line.starts_with("[ggs]") {
        Style::default().fg(Color::Yellow)
    } else if line.starts_with('|') {
        Style::default().fg(Color::Gray)
    } else {
        Style::default()
    }
}

#[derive(Default)]
struct Editor {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    draft: String,
}

impl Editor {
    fn insert(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('u')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.text.clear();
                self.cursor = 0;
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    || key
                        .modifiers
                        .contains(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(&character.to_string())
            }
            KeyCode::Left => self.cursor = self.previous(),
            KeyCode::Right => self.cursor = self.next(),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.text.len(),
            KeyCode::Backspace if self.cursor > 0 => {
                let previous = self.previous();
                self.text.replace_range(previous..self.cursor, "");
                self.cursor = previous;
            }
            KeyCode::Delete if self.cursor < self.text.len() => {
                self.text.replace_range(self.cursor..self.next(), "");
            }
            KeyCode::Up => self.history_up(),
            KeyCode::Down => self.history_down(),
            _ => {}
        }
    }

    fn previous(&self) -> usize {
        self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(index, _)| index)
    }
    fn next(&self) -> usize {
        self.cursor
            + self.text[self.cursor..]
                .chars()
                .next()
                .map_or(0, char::len_utf8)
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_index {
            None => {
                self.draft = self.text.clone();
                self.history.len() - 1
            }
            Some(index) => index.saturating_sub(1),
        };
        self.history_index = Some(index);
        self.text.clone_from(&self.history[index]);
        self.cursor = self.text.len();
    }

    fn history_down(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.history.len() {
            self.history_index = Some(index + 1);
            self.text.clone_from(&self.history[index + 1]);
        } else {
            self.history_index = None;
            self.text.clone_from(&self.draft);
        }
        self.cursor = self.text.len();
    }

    fn submit(&mut self) -> Vec<String> {
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.history_index = None;
        self.draft.clear();
        self.history.push(text.clone());
        text.split('\n').map(str::to_string).collect()
    }
}

fn visible(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for character in text.chars() {
        if character.is_control() {
            result.extend(character.escape_default());
        } else {
            result.push(character);
        }
    }
    result
}

fn width(text: &str) -> usize {
    Line::raw(text).width()
}

fn clip(text: &str, offset: usize, maximum: usize) -> String {
    let mut position: usize = 0;
    let mut result = String::new();
    let end = offset.saturating_add(maximum);
    let line = Line::raw(text);
    for grapheme in line.styled_graphemes(Style::default()) {
        let size = width(grapheme.symbol);
        let right = position.saturating_add(size);
        if position >= offset && right <= end {
            result.push_str(grapheme.symbol);
        } else if right > offset && position < end {
            result.extend(std::iter::repeat_n(
                ' ',
                right.min(end) - position.max(offset),
            ));
        }
        position = right;
        if position >= end {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use reversi_core::board::Board;

    use crate::ggs::protocol::PlayerClock;

    fn key(model: &mut Model, code: KeyCode, modifiers: KeyModifiers) -> Input {
        model.input(event::Event::Key(KeyEvent::new(code, modifiers)))
    }

    fn screen(model: &mut Model, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| model.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                let mut line = String::new();
                for x in 0..width {
                    line.push_str(buffer.cell((x, y)).unwrap().symbol());
                }
                line.trim_end().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn game(id: &str) -> DisplayGame {
        DisplayGame {
            state: Some(MatchState {
                id: id.into(),
                variant: Some("8b".into()),
                opponent: Some("bob".into()),
                board: Board::new(),
                side_to_move: Disc::Black,
                my_color: Some(Disc::Black),
                my_clock: Some(PlayerClock {
                    main_ms: 299_125,
                    increment_ms: 5_000,
                }),
            }),
            ..DisplayGame::default()
        }
    }

    #[test]
    fn unicode_middle_edits_history_and_draft_preserve_commands() {
        let mut model = Model::new("test".into());
        model.input(event::Event::Paste(" tell bob 日本語  ".into()));
        key(&mut model, KeyCode::Left, KeyModifiers::NONE);
        key(&mut model, KeyCode::Left, KeyModifiers::NONE);
        key(&mut model, KeyCode::Backspace, KeyModifiers::NONE);
        key(&mut model, KeyCode::Char('界'), KeyModifiers::NONE);
        let Input::Commands(lines) = key(&mut model, KeyCode::Enter, KeyModifiers::NONE) else {
            panic!("expected command");
        };
        assert_eq!(lines, [" tell bob 日本界  "]);
        model.input(event::Event::Paste("current draft".into()));
        key(&mut model, KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(model.editor.text, " tell bob 日本界  ");
        key(&mut model, KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(model.editor.text, "current draft");
        assert_eq!(model.editor.cursor, "current draft".len());
    }

    #[test]
    fn paste_waits_for_enter_and_preserves_blank_lines_and_spaces() {
        let mut model = Model::new("test".into());
        assert!(matches!(
            model.input(event::Event::Paste(" help  \r\n\r\nquit\n".into())),
            Input::None
        ));
        let Input::Commands(lines) = key(&mut model, KeyCode::Enter, KeyModifiers::NONE) else {
            panic!("expected command");
        };
        assert_eq!(lines, [" help  ", "", "quit", ""]);
        let Input::Commands(lines) = key(&mut model, KeyCode::Enter, KeyModifiers::NONE) else {
            panic!("expected blank command");
        };
        assert_eq!(lines, [""]);
    }

    #[test]
    fn altgr_formula_characters_are_forwarded_and_crlf_does_not_send_twice() {
        let mut model = Model::new("test".into());
        for character in "|@\\{".chars() {
            key(
                &mut model,
                KeyCode::Char(character),
                KeyModifiers::CONTROL | KeyModifiers::ALT,
            );
        }
        let Input::Commands(lines) = key(&mut model, KeyCode::Enter, KeyModifiers::NONE) else {
            panic!("expected command");
        };
        assert_eq!(lines, ["|@\\{"]);
        assert!(matches!(
            key(&mut model, KeyCode::Char('j'), KeyModifiers::CONTROL),
            Input::None
        ));
        assert!(model.editor.text.is_empty());
    }

    #[test]
    fn crlf_and_lf_only_keystroke_paste_send_each_line_once() {
        for lf_code in [KeyCode::Char('j'), KeyCode::Enter] {
            for crlf in [false, true] {
                let mut model = Model::new("test".into());
                let mut commands = Vec::new();
                for text in [" first ", "", "last"] {
                    for character in text.chars() {
                        key(&mut model, KeyCode::Char(character), KeyModifiers::NONE);
                    }
                    if crlf {
                        let Input::Commands(lines) =
                            key(&mut model, KeyCode::Enter, KeyModifiers::NONE)
                        else {
                            panic!("CR must submit");
                        };
                        commands.extend(lines);
                        model.input(event::Event::Key(KeyEvent::new_with_kind(
                            KeyCode::Enter,
                            KeyModifiers::NONE,
                            KeyEventKind::Release,
                        )));
                        assert!(matches!(
                            key(&mut model, lf_code, KeyModifiers::CONTROL),
                            Input::None
                        ));
                    } else {
                        let Input::Commands(lines) =
                            key(&mut model, lf_code, KeyModifiers::CONTROL)
                        else {
                            panic!("LF must submit");
                        };
                        commands.extend(lines);
                    }
                }
                assert_eq!(commands, [" first ", "", "last"]);
            }
        }
    }

    #[test]
    fn clipped_unicode_preserves_graphemes_padding_and_editor_cursor() {
        assert_eq!(clip("日本A", 1, 4), " 本A");
        assert_eq!(clip("A日本", 0, 4), "A日 ");
        assert_eq!(clip("ae\u{301}👩‍💻z", 1, 3), "e\u{301}👩‍💻");
        let mut model = Model::new("test".into());
        model.editor.insert(&"日本".repeat(20));
        let mut terminal = Terminal::new(TestBackend::new(20, 12)).unwrap();
        terminal.draw(|frame| model.draw(frame)).unwrap();
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!(cursor.x, 18);
        assert_eq!(
            terminal
                .backend()
                .buffer()
                .cell((1, cursor.y))
                .unwrap()
                .symbol(),
            " "
        );
        assert_eq!(
            terminal
                .backend()
                .buffer()
                .cell((2, cursor.y))
                .unwrap()
                .symbol(),
            "日"
        );
    }

    #[test]
    fn incoming_output_keeps_scrolled_view_and_editing_cursor() {
        let mut model = Model::new("test".into());
        model.lines = (0..90).map(|index| format!("line {index:03}")).collect();
        model.editor.insert("日本 draft");
        screen(&mut model, 100, 30);
        key(&mut model, KeyCode::PageUp, KeyModifiers::NONE);
        let top = model.top();
        let cursor = model.editor.cursor;
        model
            .lines
            .extend(["new server line".into(), "[ggs] diagnostic".into()]);
        let view = screen(&mut model, 100, 30);
        assert_eq!(model.top(), top);
        assert!(view.contains(&format!("line {top:03}")));
        assert_eq!(model.editor.cursor, cursor);
        assert_eq!(model.editor.text, "日本 draft");
        key(&mut model, KeyCode::End, KeyModifiers::CONTROL);
        assert!(screen(&mut model, 100, 30).contains("new server line"));
    }

    #[test]
    fn capped_transcript_keeps_the_scrolled_view_on_its_line() {
        let mut model = Model::new("test".into());
        for index in 0..MAX_TRANSCRIPT_LINES {
            model.push_line(format!("line {index}"));
        }
        screen(&mut model, 100, 30);
        key(&mut model, KeyCode::PageUp, KeyModifiers::NONE);
        let viewed = model.lines[model.top()].clone();

        model.push_line("newest".into());
        assert_eq!(model.lines.len(), MAX_TRANSCRIPT_LINES);
        assert_eq!(model.lines[0], "line 1");
        assert_eq!(model.lines[model.top()], viewed);
    }

    #[test]
    fn long_transcript_lines_scroll_and_control_bytes_render_as_text() {
        let mut model = Model::new("test".into());
        let raw = format!("{}TAIL\x1b[2J\x07", "prefix".repeat(20));
        model.lines.push_back(raw.clone());
        model.horizontal = 120;
        let view = screen(&mut model, 80, 25);
        assert!(view.contains("TAIL\\u{1b}[2J\\u{7}"));
        assert_eq!(model.lines, [raw]);
        assert!(!view.contains('\x1b'));
    }

    #[test]
    fn board_uses_absolute_colors_and_narrow_resize_keeps_editor() {
        let mut model = Model::new("test".into());
        model.games.insert(".5".into(), game(".5"));
        model.editor.insert("tell bob hello");
        let wide = screen(&mut model, 110, 30);
        assert!(wide.contains("Board  ● Black  ● White"));
        assert!(wide.contains("4  ·  ·  ·  ●  ●  ·  ·  ·"));
        assert!(wide.contains("Turn     ● Black (you)"));
        assert!(wide.contains("Discs    ● 2  ○ 2"));
        // d4 is white and e4 black, whatever the side to move.
        let row = wide.lines().position(|line| line.contains("4  ·")).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(110, 30)).unwrap();
        terminal.draw(|frame| model.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((13, row as u16)).unwrap().fg, WHITE_DISC);
        assert_eq!(buffer.cell((16, row as u16)).unwrap().fg, BLACK_DISC);
        let narrow = screen(&mut model, 30, 12);
        assert!(narrow.contains("tell bob hello"));
        assert!(narrow.contains("Transcript"));
        assert!(narrow.contains("Enter send  Ctrl+C exit"));
        assert!(screen(&mut model, 110, 30).contains("4  ·  ·  ·  ●  ●"));
    }

    #[test]
    fn synchro_selection_end_and_recycled_ids_do_not_reuse_old_boards() {
        let mut model = Model::new("test".into());
        for id in [".5", ".5.0", ".5.1", ".50"] {
            model.games.insert(id.into(), game(id));
        }
        key(&mut model, KeyCode::F(6), KeyModifiers::NONE);
        assert!(screen(&mut model, 110, 30).contains("Game .5.0"));
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5.0".into(),
            result: "+4.00".into(),
            score: None,
        });
        let ended = screen(&mut model, 110, 30);
        assert!(ended.contains("Status   +4.00"));
        assert!(model.games[".5.1"].result.is_none());
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5".into(),
            result: "+2.00".into(),
            score: None,
        });
        assert_eq!(model.games[".5.1"].result.as_deref(), Some("+2.00"));
        assert!(model.games[".50"].result.is_none());
        model.on_os_event(&OsEvent::MatchCreated {
            id: ".5".into(),
            opponent: Some("bob".into()),
            variant: Some("s8".into()),
        });
        assert!(!model.games.contains_key(".5.0"));
        assert!(!model.games.contains_key(".5.1"));
        assert!(model.games[".5"].state.is_none());
        assert!(model.games[".5"].result.is_none());
    }

    #[test]
    fn earlier_game_arrival_keeps_the_selected_board() {
        let mut model = Model::new("test".into());
        for id in [".5", ".6"] {
            model.games.insert(id.into(), game(id));
        }
        key(&mut model, KeyCode::F(6), KeyModifiers::NONE);
        assert!(screen(&mut model, 110, 30).contains("Game .6"));
        model.on_os_event(&OsEvent::MatchCreated {
            id: ".1".into(),
            opponent: Some("carol".into()),
            variant: Some("8b".into()),
        });
        assert!(screen(&mut model, 110, 30).contains("Game .6"));
        key(&mut model, KeyCode::BackTab, KeyModifiers::SHIFT);
        assert!(screen(&mut model, 110, 30).contains("Game .5"));
    }

    #[test]
    fn completed_game_stays_ended_when_ponder_stop_arrives() {
        let mut model = Model::new("test".into());
        model.games.insert(".5".into(), game(".5"));
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5".into(),
            result: "+4.00".into(),
            score: None,
        });
        model.on_action(&SessionAction::StopPonder {
            game_id: ".5".into(),
        });
        let view = screen(&mut model, 110, 30);
        assert!(view.contains("Status   +4.00"));
        assert!(!view.contains("Ponder stopping"));
    }

    #[test]
    fn idle_and_narrow_screens_keep_commands_and_exit_visible() {
        let mut model = Model::new("GGS alice@example:5000".into());
        model.lines.push_back("Welcome to GGS".into());
        model.editor.insert("tell bob hello");

        let idle = screen(&mut model, 80, 25);
        assert!(idle.contains("No active board"));
        assert!(idle.contains("Try: help    tell /os who"));
        assert!(idle.contains("Chat: tell <user> <text>"));
        assert!(idle.contains("Welcome to GGS"));
        assert!(idle.contains("GGS command"));
        assert!(idle.contains("Enter send  Ctrl+C exit  F1 help"));

        let narrow = screen(&mut model, 30, 12);
        assert!(narrow.contains("No board yet. Try: help"));
        assert!(narrow.contains("Transcript"));
        assert!(narrow.contains("tell bob hello"));
        assert!(narrow.contains("Enter send  Ctrl+C exit"));
    }

    #[test]
    fn eighty_column_game_shows_board_and_each_key_fact() {
        let mut model = Model::new("GGS alice@example:5000".into());
        model.games.insert(".5".into(), game(".5"));
        model.selected = Some(".5".into());
        model.snapshot(DisplayState {
            state: game(".5").state.unwrap(),
            clocks: Vec::new(),
            moves: vec![(1, "f5".into()), (2, "d6".into()), (3, "c3".into())],
        });
        model.snapshot(DisplayState {
            state: game(".5").state.unwrap(),
            clocks: vec![
                DisplayClock {
                    name: "alice".into(),
                    color: Disc::Black,
                    main_ms: Some(299_000),
                },
                DisplayClock {
                    name: "bob".into(),
                    color: Disc::White,
                    main_ms: Some(250_000),
                },
            ],
            moves: vec![(4, "d3".into())],
        });
        model.games.get_mut(".5").unwrap().clocks_at =
            Instant::now().checked_sub(Duration::from_millis(1_001));
        let game = model.games.get_mut(".5").unwrap();
        game.activity = Activity::Searching;
        game.eval = Some(2.0);
        game.last_search =
            Some(".5 D3 +2 | depth 18@87% | 123.4M nodes | 2.34s (wait 0.12s) | 52.7 Mnps".into());
        model.lines.extend([
            "> tell /os who".into(),
            "[ggs] engine search started".into(),
            "/os: error .5 illegal move".into(),
        ]);
        model.editor.insert("tell bob good game");

        let view = screen(&mut model, 80, 25);
        for expected in [
            "▶ .5     vs bob       ●  2-2  ○  you ●  eval  +2  Searching",
            "Board  ● Black  ● White",
            "Game .5 · vs bob · 8b",
            "Status   Searching",
            "Turn     ● Black (you)",
            "Discs    ● 2  ○ 2",
            "● alice  04:59 +5s/move  ◀",
            "○ bob    04:10",
            "Elapsed  00:01",
            "Eval     D3 +2",
            "Search   depth 18@87% · 123.4M nodes",
            "Time     2.34s (wait 0.12s) · 52.7 Mnps",
            "tell bob good game",
            "Enter send  Ctrl+C exit  F1 help",
        ] {
            assert!(view.contains(expected), "missing {expected:?}\n{view}");
        }
        let mut terminal = Terminal::new(TestBackend::new(80, 25)).unwrap();
        terminal.draw(|frame| model.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        let d3 = buffer.cell((13, 8)).unwrap();
        assert_eq!(
            view.lines().nth(5).unwrap().chars().position(|c| c == 'd'),
            Some(13)
        );
        assert_eq!(d3.bg, LAST_MOVE);
        assert_ne!(buffer.cell((13, 9)).unwrap().bg, LAST_MOVE);
    }

    #[test]
    fn synchro_games_show_both_results_from_our_side_without_switching() {
        let mut model = Model::new("test".into());
        for id in [".5.0", ".5.1"] {
            model.games.insert(id.into(), game(id));
        }
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5.0".into(),
            result: "( bob vs. alice ) +4.00".into(),
            score: Some(-4.0),
        });
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5.1".into(),
            result: "( bob vs. alice ) -10.00".into(),
            score: Some(10.0),
        });
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5".into(),
            result: "-3.00".into(),
            score: Some(3.0),
        });
        model.lines.push_back("bob: good game".into());
        let view = screen(&mut model, 80, 24);
        assert!(view.contains("bob: good game"), "{view}");
        let tall = screen(&mut model, 80, 40);
        assert!(
            tall.contains("Game .5.0") && tall.contains("Game .5.1"),
            "{tall}"
        );
        assert!(
            view.contains(".5.0   vs bob       ●  2-2  ○  you ●  Loss -4"),
            "{view}"
        );
        assert!(
            view.contains(".5.1   vs bob       ●  2-2  ○  you ●  Win +10"),
            "{view}"
        );

        model.on_os_event(&OsEvent::MatchCreated {
            id: ".6".into(),
            opponent: Some("carol".into()),
            variant: Some("8".into()),
        });
        assert_eq!(model.games.keys().collect::<Vec<_>>(), [".6"]);
    }

    #[test]
    fn synchro_parent_yields_focus_to_children_without_stealing_it_later() {
        let mut model = Model::new("test".into());
        model.on_os_event(&OsEvent::MatchCreated {
            id: ".20".into(),
            opponent: Some("alice".into()),
            variant: Some("s8".into()),
        });
        assert_eq!(model.selected.as_deref(), Some(".20"));

        let first = game(".20.0").state.unwrap();
        model.snapshot(DisplayState {
            state: first,
            clocks: Vec::new(),
            moves: Vec::new(),
        });
        assert_eq!(model.selected.as_deref(), Some(".20.0"));

        let second = game(".20.1").state.unwrap();
        model.snapshot(DisplayState {
            state: second,
            clocks: Vec::new(),
            moves: Vec::new(),
        });
        assert_eq!(model.selected.as_deref(), Some(".20.0"));
        let view = screen(&mut model, 120, 30);
        assert!(view.contains("▶ .20.0  vs bob       ●  2-2  ○  you ●"));
        assert!(view.contains("  .20.1  vs bob       ●  2-2  ○  you ●"));
        assert!(!view.contains(".20 vs alice"));

        key(&mut model, KeyCode::F(6), KeyModifiers::NONE);
        assert_eq!(model.selected.as_deref(), Some(".20.1"));
    }

    #[test]
    fn worker_events_and_duplicate_snapshots_preserve_newer_activity() {
        let mut model = Model::new("test".into());
        model.games.insert(".5".into(), game(".5"));
        model.on_action(&SessionAction::StartPonder {
            game_id: ".5".into(),
            search_id: 0,
            board: Board::new(),
        });
        model.worker_started(".5", 0, true);
        model.snapshot(DisplayState {
            state: game(".5").state.unwrap(),
            clocks: Vec::new(),
            moves: Vec::new(),
        });
        assert_eq!(model.games[".5"].activity, Activity::Pondering);

        model.on_action(&SessionAction::StopPonder {
            game_id: ".5".into(),
        });
        model.on_action(&SessionAction::StartSearch {
            game_id: ".5".into(),
            search_id: 1,
            board: Board::new(),
            search_limit: session::SearchLimit::Level { level: 1 },
        });
        model.worker_finished(".5", 0);
        assert_eq!(model.games[".5"].activity, Activity::SearchQueued);

        model.worker_started(".5", 1, false);
        model.snapshot(DisplayState {
            state: game(".5").state.unwrap(),
            clocks: Vec::new(),
            moves: Vec::new(),
        });
        assert_eq!(model.games[".5"].activity, Activity::Searching);

        model.on_action(&SessionAction::StartPonder {
            game_id: ".5".into(),
            search_id: 2,
            board: Board::new(),
        });
        model.worker_finished(".5", 1);
        assert_eq!(model.games[".5"].activity, Activity::PonderQueued);
    }

    #[test]
    fn selected_game_remains_visible_when_tabs_exceed_the_screen_width() {
        let mut model = Model::new("test".into());
        for id in [".5", ".6", ".7", ".8"] {
            model.games.insert(id.into(), game(id));
        }
        model.selected = Some(".8".into());
        let view = screen(&mut model, 80, 25);
        assert!(view.contains("▶ .8     vs bob"));
        assert!(view.contains("Game .8"));
        assert!(view.contains("Games 4/4"));
    }

    #[test]
    fn worker_that_cannot_start_does_not_remain_queued() {
        let mut model = Model::new("test".into());
        model.games.insert(".5".into(), game(".5"));
        for activity in [Activity::SearchQueued, Activity::PonderQueued] {
            let game = model.games.get_mut(".5").unwrap();
            game.activity = activity;
            game.worker = Some(0);
            model.worker_finished(".5", 0);
            let view = screen(&mut model, 80, 25);
            assert!(!view.contains("Status:"));
            assert!(!view.contains("queued"));
        }
    }

    #[test]
    fn late_snapshot_and_worker_events_do_not_reopen_an_ended_game() {
        let mut model = Model::new("test".into());
        model.games.insert(".5".into(), game(".5"));
        model.on_os_event(&OsEvent::MatchEnd {
            id: ".5".into(),
            result: "+4.00".into(),
            score: None,
        });
        model.snapshot(DisplayState {
            state: game(".5").state.unwrap(),
            clocks: Vec::new(),
            moves: Vec::new(),
        });
        model.games.get_mut(".5").unwrap().worker = Some(0);
        model.worker_started(".5", 0, false);
        model.worker_finished(".5", 0);
        assert_eq!(model.games[".5"].result.as_deref(), Some("+4.00"));
        assert_eq!(game_status(&model.games[".5"]), "+4.00");
    }

    #[test]
    fn help_overlay_does_not_capture_typed_commands() {
        let mut model = Model::new("test".into());
        key(&mut model, KeyCode::F(1), KeyModifiers::NONE);
        assert!(screen(&mut model, 80, 25).contains("Keyboard help · F1 close"));
        key(&mut model, KeyCode::Char('x'), KeyModifiers::NONE);
        assert!(!model.help_open);
        assert_eq!(model.editor.text, "x");
    }

    #[test]
    fn transcript_categories_have_distinct_styles_without_changing_text() {
        let cases = [
            ("> help", Color::Cyan),
            ("[ggs] search report", Color::Yellow),
            ("/os: error .5 illegal move", Color::Red),
            ("raw server line", Color::Reset),
        ];
        let mut model = Model::new("test".into());
        model.lines = cases.iter().map(|(text, _)| text.to_string()).collect();
        let mut terminal = Terminal::new(TestBackend::new(80, 25)).unwrap();
        terminal.draw(|frame| model.draw(frame)).unwrap();
        for (text, color) in cases {
            let row = terminal
                .backend()
                .buffer()
                .content()
                .chunks(80)
                .find(|row| {
                    row.iter()
                        .map(|cell| cell.symbol())
                        .collect::<String>()
                        .contains(text)
                })
                .unwrap();
            assert!(row[1..=text.len()].iter().all(|cell| cell.fg == color));
        }
        assert_eq!(model.lines, cases.map(|(text, _)| text));
    }

    #[test]
    fn mouse_scrolls_transcript_and_selects_games() {
        let mouse = |kind, column, row| {
            event::Event::Mouse(MouseEvent {
                kind,
                column,
                row,
                modifiers: KeyModifiers::NONE,
            })
        };
        let mut model = Model::new("test".into());
        for id in [".5.0", ".5.1"] {
            model.games.insert(id.into(), game(id));
        }
        model.selected = Some(".5.0".into());
        model.lines = (0..200).map(|index| format!("line {index:03}")).collect();
        let view = screen(&mut model, 80, 30);
        assert!(view.contains('▼'), "{view}");
        let transcript = model.transcript_area;

        model.input(mouse(MouseEventKind::ScrollUp, 10, transcript.y));
        assert!(matches!(model.scroll, Scroll::Row(row) if row == 200 - model.page_height - 3));
        model.input(mouse(MouseEventKind::ScrollDown, 10, transcript.y));
        assert!(matches!(model.scroll, Scroll::Follow));

        let scrollbar = transcript.right();
        model.input(mouse(
            MouseEventKind::Down(MouseButton::Left),
            scrollbar,
            transcript.y,
        ));
        assert!(matches!(model.scroll, Scroll::Row(0)));
        assert!(screen(&mut model, 80, 30).contains("line 000"));
        model.input(mouse(
            MouseEventKind::Drag(MouseButton::Left),
            scrollbar,
            transcript.bottom() - 1,
        ));
        assert!(matches!(model.scroll, Scroll::Follow));
        model.input(mouse(MouseEventKind::Up(MouseButton::Left), scrollbar, 0));

        // Title row, then the list border; the second game is on the third row.
        model.input(mouse(MouseEventKind::Down(MouseButton::Left), 10, 3));
        assert_eq!(model.selected.as_deref(), Some(".5.1"));
        model.input(mouse(MouseEventKind::ScrollDown, 10, 2));
        assert_eq!(model.selected.as_deref(), Some(".5.0"));
    }
}
