//! JSON-lines engine for the `--engine` option of the FlipSight bot connector.
//!
//! The connector starts one process per game and keeps it running, so the
//! transposition table and endgame tracking carry over from move to move.

use std::io::{self, BufRead, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources, StopToken};
use reversi_core::square::Square;
use serde::Deserialize;
use serde_json::json;

use crate::config::EngineConfig;

/// Clock kept back from `remaining_ms` whatever the position.
const CLOCK_RESERVE_MS: u64 = 1000;

/// Clock kept back for each move we have yet to play: the server charges a
/// move's trip over the network to our clock.
const MOVE_RESERVE_MS: u64 = 100;

/// Bank under which a move gets [`LOW_CLOCK_MOVE_MS`] instead of a share of the
/// bank: the sudden-death allocation gives such a bank no search time at all.
const LOW_CLOCK_BANK_MS: u64 = 1000;

/// Search time of a move played from the reserve.
const LOW_CLOCK_MOVE_MS: u64 = 150;

/// How long before the connector's `budget_ms` deadline a search is stopped.
const STOP_MARGIN_MS: u64 = 300;

/// Most search threads one game may use on FlipSight.
const MAX_THREADS: usize = 4;

#[derive(Debug, Deserialize)]
struct Request {
    board: Vec<u8>,
    color: u8,
    legal_moves: Vec<i32>,
    #[serde(default = "standard_mode")]
    game_mode: String,
    #[serde(default = "standard_size")]
    board_size: usize,
    /// The connector forwards the server's value, which may be `null`.
    blocked_squares: Option<Vec<i32>>,
    remaining_ms: u64,
    budget_ms: u64,
    engine_threads: Option<usize>,
}

fn standard_mode() -> String {
    "std".to_string()
}

fn standard_size() -> usize {
    8
}

/// Answers requests from stdin until it closes.
///
/// `process_start` is charged to the first move: the connector launches the
/// engine when that move is already on the clock. With `ponder`, the engine
/// searches the next position while it waits for the following request.
pub fn run(mut config: EngineConfig, ponder: bool, process_start: Instant) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    let mut lines = io::stdin().lock().lines();
    let mut next_line = || (lines.next(), Instant::now());
    let mut engine: Option<Engine> = None;
    let mut first_move = true;
    loop {
        let (line, received) = match engine.as_mut() {
            Some(engine) => engine.ponder_while(&mut next_line),
            None => next_line(),
        };
        let Some(line) = line else {
            break;
        };
        let line = line.map_err(|e| format!("cannot read request: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let started = if std::mem::take(&mut first_move) {
            process_start
        } else {
            received
        };
        let request: Request =
            serde_json::from_str(&line).map_err(|e| format!("invalid request: {e}"))?;
        let board = position(&request)?;
        let engine = engine.get_or_insert_with(|| {
            config.threads = Some(game_threads(config.threads, request.engine_threads));
            Engine::new(&config, ponder)
        });
        let reply = engine.play(&request, &board, started);
        writeln!(stdout, "{reply}")
            .and_then(|()| stdout.flush())
            .map_err(|e| format!("cannot write reply: {e}"))?;
    }
    Ok(())
}

/// Returns the thread count for a game: `--threads`, else the request's
/// `engine_threads`, never more than [`MAX_THREADS`].
fn game_threads(configured: Option<usize>, requested: Option<usize>) -> usize {
    configured
        .or(requested)
        .unwrap_or(MAX_THREADS)
        .min(MAX_THREADS)
}

/// Builds the side-to-move board of a standard 8x8 request.
fn position(request: &Request) -> Result<Board, String> {
    let blocked = request
        .blocked_squares
        .as_ref()
        .is_some_and(|b| !b.is_empty());
    if request.game_mode != "std" || request.board_size != 8 || blocked {
        return Err(format!(
            "unsupported game: mode {} on {}x{}",
            request.game_mode, request.board_size, request.board_size
        ));
    }
    if request.board.len() != 64 {
        return Err(format!("board has {} squares", request.board.len()));
    }
    let opponent_color = match request.color {
        1 => 2,
        2 => 1,
        color => return Err(format!("invalid color {color}")),
    };
    let mut player = Bitboard::new(0);
    let mut opponent = Bitboard::new(0);
    for (sq, &cell) in Square::iter().zip(&request.board) {
        if cell == request.color {
            player = player.set(sq);
        } else if cell == opponent_color {
            opponent = opponent.set(sq);
        } else if cell != 0 {
            return Err(format!("invalid cell {cell} at {sq}"));
        }
    }
    if player.count() + opponent.count() < 4 {
        return Err("at least 4 discs required".to_string());
    }
    Ok(Board::from_bitboards(player, opponent))
}

/// Returns the position the engine faces next after `mv` (-1 for a pass) on
/// `board`: the opponent's turn, or ours again when the opponent must pass.
fn ponder_board(board: &Board, mv: i32) -> Option<Board> {
    let next = if mv < 0 {
        board.switch_players()
    } else {
        let sq = Square::from_u8(u8::try_from(mv).ok()?)?;
        if !board.is_legal_move(sq) {
            return None;
        }
        board.make_move(sq)
    };
    if next.has_legal_moves() {
        Some(next)
    } else {
        Some(next.switch_players()).filter(Board::has_legal_moves)
    }
}

struct Engine {
    search: Search,
    n_threads: usize,
    hash_mb: usize,
    ponder: bool,
    /// The position to search while waiting for the next request.
    ponder_on: Option<Board>,
}

impl Engine {
    fn new(config: &EngineConfig, ponder: bool) -> Self {
        let shared = SearchSharedResources::new(&config.search_options());
        Engine {
            search: Search::from_shared_resources(&shared),
            n_threads: shared.n_threads(),
            hash_mb: config.hash_size,
            ponder,
            ponder_on: None,
        }
    }

    /// Runs `wait` while searching the position after the last reply, and
    /// stops that search once `wait` returns.
    fn ponder_while<T>(&mut self, wait: impl FnOnce() -> T) -> T {
        let Some(board) = self.ponder_on else {
            return wait();
        };
        let stop = StopToken::new();
        let options =
            SearchRunOptions::with_time(TimeControlMode::Infinite).stop_token(stop.clone());
        let search = &mut self.search;
        thread::scope(|scope| {
            scope.spawn(move || search.run(&board, &options));
            let result = wait();
            stop.stop();
            result
        })
    }

    /// Chooses a move for `board`, the position of `request`, whose turn
    /// began at `started`, and returns the reply line.
    fn play(&mut self, request: &Request, board: &Board, started: Instant) -> serde_json::Value {
        let spent_ms = elapsed_ms(started);
        let cap_ms = request.budget_ms.saturating_sub(spent_ms + STOP_MARGIN_MS);
        let mv = match request.legal_moves.as_slice() {
            [] => -1,
            &[only] => only,
            legal => {
                let our_moves = u64::from(board.empty_count().div_ceil(2));
                let bank_ms = request
                    .remaining_ms
                    .saturating_sub(spent_ms + CLOCK_RESERVE_MS + MOVE_RESERVE_MS * our_moves);
                let stop = StopToken::new();
                let options =
                    SearchRunOptions::with_time(clock_mode(bank_ms)).stop_token(stop.clone());
                let best = stop_after(Duration::from_millis(cap_ms), &stop, || {
                    self.search.run(board, &options).best_move()
                });
                best.map(|sq| sq.index() as i32)
                    .filter(|mv| legal.contains(mv))
                    .unwrap_or(legal[0])
            }
        };
        self.ponder_on = if self.ponder {
            ponder_board(board, mv)
        } else {
            None
        };
        json!({
            "move": mv,
            "used_ms": elapsed_ms(started),
            "hard_ms": cap_ms,
            "environment": {
                "engine_threads": self.n_threads,
                "hash_mb": self.hash_mb,
                "ponder": self.ponder,
            },
        })
    }
}

/// Returns the time control for a move with `bank_ms` of clock above the
/// reserve.
fn clock_mode(bank_ms: u64) -> TimeControlMode {
    if bank_ms < LOW_CLOCK_BANK_MS {
        TimeControlMode::Fixed {
            time_per_move_ms: LOW_CLOCK_MOVE_MS,
        }
    } else {
        TimeControlMode::Fischer {
            main_time_ms: bank_ms,
            increment_ms: 0,
        }
    }
}

/// Runs `job`, stopping `stop` if it is still running after `limit`.
fn stop_after<T>(limit: Duration, stop: &StopToken, job: impl FnOnce() -> T) -> T {
    let (done_tx, done_rx) = mpsc::channel::<()>();
    thread::scope(|scope| {
        scope.spawn(move || {
            if done_rx.recv_timeout(limit) == Err(mpsc::RecvTimeoutError::Timeout) {
                stop.stop();
            }
        });
        let result = job();
        drop(done_tx);
        result
    })
}

fn elapsed_ms(since: Instant) -> u64 {
    since.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"board":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"color":1,"legal_moves":[19,26,37,44],"game_mode":"std","board_size":8,"remaining_ms":60000,"budget_ms":10000}"#;

    fn request(line: &str) -> Request {
        serde_json::from_str(line).unwrap()
    }

    fn legal_indices(board: &Board) -> Vec<i32> {
        Square::iter()
            .filter(|&sq| board.is_legal_move(sq))
            .map(|sq| sq.index() as i32)
            .collect()
    }

    fn engine_with(ponder: bool) -> Engine {
        Engine::new(
            &EngineConfig {
                hash_size: 1,
                level: 1,
                threads: Some(1),
                eval_file: None,
                eval_sm_file: None,
            },
            ponder,
        )
    }

    fn engine() -> Engine {
        engine_with(false)
    }

    fn board(player: u64, opponent: u64) -> Board {
        Board::from_bitboards(Bitboard::new(player), Bitboard::new(opponent))
    }

    #[test]
    fn sample_request_maps_squares_and_colors_like_the_connector() {
        let req = request(SAMPLE);
        let board = position(&req).unwrap();
        assert_eq!(legal_indices(&board), req.legal_moves);

        let white = request(&SAMPLE.replace(r#""color":1"#, r#""color":2"#));
        assert_eq!(legal_indices(&position(&white).unwrap()), [20, 29, 34, 43]);
    }

    #[test]
    fn accepts_the_full_bot_match_request_with_null_fields() {
        let line = SAMPLE.replace(
            r#""budget_ms":10000"#,
            r#""budget_ms":10000,"blocked_squares":null,"placed":{},"match_id":7,"leg":1,"revision":0,"history":[],"round_no":null,"room_created_at":null,"engine_threads":4"#,
        );
        let req = request(&line);
        assert_eq!(req.engine_threads, Some(4));
        assert!(position(&req).is_ok());
    }

    #[test]
    fn never_uses_more_than_the_site_thread_limit() {
        assert_eq!(game_threads(None, None), MAX_THREADS);
        assert_eq!(game_threads(Some(16), Some(2)), MAX_THREADS);
        assert_eq!(game_threads(None, Some(32)), MAX_THREADS);
        assert_eq!(game_threads(Some(2), Some(4)), 2);
        assert_eq!(game_threads(None, Some(3)), 3);
    }

    #[test]
    fn rejects_variants_instead_of_playing_them_as_standard() {
        for (from, to) in [
            (r#""game_mode":"std""#, r#""game_mode":"anti""#),
            (r#""board_size":8"#, r#""board_size":10"#),
            (r#""color":1"#, r#""color":3"#),
        ] {
            assert!(position(&request(&SAMPLE.replace(from, to))).is_err());
        }
    }

    #[test]
    fn answers_pass_and_forced_moves_without_searching() {
        let mut engine = engine();
        let board = position(&request(SAMPLE)).unwrap();
        // A zero budget would leave no time for a search.
        let pass = request(&SAMPLE.replace("[19,26,37,44]", "[]").replace("10000", "0"));
        assert_eq!(engine.play(&pass, &board, Instant::now())["move"], -1);
        let forced = request(
            &SAMPLE
                .replace("[19,26,37,44]", "[26]")
                .replace("10000", "0"),
        );
        assert_eq!(engine.play(&forced, &board, Instant::now())["move"], 26);
    }

    #[test]
    fn ponders_the_position_we_face_next_across_passes() {
        let opening = position(&request(SAMPLE)).unwrap();
        assert_eq!(
            ponder_board(&opening, 19),
            Some(opening.make_move(Square::from_u8(19).unwrap()))
        );
        assert_eq!(ponder_board(&opening, -1), Some(opening.switch_players()));
        assert_eq!(ponder_board(&opening, 0), None);

        // C1 leaves the opponent's G8 without a move, so we move again.
        let opponent_passes = board(1 << 0 | 1 << 63, 1 << 1 | 1 << 62);
        let after = opponent_passes.make_move(Square::from_u8(2).unwrap());
        assert!(!after.has_legal_moves());
        assert_eq!(
            ponder_board(&opponent_passes, 2),
            Some(after.switch_players())
        );

        // C1 captures the opponent's last disc and ends the game.
        assert_eq!(ponder_board(&board(1 << 0, 1 << 1), 2), None);
    }

    #[test]
    fn next_request_stops_the_ponder() {
        let mut engine = engine_with(true);
        let req = request(&SAMPLE.replace("10000", "500"));
        let board = position(&req).unwrap();
        let reply = engine.play(&req, &board, Instant::now());
        assert_eq!(reply["environment"]["ponder"], true);
        assert!(engine.ponder_on.is_some());

        let waited = Duration::from_millis(200);
        let started = Instant::now();
        engine.ponder_while(|| thread::sleep(waited));
        assert!(started.elapsed() < waited + Duration::from_millis(500));

        let next = engine.ponder_on.unwrap();
        let legal = legal_indices(&next);
        let reply = engine.play(
            &request(
                &SAMPLE
                    .replace("[19,26,37,44]", &format!("{legal:?}"))
                    .replace("10000", "500"),
            ),
            &next,
            Instant::now(),
        );
        assert!(legal.contains(&(reply["move"].as_i64().unwrap() as i32)));
    }

    #[test]
    fn a_clock_inside_the_reserve_still_gets_a_short_search() {
        let low = TimeControlMode::Fixed {
            time_per_move_ms: LOW_CLOCK_MOVE_MS,
        };
        assert_eq!(clock_mode(0), low);
        assert_eq!(clock_mode(LOW_CLOCK_BANK_MS - 1), low);
        assert_eq!(
            clock_mode(LOW_CLOCK_BANK_MS),
            TimeControlMode::Fischer {
                main_time_ms: LOW_CLOCK_BANK_MS,
                increment_ms: 0,
            }
        );

        // The connector's budget still stops a search the clock cannot pay for.
        let mut engine = engine();
        let board = position(&request(SAMPLE))
            .unwrap()
            .make_move(Square::from_u8(19).unwrap());
        let legal = legal_indices(&board);
        let req = request(
            &SAMPLE
                .replace("[19,26,37,44]", &format!("{legal:?}"))
                .replace("60000", "900")
                .replace("10000", "150"),
        );
        let started = Instant::now();
        let reply = engine.play(&req, &board, started);
        assert!(started.elapsed() < Duration::from_millis(LOW_CLOCK_MOVE_MS));
        assert!(legal.contains(&(reply["move"].as_i64().unwrap() as i32)));
    }

    #[test]
    fn stops_at_the_connector_budget_even_with_a_long_clock() {
        let mut engine = engine();
        let req = request(&SAMPLE.replace("60000", "3600000").replace("10000", "500"));
        let board = position(&req).unwrap();
        let started = Instant::now();
        let reply = engine.play(&req, &board, started);
        assert!(started.elapsed() < Duration::from_millis(500));
        let mv = reply["move"].as_i64().unwrap() as i32;
        assert!(req.legal_moves.contains(&mv));
    }
}
