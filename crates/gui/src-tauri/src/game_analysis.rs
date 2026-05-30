//! Game Analysis (CONTEXT.md → Game Analysis): replay a Game's moves, score
//! each Position, and report the per-move Score Loss.
//!
//! The engine Search is injected at a seam (`search`), so the replay and the
//! backward Score-Loss propagation — the bug-prone part — are exercised with a
//! fake scorer instead of the neural-network engine. Cancellation and progress
//! delivery are likewise injected, keeping this module free of `Arc<Mutex>`,
//! `Level`, and Tauri. `analyze_game_command` in `lib.rs` owns that wiring.

use reversi_core::board::Board;
use reversi_core::square::Square;
use reversi_core::types::Scoref;

/// What the injected engine Search returns for one Position that has Legal
/// Moves.
pub struct Analysis {
    pub best_move: Square,
    pub score: Scoref,
    pub depth: u32,
}

/// One per-move Game Analysis result (CONTEXT.md → Game Analysis, Played
/// Score, Score Loss).
pub struct GameAnalysisProgress {
    pub move_index: usize,
    pub best_move: Square,
    pub best_score: Scoref,
    pub played_score: Scoref,
    pub score_loss: Scoref,
    pub depth: u32,
}

/// One Game Analysis move after decoding the command's wire format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameAnalysisMove {
    Pass,
    Play(Square),
}

/// Drive a Game Analysis over `moves` applied from `initial`, emitting one
/// [`GameAnalysisProgress`] per non-pass move, newest move first.
///
/// - `search` scores a Position that has Legal Moves — the engine seam.
/// - `is_cancelled` is polled to bail when the run is superseded; on cancel
///   the analysis stops and returns `Ok(())`.
/// - `on_progress` receives each per-move result.
///
/// Scores must already be at the caller's reporting precision: the seed and
/// each `Analysis::score` are propagated verbatim (the terminal `solve()` path
/// returns an exact whole-number Final Score, which needs no rounding).
///
/// # Errors
///
/// Returns `Err` if the decoded move list is illegal, or if `search` fails.
pub fn analyze_game(
    initial: Board,
    moves: &[GameAnalysisMove],
    search: impl Fn(&Board) -> Result<Analysis, String>,
    is_cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(GameAnalysisProgress),
) -> Result<(), String> {
    let Replay { steps, final_board } = replay(initial, moves)?;

    if is_cancelled() {
        return Ok(());
    }

    let mut score: Scoref = match classify_seed(&final_board) {
        SeedPosition::Searchable => {
            let analysis = search(&final_board)?;
            if is_cancelled() {
                return Ok(());
            }
            analysis.score
        }
        SeedPosition::Pass => {
            // Side to move has no Legal Moves but the opponent does — a forced
            // Pass the move list did not record. Score the post-pass Position
            // and flip perspective.
            let analysis = search(&final_board.switch_players())?;
            if is_cancelled() {
                return Ok(());
            }
            -analysis.score
        }
        SeedPosition::Terminal => final_board.solve(final_board.get_empty_count()) as Scoref,
    };

    for (move_index, step) in steps.iter().enumerate().rev() {
        if is_cancelled() {
            return Ok(());
        }

        let before = match step {
            Step::Pass => {
                score = -score;
                continue;
            }
            Step::Play { before } => before,
        };

        let analysis = search(before)?;
        if is_cancelled() {
            return Ok(());
        }

        let (played_score, score_loss, next_score) = propagate_step(analysis.score, score);
        on_progress(GameAnalysisProgress {
            move_index,
            best_move: analysis.best_move,
            best_score: analysis.score,
            played_score,
            score_loss,
            depth: analysis.depth,
        });
        score = next_score;
    }

    Ok(())
}

/// The decoded outcome of replaying one move from the Game's move list.
#[derive(Debug)]
enum Step {
    /// A Pass: no Position is scored — only the carried Score's sign flips.
    Pass,
    /// A played move, paired with the Board it was played from: the Position
    /// the engine seam scores for this move.
    Play { before: Board },
}

/// The replayed Game: one [`Step`] per move (in play order) plus the Board
/// after the final move — the seed Position for the backward pass.
#[derive(Debug)]
struct Replay {
    steps: Vec<Step>,
    final_board: Board,
}

/// Replay `moves` from `initial`, validating pass and move legality.
fn replay(initial: Board, moves: &[GameAnalysisMove]) -> Result<Replay, String> {
    let mut steps = Vec::with_capacity(moves.len());
    let mut current = initial;

    for &move_input in moves {
        match move_input {
            GameAnalysisMove::Pass => {
                if current.is_game_over() {
                    return Err("Cannot pass after the game is over".to_string());
                }
                if current.has_legal_moves() {
                    return Err("Cannot pass when legal moves are available".to_string());
                }
                steps.push(Step::Pass);
                current = current.switch_players();
            }
            GameAnalysisMove::Play(square) => {
                // Validate legality here: Board::make_move only debug-asserts, so an
                // illegal move from malformed command input would silently corrupt
                // the Board in release. try_make_move rejects it instead.
                let next = current
                    .try_make_move(square)
                    .ok_or_else(|| format!("Illegal move '{square}' in move list"))?;
                steps.push(Step::Play { before: current });
                current = next;
            }
        }
    }

    Ok(Replay {
        steps,
        final_board: current,
    })
}

/// How the seed Position (the Board after the final move) must be scored for
/// the backward pass.
enum SeedPosition {
    /// The side to move has Legal Moves: score it with the engine seam.
    Searchable,
    /// The side to move has no Legal Moves but the opponent does — a forced
    /// Pass the move list did not record.
    Pass,
    /// Neither side can move: the Game is over, score it exactly.
    Terminal,
}

/// Classify how to score `board` as the backward pass's seed, making the
/// Searchable / Pass / Terminal distinction explicit.
fn classify_seed(board: &Board) -> SeedPosition {
    if !board.get_moves().is_empty() {
        SeedPosition::Searchable
    } else if board.is_game_over() {
        SeedPosition::Terminal
    } else {
        SeedPosition::Pass
    }
}

/// The negamax step that turns a Position's best Score and the Score of the
/// resulting Position (`prev`, from the next ply's Side To Move) into the
/// Played Score, the Score Loss, and the Score to carry to the shallower ply.
fn propagate_step(best: Scoref, prev: Scoref) -> (Scoref, Scoref, Scoref) {
    let played = -prev;
    let loss = (best - played).max(0.0);
    let next = best.max(played);
    (played, loss, next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reversi_core::disc::Disc;
    use std::cell::Cell;

    fn play(square: Square) -> GameAnalysisMove {
        GameAnalysisMove::Play(square)
    }

    fn pass() -> GameAnalysisMove {
        GameAnalysisMove::Pass
    }

    #[test]
    fn propagate_step_played_is_negated_prev() {
        let (played, _, _) = propagate_step(3.0, 2.0);
        assert_eq!(played, -2.0);
    }

    #[test]
    fn propagate_step_loss_clamps_at_zero_when_played_beats_best() {
        // played = 5.0, best = 1.0 -> raw -4.0 -> clamped 0.0
        let (_, loss, _) = propagate_step(1.0, -5.0);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn propagate_step_loss_and_next_when_best_exceeds_played() {
        let (played, loss, next) = propagate_step(4.0, -1.0); // played 1.0
        assert_eq!(played, 1.0);
        assert_eq!(loss, 3.0);
        assert_eq!(next, 4.0); // max(best 4, played 1)
    }

    #[test]
    fn propagate_step_next_keeps_the_better_played_line() {
        let (_, _, next) = propagate_step(-3.0, -5.0); // played 5.0, best -3.0
        assert_eq!(next, 5.0);
    }

    #[test]
    fn replay_yields_one_play_step_per_move_with_its_before_board() {
        let moves = vec![play(Square::D3), play(Square::C3)];
        let replayed = replay(Board::new(), &moves).unwrap();
        assert_eq!(replayed.steps.len(), 2);
        assert!(matches!(replayed.steps[1], Step::Play { .. }));
        let Step::Play { before } = &replayed.steps[0] else {
            panic!("expected a play step");
        };
        assert_eq!(*before, Board::new());
    }

    #[test]
    fn replay_handles_a_pass() {
        let moves = vec![pass()];
        let must_pass = forced_pass_board();
        let replayed = replay(must_pass, &moves).unwrap();
        assert_eq!(replayed.steps.len(), 1);
        assert!(matches!(replayed.steps[0], Step::Pass));
        assert_eq!(replayed.final_board, must_pass.switch_players());
    }

    #[test]
    fn replay_rejects_an_illegal_move() {
        // a1 is a valid Square but not a legal opening move.
        let err = replay(Board::new(), &[play(Square::A1)]).unwrap_err();
        assert!(err.contains("Illegal move"), "got: {err}");
    }

    #[test]
    fn replay_rejects_a_pass_when_legal_moves_are_available() {
        let err = replay(Board::new(), &[pass()]).unwrap_err();
        assert!(
            err.contains("Cannot pass when legal moves are available"),
            "got: {err}"
        );
    }

    #[test]
    fn replay_rejects_a_pass_after_the_game_is_over() {
        let full = Board::from_string(&"X".repeat(64), Disc::Black).unwrap();
        let err = replay(full, &[pass()]).unwrap_err();
        assert!(
            err.contains("Cannot pass after the game is over"),
            "got: {err}"
        );
    }

    /// a1=White, b1=Black, rest empty: Black has no Legal Move, but White can
    /// play c1 (flipping b1), so the Game is not over — a forced-Pass Position.
    fn forced_pass_board() -> Board {
        Board::from_string(&format!("OX{}", "-".repeat(62)), Disc::Black).unwrap()
    }

    /// H6=Black, H7/G8=White: Black can play H8, which leaves White with no
    /// Legal Moves while Black still has one, so the next move is a forced Pass.
    fn move_then_forced_pass_board() -> Board {
        Board::from_string(
            "-----------------------------------------------X-------O------O-",
            Disc::Black,
        )
        .unwrap()
    }

    #[test]
    fn classify_seed_distinguishes_searchable_pass_and_terminal() {
        assert!(matches!(
            classify_seed(&Board::new()),
            SeedPosition::Searchable
        ));

        let full = Board::from_string(&"X".repeat(64), Disc::Black).unwrap();
        assert!(matches!(classify_seed(&full), SeedPosition::Terminal));

        let must_pass = forced_pass_board();
        assert!(must_pass.get_moves().is_empty());
        assert!(!must_pass.is_game_over());
        assert!(matches!(classify_seed(&must_pass), SeedPosition::Pass));
    }

    #[test]
    fn analyze_game_seeds_a_forced_pass_final_board_via_the_switched_position() {
        // A forced-Pass final board with no recorded moves: the seed is scored
        // from the post-pass Position (which has moves), not from solve().
        let searched = Cell::new(false);
        let search = |b: &Board| {
            searched.set(true);
            assert!(
                !b.get_moves().is_empty(),
                "seed must search a Position with moves"
            );
            Ok(Analysis {
                best_move: Square::C1,
                score: 1.0,
                depth: 1,
            })
        };
        analyze_game(forced_pass_board(), &[], search, || false, |_p| {}).unwrap();
        assert!(
            searched.get(),
            "forced-pass seed must search the switched Position"
        );
    }

    #[test]
    fn analyze_game_emits_per_move_progress_newest_first() {
        // d3 (Black) then c3 (White); the final board still has Legal Moves,
        // so search is called for the seed, then once per move backward.
        let moves = vec![play(Square::D3), play(Square::C3)];
        let call = Cell::new(0usize);
        let search = |_b: &Board| {
            let i = call.get();
            call.set(i + 1);
            // call 0: final board (seed); call 1: boards[1] (i=1); call 2: boards[0] (i=0)
            Ok(match i {
                0 => Analysis {
                    best_move: Square::A1,
                    score: 2.0,
                    depth: 5,
                },
                1 => Analysis {
                    best_move: Square::D3,
                    score: 3.0,
                    depth: 7,
                },
                _ => Analysis {
                    best_move: Square::C4,
                    score: 1.0,
                    depth: 9,
                },
            })
        };

        let mut emitted: Vec<GameAnalysisProgress> = Vec::new();
        analyze_game(Board::new(), &moves, search, || false, |p| emitted.push(p)).unwrap();

        assert_eq!(emitted.len(), 2);

        // seed score = 2.0; i=1: played=-2.0, loss=max(3-(-2),0)=5.0, next=max(3,-2)=3.0
        assert_eq!(emitted[0].move_index, 1);
        assert_eq!(emitted[0].best_move, Square::D3);
        assert_eq!(emitted[0].best_score, 3.0);
        assert_eq!(emitted[0].played_score, -2.0);
        assert_eq!(emitted[0].score_loss, 5.0);
        assert_eq!(emitted[0].depth, 7);

        // score now 3.0; i=0: played=-3.0, loss=max(1-(-3),0)=4.0
        assert_eq!(emitted[1].move_index, 0);
        assert_eq!(emitted[1].best_score, 1.0);
        assert_eq!(emitted[1].played_score, -3.0);
        assert_eq!(emitted[1].score_loss, 4.0);
    }

    #[test]
    fn analyze_game_negates_score_across_a_pass() {
        // H8 is a real move that forces the opponent to pass. The pass flips
        // the carried score sign and emits nothing; only H8 (index 0) emits.
        let moves = vec![play(Square::H8), pass()];
        let call = Cell::new(0usize);
        let search = |_b: &Board| {
            let i = call.get();
            call.set(i + 1);
            // call 0: final board (seed) = 4.0; call 1: boards[0] (i=0)
            Ok(match i {
                0 => Analysis {
                    best_move: Square::A1,
                    score: 4.0,
                    depth: 1,
                },
                _ => Analysis {
                    best_move: Square::C1,
                    score: 0.0,
                    depth: 1,
                },
            })
        };
        let mut emitted: Vec<GameAnalysisProgress> = Vec::new();
        analyze_game(
            move_then_forced_pass_board(),
            &moves,
            search,
            || false,
            |p| emitted.push(p),
        )
        .unwrap();

        // Only the real move at index 0 emits; the pass at index 1 does not.
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].move_index, 0);
        // seed 4.0 -> pass flips to -4.0 -> played = -(-4.0) = 4.0 (pass negation observed).
        assert_eq!(emitted[0].played_score, 4.0);
    }

    #[test]
    fn analyze_game_negates_an_unrecorded_forced_pass_seed() {
        // H8 (Black) leaves White with no Legal Moves while Black still has one,
        // and this move list does NOT record the resulting forced Pass. The seed
        // is then scored from the switched (post-pass) Position and negated; that
        // seed negation is what this pins — dropping the sign inverts the reported
        // Played Score and Score Loss for the move, yet leaves every other test
        // green (the only forced-pass-seed test uses an empty move list, so it
        // emits nothing and never observes the sign).
        let moves = vec![play(Square::H8)];
        let call = Cell::new(0usize);
        let search = |b: &Board| {
            let i = call.get();
            call.set(i + 1);
            assert!(
                !b.get_moves().is_empty(),
                "search must only run on a Position with Legal Moves"
            );
            // call 0: forced-pass seed, scored on the switched Position = 2.0;
            // call 1: the board H8 was played from = 5.0.
            Ok(match i {
                0 => Analysis {
                    best_move: Square::A1,
                    score: 2.0,
                    depth: 1,
                },
                _ => Analysis {
                    best_move: Square::D3,
                    score: 5.0,
                    depth: 7,
                },
            })
        };

        let mut emitted: Vec<GameAnalysisProgress> = Vec::new();
        analyze_game(
            move_then_forced_pass_board(),
            &moves,
            search,
            || false,
            |p| emitted.push(p),
        )
        .unwrap();

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].move_index, 0);
        // seed = -2.0 (negated switched-Position score); played = -seed = 2.0;
        // loss = max(best 5.0 - played 2.0, 0) = 3.0.
        assert_eq!(emitted[0].best_score, 5.0);
        assert_eq!(emitted[0].played_score, 2.0);
        assert_eq!(emitted[0].score_loss, 3.0);
        assert_eq!(emitted[0].depth, 7);
    }

    #[test]
    fn analyze_game_scores_a_terminal_final_board_without_search() {
        // A full board has no empties and no Legal Moves: the seed comes from
        // solve(), never from search.
        let full = Board::from_string(&"X".repeat(64), Disc::Black).unwrap();
        let search = |_b: &Board| -> Result<Analysis, String> {
            panic!("search must not run on a terminal board");
        };
        let mut emitted = 0;
        analyze_game(full, &[], search, || false, |_p| emitted += 1).unwrap();
        assert_eq!(emitted, 0);
    }

    #[test]
    fn analyze_game_returns_ok_and_emits_nothing_when_cancelled_upfront() {
        let moves = vec![play(Square::D3), play(Square::C3)];
        let search = |_b: &Board| {
            Ok(Analysis {
                best_move: Square::A1,
                score: 1.0,
                depth: 1,
            })
        };
        let mut emitted = 0;
        analyze_game(Board::new(), &moves, search, || true, |_p| emitted += 1).unwrap();
        assert_eq!(emitted, 0);
    }

    #[test]
    fn analyze_game_stops_emitting_when_superseded_mid_run() {
        let moves = vec![play(Square::D3), play(Square::C3)];
        let search = |_b: &Board| {
            Ok(Analysis {
                best_move: Square::A1,
                score: 1.0,
                depth: 1,
            })
        };
        // First two cancellation polls pass (after-replay + after-seed), then supersede.
        let polls = Cell::new(0usize);
        let is_cancelled = || {
            let n = polls.get();
            polls.set(n + 1);
            n >= 2
        };
        let mut emitted = 0;
        analyze_game(Board::new(), &moves, search, is_cancelled, |_p| {
            emitted += 1
        })
        .unwrap();
        assert_eq!(emitted, 0);
    }

    #[test]
    fn analyze_game_suppresses_emit_when_superseded_after_a_move_search() {
        // One played move: the after-replay, after-seed, and loop-top polls all
        // pass, so cancellation first trips on the poll immediately AFTER the
        // per-move search. That post-search check is the only thing stopping the
        // move from emitting — the prior mid-run test bails at the loop-top check
        // and never reaches here, so without this case the guard is uncovered.
        let moves = vec![play(Square::D3)];
        let search = |_b: &Board| {
            Ok(Analysis {
                best_move: Square::A1,
                score: 1.0,
                depth: 1,
            })
        };
        // Polls 0..=2 pass (after-replay, after-seed, loop-top); poll 3 (the
        // post-search check) supersedes before the move can emit.
        let polls = Cell::new(0usize);
        let is_cancelled = || {
            let n = polls.get();
            polls.set(n + 1);
            n >= 3
        };
        let mut emitted = 0;
        analyze_game(Board::new(), &moves, search, is_cancelled, |_p| {
            emitted += 1
        })
        .unwrap();
        assert_eq!(emitted, 0);
    }
}
