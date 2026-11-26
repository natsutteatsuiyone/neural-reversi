use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::aspiration::widen_aspiration_window;
use crate::engine::node_type::{NodeType, Root};
use crate::score::{DiscScore, Score};
use crate::types::Depth;

struct Board;
struct EndGameStrategy;

struct Thread {
    aborted: AtomicBool,
}

impl Thread {
    fn is_search_aborted(&self) -> bool {
        self.aborted.load(Ordering::Relaxed)
    }
}

struct SearchContext {
    n_empties: Depth,
    value: Score,
    fail_soft: bool,
    abort_at: Option<usize>,
    trials: Vec<(Score, Score, Score, bool)>,
}

fn search<NT: NodeType, Strategy>(
    ctx: &mut SearchContext,
    _: &Board,
    depth: Depth,
    alpha: Score,
    beta: Score,
    thread: &Thread,
    cut_node: bool,
) -> Score {
    assert!(NT::ROOT_NODE && !cut_node);
    assert_eq!(
        std::any::type_name::<Strategy>(),
        std::any::type_name::<EndGameStrategy>()
    );
    assert_eq!(depth, ctx.n_empties);
    assert!(alpha < beta);
    assert!(ctx.trials.len() < 11);
    let aborted = ctx.abort_at == Some(ctx.trials.len());
    thread.aborted.store(aborted, Ordering::Relaxed);
    let score = if ctx.fail_soft {
        ctx.value
    } else {
        ctx.value.clamp(alpha, beta)
    };
    ctx.trials.push((alpha, beta, score, aborted));
    score
}

include!(concat!(env!("OUT_DIR"), "/endgame_aspiration_source.rs"));

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let endpoints = [-32767, -16384, -1024, -1, 0, 1, 1024, 16384, 32767];
    let values = [-16384, -1024, -255, 0, 255, 1024, 16384];
    for (index, &lo) in endpoints.iter().enumerate() {
        for &hi in &endpoints[index + 1..] {
            for raw in values {
                for fail_soft in [false, true] {
                    for abort_at in [None, Some(0), Some(1), Some(3)] {
                        let mut ctx = SearchContext {
                            n_empties: 20,
                            value: Score::from_raw(raw),
                            fail_soft,
                            abort_at,
                            trials: Vec::new(),
                        };
                        let board = Board;
                        let thread = Arc::new(Thread {
                            aborted: AtomicBool::new(false),
                        });
                        let mut alpha = Score::from_raw(lo);
                        let mut beta = Score::from_raw(hi);
                        let result =
                            aspiration_search(&mut ctx, &board, &mut alpha, &mut beta, &thread);
                        write!(
                            out,
                            "{lo} {hi} {} {} {}",
                            alpha.value(),
                            beta.value(),
                            result.value()
                        )?;
                        for (a, b, s, aborted) in ctx.trials {
                            write!(
                                out,
                                " {},{},{},{}",
                                a.value(),
                                b.value(),
                                s.value(),
                                u8::from(aborted)
                            )?;
                        }
                        writeln!(out)?;
                    }
                }
            }
        }
    }
    Ok(())
}
