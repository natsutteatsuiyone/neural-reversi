use std::cell::Cell;
use std::io::{self, Write};

use crate::engine::node_type::{NodeType, NonPV, PV, Root};
use crate::score::Score;
use crate::square::Square;

struct Move {
    sq: Square,
}

struct State {
    alpha: Cell<Score>,
    best: Cell<Score>,
    raised: Cell<bool>,
    improved: Cell<bool>,
    cutoff: Cell<bool>,
}

impl State {
    fn alpha(&self) -> Score {
        self.alpha.get()
    }

    fn set_alpha(&self, value: Score) {
        self.raised.set(true);
        self.alpha.set(value);
    }

    fn best_score(&self) -> Score {
        self.best.get()
    }

    fn set_best_score(&self, value: Score) {
        self.best.set(value);
    }

    fn set_best_move(&self, square: Option<Square>) {
        self.improved.set(square.is_some());
    }
}

struct Context;

impl Context {
    fn update_pv(&self, _: Square) {}

    fn pv(&self) -> &[Square] {
        &[]
    }
}

struct SplitPoint;

impl SplitPoint {
    fn copy_pv(&self, _: &[Square]) {}
}

struct Thread;

impl Thread {
    fn mark_split_point_cutoff(&self, sp: &State) {
        sp.cutoff.set(true);
    }
}

fn publish<NT: NodeType>(
    out: &mut impl Write,
    alpha: i32,
    best: i32,
    beta: i32,
    returned: i32,
) -> io::Result<()> {
    let sp = &State {
        alpha: Cell::new(Score::from_raw(alpha)),
        best: Cell::new(Score::from_raw(best)),
        raised: Cell::new(false),
        improved: Cell::new(false),
        cutoff: Cell::new(false),
    };
    let mv = Move { sq: Square::A1 };
    let ctx = Context;
    let split_point = SplitPoint;
    let thread = Thread;
    let score = Score::from_raw(returned);
    let beta = Score::from_raw(beta);
    let cutoff = include!(concat!(env!("OUT_DIR"), "/split_report_source.rs"));
    assert_eq!(cutoff, sp.cutoff.get());
    writeln!(
        out,
        "{} {alpha} {best} {} {returned} {} {} {} {} {}",
        u8::from(NT::PV_NODE),
        beta.value(),
        sp.alpha().value(),
        sp.best_score().value(),
        u8::from(sp.raised.get()),
        u8::from(sp.improved.get()),
        u8::from(cutoff),
    )
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let scores = [-16384, -1, 0, 1, 16382, 16383, 16384];
    for alpha in [-32767, -16384, -1, 0, 1, 16382, 16383, 16384] {
        for best in std::iter::once(-32767).chain(scores) {
            for score in scores {
                publish::<NonPV>(out, alpha, best, alpha + 1, score)?;
                for beta in [-16383, 0, 1, 16383, 16384, 16385, 32767] {
                    if alpha < beta {
                        publish::<PV>(out, alpha, best, beta, score)?;
                        publish::<Root>(out, alpha, best, beta, score)?;
                    }
                }
            }
        }
    }
    Ok(())
}
