use std::io::{self, Write};
use std::sync::Arc;

#[allow(dead_code)]
#[path = "../../../../../crates/reversi-core/src/search/threading/abort.rs"]
mod abort;
#[path = "../../../../../crates/reversi-core/src/search/stop.rs"]
mod stop;

mod threading {
    pub(crate) use super::abort::AbortState;
}

use abort::AbortState;
use stop::{StopAttachment, StopToken};

#[derive(Clone, Copy)]
enum Action {
    Begin(usize),
    Attach(usize),
    Detach(usize),
    Stop,
}

fn write_case(out: &mut impl Write, actions: &[Action]) -> io::Result<()> {
    let pools = [Arc::new(AbortState::new()), Arc::new(AbortState::new())];
    let mut generations = [0; 2];
    let mut attachments: [Option<StopAttachment>; 2] = [None, None];
    let token = StopToken::new();
    for action in actions {
        match *action {
            Action::Begin(p) => {
                generations[p] = pools[p].begin_search();
                write!(out, "b{p} ")?;
            }
            Action::Attach(p) => {
                assert!(attachments[p].is_none());
                attachments[p] = Some(token.attach(&pools[p], generations[p]));
                write!(out, "a{p} ")?;
            }
            Action::Detach(p) => {
                drop(attachments[p].take().unwrap());
                write!(out, "d{p} ")?;
            }
            Action::Stop => {
                token.stop();
                write!(out, "s ")?;
            }
        }
        let bits = u8::from(pools[0].is_aborted()) | (u8::from(pools[1].is_aborted()) << 1);
        write!(out, "{bits} ")?;
    }
    writeln!(out)
}

fn interleavings(
    out: &mut impl Write,
    a: &[Action],
    b: &[Action],
    prefix: &mut Vec<Action>,
) -> io::Result<()> {
    if let Some((first, rest)) = a.split_first() {
        prefix.push(*first);
        interleavings(out, rest, b, prefix)?;
        prefix.pop();
    }
    if let Some((first, rest)) = b.split_first() {
        prefix.push(*first);
        interleavings(out, a, rest, prefix)?;
        prefix.pop();
    }
    if a.is_empty() && b.is_empty() {
        for stop_at in 0..=prefix.len() {
            prefix.insert(stop_at, Action::Stop);
            write_case(out, prefix)?;
            prefix.remove(stop_at);
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    use Action::{Attach, Begin, Detach, Stop};

    flip_scalar_difftest::run_buffered(|out| {
        let full = |p| {
            [
                Begin(p),
                Attach(p),
                Detach(p),
                Begin(p),
                Attach(p),
                Detach(p),
            ]
        };
        interleavings(out, &full(0), &full(1), &mut Vec::new())?;
        write_case(out, &[Begin(0), Attach(0), Begin(0), Stop, Detach(0)])?;
        write_case(
            out,
            &[Stop, Begin(1), Attach(1), Stop, Detach(1), Begin(1), Stop],
        )
    })
}
