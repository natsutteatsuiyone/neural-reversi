module
public import NeuralReversi.PvsParallel.SlotFrameMir.Syntax

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver

inductive Site
  | abandon | assign | book | bookJob | detach | dispatch | finalize | helper | helperDrop | idle | late | mainEntry | mainLoop | other | ownerInit | pvs | scout | split | unwindSplit | unwindWork | work | workerEntry
  deriving DecidableEq

def incoming : Site → Option (List Site)
  | .abandon => some [.unwindSplit, .unwindWork]
  | .assign => some [.ownerInit]
  | .book => some [.assign, .late]
  | .detach => some [.abandon, .work]
  | .dispatch => some [.work]
  | .finalize => some [.split, .unwindSplit]
  | .helper => some [.idle]
  | .helperDrop => some [.helper]
  | .idle => some [.split, .workerEntry]
  | .late => some [.work]
  | .mainEntry => some []
  | .mainLoop => some [.mainEntry]
  | .ownerInit => some [.split]
  | .split => some [.pvs]
  | .unwindSplit => some [.ownerInit, .split]
  | .unwindWork => some [.work]
  | .work => some [.idle]
  | .workerEntry => some []
  | _ => none

def pointerWriters : List Site := [.book, .detach, .finalize, .ownerInit]

def searchingWriters : List (Site × Bool) := [(.book, true), (.bookJob, true), (.detach, false), (.finalize, true), (.helperDrop, false), (.mainLoop, false), (.mainLoop, true)]

def workKind : Site → WorkerFrameMir.Kind
  | .abandon => .abandon
  | .book => .book
  | .bookJob => .bookJob
  | .detach => .detach
  | .finalize => .finalize
  | .helper => .helper
  | .helperDrop => .helperDrop
  | .idle => .idle
  | .mainEntry => .mainEntry
  | .mainLoop => .mainLoop
  | .split => .split
  | .work => .work
  | .workerEntry => .workerEntry
  | _ => .other

def guardReceiver (active : Option Slot) := active

end NeuralReversi.PvsParallel.SlotFrameMir.Receiver
