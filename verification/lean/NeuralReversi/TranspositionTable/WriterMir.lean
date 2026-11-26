import NeuralReversi.TranspositionTable.WriterMir.Program
import NeuralReversi.PvsParallel.ConcurrentReplay

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word)
open PvsParallel

/-- `event i` labels the i-th atomic action in this invocation. -/
def publication (event : Nat → ε) : List Atom × Bool → Option (WriterTrace.Call ε)
  | ([.cas .sequence expected claim .acqRel .relaxed, .fence .release,
      .store (.payload .player) player .relaxed, .store (.payload .opponent) opponent .relaxed,
      .store (.payload .metadata) metadata .relaxed, .store .sequence finish .release], true) =>
    if claim = expected ||| 1 ∧ finish = SeqLock.publish expected then
      some ⟨expected, ⟨player, opponent, metadata⟩, event 0, event 1,
        (fun | .player => event 2 | .opponent => event 3 | .metadata => event 4), event 5⟩
    else none
  | _ => none

theorem published (event : Nat → ε) (endian seq p o d) :
    publication event (writeStable endian seq p o d true) =
      some ⟨seq.val, ⟨p, o, Packing.pack endian d⟩, event 0, event 1,
        (fun | .player => event 2 | .opponent => event 3 | .metadata => event 4), event 5⟩ := by
  simp [writeStable, publication, SeqLock.publish]

theorem cas_failure (endian seq p o d) :
    writeStable endian seq p o d false =
      ([.cas .sequence seq.val (seq.val ||| 1) .acqRel .relaxed], false) := rfl

theorem stored_words {board : PvsBoardTree.Position} (frame : Option Nat)
    {event endian seq d success call}
    (hit : publication event (writeStable endian seq board.val.player board.val.opponent d success) =
      some call) : ConcurrentReplay.Action.Words endian (.stored frame call board d) := by
  cases success with
  | false => simp [writeStable, publication] at hit
  | true =>
    rw [published] at hit
    cases hit
    rfl

theorem cleared_words {event endian seq success call}
    (hit : publication event (writeStable endian seq 0 0 (Packing.unpack endian 0) success) =
      some call) : ConcurrentReplay.Action.Words endian (.cleared call) := by
  cases success with
  | false => simp [writeStable, publication] at hit
  | true =>
    rw [published] at hit
    cases hit
    simp [ConcurrentReplay.Action.Words, Packing.pack_unpack]

end NeuralReversi.TranspositionTable.WriterMir
