import NeuralReversi.TranspositionTable.WriterMir.StoreProof

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word)
open PvsParallel
open PvsBoardTree (Position)

theorem decode_success {event endian current step req call}
    (hit : publishRound event endian current step = some (req,call)) :
    requested step.ops = some req ∧ req.expected &&& 1 = 0 ∧ current = req.expected ∧
      call.expected = req.expected ∧ call.payload = ⟨req.player, req.opponent, Packing.pack endian req.entry⟩ := by
  unfold publishRound at hit
  cases issued : requested step.ops with
  | none => simp [issued] at hit
  | some candidate =>
    rw [issued] at hit
    change (if even : candidate.expected &&& 1 = 0 then (publication event (writeStable endian ⟨_,even⟩
      candidate.player candidate.opponent candidate.entry (decide (current = candidate.expected)))).bind
      (fun c => some (candidate,c)) else none) = some (req,call) at hit
    split at hit <;> try contradiction
    rename_i even
    by_cases fresh : current = candidate.expected
    · simp [fresh, published] at hit
      obtain ⟨rfl,rfl⟩ := hit
      exact ⟨rfl,even,fresh,rfl,rfl⟩
    · simp [fresh, writeStable, publication] at hit

theorem clear_event {event endian current more seq metadata last req call}
    (hit : publishRound event endian current
      (clearStep endian more (decide (current = seq)) seq metadata last) = some (req,call)) :
    ConcurrentReplay.Action.Words endian (.cleared call) := by
  obtain ⟨issued,_,_,_,words⟩ := decode_success hit
  rw [clear_request] at issued
  split at issued <;> cases issued
  simpa [ConcurrentReplay.Action.Words, Packing.pack_unpack] using words

theorem penalty_event {board : Position} {event endian current expected req call} {sample : Option SeqLock.Snapshot}
    (frame : Option Nat) {mc fc computed source}
    (previous : TableReads.SlotLookupFrom source board (some expected))
    (hit : publishRound event endian current
      (penaltyStep sample.isSome board.val.player board.val.opponent expected (sample.getD dummy)
        (Packing.unpack endian (sample.getD dummy).metadata)) = some (req,call)) :
    CausalReplay.PrefixWriteFrom mc fc endian computed source board req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  obtain ⟨issued,_,fresh,_,words⟩ := decode_success hit
  have written := penalty_agrees endian board.val.player board.val.opponent current expected sample
  rw [issued] at written
  have accepts : Penalty.tryPenalize endian board.val.player board.val.opponent expected sample current =
      some req.entry := by simpa [fresh] using written.symm
  refine ⟨.penalty expected sample current previous accepts,?_⟩
  simp only [penaltyStep] at issued
  repeat' split at issued
  all_goals simp [requested] at issued
  cases issued
  exact words

theorem regular_event {board : Position} {event endian current entry obs req call} (frame : Option Nat)
    {mc fc computed source move}
    (other : ¬Cutoffs.ExactEndgame entry) (hint : CausalReplay.OtherHint mc fc board move)
    (encoded : entry.bestMove = Packing.encodeBestMove move)
    (weRead : observations source endian obs
      (saveRound endian board.val.player board.val.opponent entry obs))
    (currentValue : current = obs.current)
    (hit : publishRound event endian current
      (saveRound endian board.val.player board.val.opponent entry obs) = some (req,call)) :
    CausalReplay.PrefixWriteFrom mc fc endian computed source board req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  subst current
  have pair := save_publication frame hit
  exact ⟨.other entry obs _ move other hint encoded (preserves_observed_hint weRead) pair.1,pair.2⟩

end NeuralReversi.TranspositionTable.WriterMir

