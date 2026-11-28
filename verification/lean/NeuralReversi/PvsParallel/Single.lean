module
public import NeuralReversi.PvsParallel.ConcurrentReplay
public import NeuralReversi.MoveList.Generation
public import NeuralReversi.PvsParallel.SingleExit

@[expose] public section

namespace NeuralReversi.PvsParallel.ConcurrentReplay

open Bitboard (Square)
open PvsBoardTree (Position GeneratedSquare build play)
open PvsMoveLoop (scoreInf ScanCalls MoveCalls)
open RootReplay (serial start)

theorem generate_single_root_any (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (roots : List Square) (idx : Nat) (move : GeneratedSquare mc b.val) (covers : Root.Covers mc b roots)
    (only : Root.choices mc b roots idx = [move])
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) :
    let mask := Root.remainingMask mc b.val roots idx
    let list := MoveList.Generation.fill false paired fc b.val mask mem
    mask ≠ 0 ∧ list.len = some 1 ∧ list.slots 0 = some (MoveList.Generation.entry fc b.val move.val) ∧
      list.wipeout = some (if EndgameCutoffs.wipesOut fc b.val move.val then some move.val else none) := by
  let mask := Root.remainingMask mc b.val roots idx
  have exactBit (sq : Square) : mask.getLsbD sq.val = true ↔ sq = move.val := by
    rw [(Root.maskContract_proven mc b roots idx covers).1 sq, ← Root.choices_squares_mem covers, only]
    simp
  have iffMember (sq : Square) : sq ∈ MoveList.Generation.squares mask ↔ sq = move.val := by
    simpa [MoveList.Generation.squares] using exactBit sq
  have distinct : (MoveList.Generation.squares mask).Nodup := (List.nodup_finRange 64).filter _
  have oneSq : MoveList.Generation.squares mask = [move.val] := by
    cases splitList : MoveList.Generation.squares mask with
    | nil =>
      have h := (iffMember move.val).mpr rfl
      simp [splitList] at h
    | cons head tail =>
      have same := (iffMember head).mp (by simp [splitList])
      have noOther : tail = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro next rest
        have sameNext := (iffMember next).mp (by simp [splitList, rest])
        have noHead := (List.nodup_cons.mp (splitList ▸ distinct)).1
        exact noHead (same.trans sameNext.symm ▸ rest)
      simp [same, noOther]
  have subset : MoveList.Generation.IsSubset mask (Board.getMoves mc b.val) := by
    intro i
    simp only [mask, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true]
    exact And.left
  have generated := MoveList.Generation.generationContract_proven mc fc b.val mask false paired mem
    b.property subset (by simp)
  have count := generated.1
  have slot := generated.2.1 0
  have cache := generated.2.2.2.2
  have nonempty : mask ≠ 0 := by
    intro zero
    have setBit := (exactBit move.val).mpr rfl
    simp [zero] at setBit
  refine ⟨nonempty, ?_, ?_, ?_⟩
  · simpa [MoveList.Generation.records, oneSq] using count
  · simpa [MoveList.Generation.records, oneSq] using slot
  · simpa [MoveList.Generation.records, oneSq, MoveList.Generation.candidatesForCache,
      MoveList.Wipeout.isMatch, MoveList.Generation.entry, EndgameCutoffs.wipesOut] using cache

theorem Consumed.sound_pv (observed : Observations mc fc endian events)
    (used : Consumed observed.before events consumer tree true alpha beta child)
    (window : Aspiration.Window alpha beta) :
    Aspiration.SoundResult (PvsMoveLoop.minimax tree) alpha beta child := by
  obtain ⟨linear, perm, _, run⟩ := contract_proven _ _ _ _ observed
  have recorded := observed.consumedScore used
  obtain ⟨event, previous, saved⟩ := List.mem_filterMap.mp recorded
  have earlier := List.mem_reverse.mp previous
  have member : event ∈ events := (List.mem_filter.mp earlier).1
  have found : CausalReplay.Recorded (scores linear) tree true alpha beta child :=
    List.mem_filterMap.mpr ⟨event, List.mem_reverse.mpr (perm.mem_iff.mp member), saved⟩
  exact replayContract_proven _ _ _ _ _ _ window (by simp)
    ((CausalReplay.contract_proven _ _ _ _ _ _ run).1 _ _ _ _ _ found)

theorem single_score_of_reply (observed : Observations mc fc endian events)
    (used : Consumed observed.before events consumer tree true (-beta) (-alpha) child)
    (window : Aspiration.Window alpha beta) (sq : Square) :
    (singleRootExit sq child).2 = -child ∧
    -32767 < -child ∧ -child < 32767 := by
  have bounds := Consumed.sound_pv observed used (PvsMoveLoop.reflectedWindow window)
  have lower := bounds.lower
  have upper := bounds.upper
  have above : -32767 < -child := by omega
  exact ⟨by simp only [singleRootExit, ite_eq_left above], above, by omega⟩

theorem rootAttempt_of_single_child (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (endian : Bool) (b : Position) (roots : List Square) (idx : Nat) (alpha beta child : Int)
    (move : GeneratedSquare mc b.val) (only : Root.choices mc b roots idx = [move])
    (observed : Observations mc fc endian events) (consumer : Nat)
    (returned : Consumed observed.before events consumer (build mc fc (play mc fc b move).val)
      true (-beta) (-alpha) child) :
    let (writes, score) := singleRootExit move.val child
    let payload := writes.foldl (fun rows (sq, value, pv) =>
      RootOrder.writeScore rows sq (if pv then value else -scoreInf)) (fun _ => -scoreInf)
    RootAttempt mc fc endian b roots idx alpha beta score payload := by
  let observe : PvsMoveLoop.ChildSearch (GeneratedSquare mc b.val) := fun _ _ _ _ => child
  let used := UsedChild observed.identify observed.before events consumer
  have calls : ScanCalls (fun m : GeneratedSquare mc b.val =>
      used .serial (build mc fc (play mc fc b m).val)) observe true beta [move] true alpha (-scoreInf) := by
    rw [ScanCalls]
    refine ⟨?_, ?_⟩
    · exact returned
    · split <;> trivial
  have attempt := RootEvidence.AttemptWithUses.sequential (mc := mc) (fc := fc) (b := b)
    (roots := roots) (idx := idx) (alpha := alpha) (beta := beta) (completed := used)
    [move] observe (only ▸ List.Perm.refl _) calls
  let after := (start alpha).record beta ⟨move.val, alpha, -child⟩
  have stateEq : serial Subtype.val observe beta [move] (start alpha) = after := by
    rw [serial]
    split <;> rfl
  rw [stateEq] at attempt
  have reached : RootAttempt mc fc endian b roots idx alpha beta
      (max (-scoreInf) (-child)) (RootOrder.writeScore (fun _ => -scoreInf) move.val (-child)) :=
    ⟨events, observed, consumer, attempt⟩
  by_cases high : -scoreInf < -child
  · rw [Int.max_eq_right (Int.le_of_lt high)] at reached
    simp only [singleRootExit, ite_eq_left (show -32767 < -child from high)]
    exact reached
  · rw [Int.max_eq_left (Int.le_of_not_gt high)] at reached
    simp only [singleRootExit, ite_eq_right (show ¬ -32767 < -child from high)]
    exact reached

end NeuralReversi.PvsParallel.ConcurrentReplay
