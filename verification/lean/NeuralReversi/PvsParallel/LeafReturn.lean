module
public import NeuralReversi.PvsParallel.ConcurrentReplay
public import NeuralReversi.PvsParallel.ExactDepth
public import NeuralReversi.PvsParallel.NonPvMir.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.ConcurrentReplay

open PvsBoardTree (Position build)

/-- A serial callee that needs no earlier child or TT read emits its score before
returning. Both events precede the caller consuming the return. -/
theorem consumed_of_leaf (r : CausalReplay.ScoreResult)
    (computed : ExecutionWithUses (fun _ _ _ _ _ _ _ => False) (fun _ _ _ _ _ _ => False)
      r.tree r.pv r.alpha r.beta r.score) :
    ∃ (events : List Event) (o : Observations mc fc endian events),
      Consumed o.before events 2 r.tree r.pv r.alpha r.beta r.score := by
  classical
  let score : Event := ⟨0, .scored 0 r⟩
  let ret : Event := ⟨1, .returned 0 r⟩
  let events := [score, ret]
  let o : Observations mc fc endian events := {
    identify := fun _ => 0
    before := Nat.lt
    irrefl := Nat.lt_irrefl
    trans := Nat.lt_trans
    words := by simp [events, score, ret, Action.Words]
    tablePublications := by simp [events, score, ret, Event.call]
    evidence := by
      intro e member
      rcases List.mem_cons.mp member with rfl | tail
      · exact computed.weaken (fun _ _ _ _ _ _ h => h.elim) (fun _ _ _ _ _ _ _ h => h.elim)
      · obtain rfl := List.mem_singleton.mp tail
        exact ⟨score, List.mem_filter.mpr ⟨List.mem_cons_self, decide_eq_true (Nat.lt_succ_self 0)⟩, rfl⟩ }
  exact ⟨events, o, ret, by simp [events], Nat.lt_succ_self 1, 0, rfl⟩

theorem consumed_zero_depth (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (endian checked : Bool) (b : Position) (alpha beta : Int) (zero : ExactDepth.Full b 0) :
    ∃ raw, NonPvMir.pvZeroScore checked b.val = .ok raw ∧
      ∃ (events : List Event) (o : Observations mc fc endian events),
        Consumed o.before events 2 (build mc fc b) true alpha beta raw.raw.toInt := by
  obtain ⟨noMoves, noPass, value⟩ := ExactDepth.zeroDepthContract_proven mc fc b checked zero
  have leaf := PvsBoardTree.terminal_tree mc fc b (PvsBoardTree.nil_of_stuck noMoves) noPass
  unfold NonPvMir.pvZeroScore
  cases h : ScoreRefresh.BoardBridge.finalScore checked b.val with
  | error err => rw [h] at value; cases value
  | ok raw =>
    refine ⟨raw, rfl, consumed_of_leaf ⟨build mc fc b, true, alpha, beta, raw.raw.toInt⟩ ?_⟩
    have numeric : raw.raw.toInt = PvsBoardTree.minimax mc fc b := by simpa [h, Except.map] using value
    change ExecutionWithUses _ _ (build mc fc b) true alpha beta raw.raw.toInt
    rw [numeric, PvsBoardTree.minimax, leaf, PvsMoveLoop.minimax.eq_def]
    exact .leaf

end NeuralReversi.PvsParallel.ConcurrentReplay
