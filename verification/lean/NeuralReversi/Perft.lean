module
public import NeuralReversi.PerftModel
public import NeuralReversi.PvsBoardTree
public import NeuralReversi.Board.InitializationProperties
public import NeuralReversi.Eval.PatternUpdate.Storage

@[expose] public section

namespace NeuralReversi.Perft

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares pass play build)
open PvsMoveLoop (Tree)

def frontierCount (tree : Tree) (depth : Nat) : Nat :=
  match depth, tree with
  | 0, _ => 1
  | _ + 1, .leaf _ => 1
  | depth + 1, .pass child => frontierCount child (depth + 1)
  | depth + 1, .turn first rest =>
    ((first :: rest).attach.map fun child => frontierCount child.val depth).sum
termination_by sizeOf tree
decreasing_by
  · simp_wf
  · exact Tree.earlier child

def count (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) : Nat :=
  match depth with
  | 0 => 1
  | nextDepth + 1 =>
    match legal : legalSquares mc board.val with
    | [] =>
      let next := pass board
      if other : Board.hasLegalMoves mc next.val = true then count mc fc next (nextDepth + 1)
      else 1
    | first :: rest =>
      count mc fc (play mc fc board first).val nextDepth +
        (rest.map fun sq => count mc fc (play mc fc board sq).val nextDepth).sum
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil legal, other⟩) board.property
  · exact (play mc fc board first).property
  · exact (play mc fc board sq).property

theorem frontier_turn (first : Tree) (rest : List Tree) (depth : Nat) :
    frontierCount (.turn first rest) (depth + 1) =
      frontierCount first depth + (rest.map fun t => frontierCount t depth).sum := by
  rw [frontierCount.eq_def]
  change (List.map ((fun t => frontierCount t depth) ∘ Subtype.val)
    (first :: rest).attach).sum = _
  rw [← List.map_map]
  simp

def NodeCountContract : Prop :=
  ∀ mc fc (board : Position) depth, count mc fc board depth = frontierCount (build mc fc board) depth

theorem count_eq_frontier (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) :
    count mc fc board depth = frontierCount (build mc fc board) depth := by
  cases depth with
  | zero => simp [count.eq_def, frontierCount.eq_def]
  | succ nextDepth =>
    rw [count.eq_def, build.eq_def]
    cases moves : legalSquares mc board.val with
    | nil =>
      by_cases mobile : Board.hasLegalMoves mc (pass board).val = true
      · simp only [mobile, dite_true]
        rw [frontierCount.eq_def]
        exact count_eq_frontier mc fc (pass board) (nextDepth + 1)
      · simp [mobile, frontierCount.eq_def]
    | cons first rest =>
      rw [frontier_turn, List.map_map]
      dsimp
      have head := count_eq_frontier mc fc (play mc fc board first).val nextDepth
      rw [head]
      congr 2
      apply List.map_congr_left
      intro square _member
      exact count_eq_frontier mc fc (play mc fc board square).val nextDepth
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc
      (.pass ⟨PvsBoardTree.stuck_of_nil moves, mobile⟩) board.property
  · exact (play mc fc board first).property
  · exact (play mc fc board square).property

theorem nodeCountContract_proven : NodeCountContract := count_eq_frontier

def root : Position :=
  ⟨Board.newBoard, (Board.initialBoardContract_proven _ (by simp)).1⟩

def sumWords (children : List Nat) (start : BitVec 64) : BitVec 64 :=
  children.foldl (fun acc n => acc + BitVec.ofNat 64 n) start

def AccumulatorContract : Prop :=
  ∀ children (start : BitVec 64), start.toNat + children.sum < 2 ^ 64 →
    (sumWords children start).toNat = start.toNat + children.sum

theorem accumulatorContract_proven : AccumulatorContract := by
  intro children start fits
  induction children generalizing start with
  | nil => simp [sumWords]
  | cons first rest ih =>
    have total : start.toNat + first + rest.sum < 2 ^ 64 := by
      simpa only [List.sum_cons, Nat.add_assoc] using fits
    have firstFits : first < 2 ^ 64 := by omega
    have partialFits : start.toNat + first < 2 ^ 64 := by omega
    have exactAdd : (start + BitVec.ofNat 64 first).toNat = start.toNat + first := by
      rw [BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt firstFits,
        Nat.mod_eq_of_lt partialFits]
    simp only [sumWords, List.foldl_cons, List.sum_cons, ← Nat.add_assoc]
    have nextFits : (start + BitVec.ofNat 64 first).toNat + rest.sum < 2 ^ 64 := by
      rwa [exactAdd]
    change (sumWords rest (start + BitVec.ofNat 64 first)).toNat = _
    rw [ih _ nextFits, exactAdd]

private theorem sum_map_le {α : Type} (xs : List α) (f : α → Nat) (bound : Nat)
    (each : ∀ x ∈ xs, f x ≤ bound) :
    (xs.map f).sum ≤ xs.length * bound := by
  induction xs with
  | nil => simp
  | cons head tail ih =>
    have a := each head (by simp)
    have b := ih (fun x inTail => each x (by simp [inTail]))
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.add_mul, Nat.one_mul]
    omega

theorem branching_bound (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) : count mc fc board depth ≤ 64 ^ depth := by
  cases depth with
  | zero => simp [count.eq_def]
  | succ nextDepth =>
    rw [count.eq_def]
    cases moves : legalSquares mc board.val with
    | nil =>
      by_cases mobile : Board.hasLegalMoves mc (pass board).val = true
      · simp only [mobile, dite_true]
        exact branching_bound mc fc (pass board) (nextDepth + 1)
      · simpa [mobile] using Nat.succ_le_of_lt
          (Nat.pow_pos (n := nextDepth + 1) (by decide : 0 < 64))
    | cons first rest =>
      dsimp
      have firstBound := branching_bound mc fc (play mc fc board first).val nextDepth
      have tailBound := sum_map_le rest (fun square => count mc fc (play mc fc board square).val nextDepth)
        (64 ^ nextDepth) (fun square _ => branching_bound mc fc (play mc fc board square).val nextDepth)
      have lengthBound := List.length_filter_le
        (fun square : Bitboard.Square => (Board.getMoves mc board.val).getLsbD square.val)
        (List.finRange 64)
      have totalMoves : rest.length + 1 ≤ 64 := by
        have all := PvsBoardTree.legalSquares_values mc board.val
        rw [moves] at all
        rw [← all] at lengthBound
        simpa using lengthBound
      rw [Nat.pow_succ']
      calc
        _ ≤ (rest.length + 1) * 64 ^ nextDepth := by rw [Nat.add_mul, Nat.one_mul]; omega
        _ ≤ _ := Nat.mul_le_mul_right _ totalMoves
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc
      (.pass ⟨PvsBoardTree.stuck_of_nil moves, mobile⟩) board.property
  · exact (play mc fc board first).property
  · exact (play mc fc board square).property

theorem depth_ten_fits (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) (short : depth ≤ 10) :
    count mc fc board depth < 2 ^ 64 := by
  have bound := branching_bound mc fc board depth
  have depthBound := Nat.pow_le_pow_right (by decide : 0 < 64) short
  omega

def PlyContract : Prop :=
  (Board.emptyCount root.val).toNat = 60 ∧
  (∀ mc fc (board : Position) (ply : Nat) (sq : GeneratedSquare mc board.val),
    ply + (Board.emptyCount board.val).toNat = 60 →
    let flipped := Flip.Dispatch.resultValue fc sq.val board.val.player board.val.opponent
    Eval.PatternUpdate.updateAllowed ply flipped = true ∧
    ply + 1 + (Board.emptyCount (play mc fc board sq).val.val).toNat = 60) ∧
  (∀ board : Position, Board.emptyCount (pass board).val = Board.emptyCount board.val)

theorem plyContract_proven : PlyContract := by
  refine ⟨(Board.initialBoardContract_proven _ (by simp [root])).2.2.2.1, ?_, ?_⟩
  · intro mc fc board ply sq total flipped
    have accepted := (Board.precomputedMoveContract_proven mc fc true true _ sq.val
      board.property sq.property).1
    have step := (Board.checkedPlacementProgress_proven true _ _ _ _ board.property accepted).2
    change (Board.emptyCount (play mc fc board sq).val.val).toNat + 1 =
      (Board.emptyCount board.val).toNat at step
    have nonempty := (Board.checkedAcceptsGeneratedMoves_proven mc fc _ sq.val
      board.property sq.property).1
    have validPly : ply < 60 := by omega
    refine ⟨Eval.PatternUpdate.updateGuardContract_proven _ _ |>.mpr ?_, by omega⟩
    exact ⟨by omega, by omega, nonempty, BitVec.ctz_lt_iff_ne_zero.mpr nonempty⟩
  · intro board
    exact Board.switchPlayers_emptyCount board.val

private theorem countRaw_eq_count_aux (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) (canPass : Bool)
    (passValid : canPass = false → Board.hasLegalMoves mc board.val = true) :
    countRaw mc fc board.val depth canPass = count mc fc board depth := by
  cases depth with
  | zero => simp [countRaw.eq_def, count.eq_def]
  | succ nextDepth =>
    rw [countRaw.eq_def, count.eq_def]
    have hvals := PvsBoardTree.legalSquares_values mc board.val
    cases moves : legalSquares mc board.val with
    | nil =>
      rw [moves] at hvals
      have hfilter : (List.finRange 64).filter (fun sq => (Board.getMoves mc board.val).getLsbD sq.val) = [] :=
        hvals.symm
      simp only [hfilter]
      have hcan : canPass = true := by
        cases canPass with
        | true => rfl
        | false =>
          have mobile := passValid rfl
          have stuck := PvsBoardTree.stuck_of_nil moves
          rw [stuck] at mobile
          contradiction
      simp only [hcan, ↓reduceIte]
      by_cases mobile : Board.hasLegalMoves mc (pass board).val = true
      · have mobile' : Board.hasLegalMoves mc (Board.switchPlayers board.val) = true := mobile
        simp only [mobile, mobile', ↓reduceIte, dite_true]
        exact countRaw_eq_count_aux mc fc (pass board) (nextDepth + 1) false (fun _ => mobile)
      · have mobile' : ¬ Board.hasLegalMoves mc (Board.switchPlayers board.val) = true := mobile
        simp [mobile, mobile']
    | cons first rest =>
      rw [moves] at hvals
      have hfilter : (List.finRange 64).filter (fun sq => (Board.getMoves mc board.val).getLsbD sq.val) =
          first.val :: rest.map Subtype.val := by simpa using hvals.symm
      simp only [hfilter, List.map_map]
      have head := countRaw_eq_count_aux mc fc (play mc fc board first).val nextDepth true (by simp)
      change countRaw mc fc (play mc fc board first).val.val nextDepth true + _ = _
      rw [head]
      congr 2
      apply List.map_congr_left
      intro sq _
      exact countRaw_eq_count_aux mc fc (play mc fc board sq).val nextDepth true (by simp)
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc
      (.pass ⟨PvsBoardTree.stuck_of_nil moves, mobile⟩) board.property
  · exact (play mc fc board first).property
  · exact (play mc fc board sq).property

theorem countRaw_eq_count (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (depth : Nat) :
    countRaw mc fc board.val depth true = count mc fc board depth :=
  countRaw_eq_count_aux mc fc board depth true (by simp)

end NeuralReversi.Perft
