module
public import NeuralReversi.Solve1.TreeBridge
public import NeuralReversi.Solve34.Nodes

@[expose] public section

namespace NeuralReversi.Solve34

open Bitboard.Moves (X86Config LegalMove)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open EndgameTreeCache (discValue)

abbrev FlipConfig := Flip.Dispatch.X86Config

def position (b : EmptyBoard n) : Position := ⟨b.asBoard, b.disjoint⟩

theorem legal_iff_emptyList_flip (b : EmptyBoard n) (sq : Square) :
    LegalMove b.player b.opponent sq.val ↔ sq ∈ b.squares ∧ generatedFlip b sq ≠ 0 := by
  exact (Board.legal_iff_empty_flip b.asBoard sq b.disjoint).trans
    (and_congr (empty_bits b sq) Iff.rfl)

theorem moveChild_eq_play (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard n)
    (move : GeneratedSquare mc b.asBoard) (mem : move.val ∈ b.squares) :
    (moveChild b ⟨move.val, mem⟩).asBoard = (play mc fc (position b) move).val.val := by
  change childBoard b move.val (generatedFlip b move.val) =
    Board.place b.asBoard (Flip.Dispatch.resultValue fc move.val b.player b.opponent) move.val
  rw [Flip.Dispatch.value_disjoint_scalar fc move.val b.player b.opponent b.disjoint]
  exact child_eq_setPlacement b move.val mem _ (generatedFlip_subset b move.val)

private theorem candidateValues_match (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard n)
    (child : EmptyBoard (n - 1) → Int) (correct : ∀ cb, child cb = discValue mc fc cb.asBoard)
    (score : Int) :
    some score ∈ candidateValues b child ↔
      score ∈ (legalSquares mc b.asBoard).map (fun sq => discValue mc fc (play mc fc (position b) sq).val.val) := by
  simp only [candidateValues]
  constructor
  · intro candidate
    obtain ⟨sq, _, chosen⟩ := List.mem_map.mp candidate
    by_cases zero : generatedFlip b sq.val = 0
    · simp [zero] at chosen
    · have valid := (legal_iff_emptyList_flip b sq.val).mpr ⟨sq.property, zero⟩
      obtain ⟨move, member, eq⟩ := (PvsBoardTree.legalSquares_complete mc b.asBoard sq.val).mpr valid
      apply List.mem_map.mpr
      refine ⟨move, member, ?_⟩
      rw [ite_eq_right zero, Option.some.injEq, correct] at chosen
      have empty : move.val ∈ b.squares := by rw [eq]; exact sq.property
      have same : (⟨move.val, empty⟩ : {s // s ∈ b.squares}) = sq := Subtype.ext eq
      rwa [← moveChild_eq_play mc fc b move empty, same]
  · intro generated
    obtain ⟨move, _, rfl⟩ := List.mem_map.mp generated
    have valid := (Board.getMoves_exact mc b.asBoard move.val.val).mp move.property
    obtain ⟨member, flip⟩ := (legal_iff_emptyList_flip b move.val).mp valid
    apply List.mem_map.mpr
    refine ⟨⟨move.val, member⟩, by simp, ?_⟩
    change (if generatedFlip b move.val = 0 then none else some (child (moveChild b ⟨move.val, member⟩))) = _
    rw [ite_eq_right flip, correct, moveChild_eq_play mc fc b move member]

private theorem foldMax_list (xs : List (Option Int)) (best : Option Int) :
    foldMax best xs = (best.toList ++ xs.filterMap id).max? := by
  induction xs generalizing best with
  | nil => cases best <;> rfl
  | cons x xs ih =>
    rw [foldMax, ih]
    cases x with
    | none => simp
    | some v =>
      cases best with
      | none => rfl
      | some w =>
        change some ((xs.filterMap id).foldl max (max v w)) =
          some ((v :: xs.filterMap id).foldl max w)
        simp [Int.max_comm]

private theorem foldMin_negate (xs : List (Option Int)) (best : Option Int) :
    (foldMin best xs).map (- ·) = foldMax (best.map (- ·)) (xs.map (Option.map (- ·))) := by
  induction xs generalizing best with
  | nil => rfl
  | cons x xs ih =>
    rw [foldMin, List.map_cons, foldMax, ih]
    congr 1
    cases best <;> cases x <;> simp [mergeMin, mergeMax]

private theorem listMax_eq_of_same_values (xs ys : List Int)
    (same : ∀ value, value ∈ xs ↔ value ∈ ys) : xs.max? = ys.max? := by
  cases h : xs.max? with
  | none =>
    symm
    rw [List.max?_eq_none_iff] at h ⊢
    cases ys with
    | nil => rfl
    | cons v vs =>
      have found := (same v).mpr (by simp)
      simp [h] at found
  | some v =>
    symm
    rw [List.max?_eq_some_iff] at h ⊢
    exact ⟨(same v).mp h.1, fun w found => h.2 w ((same w).mpr found)⟩

private theorem bestValue_as_fold {Move : Type} (v : Move → Int) (moves : List Move) (seed : Int) :
    PvsMoveLoop.bestValue v moves seed = (moves.map v).foldl max seed := by
  induction moves generalizing seed with
  | nil => rfl
  | cons move rest ih => simp [PvsMoveLoop.bestValue, ih]

private theorem candidateMax (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard n)
    (child : EmptyBoard (n - 1) → Int) (correct : ∀ cb, child cb = discValue mc fc cb.asBoard) :
    foldMax none ((candidateValues b child).map (Option.map (- ·))) =
      if legalSquares mc b.asBoard = [] then none else some (discValue mc fc b.asBoard) := by
  let v : GeneratedSquare mc b.asBoard → Int := fun sq => -discValue mc fc (play mc fc (position b) sq).val.val
  rw [foldMax_list]
  change (((candidateValues b child).map (Option.map (- ·))).filterMap id).max? = _
  have matchValues (score : Int) :
      score ∈ ((candidateValues b child).map (Option.map (- ·))).filterMap id ↔
        score ∈ (legalSquares mc b.asBoard).map v := by
    have eq := candidateValues_match mc fc b child correct (-score)
    simp only [List.mem_filterMap, List.mem_map, id_eq]
    constructor
    · rintro ⟨o, ⟨source, found, rfl⟩, value⟩
      cases source with
      | none => cases value
      | some actual =>
        cases value
        obtain ⟨sq, mem, h⟩ := List.mem_map.mp (eq.mp (by simpa using found))
        exact ⟨sq, mem, by simpa [v] using congrArg (fun x : Int => -x) h⟩
    · rintro ⟨sq, found, rfl⟩
      have appears := eq.mpr (List.mem_map.mpr ⟨sq, found, by simp [v]⟩)
      exact ⟨some (v sq), ⟨some (-v sq), appears, by simp⟩, rfl⟩
  rw [listMax_eq_of_same_values _ _ matchValues]
  cases moves : legalSquares mc b.asBoard with
  | nil => simp
  | cons first rest =>
    have eq := EndgameTreeCache.discValue_moves mc fc (position b) first rest moves
    change discValue mc fc b.asBoard = PvsMoveLoop.bestValue v (first :: rest) (-127) at eq
    have range := (EndgameTreeCache.discValue_range mc fc (play mc fc (position b) first).val.val).2
    have seed : -127 ≤ v first := by dsimp [v]; omega
    rw [PvsMoveLoop.bestValue, Int.max_eq_right seed, bestValue_as_fold] at eq
    simp only [List.cons_ne_nil, ↓reduceIte, List.map_cons, List.max?_cons', ← eq]

theorem nextValue_eq_tree (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard n)
    (child : EmptyBoard (n - 1) → Int) (correct : ∀ cb, child cb = discValue mc fc cb.asBoard) :
    nextValue b child = discValue mc fc b.asBoard := by
  unfold nextValue value
  rw [candidateMax mc fc b child correct]
  cases moves : legalSquares mc b.asBoard with
  | cons first rest => simp
  | nil =>
    simp only [↓reduceIte, Option.getD_none]
    have other := (foldMin_negate (candidateValues b.pass child) none).trans
      (candidateMax mc fc b.pass child correct)
    have exactValue := EndgameTreeCache.scaled_discValue mc fc (position b)
    cases mobile : Board.hasLegalMoves mc b.pass.asBoard
    · have empty := PvsBoardTree.nil_of_stuck mobile
      rw [empty] at other
      simp only [↓reduceIte] at other
      rw [Option.map_eq_none_iff] at other
      rw [other, Option.getD_none]
      rw [PvsBoardTree.terminal_value mc fc (position b) moves mobile] at exactValue
      change 256 * discValue mc fc b.asBoard = 256 * Board.AwardedScore b.asBoard at exactValue
      omega
    · have nonempty : legalSquares mc b.pass.asBoard ≠ [] := by
        intro empty
        have stuck := PvsBoardTree.stuck_of_nil empty
        rw [stuck] at mobile
        cases mobile
      rw [ite_eq_right nonempty] at other
      rw [PvsBoardTree.minimax_pass mc fc (position b) moves mobile] at exactValue
      have opposite := EndgameTreeCache.scaled_discValue mc fc (pass (position b))
      have eq : discValue mc fc b.asBoard = -discValue mc fc b.pass.asBoard := by
        change 256 * discValue mc fc b.asBoard = -PvsBoardTree.minimax mc fc (pass (position b)) at exactValue
        change 256 * discValue mc fc b.pass.asBoard = PvsBoardTree.minimax mc fc (pass (position b)) at opposite
        omega
      cases h : foldMin none (candidateValues b.pass child) with
      | none => simp [h] at other
      | some score =>
        simp only [h, Option.map_some, Option.some.injEq] at other
        simp only [Option.getD_some]
        omega

private theorem twoChildValue_eq_tree (mc : X86Config) (fc : FlipConfig)
    (b : Solve2.TwoEmpty) :
    Solve2.childValue b (Solve2.generatedFlip b) (Solve2.generatedFlip_subset b) =
      discValue mc fc (Solve2.childBoard b (Solve2.generatedFlip b)) := by
  let f := Solve2.generatedFlip b
  let subset := Solve2.generatedFlip_subset b
  rw [Solve2.childValue, Solve1.exactScore_eq_tree mc fc (Solve2.childOneEmpty b f subset)]
  congr 1
  exact (Board.Board.mk.injEq ..).mpr ⟨rfl, (Solve2.childOneEmpty_exact_board b f subset).symm⟩

private theorem twoCandidates_eq (mc : X86Config) (fc : FlipConfig)
    (b : EmptyBoard 2) :
    candidateValues b (fun cb => discValue mc fc cb.asBoard) =
      [Solve2.candidateValue b.toTwoEmpty, Solve2.candidateValue b.toTwoEmpty.exchange] := by
  let slots := b.toTwoEmpty
  let eval := fun (sq : {s // s ∈ b.squares}) =>
    if generatedFlip b sq.val = 0 then none else some (discValue mc fc (moveChild b sq).asBoard)
  have first : slots.first ∈ b.squares := List.getElem_mem _
  have second : slots.second ∈ b.squares := List.getElem_mem _
  have attached : b.squares.attach = [⟨slots.first, first⟩, ⟨slots.second, second⟩] := by
    apply List.ext_getElem
    · simp [b.length_eq]
    · intro i hi hj
      have bound : i < 2 := by simpa using hj
      apply Subtype.ext
      rw [List.getElem_attach]
      cases i with
      | zero => rfl
      | succ i =>
        have eq : i = 0 := by omega
        subst i
        rfl
  change b.squares.attach.map eval = _
  rw [attached, List.map_cons, List.map_cons, List.map_nil]
  have firstEq : eval ⟨slots.first, first⟩ = Solve2.candidateValue slots := by
    unfold Solve2.candidateValue eval
    change (if Solve2.generatedFlip slots = 0 then none
      else some (discValue mc fc (Solve2.childBoard slots (Solve2.generatedFlip slots)))) = _
    rw [twoChildValue_eq_tree mc fc]
  have secondEq : eval ⟨slots.second, second⟩ = Solve2.candidateValue slots.exchange := by
    unfold Solve2.candidateValue eval
    change (if Solve2.generatedFlip slots.exchange = 0 then none
      else some (discValue mc fc (Solve2.childBoard slots.exchange (Solve2.generatedFlip slots.exchange)))) = _
    rw [twoChildValue_eq_tree mc fc]
  rw [firstEq, secondEq]

theorem twoValue_eq_tree (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard 2) :
    twoValue b = discValue mc fc b.asBoard := by
  let child : EmptyBoard 1 → Int := fun cb => discValue mc fc cb.asBoard
  rw [← nextValue_eq_tree mc fc b child (fun _ => rfl)]
  unfold twoValue Solve2.exactTwoValue nextValue
  rw [twoCandidates_eq mc fc b, twoCandidates_eq mc fc b.pass]
  have order : b.pass.toTwoEmpty = b.toTwoEmpty.pass := rfl
  rw [order]
  cases Solve2.candidateValue b.toTwoEmpty <;> cases Solve2.candidateValue b.toTwoEmpty.exchange <;>
    cases Solve2.candidateValue b.toTwoEmpty.pass <;> cases Solve2.candidateValue b.toTwoEmpty.pass.exchange <;>
    rfl

theorem exactFourValue_eq_tree (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard 4) :
    exactFourValue b = discValue mc fc b.asBoard :=
  nextValue_eq_tree mc fc b exactThreeValue fun cb =>
    nextValue_eq_tree mc fc cb twoValue fun base => twoValue_eq_tree mc fc base

def TreeBoundContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard 4) (alpha : Score),
    WindowSafe alpha →
    EndgameCache.SearchRange (source4Run b alpha).score.toInt ∧
      Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) (source4Run b alpha).score.toInt

private theorem source4Run_tree_nws (mc : X86Config) (fc : FlipConfig) (b : EmptyBoard 4)
    (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) (source4Run b alpha).score.toInt :=
  exactFourValue_eq_tree mc fc b ▸ source4Run_nws b alpha safe

theorem treeBoundContract_proven : TreeBoundContract := by
  intro mc fc b alpha safe
  have ⟨lo, hi⟩ := solve4Concrete_bounds b alpha safe
  rw [source4Run_score]
  exact ⟨⟨by omega, by omega⟩, source4Run_score b alpha ▸ source4Run_tree_nws mc fc b alpha safe⟩

end NeuralReversi.Solve34
