module
public import NeuralReversi.PvsParallel

@[expose] public section

namespace NeuralReversi.PvsParallel.Root

open Aspiration (SoundResult Window)
open Bitboard (Bits Square squareMask)
open Bitboard.Moves (LegalMove X86Config)
open PvsBoardTree (GeneratedSquare Position legalSquares play)
open PvsMoveLoop (Bound Tree bestValue scoreInf scoreMax)

abbrev FlipConfig := Flip.Dispatch.X86Config

def mask (moves : List Square) : Bits :=
  moves.foldl (fun bits sq => bits ||| squareMask sq) 0

private theorem fold_bit (moves : List Square) (bits : Bits) (target : Square) :
    (moves.foldl (fun acc sq => acc ||| squareMask sq) bits).getLsbD target.val = true ↔
      bits.getLsbD target.val = true ∨ target ∈ moves := by
  induction moves generalizing bits with
  | nil => simp
  | cons move tail ih =>
    rw [List.foldl_cons, ih, BitVec.getLsbD_or, Bitboard.squareMask_bit]
    simp only [Bool.or_eq_true, decide_eq_true_eq, List.mem_cons]
    rw [Fin.ext_iff]
    exact or_assoc

theorem mask_bit (moves : List Square) (target : Square) :
    (mask moves).getLsbD target.val = true ↔ target ∈ moves := by
  unfold mask
  rw [fold_bit]
  simp

def remainingMask (mc : X86Config) (b : Board.Board) (roots : List Square) (idx : Nat) : Bits :=
  Board.getMoves mc b &&& mask (roots.drop idx)

def choices (mc : X86Config) (b : Position) (roots : List Square) (idx : Nat) :
    List (GeneratedSquare mc b.val) :=
  (legalSquares mc b.val).filter fun m => (mask (roots.drop idx)).getLsbD m.val.val

private theorem choices_bit (mc : X86Config) (b : Position) (roots : List Square) (idx : Nat)
    (sq : Square) :
    (∃ m ∈ choices mc b roots idx, m.val = sq) ↔
      (remainingMask mc b.val roots idx).getLsbD sq.val = true := by
  rw [remainingMask, BitVec.getLsbD_and, Bool.and_eq_true, mask_bit, Board.getMoves_exact,
    ← PvsBoardTree.legalSquares_complete]
  constructor
  · rintro ⟨m, inList, rfl⟩
    obtain ⟨legal, kept⟩ := List.mem_filter.mp inList
    exact ⟨⟨m, legal, rfl⟩, (mask_bit _ _).mp kept⟩
  · rintro ⟨⟨m, legal, rfl⟩, kept⟩
    exact ⟨m, List.mem_filter.mpr ⟨legal, (mask_bit _ _).mpr kept⟩, rfl⟩

abbrev Covers (mc : X86Config) (b : Position) (roots : List Square) : Prop :=
  roots.Perm ((legalSquares mc b.val).map Subtype.val)

theorem roots_nodup (covers : Covers mc b roots) : roots.Nodup := by
  apply covers.nodup_iff.mpr
  rw [PvsBoardTree.legalSquares_values]
  exact (List.nodup_finRange 64).filter _

private theorem prefix_excluded (covers : Covers mc b roots) (idx : Nat)
    (earlier : sq ∈ roots.take idx) : sq ∉ roots.drop idx := by
  have distinct := roots_nodup covers
  rw [← List.take_append_drop idx roots, List.nodup_append] at distinct
  exact fun kept => (distinct.2.2 sq earlier sq kept) rfl

private theorem remaining_legal (covers : Covers mc b roots) (kept : sq ∈ roots.drop idx) :
    LegalMove b.val.player b.val.opponent sq.val := by
  obtain ⟨m, legal, same⟩ := List.mem_map.mp (covers.mem_iff.mp (List.mem_of_mem_drop kept))
  exact (PvsBoardTree.legalSquares_complete mc b.val sq).mp ⟨m, legal, same⟩

def MaskContract : Prop :=
  ∀ (mc : X86Config) (b : Position) (roots : List Square) (idx : Nat),
    Covers mc b roots →
    (∀ sq, (remainingMask mc b.val roots idx).getLsbD sq.val = true ↔ sq ∈ roots.drop idx) ∧
    (∀ sq ∈ roots.take idx, (remainingMask mc b.val roots idx).getLsbD sq.val = false) ∧
    (choices mc b roots idx ≠ [] ↔ idx < roots.length)

theorem maskContract_proven : MaskContract := by
  intro mc b roots idx covers
  have exactMask (sq : Square) :
      (remainingMask mc b.val roots idx).getLsbD sq.val = true ↔ sq ∈ roots.drop idx := by
    rw [remainingMask, BitVec.getLsbD_and, Bool.and_eq_true, mask_bit, Board.getMoves_exact]
    exact ⟨And.right, fun kept => ⟨remaining_legal covers kept, kept⟩⟩
  refine ⟨exactMask, fun sq earlier => ?_, ?_⟩
  · cases bit : (remainingMask mc b.val roots idx).getLsbD sq.val
    · rfl
    · exact False.elim (prefix_excluded covers idx earlier ((exactMask sq).mp bit))
  · constructor
    · intro nonempty
      by_cases inside : idx < roots.length
      · exact inside
      · have absent : roots.drop idx = [] := List.drop_eq_nil_of_le (by omega)
        unfold choices at nonempty
        simp [absent, mask] at nonempty
    · intro inside empty
      have suffix : roots.drop idx ≠ [] := by
        rw [ne_eq, List.drop_eq_nil_iff]
        omega
      obtain ⟨sq, rest, remains⟩ := List.exists_cons_of_ne_nil suffix
      have member : sq ∈ roots.drop idx := by simp [remains]
      obtain ⟨m, inList, _⟩ :=
        (choices_bit mc b roots idx sq).mpr ((exactMask sq).mpr member)
      simp [empty] at inList

def childValue (mc : X86Config) (fc : FlipConfig) (b : Position)
    (move : GeneratedSquare mc b.val) : Int :=
  -PvsBoardTree.minimax mc fc (play mc fc b move).val

def squareValue (mc : X86Config) (fc : FlipConfig) (b : Position) (square : Square) : Int :=
  if legal : (Board.getMoves mc b.val).getLsbD square.val = true
  then childValue mc fc b ⟨square, legal⟩ else -scoreInf

theorem square_value_eq_child {mc : X86Config} {fc : FlipConfig} {b : Position}
    (move : GeneratedSquare mc b.val) :
    squareValue mc fc b move.val = childValue mc fc b move := by
  rw [squareValue, dite_eq_left move.property]

def value (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square) (idx : Nat) : Int :=
  bestValue (childValue mc fc b) (choices mc b roots idx) (-scoreInf)

theorem choices_squares_nodup (mc : X86Config) (b : Position) (roots : List Square) (idx : Nat) :
    ((choices mc b roots idx).map Subtype.val).Nodup := by
  unfold choices
  apply (List.filter_sublist.map Subtype.val).nodup
  rw [PvsBoardTree.legalSquares_values]
  exact (List.nodup_finRange 64).filter _

theorem choices_squares_mem (covers : Covers mc b roots) (sq : Square) :
    sq ∈ (choices mc b roots idx).map Subtype.val ↔ sq ∈ roots.drop idx := by
  rw [List.mem_map, choices_bit]
  exact (maskContract_proven mc b roots idx covers).1 sq

theorem square_value_le (covers : Covers mc b roots) (member : square ∈ roots.drop idx) :
    squareValue mc fc b square ≤ value mc fc b roots idx := by
  have bit := ((maskContract_proven mc b roots idx covers).1 square).mpr member
  obtain ⟨move, kept, rfl⟩ := (choices_bit mc b roots idx square).mpr bit
  rw [square_value_eq_child move]
  exact PvsMoveLoop.bestValue_ge_move (childValue mc fc b) _ (-scoreInf) kept

def tree (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square) (idx : Nat) :
    Option Tree :=
  match choices mc b roots idx with
  | [] => none
  | first :: rest => some (.turn (PvsBoardTree.build mc fc (play mc fc b first).val)
      (rest.map fun m => PvsBoardTree.build mc fc (play mc fc b m).val))

private theorem tree_value (present : tree mc fc b roots idx = some node) :
    PvsMoveLoop.minimax node = value mc fc b roots idx := by
  unfold tree at present
  cases list : choices mc b roots idx with
  | nil => simp [list] at present
  | cons first rest =>
    rw [list] at present
    cases Option.some.inj present
    rw [PvsMoveLoop.minimax_turn, value, list]
    exact PvsMoveLoop.bestValue_map (fun child => -PvsMoveLoop.minimax child)
      (fun m => PvsBoardTree.build mc fc (play mc fc b m).val) (first :: rest) _

theorem value_range (covers : Covers mc b roots) (inside : idx < roots.length) :
    -scoreMax ≤ value mc fc b roots idx ∧ value mc fc b roots idx ≤ scoreMax := by
  have nonempty := (maskContract_proven mc b roots idx covers).2.2.mpr inside
  cases legal : choices mc b roots idx with
  | nil => exact False.elim (nonempty legal)
  | cons first rest =>
    have present : tree mc fc b roots idx = some
        (.turn (PvsBoardTree.build mc fc (play mc fc b first).val)
          (rest.map fun m => PvsBoardTree.build mc fc (play mc fc b m).val)) := by
      simp only [tree, legal]
    rw [← tree_value present]
    exact PvsMoveLoop.minimax_range _

theorem choices_zero (covers : Covers mc b roots) : choices mc b roots 0 = legalSquares mc b.val := by
  apply List.filter_eq_self.mpr
  intro m mem
  exact (mask_bit _ _).mpr (by simpa using covers.mem_iff.mpr (List.mem_map_of_mem mem))

theorem value_zero (covers : Covers mc b roots) (inside : 0 < roots.length) :
    value mc fc b roots 0 = PvsBoardTree.minimax mc fc b := by
  have selected := choices_zero covers
  cases legal : legalSquares mc b.val with
  | nil =>
    have length : roots.length = 0 := by rw [covers.length_eq, legal]; rfl
    omega
  | cons first rest =>
    rw [value, selected, PvsBoardTree.minimax_moves mc fc b first rest legal, legal]
    rfl

theorem wipeout_value (chosen : move ∈ choices mc b roots idx)
    (wipes : EndgameCutoffs.wipesOut fc b.val move.val = true) : value mc fc b roots idx = scoreMax := by
  have atMost : value mc fc b roots idx ≤ scoreMax := by
    apply PvsMoveLoop.bestValue_le _ _ _ _ (by decide)
    intro m _
    have lower := (PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (play mc fc b m).val)).1
    change -scoreMax ≤ PvsBoardTree.minimax mc fc (play mc fc b m).val at lower
    unfold childValue
    omega
  have winning := EndgameCutoffs.wipeout_child_value mc fc b move wipes
  have dominates : childValue mc fc b move ≤ value mc fc b roots idx :=
    PvsMoveLoop.bestValue_ge_move _ _ _ chosen
  unfold childValue at dominates
  rw [winning] at dominates
  exact Int.le_antisymm atMost (by unfold scoreMax; omega)

inductive Execution (completed : Search) (mc : X86Config) (fc : FlipConfig) (b : Position)
    (roots : List Square) (idx : Nat) (alpha beta : Int) : Int → Prop
  | searched : tree mc fc b roots idx = some node → completed node true alpha beta score →
      Execution completed mc fc b roots idx alpha beta score
  | wipeout (move : GeneratedSquare mc b.val) : move ∈ choices mc b roots idx →
      EndgameCutoffs.wipesOut fc b.val move.val = true →
      Execution completed mc fc b roots idx alpha beta scoreMax

def ScoreContract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square) (idx : Nat)
      (alpha beta score : Int), Window alpha beta →
      Execution completed mc fc b roots idx alpha beta score →
      SoundResult (value mc fc b roots idx) alpha beta score

theorem scoreContract_proven : ScoreContract := by
  intro completed follows mc fc b roots idx alpha beta score window run
  cases run with
  | searched present trace =>
    rw [← tree_value present]
    exact PvsParallel.sound completed follows _ true alpha beta score window (by simp) trace
  | wipeout move chosen wipes =>
    rw [wipeout_value chosen wipes]
    exact ⟨by decide, by decide, fun _ => Int.le_refl _, fun _ => Int.le_refl _, fun _ _ => rfl⟩

def storeAllowed (root : Bool) (idx : Nat) : Bool := !root || idx == 0

def StoreGateContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square) (idx : Nat),
    Covers mc b roots → idx < roots.length →
    (storeAllowed true idx = true ↔ idx = 0) ∧
    (storeAllowed true idx = true → value mc fc b roots idx = PvsBoardTree.minimax mc fc b)

theorem storeGateContract_proven : StoreGateContract := by
  intro mc fc b roots idx covered inBounds
  have allowed : storeAllowed true idx = true ↔ idx = 0 := by simp [storeAllowed]
  refine ⟨allowed, fun stored => ?_⟩
  cases allowed.mp stored
  exact value_zero covered inBounds

end NeuralReversi.PvsParallel.Root
