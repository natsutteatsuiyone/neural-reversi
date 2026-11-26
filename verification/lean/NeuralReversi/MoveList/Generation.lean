import NeuralReversi.MoveList.SmallGenerate
import NeuralReversi.MoveList.Wipeout
import NeuralReversi.LegalMoves.Encoding
import NeuralReversi.Move.Model

namespace NeuralReversi.MoveList.Generation
open Capacity Initialization
open Bitboard (Bits setIndices)
open Board (Board)

def squares (selected : Bits) : List (Fin 64) :=
  (List.finRange 64).filter (selected.getLsbD ∘ Fin.val)

theorem squareIndices (selected : Bits) : (squares selected).map Fin.val = setIndices selected := by
  have indices : (List.finRange 64).map Fin.val = List.range 64 := by decide
  rw [squares, ← List.filter_map, indices, Bitboard.setIndices]

def entry (fc : Flip.Dispatch.X86Config) (board : Board) (sq : Fin 64) : Move.Move :=
  ⟨Flip.Dispatch.resultValue fc sq board.player board.opponent, 0, sq⟩

def records (fc : Flip.Dispatch.X86Config) (board : Board) (selected : Bits) : List Move.Move :=
  (squares selected).map (entry fc board)

theorem records_length (fc : Flip.Dispatch.X86Config) (board : Board) (selected : Bits) :
    (records fc board selected).length = population selected := by
  have len := congrArg List.length (squareIndices selected)
  rw [List.length_map] at len
  rw [records, List.length_map, len]
  exact (Bitboard.iteratorEnumerationContract_proven selected false).2.2.2.1

def IsSubset (selected legal : Bits) : Prop :=
  ∀ i, selected.getLsbD i = true → legal.getLsbD i = true

theorem entry_checked (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Board) (wf : Board.wellFormed board) (selected : Bits)
    (subset : IsSubset selected (Board.getMoves mc board)) (mv : Move.Move)
    (member : mv ∈ records fc board selected) : Move.create true mv.sq mv.flipped = .ok mv := by
  obtain ⟨sq, hs, rfl⟩ := List.mem_map.mp member
  have flip := (Board.checkedAcceptsGeneratedMoves_proven mc fc board sq wf
    (subset sq.val (List.mem_filter.mp hs).2)).1
  have nonzero : Flip.Dispatch.resultValue fc sq board.player board.opponent ≠ 0#64 := flip
  simp [entry, Move.create, nonzero]

def candidatesForCache (moves : List Move.Move) : List Wipeout.Move :=
  moves.map fun mv => ⟨mv.sq, mv.flipped⟩

theorem generationCache_last (opponent : Bits) (moves : List Wipeout.Move)
    (initial : Option (Fin 64)) :
    Wipeout.generationCache opponent moves initial =
      ((moves.reverse.find? (Wipeout.isMatch opponent)).map Wipeout.Move.square).or initial := by
  induction moves generalizing initial with
  | nil => rfl
  | cons mv rest ih =>
    rw [Wipeout.generationCache, ih, List.reverse_cons, List.find?_append]
    cases rest.reverse.find? (Wipeout.isMatch opponent) <;>
      cases hit : Wipeout.isMatch opponent mv <;> simp [hit]

def fill (skipEmpty paired : Bool) (fc : Flip.Dispatch.X86Config) (board : Board)
    (selected : Bits) (mem : Memory Move.Move) : Result Move.Move (Fin 64) :=
  let moves := records fc board selected
  let values := fun i => moves[i]?.getD ⟨0, 0, 0⟩
  let cache := Wipeout.generationCache board.opponent (candidatesForCache moves) none
  match dispatch skipEmpty selected with
  | .empty => finish mem 0 none
  | .single => finish (scalar values 1 0 mem) 1 cache
  | .bulk =>
    if paired then finish (sourcePairRun values selected mem) (pairFinalLen selected) cache
    else finish (scalar values (population selected) 0 mem) (population selected) cache

private theorem empty_dispatch (skipEmpty : Bool) (selected : Bits)
    (empty : dispatch skipEmpty selected = .empty) : selected = 0 := by
  unfold dispatch at empty
  split at empty
  · exact eq_of_beq (Bool.and_eq_true_iff.mp ‹_›).2
  · split at empty <;> contradiction

private theorem fill_normalized (skipEmpty paired : Bool) (fc : Flip.Dispatch.X86Config)
    (board : Board) (selected : Bits) (mem : Memory Move.Move) (nonempty : skipEmpty = true → selected ≠ 0) :
    let moves := records fc board selected
    fill skipEmpty paired fc board selected mem =
      finish (scalar (fun i => moves[i]?.getD ⟨0, 0, 0⟩) (population selected) 0 mem)
        (population selected) (Wipeout.generationCache board.opponent (candidatesForCache moves) none) := by
  dsimp only
  cases choice : dispatch skipEmpty selected with
  | empty =>
    have zero := empty_dispatch skipEmpty selected choice
    subst selected
    simp only [fill, choice]
    rfl
  | single => simp only [fill, choice, (single_dispatch_safe skipEmpty selected nonempty choice).1]
  | bulk => cases paired <;> simp [fill, choice, sourcePairRun_eq_scalar, pair_final_length]

def GenerationContract : Prop :=
  ∀ mc fc board selected skipEmpty paired mem,
    Board.wellFormed board → IsSubset selected (Board.getMoves mc board) →
    (skipEmpty = true → selected ≠ 0) →
    let moves := records fc board selected
    let result := fill skipEmpty paired fc board selected mem
    result.len = some moves.length ∧
    (∀ i, result.slots i = if i < moves.length then moves[i]? else mem i) ∧
    moves.map (fun mv => mv.sq.val) = setIndices selected ∧
    (∀ mv ∈ moves, Move.create true mv.sq mv.flipped = .ok mv) ∧
    result.wipeout = some (((candidatesForCache moves).reverse.find?
      (Wipeout.isMatch board.opponent)).map Wipeout.Move.square)

theorem generationContract_proven : GenerationContract := by
  intro mc fc board selected skipEmpty paired mem wf subset nonempty
  dsimp only
  rw [fill_normalized skipEmpty paired fc board selected mem nonempty]
  have len := records_length fc board selected
  refine ⟨congrArg some len.symm, ?_, ?_, entry_checked mc fc board wf selected subset, ?_⟩
  · intro i
    change scalar _ _ 0 mem i = _
    rw [scalar_prefix, ← len]
    split
    next hit => simp [List.getElem?_eq_getElem hit]
    next => rfl
  · simpa only [records, List.map_map, Function.comp_def, entry] using squareIndices selected
  · simp [finish, generationCache_last]

private theorem popIndices (selected : Bits) (nonzero : selected ≠ 0) :
    setIndices selected = selected.ctz.toNat :: setIndices (Bitboard.clearLsb selected) := by
  let sq : Fin 64 := ⟨selected.ctz.toNat, (pop_progress selected nonzero).2⟩
  exact Bitboard.setIndices_pop selected (Bitboard.clearLsb selected) sq
    ⟨sq.isLt, BitVec.getLsbD_true_ctz_of_ne_zero nonzero,
      fun _ hi => BitVec.getLsbD_false_of_lt_ctz hi⟩ (Bitboard.clearLsb_ctz selected nonzero)

private theorem remaining_clear (bits : Bits) (i : Nat) :
    remaining (Bitboard.clearLsb bits) i = remaining bits (i + 1) := by
  induction i with
  | zero => rfl
  | succ i ih => exact congrArg Bitboard.clearLsb ih

private theorem nextIndex (selected : Bits) (i : Nat) (active : i < population selected) :
    (setIndices selected)[i]? = some (remaining selected i).ctz.toNat := by
  induction i generalizing selected with
  | zero =>
    have nonzero : selected ≠ 0 := by
      intro zero
      have := (population_zero_iff _).mpr zero
      omega
    rw [popIndices selected nonzero]; rfl
  | succ i ih =>
    have nonzero : selected ≠ 0 := by
      intro zero
      have := (population_zero_iff _).mpr zero
      omega
    have progress := (pop_progress selected nonzero).1
    rw [popIndices selected nonzero, List.getElem?_cons_succ, ← remaining_clear]
    exact ih (Bitboard.clearLsb selected) (by omega)

def EnumerationContract : Prop :=
  (∀ selected i, i < population selected → (setIndices selected)[i]? = some (remaining selected i).ctz.toNat) ∧
  (∀ fc board x y, Board.wellFormed board →
    Flip.Avx512.WidePair.flipPair x y board.player board.opponent =
      ((entry fc board x).flipped, (entry fc board y).flipped) ∧
    Flip.Avx512.WidePair.singleMin x board.player board.opponent = (entry fc board x).flipped)

theorem enumerationContract_proven : EnumerationContract := by
  refine ⟨nextIndex, ?_⟩
  intro fc board x y wf
  refine ⟨SmallGenerate.widePair_eq_generatedMasks fc board wf x y, ?_⟩
  exact (Flip.Avx512.WidePair.singleMin_eq _ _ _).trans (Flip.Dispatch.value_disjoint_scalar _ _ _ _ wf).symm

def WriteSafetyContract : Prop :=
  ∀ selected skipEmpty, population selected ≤ 35 → (skipEmpty = true → selected ≠ 0) →
    (dispatch skipEmpty selected = .empty → population selected = 0) ∧
    (dispatch skipEmpty selected = .single → population selected = 1 ∧ selected.ctz.toNat < 64) ∧
    (∀ i, i < population selected → i < 35 ∧ (remaining selected i).ctz.toNat < 64) ∧
    (∀ pair, pair < population selected / 2 →
      2 * pair + 1 < 35 ∧ (remaining selected (2 * pair)).ctz.toNat < 64 ∧
      (remaining selected (2 * pair + 1)).ctz.toNat < 64) ∧
    (remaining selected (2 * (population selected / 2)) ≠ 0 →
      2 * (population selected / 2) < 35 ∧
      (remaining selected (2 * (population selected / 2))).ctz.toNat < 64) ∧
    pairFinalLen selected = population selected

theorem writeSafetyContract_proven : WriteSafetyContract := by
  intro selected skipEmpty bound nonempty
  refine ⟨fun empty => (population_zero_iff _).mpr (empty_dispatch _ _ empty),
    single_dispatch_safe _ _ nonempty, ?_, ?_, ?_, pair_final_length _⟩
  · intro i active
    obtain ⟨slot, square, _⟩ := scalar_write_safe selected bound i active
    exact ⟨slot, square⟩
  · intro pair active
    obtain ⟨_, slot, first, second, _⟩ := pair_write_safe selected bound pair active
    exact ⟨slot, first, second⟩
  · intro tail
    obtain ⟨slot, square, _⟩ := odd_tail_safe selected bound tail
    exact ⟨slot, square⟩

def CertifiedCapacityContract : Prop :=
  ∀ proof, Std.Tactic.BVDecide.LRAT.check proof (LegalMoves.MoveBound.dimacsFormula 35) = true →
    ∀ mc board selected, Board.wellFormed board → IsSubset selected (Board.getMoves mc board) →
      population selected ≤ 35

theorem certifiedCapacityContract_proven : CertifiedCapacityContract := by
  intro proof checked mc board selected wf subset
  have legal := LegalMoves.MoveBound.certifiedBoundContract_proven 35 proof (by decide) checked
    (.x86_64 ⟨false, mc.avx512vl, mc.avx2⟩) board.player board.opponent wf
  change population (Board.getMoves mc board) ≤ 35 at legal
  refine Nat.le_trans ?_ legal
  simp only [population, ← (Bitboard.iteratorEnumerationContract_proven selected false).2.2.2.1,
    ← (Bitboard.iteratorEnumerationContract_proven (Board.getMoves mc board) false).2.2.2.1,
    Bitboard.setIndices, ← List.countP_eq_length_filter]
  exact List.countP_mono_left (fun i _ h => subset i h)

end NeuralReversi.MoveList.Generation
