module
public import NeuralReversi.PvsParallel.RootFrameMir.Program
public import NeuralReversi.MoveList.Generation
public import NeuralReversi.MoveList.Ordering
public import NeuralReversi.PvsParallel.Root

@[expose] public section

namespace NeuralReversi.PvsParallel.RootFrameMir

open Bitboard (Square)
open PvsBoardTree (Position GeneratedSquare)
open MoveList

theorem generated_squares (mc : Bitboard.Moves.X86Config) (b : Position) (roots : List Square) (idx : Nat) :
    Generation.squares (Root.remainingMask mc b.val roots idx) =
      (Root.choices mc b roots idx).map Subtype.val := by
  unfold Root.choices
  let suffix := Root.mask (roots.drop idx)
  change _ = List.map Subtype.val (List.filter ((fun (s : Square) => suffix.getLsbD s.val) ∘ Subtype.val) _)
  rw [← List.filter_map, PvsBoardTree.legalSquares_values, List.filter_filter]
  apply congrArg (fun test => (List.finRange 64).filter test)
  funext sq
  simp [Root.remainingMask, suffix, Bool.and_comm]

def asChoice (mc : Bitboard.Moves.X86Config) (b : Position) (mv : Move.Move) :
    Option (GeneratedSquare mc b.val) :=
  if valid : (Board.getMoves mc b.val).getLsbD mv.sq.val = true then some ⟨mv.sq, valid⟩ else none

def setKey (key : Square → BitVec 32) (mv : Move.Move) : Move.Move :=
  { mv with value := key mv.sq }

theorem exists_key (before after : List Move.Move)
    (unchanged : before.map (fun m => (m.sq, m.flipped)) = after.map (fun m => (m.sq, m.flipped)))
    (unique : (before.map Move.Move.sq).Nodup) :
    ∃ key : Square → BitVec 32, before.map (setKey key) = after := by
  induction before generalizing after with
  | nil =>
    cases after <;> simp_all
  | cons m tail ih =>
    cases after with
    | nil => simp at unchanged
    | cons n rest =>
      simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at unchanged
      obtain ⟨⟨square, flips⟩, restSame⟩ := unchanged
      obtain ⟨absent, distinct⟩ := List.nodup_cons.mp unique
      obtain ⟨restKey, keyed⟩ := ih rest restSame distinct
      refine ⟨fun sq => if sq = m.sq then n.value else restKey sq, ?_⟩
      simp only [List.map_cons, List.cons.injEq]
      refine ⟨?_, ?_⟩
      · cases m
        cases n
        simp_all [setKey]
      · rw [← keyed]
        apply List.map_congr_left
        intro h member
        have different : h.sq ≠ m.sq := fun eq => absent (eq ▸ List.mem_map_of_mem member)
        simp [setKey, different]

theorem generated_slots (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (b : Position) (roots : List Square) (idx : Nat) (paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) :
    let choices := Root.choices mc b roots idx
    let filled := Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem
    filled.len = some choices.length ∧
      ∀ i, i < choices.length → filled.slots i = (choices[i]?).map (fun m => Generation.entry fc b.val m.val) := by
  dsimp only
  let mask := Root.remainingMask mc b.val roots idx
  have subset : Generation.IsSubset mask (Board.getMoves mc b.val) := by
    intro i
    simp only [mask, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true]
    exact And.left
  have h := Generation.generationContract_proven mc fc b.val mask false paired mem b.property subset (by simp)
  have list : Generation.records fc b.val mask =
    (Root.choices mc b roots idx).map (fun m => Generation.entry fc b.val m.val) := by
    simp [Generation.records, mask, generated_squares]
  refine ⟨by simpa [list] using h.1, ?_⟩
  intro i inside
  simpa [list, inside] using h.2.1 i

theorem order_of_permutation (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (b : Position) (roots : List Square) (idx : Nat) (key : Square → BitVec 32) (moves : List Move.Move)
    (order : moves.Perm ((Generation.records fc b.val (Root.remainingMask mc b.val roots idx)).map (setKey key))) :
    (moves.filterMap (asChoice mc b)).Perm (Root.choices mc b roots idx) ∧
      ∀ m ∈ moves, m.flipped = Flip.Dispatch.resultValue fc m.sq b.val.player b.val.opponent := by
  have generated : (Generation.records fc b.val (Root.remainingMask mc b.val roots idx)).map (setKey key) =
      (Root.choices mc b roots idx).map (fun m => setKey key (Generation.entry fc b.val m.val)) := by
    simp [Generation.records, generated_squares, List.map_map, Function.comp_def]
  rw [generated] at order
  have choice (m : GeneratedSquare mc b.val) :
      asChoice mc b (setKey key (Generation.entry fc b.val m.val)) = some m := by
    simp [asChoice, setKey, Generation.entry, m.property]
  refine ⟨?_, ?_⟩
  · have result := order.filterMap (asChoice mc b)
    simpa only [List.filterMap_map, Function.comp_def, choice, List.filterMap_some] using result
  · intro m hm
    obtain ⟨candidate, _, rfl⟩ := List.mem_map.mp (order.mem_iff.mp hm)
    rfl

theorem sort_order (key : Square → BitVec 32) (records : List Move.Move) :
    (MoveList.Ordering.sortByKeyDesc (fun move => move.value.toInt) (records.map (setKey key))).Perm
      (records.map (setKey key)) :=
  MoveList.Ordering.sortByKeyDesc_perm _ _

/-- The library may choose any tie order for whole moves in the initialized slice. -/
abbrev LibraryOrder := (records : List Move.Move) → {output : List Move.Move // output.Perm records}

def orderedRecords (key : Square → BitVec 32) (library : LibraryOrder) (records : List Move.Move) :
    List Move.Move :=
  if 1 < records.length then
    let keyed := records.map (setKey key)
    if records.length ≤ 3 then MoveList.Ordering.sortByKeyDesc (fun m => m.value.toInt) keyed
    else (library keyed).val
  else records

theorem ordered_candidates (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (b : Position) (roots : List Square) (idx : Nat) (key : Square → BitVec 32) (library : LibraryOrder) :
    let moves := orderedRecords key library (Generation.records fc b.val (Root.remainingMask mc b.val roots idx))
    (moves.filterMap (asChoice mc b)).Perm (Root.choices mc b roots idx) ∧
    ∀ m ∈ moves, m.flipped = Flip.Dispatch.resultValue fc m.sq b.val.player b.val.opponent := by
  dsimp only
  let records := Generation.records fc b.val (Root.remainingMask mc b.val roots idx)
  by_cases multiple : 1 < records.length
  · apply order_of_permutation mc fc b roots idx key
    simp only [orderedRecords, show 1 < (Generation.records fc b.val (Root.remainingMask mc b.val roots idx)).length
      from multiple, ↓reduceIte]
    split
    · exact sort_order _ _
    · exact (library _).property
  · apply order_of_permutation mc fc b roots idx (fun _ => 0)
    rw [orderedRecords, ite_eq_right (show ¬ 1 < (Generation.records fc b.val (Root.remainingMask mc b.val roots idx)).length
      from multiple)]
    unfold Generation.records
    simp [List.map_map, setKey, Generation.entry, Function.comp_def]
    exact .refl _

end NeuralReversi.PvsParallel.RootFrameMir
