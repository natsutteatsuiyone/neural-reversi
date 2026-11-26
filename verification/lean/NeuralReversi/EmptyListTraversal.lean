import NeuralReversi.EmptyListRemoval
import NeuralReversi.Square.Model

namespace NeuralReversi.EmptyListTraversal

open EmptyListLinks EmptyListInit

def noSuccessor (node : Index) : String :=
  s!"no empty square follows node {node.val}"

def squareAfter (links : Links) (node : Index) : Except String Square :=
  if h : (links node).next.val < 64 then .ok ⟨(links node).next.val, h⟩
  else .error (noSuccessor node)

private theorem chain_step {links : Links} {last : Index} {xs : List Square}
    (valid : chain links last (xs.map slot)) :
    ∀ (i : Nat) (hi : i + 1 < xs.length),
      (links (slot xs[i])).next = slot xs[i + 1] := by
  induction xs generalizing last with
  | nil =>
    intro i hi
    simp at hi
  | cons a rest ih =>
    intro i hi
    cases rest with
    | nil => simp at hi
    | cons b tail =>
      obtain ⟨_, restChain⟩ := valid
      cases i with
      | zero =>
        exact restChain.1.1
      | succ j =>
        exact ih restChain j (by simpa using hi)

def SuccessorContract : Prop :=
  (∀ (links : Links) (node : Index),
    ((links node).next ≠ head ↔ (links node).next.val < 64) ∧
    ∀ square, squareAfter links node = .ok square ↔ (links node).next = slot square) ∧
  (∀ (links : Links) (xs : List Square), chain links head (xs.map slot) →
    (∀ (h : 0 < xs.length),
      (links head).next = slot xs[0] ∧ (links head).next ≠ head ∧ (links head).next.val < 64) ∧
    (∀ (i : Nat) (hi : i + 1 < xs.length),
      (links (slot xs[i])).next = slot xs[i + 1] ∧
      (links (slot xs[i])).next ≠ head ∧
      (links (slot xs[i])).next.val < 64))

theorem successorContract_proven : SuccessorContract := by
  refine ⟨?_, ?_⟩
  · intro links node
    have max := (links node).next.isLt
    have headValue : head.val = 64 := by decide
    refine ⟨?_, ?_⟩
    · constructor
      · intro neHead
        have neVal : (links node).next.val ≠ 64 := fun h =>
          neHead (Fin.ext (h.trans headValue.symm))
        omega
      · intro below same
        rw [same, headValue] at below
        omega
    · intro square
      unfold squareAfter
      by_cases playable : (links node).next.val < 64
      · simp only [playable, ↓reduceDIte, Except.ok.injEq]
        exact ⟨fun equal => Fin.ext (congrArg (fun i : Square => i.val) equal),
          fun equal => Fin.ext (congrArg (fun i : Index => i.val) equal)⟩
      · have isHead : (links node).next = head := Fin.ext (by omega)
        rw [dite_eq_right playable]
        simp [isHead, (slot_ne_head square).symm]
  · intro links xs valid
    constructor
    · intro nonempty
      cases xs with
      | nil => simp at nonempty
      | cons first rest =>
        have edge := valid.1.1
        refine ⟨edge, ?_, ?_⟩
        · rw [edge]
          exact slot_ne_head first
        · rw [edge]
          exact first.isLt
    · intro i hi
      have edge := chain_step valid i hi
      refine ⟨edge, ?_, ?_⟩
      · rw [edge]
        exact slot_ne_head xs[i + 1]
      · rw [edge]
        exact xs[i + 1].isLt

theorem initial_first (empties : Bitboard.Bits) :
    match scanOrder empties with
    | [] => (newLinks empties head).next = head
    | first :: _ =>
      ∃ square, first = slot square ∧ (newLinks empties head).next = slot square ∧
        (newLinks empties head).next.val < 64 := by
  have path := (initContract_proven empties).1
  cases order : scanOrder empties with
  | nil =>
    rw [order] at path
    exact path.1
  | cons first rest =>
    rw [order] at path
    have listed : first ∈ scanOrder empties := by simp [order]
    obtain ⟨square, _, same⟩ := List.mem_map.mp listed
    have edge := path.1.1.trans same.symm
    exact ⟨square, same.symm, edge, by rw [edge]; exact square.isLt⟩

def RestoreCapacityContract : Prop :=
  ∀ empties square, slot square ∈ scanOrder empties →
    ((scanOrder empties).erase (slot square)).length < 64 ∧
    ((scanOrder empties).erase (slot square)).length + 1 = (scanOrder empties).length

theorem restoreCapacityContract_proven : RestoreCapacityContract := by
  intro empties square present
  have bound : (scanOrder empties).length ≤ 64 := by
    have full : EmptyListParity.presorted.length = 64 := by decide
    simpa [scanOrder, full] using List.length_filter_le
      (fun sq : Square => empties.getLsbD sq.val) EmptyListParity.presorted
  have nonempty : 0 < (scanOrder empties).length := List.length_pos_iff.mpr (by
    intro nil
    simp [nil] at present)
  rw [List.length_erase_of_mem present]
  omega

def QuadrantContract : Prop :=
  ∀ square : Square, (EmptyListParity.quadrantId square).toNat =
    2 ^ (2 * (square.val / 32) + square.val % 8 / 4)

theorem quadrantContract_proven : QuadrantContract := by
  unfold QuadrantContract
  decide

end NeuralReversi.EmptyListTraversal
