module
public import NeuralReversi.PrincipalVariation.StackAccessModel
public import NeuralReversi.SearchContextState

@[expose] public section

namespace NeuralReversi.PrincipalVariation.StackAccess

def Shaped (stack : Stack) : Prop :=
  stack.length = capacity ∧ ∀ (i : Nat) (row : Buffer), stack[i]? = some row → row.length = capacity

private theorem replaced (stack : Stack) (ply : Nat) (row : Buffer)
    (shape : Shaped stack) (size : row.length = capacity) : Shaped (stack.set ply row) := by
  refine ⟨by simp [shape.1], ?_⟩
  intro i current hit
  by_cases same : i = ply
  · subst i
    have existsRow : ply < (stack.set ply row).length := List.getElem?_eq_some_iff.mp hit |>.1
    rw [List.length_set] at existsRow
    rw [List.getElem?_set_self existsRow] at hit
    exact (Option.some.inj hit) ▸ size
  · rw [List.getElem?_set_ne (Ne.symm same)] at hit
    exact shape.2 _ _ hit

private theorem fetch (stack : Stack) (ply : Nat) (shape : Shaped stack) :
    (stack[ply]? = none ↔ capacity ≤ ply) ∧
    (ply < capacity → ∃ row, stack[ply]? = some row ∧ row.length = capacity) := by
  refine ⟨by rw [List.getElem?_eq_none_iff, shape.1], ?_⟩
  intro inside
  have valid : ply < stack.length := by rwa [shape.1]
  have read := List.getElem?_eq_getElem valid
  exact ⟨_, read, shape.2 _ _ read⟩

def AccessContract : Prop :=
  Shaped initial ∧ (∀ ply, ply < capacity → initial[ply]? = some (List.replicate capacity none)) ∧
  ∀ stack, Shaped stack → ∀ ply row, row.length = capacity →
    (stack[ply]? = none ↔ capacity ≤ ply) ∧
    (put stack ply row = none ↔ capacity ≤ ply) ∧
    ∀ next, put stack ply row = some next →
      Shaped next ∧ next[ply]? = some row ∧ (∀ i, i ≠ ply → next[i]? = stack[i]?)

theorem accessContract_proven : AccessContract := by
  have emptyRow : ∀ ply, ply < capacity → initial[ply]? = some (List.replicate capacity none) := by
    intro ply inside
    simp [initial, inside]
  refine ⟨⟨by simp [initial], ?_⟩, emptyRow, ?_⟩
  · intro i row hit
    have inside := (List.getElem?_eq_some_iff.mp hit).1
    have valid : i < capacity := by simpa [initial] using inside
    exact (Option.some.inj ((emptyRow i valid).symm.trans hit)) ▸ List.length_replicate
  · intro stack shape ply row size
    obtain ⟨miss, hit⟩ := fetch stack ply shape
    refine ⟨miss, ?_, ?_⟩
    · simpa only [put, Option.map_eq_none_iff] using miss
    · intro next assigned
      obtain ⟨prior, priorHit, rfl⟩ := Option.map_eq_some_iff.mp assigned
      exact ⟨replaced stack ply row shape size,
        List.getElem?_set_self (List.getElem?_eq_some_iff.mp priorHit).1,
        fun _ different => List.getElem?_set_ne different.symm⟩

def ClearContract : Prop :=
  ∀ stack ply next, Shaped stack → clearAt stack ply = some next →
    Shaped next ∧ (∃ row, stack[ply]? = some row ∧ next[ply]? = some (clear row) ∧ visible (clear row) = []) ∧
    ∀ i, i ≠ ply → next[i]? = stack[i]?

theorem clearContract_proven : ClearContract := by
  intro stack ply next shape done
  obtain ⟨row, read, rfl⟩ := Option.map_eq_some_iff.mp done
  have len := shape.2 _ _ read
  have inside := (List.getElem?_eq_some_iff.mp read).1
  refine ⟨replaced stack ply (clear row) shape (by simp [clear, len, capacity]),
    ⟨row, read, List.getElem?_set_self inside, rfl⟩, fun _ different => List.getElem?_set_ne different.symm⟩

def PrepareAtContract : Prop :=
  ∀ stack ply, Shaped stack →
    (prepareAt stack ply = none ↔ capacity ≤ ply) ∧
    ∀ next, prepareAt stack ply = some next →
      Shaped next ∧
      (∀ i, i < capacity → (i = ply ∨ i = ply + 1) → ∃ row, next[i]? = some row ∧ visible row = []) ∧
      ∀ i, i ≠ ply → i ≠ ply + 1 → next[i]? = stack[i]?

theorem prepareAtContract_proven : PrepareAtContract := by
  intro stack ply shape
  have miss := (fetch stack ply shape).1
  refine ⟨by simpa only [prepareAt, Option.map_eq_none_iff] using miss, ?_⟩
  intro next done
  obtain ⟨_, _, rfl⟩ := Option.map_eq_some_iff.mp done
  have spec := prepareContract_proven stack ply
  have sized := spec.1.trans shape.1
  refine ⟨⟨sized, ?_⟩, ?_, spec.2.2⟩
  · intro i row hit
    have inside := (List.getElem?_eq_some_iff.mp hit).1
    have existsOld : i < stack.length := by rwa [spec.1] at inside
    have len := shape.2 i _ (List.getElem?_eq_getElem existsOld)
    have clearedSize : (clear stack[i]).length = capacity := by simp [clear, len, capacity]
    by_cases same : i = ply
    · subst i
      have before : (prepare stack ply)[ply]? = some (clear stack[ply]) := by simp [prepare, existsOld]
      exact Option.some.inj (before.symm.trans hit) ▸ clearedSize
    · by_cases child : i = ply + 1
      · subst i
        have before : (prepare stack ply)[ply + 1]? = some (clear stack[ply + 1]) := by simp [prepare, existsOld]
        exact Option.some.inj (before.symm.trans hit) ▸ clearedSize
      · exact shape.2 _ _ ((spec.2.2 i same child).symm.trans hit)
  · intro i inside affected
    exact spec.2.1 i (by rwa [shape.1]) affected

def PromoteContract : Prop :=
  ∀ stack ply sq, Shaped stack →
    (promote stack ply sq = none ↔ capacity ≤ ply + 1) ∧
    ∀ next, promote stack ply sq = some next →
      Shaped next ∧
      (∃ parent child, stack[ply]? = some parent ∧ stack[ply+1]? = some child ∧
        next[ply]? = some (prepend sq child parent) ∧
        visible (prepend sq child parent) = sq :: (visible child).take (capacity - 2)) ∧
      ∀ i, i ≠ ply → next[i]? = stack[i]?

theorem promoteContract_proven : PromoteContract := by
  intro stack ply sq shape
  obtain ⟨parentNone, parentSome⟩ := fetch stack ply shape
  obtain ⟨childNone, childSome⟩ := fetch stack (ply + 1) shape
  refine ⟨?_, ?_⟩
  · by_cases inside : ply + 1 < capacity
    · obtain ⟨parent, p, _⟩ := parentSome (by omega)
      obtain ⟨child, c, _⟩ := childSome inside
      simp [promote, p, c, Nat.not_le_of_lt inside]
    · have miss := childNone.mpr (by omega)
      cases p : stack[ply]? <;> simp [promote, p, miss, show capacity ≤ ply + 1 by omega]
  · intro next written
    obtain ⟨parent, p, rest⟩ := Option.bind_eq_some_iff.mp written
    obtain ⟨child, c, same⟩ := Option.bind_eq_some_iff.mp rest
    have sameStack : stack.set ply (prepend sq child parent) = next := Option.some.inj same
    subst next
    have buffer := bufferContract_proven sq child parent (shape.2 _ _ p)
    exact ⟨replaced stack ply _ shape buffer.2.1,
      ⟨parent, child, p, c, List.getElem?_set_self (List.getElem?_eq_some_iff.mp p).1, buffer.2.2.1⟩,
      fun _ different => List.getElem?_set_ne different.symm⟩

def SiblingContract : Prop :=
  ∀ stack ply first second searched clean next, Shaped stack →
    promote stack ply first = some searched → clearAt searched (ply + 1) = some clean →
    promote clean ply second = some next →
    (∃ row, next[ply]? = some row ∧ visible row = [second]) ∧
    (∀ i, i ≠ ply → i ≠ ply + 1 → next[i]? = stack[i]?)

theorem siblingContract_proven : SiblingContract := by
  intro stack ply first second searched clean next shape firstResult cleared secondResult
  have firstSpec := (promoteContract_proven stack ply first shape).2 searched firstResult
  have clearSpec := clearContract_proven searched (ply + 1) clean firstSpec.1 cleared
  have secondSpec := (promoteContract_proven clean ply second clearSpec.1).2 next secondResult
  obtain ⟨row, _, blank, empty⟩ := clearSpec.2.1
  obtain ⟨parent, child, _, read, published, line⟩ := secondSpec.2.1
  have childEmpty : visible child = [] := Option.some.inj (read.symm.trans blank) ▸ empty
  refine ⟨⟨_, published, by simp [line, childEmpty]⟩, ?_⟩
  intro i parentOther childOther
  rw [secondSpec.2.2 _ parentOther, clearSpec.2.2 _ childOther, firstSpec.2.2 _ parentOther]

open SearchContextState (State Matches ply passTurn placement undo)
open PvsBoardTree (Position GeneratedSquare)

def ContextRowsContract : Prop :=
  ∀ (s : State) (board : Position) stack, Matches s board → Shaped stack →
    (∃ row, stack[ply s]? = some row) ∧ stack[ply (passTurn s)]? = stack[ply s]? ∧
    ∀ mc fc (move : GeneratedSquare mc board.val),
      let flipped := Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent
      let child := placement s move.val flipped
      (∃ row, stack[ply child]? = some row) ∧ ply child = ply s + 1 ∧
      ply (undo child) = ply s ∧ ∃ next, promote stack (ply s) move.val = some next

theorem contextRowsContract_proven : ContextRowsContract := by
  intro s board stack agrees shape
  have parentIn : ply s < capacity := by unfold ply capacity; omega
  obtain ⟨row, parent, _⟩ := (fetch stack (ply s) shape).2 parentIn
  refine ⟨⟨row, parent⟩, rfl, ?_⟩
  intro mc fc move flipped child
  obtain ⟨_, _, childPly, _, restored, _, _, _, _, below⟩ :=
    SearchContextState.placementContract_proven mc fc board move s agrees
  have childIn : ply child < capacity := by unfold capacity; exact below
  have restoredPly : ply (undo child) = ply s := by unfold ply; rw [restored]
  obtain ⟨nextRow, childRead, _⟩ := (fetch stack (ply child) shape).2 childIn
  refine ⟨⟨nextRow, childRead⟩, childPly, restoredPly, ?_⟩
  have hc : stack[ply s + 1]? = some nextRow := childPly ▸ childRead
  exact ⟨stack.set (ply s) (prepend move.val nextRow row), by simp [promote, parent, hc]⟩

end NeuralReversi.PrincipalVariation.StackAccess
