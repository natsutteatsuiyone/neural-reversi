module
public import NeuralReversi.Threading.SplitWindow

@[expose] public section

namespace NeuralReversi.Threading.SplitWindow

open AtomicWord (Write signed32At)
open Aspiration (Window)

inductive AlphaStores (pv : Bool) (beta : Int) (initial : State) : List Int → List Write → Prop
  | initialized (write : Write) : write.value.setWidth 32 = BitVec.ofInt 32 initial.alpha →
      AlphaStores pv beta initial [] [write]
  | unchanged (score : Int) :
      AlphaStores pv beta initial reports writes →
      ¬raisesAlpha pv beta (scan pv beta initial reports) score →
      AlphaStores pv beta initial (reports ++ [score]) writes
  | raised (score : Int) (write : Write) :
      AlphaStores pv beta initial reports writes →
      raisesAlpha pv beta (scan pv beta initial reports) score →
      write.value.setWidth 32 = BitVec.ofInt 32 score →
      AlphaStores pv beta initial (reports ++ [score]) (writes ++ [write])

private theorem AlphaStores.prefix (stores : AlphaStores pv beta initial reports writes)
    {i : Nat} {write : Write} (found : writes[i]? = some write) :
    ∃ earlier rest, reports = earlier ++ rest ∧
      write.value.setWidth 32 = BitVec.ofInt 32 (scan pv beta initial earlier).alpha := by
  induction stores with
  | initialized first initialized =>
    obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp found
    have zero : i = 0 := by simp at inside; omega
    subst i
    have same : first = write := same
    subst write
    exact ⟨[], [], rfl, initialized⟩
  | unchanged score _ _ ih =>
    obtain ⟨earlier, rest, reportsEq, value⟩ := ih found
    exact ⟨earlier, rest ++ [score], by rw [reportsEq, List.append_assoc], value⟩
  | @raised reports writes score latest _ raises value ih =>
    by_cases prior : i < writes.length
    · obtain ⟨earlier, rest, reportsEq, word⟩ := ih ((List.getElem?_append_left prior).symm.trans found)
      exact ⟨earlier, rest ++ [score], by rw [reportsEq, List.append_assoc], word⟩
    · obtain ⟨inside, _⟩ := List.getElem?_eq_some_iff.mp found
      have last : i = writes.length := by simp at inside; omega
      subst i
      have same := Option.some.inj ((show (writes ++ [latest])[writes.length]? = some latest by
        simp).symm.trans found)
      have scanned : scan pv beta initial (reports ++ [score]) =
          publish pv beta (scan pv beta initial reports) score := by
        simp only [scan, List.foldl_append, List.foldl_cons, List.foldl_nil]
      refine ⟨reports ++ [score], [], by simp, ?_⟩
      rw [← same, scanned]
      change _ = BitVec.ofInt 32 (if raisesAlpha pv beta (scan pv beta initial reports) score
        then score else (scan pv beta initial reports).alpha)
      rw [ite_eq_left raises]
      exact value

def StoreWindow (writes : List Write) (first last : Fin writes.length) (pv : Bool)
    (beta : Int) (initial : State) (reports : List Int) : Prop :=
  ∃ before current after, writes = before ++ current ++ after ∧
    first.val = before.length ∧ last.val + 1 = before.length + current.length ∧
    AlphaStores pv beta initial reports current

def StoreContract : Prop :=
  ∀ (writes : List Write) (first last : Fin writes.length) pv original beta initial reports,
    StoreWindow writes first last pv beta initial reports →
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    Matches writes first last pv beta initial reports

theorem storeContract_proven : StoreContract := by
  intro writes first last pv original beta initial reports stores window capped i afterInit beforeEnd
  obtain ⟨before, current, after, writeOrder, firstEq, lastEq, trace⟩ := stores
  let index := i.val - before.length
  have inside : index < current.length := by dsimp [index]; omega
  have currentRead : writes[i.val]? = some (current[index]'inside) := by
    calc
      writes[i.val]? = (before ++ current ++ after)[i.val]? :=
        congrArg (fun (ws : List Write) => ws[i.val]?) writeOrder
      _ = some (current[index]'inside) := by
        rw [List.getElem?_append_left (by simp; omega), List.getElem?_append_right (by omega)]
        exact List.getElem?_eq_getElem inside
  have same : writes.get i = current[index] := Option.some.inj
    ((List.getElem?_eq_getElem i.isLt).symm.trans currentRead)
  obtain ⟨earlier, rest, order, raw⟩ := trace.prefix (List.getElem?_eq_getElem inside)
  obtain ⟨_, window, _, _⟩ := stateContract_proven pv original beta initial earlier window capped
  have ⟨lower, nonempty, upper⟩ := window
  refine ⟨earlier, rest, order, ?_⟩
  unfold signed32At
  rw [same, raw]
  exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

private theorem scan_best (pv : Bool) (beta : Int) (initial : State) (reports : List Int) :
    (scan pv beta initial reports).best = reports.foldl max initial.best := by
  induction reports generalizing initial with
  | nil => rfl
  | cons score rest ih =>
    change (scan pv beta (publish pv beta initial score) rest).best =
      rest.foldl max (max initial.best score)
    exact ih _

inductive BestStores (initial : Int) : List Int → List Write → Prop
  | initialized (write : Write) : (write.value.setWidth 32).toInt = initial →
      BestStores initial [] [write]
  | unchanged (score : Int) : BestStores initial reports writes →
      score ≤ reports.foldl max initial →
      BestStores initial (reports ++ [score]) writes
  | improved (score : Int) (write : Write) : BestStores initial reports writes →
      reports.foldl max initial < score → (write.value.setWidth 32).toInt = score →
      BestStores initial (reports ++ [score]) (writes ++ [write])

private theorem BestStores.lastWrite (trace : BestStores initial reports writes) :
    ∃ earlier write, writes = earlier ++ [write] ∧
      (write.value.setWidth 32).toInt = reports.foldl max initial := by
  induction trace with
  | initialized write initialValue => exact ⟨[], write, rfl, initialValue⟩
  | unchanged score _ kept ih =>
    obtain ⟨earlier, write, order, value⟩ := ih
    exact ⟨earlier, write, order, by
      simpa only [List.foldl_append, List.foldl_cons, List.foldl_nil, Int.max_eq_left kept] using value⟩
  | improved score write _ better value _ =>
    exact ⟨_, write, rfl, by
      simpa only [List.foldl_append, List.foldl_cons, List.foldl_nil,
        Int.max_eq_right (Int.le_of_lt better)] using value⟩

def BestStoreWindow (writes : List Write) (last : Fin writes.length)
    (initial : Int) (reports : List Int) : Prop :=
  ∃ before current after, writes = before ++ current ++ after ∧
    last.val + 1 = before.length + current.length ∧ BestStores initial reports current

def BestStoreContract : Prop :=
  ∀ (writes : List Write) (last : Fin writes.length) pv beta initial reports,
    BestStoreWindow writes last initial.best reports →
    signed32At writes last = (scan pv beta initial reports).best

theorem bestStoreContract_proven : BestStoreContract := by
  intro writes last pv beta initial reports stores
  obtain ⟨before, current, after, writeOrder, lastEq, trace⟩ := stores
  obtain ⟨earlier, write, rfl, value⟩ := trace.lastWrite
  have index : last.val = (before ++ earlier).length := by simp at lastEq ⊢; omega
  have actual : writes[last.val]? = some write := calc
    writes[last.val]? = ((before ++ earlier) ++ write :: after)[last.val]? := by
      apply congrArg (fun (ws : List Write) => ws[last.val]?)
      simpa only [List.append_assoc, List.singleton_append] using writeOrder
    _ = some write := by
      rw [index, List.getElem?_append_right (Nat.le_refl _)]
      simp
  have same : writes.get last = write := Option.some.inj
    ((List.getElem?_eq_getElem last.isLt).symm.trans actual)
  unfold signed32At
  rw [same, scan_best]
  exact value

open SplitResult (Report)

inductive MoveStores (encode : Option Move → BitVec 8) (moveAt : Nat → Option Move)
    (pv : Bool) (beta : Int) (initial : State) (seed : Option Move) :
    List Report → Option Move → List Write → Prop
  | initialized (write : Write) : write.value.setWidth 8 = encode seed →
      MoveStores encode moveAt pv beta initial seed [] seed [write]
  | unchanged (report : Report) :
      MoveStores encode moveAt pv beta initial seed reports picked writes →
      ¬choosesMove (scan pv beta initial (reports.map Report.score)) report.score →
      MoveStores encode moveAt pv beta initial seed (reports ++ [report]) picked writes
  | improved (report : Report) (write : Write) :
      MoveStores encode moveAt pv beta initial seed reports picked writes →
      choosesMove (scan pv beta initial (reports.map Report.score)) report.score →
      write.value.setWidth 8 = encode (moveAt report.index) →
      MoveStores encode moveAt pv beta initial seed (reports ++ [report]) (moveAt report.index)
        (writes ++ [write])

private theorem MoveStores.lastWrite (trace : MoveStores encode moveAt pv beta initial seed reports picked writes) :
    ∃ earlier write, writes = earlier ++ [write] ∧ write.value.setWidth 8 = encode picked := by
  induction trace with
  | initialized write value => exact ⟨[], write, rfl, value⟩
  | unchanged _ _ _ ih => exact ih
  | improved _ write _ _ value _ => exact ⟨_, write, rfl, value⟩

theorem MoveStores.member (trace : MoveStores encode moveAt pv beta initial seed reports picked writes) :
    picked = seed ∨ ∃ report ∈ reports, picked = moveAt report.index := by
  induction trace with
  | initialized => exact Or.inl rfl
  | unchanged _ _ _ ih =>
    rcases ih with keep | ⟨report, seen, move⟩
    · exact Or.inl keep
    · exact Or.inr ⟨report, List.mem_append_left _ seen, move⟩
  | improved report _ _ _ _ _ => exact Or.inr ⟨report, by simp, rfl⟩

def MoveStoreWindow (encode : Option Move → BitVec 8) (moveAt : Nat → Option Move)
    (pv : Bool) (beta : Int) (initial : State) (seed : Option Move) (reports : List Report)
    (picked : Option Move) (writes : List Write) (last : Fin writes.length) : Prop :=
  ∃ before current after, writes = before ++ current ++ after ∧
    last.val + 1 = before.length + current.length ∧
    MoveStores encode moveAt pv beta initial seed reports picked current

theorem MoveStoreWindow.lastValue
    (window : MoveStoreWindow encode moveAt pv beta initial seed reports picked writes last) :
    (writes.get last).value.setWidth 8 = encode picked := by
  obtain ⟨before, current, after, order, lastEq, trace⟩ := window
  obtain ⟨earlier, write, rfl, value⟩ := trace.lastWrite
  have index : last.val = (before ++ earlier).length := by simp at lastEq ⊢; omega
  have stored : writes[last.val]? = some write := calc
    writes[last.val]? = ((before ++ earlier) ++ write :: after)[last.val]? := by
      apply congrArg (fun (ws : List Write) => ws[last.val]?)
      simpa only [List.append_assoc, List.singleton_append] using order
    _ = some write := by rw [index, List.getElem?_append_right (Nat.le_refl _)]; simp
  have same := Option.some.inj ((List.getElem?_eq_getElem last.isLt).symm.trans stored)
  rw [List.get_eq_getElem, same]
  exact value

theorem MoveStoreWindow.member
    (window : MoveStoreWindow encode moveAt pv beta initial seed reports picked writes last) :
    picked = seed ∨ ∃ report ∈ reports, picked = moveAt report.index := by
  obtain ⟨_, _, _, _, _, trace⟩ := window
  exact trace.member

end NeuralReversi.Threading.SplitWindow
