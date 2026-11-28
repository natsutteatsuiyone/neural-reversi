module
public import NeuralReversi.Threading.SpinLock

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

open Threading
open AtomicWord (Event Write SequencedBefore InReleaseSequence)
open SpinLock (Operation)
open CutoffVisibility (History)

def pairOps (g : Event × Event) : List Operation := [.acquire g.1, .release g.2]

/-- Pair the successful false-to-true CAS with the next guard drop on this mutex.
    The observation may end with a worker still holding the guard. -/
def guards : List Operation → Option (List (Event × Event))
  | [] | [.acquire _] => some []
  | .acquire a :: .release r :: rest =>
    if a.worker = r.worker ∧ a.step < r.step then (guards rest).map ((a,r) :: ·) else none
  | _ => none

def covers (ops : List Operation) (rows : List (Event × Event)) : Bool :=
  (guards ops).any fun pairs => decide (rows.Sublist pairs)

theorem paired {ops : List Operation} (ok : guards ops = some pairs) :
    SpinLock.Legal history none ops ∧ (pairs.flatMap pairOps).Sublist ops := by
  match ops with
  | [] => cases ok; exact ⟨.stop,.slnil⟩
  | [.acquire _] => cases ok; exact ⟨.acquired .stop,List.nil_sublist _⟩
  | .release _ :: _ | .acquire _ :: .acquire _ :: _ => contradiction
  | .acquire a :: .release r :: rest =>
    rw [guards] at ok
    split at ok <;> try contradiction
    obtain ⟨tail, got, eq⟩ := Option.map_eq_some_iff.mp ok
    cases eq
    obtain ⟨legal, contained⟩ := paired (history := history) got
    exact ⟨.acquired (.released (.sequenced (by assumption)) legal), (contained.cons_cons _).cons_cons _⟩
termination_by ops.length

def word : Operation → Write
  | .acquire a => ⟨a, 1, false, true⟩
  | .release r => ⟨r, 0, true, false⟩

def words (init : Event) (ops : List Operation) : List Write :=
  ⟨init,0,false,false⟩ :: ops.map word

private theorem free_at (ops : List Operation) {k : Nat} (ok : guards ops = some pairs)
    (found : ops[k]? = some (Operation.acquire a)) : ((words init ops)[k]?).map Write.value = some 0 := by
  match ops with
  | [] => simp at found
  | [.acquire _] => cases k <;> simp_all [words]
  | .release _ :: _ | .acquire _ :: .acquire _ :: _ => contradiction
  | .acquire first :: .release released :: rest =>
    rw [guards] at ok
    split at ok <;> try contradiction
    obtain ⟨tail, sub, _⟩ := Option.map_eq_some_iff.mp ok
    cases k with
    | zero => rfl
    | succ k =>
      cases k with
      | zero => simp at found
      | succ k =>
        cases k with
        | zero => rfl
        | succ k => exact free_at rest sub (k := k+1) found (init := released)
termination_by ops.length

structure Capture (history : History) where
  init : Event
  ops : List Operation
  /-- The releasing store synchronizes with the acquire CAS that reads it. -/
  sync : ∀ (n : Nat) r a, ops[n]? = some (.release r) → ops[n+1]? = some (.acquire a) →
    history.otherSync r a

private theorem Capture.acquired_zero (c : Capture history) (ok : guards c.ops = some pairs)
    (i : Fin c.ops.length) (event : c.ops.get i = .acquire a) :
    ((words c.init c.ops).get ⟨i.val, by simp [words]; omega⟩).value = 0 := by
  have read := free_at c.ops ok ((List.getElem?_eq_getElem i.isLt).trans (congrArg some event)) (init := c.init)
  rw [List.getElem?_eq_getElem (by simp [words]; omega)] at read
  exact Option.some.inj read

private theorem Capture.acquired_sync (c : Capture history) (ok : guards c.ops = some pairs)
    (i : Fin c.ops.length) (event : c.ops.get i = .acquire a) (head : Fin (words c.init c.ops).length)
    (seq : InReleaseSequence (words c.init c.ops) head ⟨i.val, by simp [words]; omega⟩) :
    history.otherSync ((words c.init c.ops).get head).event a := by
  let src : Fin (words c.init c.ops).length := ⟨i.val, by simp [words]; omega⟩
  have zero := c.acquired_zero ok i event
  change ((words c.init c.ops).get src).value = 0 at zero
  obtain ⟨released, before, rmws⟩ := seq
  change head.val ≤ src.val at before
  have same (plain : ((words c.init c.ops).get src).rmw = false) : head = src := by
    apply Fin.ext
    by_cases lt : head.val < src.val
    · have busy := rmws src lt (Nat.le_refl _)
      rw [plain] at busy
      contradiction
    · omega
  cases index : i.val with
  | zero =>
    have first : src = ⟨0, by simp [words]⟩ := Fin.ext index
    have eq := same (by rw [first]; rfl)
    rw [eq, first] at released
    contradiction
  | succ n =>
    have bound : n < c.ops.length := by have := i.isLt; omega
    have atRead : c.ops[n+1]? = some (.acquire a) := by
      rw [←index]; exact (List.getElem?_eq_getElem i.isLt).trans (congrArg some event)
    have prev : (words c.init c.ops).get src = word c.ops[n] := by
      have atPrev : (words c.init c.ops)[i.val]? = some (word c.ops[n]) := by simp [words,index,List.getElem?_eq_getElem bound]
      exact (List.getElem?_eq_some_iff.mp atPrev).2
    cases op : c.ops[n] with
    | acquire entering =>
      rw [prev, op] at zero
      exact False.elim ((by decide : (1#64) ≠ 0) zero)
    | release r =>
      rw [same (by rw [prev,op]; rfl), prev, op]
      exact c.sync n r a ((List.getElem?_eq_getElem bound).trans (congrArg some op)) atRead

def Capture.trace (c : Capture history) (ok : guards c.ops = some pairs) : SpinLock.Trace history where
  operations := c.ops
  writes := words c.init c.ops
  length := by simp [words]
  initial := rfl
  encode := by
    intro i
    change (match c.ops.get i with | .acquire _ => _ | .release _ => _)
    simp only [List.get_eq_getElem, words, List.getElem_cons_succ, List.getElem_map]
    cases c.ops[i] <;> rfl
  casRead := by
    intro i a got
    have bound : i.val < (words c.init c.ops).length := by simp [words]; omega
    exact ⟨⟨a, ⟨i.val, bound⟩, true⟩, rfl, rfl, rfl, c.acquired_zero ok i got⟩
  sync := by
    rintro h read ⟨i, same, pos⟩ _ seq
    apply c.acquired_sync ok i same h
    exact (show read.source = ⟨i.val, by simp [words]; omega⟩ from Fin.ext pos) ▸ seq
  legal := (paired ok).1

theorem Capture.sections (c : Capture history) (ok : covers c.ops rows) :
    ∃ pairs, guards c.ops = some pairs ∧ (rows.flatMap pairOps).Sublist c.ops := by
  obtain ⟨pairs, decoded, included⟩ := (Option.any_eq_true _ _).mp ok
  refine ⟨pairs, decoded, List.Sublist.trans ?_ (paired decoded (history := history)).2⟩
  have subset := of_decide_eq_true included
  clear ok decoded included
  induction subset with
  | slnil => exact .slnil
  | cons _ _ ih => exact ih.trans (List.sublist_append_right ..)
  | cons_cons _ _ ih => exact ih.append_left _

end NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex
