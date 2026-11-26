import NeuralReversi.PvsParallel.WorkerFrameMir.Program

namespace NeuralReversi.PvsParallel.WorkerFrameMir

open Threading.AtomicWord (Event SequencedBefore)

def permits (p child : Kind) : Bool :=
  match callers child with
  | none => true
  | some origins => origins.contains p

def top (root : Frame) (frames : List Frame) := frames.headD root

inductive Stack (root : Frame) : List Frame → Prop
  | start : root.kind = .work → Stack root []
  | call : Stack root frames → permits (top root frames).kind child.kind = true →
      SequencedBefore (top root frames).entered child.entered → Stack root (child :: frames)

def FirstInner (frames : List Frame) (f : Frame) : Prop :=
  f.kind = .split ∧ ∃ inner outer, frames = inner ++ f :: outer ∧ ∀ x ∈ outer, x.kind ≠ .split

private theorem first_or_none (frames : List Frame) :
    (∃ f, FirstInner frames f) ∨ ∀ f ∈ frames, f.kind ≠ .split := by
  induction frames with
  | nil => exact Or.inr (by simp)
  | cons x xs ih =>
    rcases ih with ⟨f,kind,part,outer,eq,ok⟩ | no
    · exact Or.inl ⟨f,kind,x::part,outer,by simp [eq],ok⟩
    · by_cases isSplit : x.kind = .split
      · exact Or.inl ⟨x,isSplit,[],xs,rfl,no⟩
      · exact Or.inr (fun f mem => (List.mem_cons.mp mem).elim (fun eq => eq ▸ isSplit) (no f))

theorem FirstInner.mem (x : FirstInner frames f) : f ∈ frames := by
  obtain ⟨_,left,_,rfl,_⟩ := x
  exact List.mem_append_right _ (List.mem_cons_self ..)

theorem Stack.before_write (stack : Stack root frames) (member : f ∈ frames)
    (store : SequencedBefore (top root frames).entered atWrite) : SequencedBefore f.entered atWrite := by
  induction stack with
  | start => cases member
  | call _ _ stamp ih =>
    rcases List.mem_cons.mp member with same | prior
    · subst f; exact store
    · exact ih prior ⟨stamp.1.trans store.1, Nat.lt_trans stamp.2 store.2⟩

private def normal : Kind → Bool
  | .mainEntry | .workerEntry | .mainLoop | .idle | .helper | .helperDrop | .split => false
  | _ => true

private theorem retained (p q : Kind) (start : normal p = true)
    (yes : permits p q = true) (no : q ≠ .split) : normal q = true ∧ q ≠ .work := by
  revert start yes no; cases p <;> cases q <;> decide

private theorem Stack.no_split (stack : Stack root frames) (no : ∀ f ∈ frames, f.kind ≠ .split) :
    normal (top root frames).kind = true ∧ ((top root frames).kind = .work → frames = []) := by
  induction stack with
  | start rootKind => exact ⟨by change normal root.kind = true; rw [rootKind]; rfl, fun _ => rfl⟩
  | @call frames x _ ok _ ih =>
    have previous := ih (fun f member => no f (List.mem_cons_of_mem _ member))
    have allowed := retained _ _ previous.1 ok (no x (List.mem_cons_self ..))
    exact ⟨allowed.1,fun h => (allowed.2 h).elim⟩

private theorem only_detach (k : Kind) (ok : normal k = true) (writes : (k,false) ∈ searchingStores) :
    k = .detach := by revert ok writes; cases k <;> decide

inductive Origin (root : Frame) : List Frame → Prop
  | nested : FirstInner fs f → Origin root fs
  | own (leave : Frame) : leave.kind = .detach → Origin root [leave]
  | unwind (leave parent : Frame) : leave.kind = .detach → parent.kind = .abandon →
      Origin root (leave :: parent :: outer)

theorem origin (stack : Stack root frames) (writes : ((top root frames).kind,false) ∈ searchingStores) :
    Origin root frames := by
  rcases first_or_none frames with ⟨f,first⟩ | absent
  · exact .nested first
  have kind := only_detach _ (stack.no_split absent).1 writes
  cases stack with
  | start h => change root.kind = .detach at kind; rw [h] at kind; contradiction
  | @call prior leave outerStack permitted _ =>
    change leave.kind = .detach at kind
    have parents : (top root prior).kind = .abandon ∨ (top root prior).kind = .work := by
      rw [kind] at permitted
      simpa [permits,callers] using permitted
    rcases parents with failed | own
    · cases prior with
      | nil =>
        cases outerStack with
        | start h => change root.kind = .abandon at failed; rw [h] at failed; contradiction
      | cons parent tail => exact .unwind leave parent kind failed
    · have empty := (outerStack.no_split (fun f member => absent f (List.mem_cons_of_mem _ member))).2 own
      subst prior; exact .own leave kind

end NeuralReversi.PvsParallel.WorkerFrameMir
