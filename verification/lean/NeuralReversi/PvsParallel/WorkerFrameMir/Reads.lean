import NeuralReversi.PvsParallel.WorkerFrameMir.Stack

namespace NeuralReversi.PvsParallel.WorkerFrameMir

open Threading
open AtomicWord (Event Write Read ReadWindow SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitRebookingMir (Coherence AtOrAfter Prefix FirstNested)
open SplitCompletion (Worker MaskState Action)

structure LocalStore (root : Frame) where
  frames : List Frame
  stack : Stack root frames
  bit : Bool
  emitted : ((top root frames).kind,bit) ∈ searchingStores
  time : Event
  afterEntry : SequencedBefore (top root frames).entered time

inductive Record (root : Frame)
  | localWrite (store : LocalStore root)
  | externalBusy (e : Event)

def Record.encode : Record root → Write
  | .localWrite s => ⟨s.time,if s.bit then 1 else 0,true,false⟩
  | .externalBusy e => ⟨e,1,true,false⟩

/-- The span is taken from the target searching atomic's modification order. Post-booking
    events on other threads can publish true. Local writes carry their Rust call stacks. -/
structure Span (root : Frame) (ws : List Write) where
  before : List Write
  busy : Write
  posted : busy.value = 1
  records : List (Record root)
  after : List Write
  writes : ws = before ++ busy :: records.map Record.encode ++ after
  first : Fin ws.length
  last : Fin ws.length
  lower : first.val = before.length
  upper : last.val = before.length + records.length

theorem Span.selected (span : Span root ws) (source : Fin ws.length)
    (within : span.first.val ≤ source.val ∧ source.val ≤ span.last.val)
    (idle : (ws.get source).value ≠ 1) :
    ∃ st : LocalStore root, .localWrite st ∈ span.records ∧ st.bit = false ∧
      (ws.get source).event = st.time ∧ (ws.get source).release = true := by
  have lo := span.lower; have hi := span.upper
  have pos : source.val < (span.before ++ span.busy :: span.records.map Record.encode).length := by simp; omega
  have get : ws[source.val]? = some (ws.get source) := List.getElem?_eq_getElem source.isLt
  replace get := (congrArg (fun (ys : List Write) => ys[source.val]?) span.writes).symm.trans get
  rw [List.getElem?_append_left pos,List.getElem?_append_right (by omega)] at get
  let n := source.val - span.before.length
  change (span.busy :: span.records.map Record.encode)[n]? = _ at get
  cases eq : n with
  | zero =>
    rw [eq] at get
    have same := Option.some.inj get
    have bad := span.posted
    exact (idle (same ▸ bad)).elim
  | succ j =>
    rw [eq] at get
    change (span.records.map Record.encode)[j]? = _ at get
    rw [List.getElem?_map] at get
    obtain ⟨r,found,same⟩ := Option.map_eq_some_iff.mp get
    have member := List.mem_of_getElem? found
    cases r with
    | externalBusy e =>
      change (⟨e,1,true,false⟩ : Write) = ws.get source at same
      exact (idle (same ▸ rfl)).elim
    | localWrite st =>
      refine ⟨st,member,?_,?_,?_⟩
      · rw [← same] at idle
        change (if st.bit then (1 : BitVec 64) else 0) ≠ 1 at idle
        cases bit : st.bit <;> simp_all
      · rw [← same]; rfl
      · rw [← same]; rfl

def failures (birth : Event) (published : List Event) : List Write :=
  ⟨birth,0,false,false⟩ :: published.map (fun e => ⟨e,1,false,false⟩)

theorem failed_after_unwind (history : History) (birth : Event) (published : List Event)
    (load : Read (failures birth published)) (stamp : Fin published.length)
    (fromMark : HappensBefore history (published.get stamp) load.event)
    (memory : Coherence history (failures birth published) load) :
    ((failures birth published).get load.source).value = 1 := by
  let i : Fin (failures birth published).length := ⟨stamp.val+1,by simp [failures,stamp.isLt]⟩
  have mark : ((failures birth published).get i).event = published.get stamp := by simp [failures,i]
  have lower := memory.writeRead i (mark ▸ fromMark)
  have pos : 0 < load.source.val := by dsimp [i] at lower; omega
  cases idx : load.source.val with
  | zero => omega
  | succ n =>
    have hn : n < published.length := by have := load.source.isLt; simp [failures,idx] at this; omega
    have atN : (failures birth published)[load.source.val]? = some (⟨published[n],1,false,false⟩ : Write) := by
      simp [failures, idx, hn]
    have same := Option.some.inj ((List.getElem?_eq_getElem load.source.isLt).symm.trans atN)
    change ((failures birth published)[load.source.val]).value = 1
    rw [same]

/-- The inner guard carries actual machine-word inputs and the branch selected by the
    compiler. The edge to a competing booking will follow from the attributed false read. -/
inductive GuardSample (history : History) (s : MaskState) (acts : List Action) (book : Event) (f : Frame) : Type
  | closed (words : List Write) (r : Read words) (p : Prefix words .accepting s acts) (pool depth : Nat) :
      SplitRebookingMir.nested pool depth false 0 = true → (words.get r.source).value = 0 →
      Coherence history words r → HappensBefore history (words.get p.first).event r.event →
      SequencedBefore r.event f.entered →
      (∀ i : Fin words.length, i.val = p.last.val + 1 → AtOrAfter history book (words.get i).event) →
      GuardSample history s acts book f
  | counted (words : List Write) (r : Read words) (p : Prefix words .mask s acts) (pool depth : Nat) :
      SplitRebookingMir.nested pool depth true (AtomicBitSet.count (words.get r.source).value) = true →
      Coherence history words r → HappensBefore history (words.get p.first).event r.event →
      SequencedBefore r.event f.entered →
      (∀ i : Fin words.length, i.val = p.last.val + 1 → AtOrAfter history book (words.get i).event) →
      GuardSample history s acts book f

theorem GuardSample.entered (sample : GuardSample history s acts book f)
    (ahead : HappensBefore history f.entered book) : FirstNested history s acts book := by
  cases sample with
  | closed ws r p n d grants zero memory pub order next =>
    exact .closed ws r p n d grants zero memory pub (.trans (.sequenced order) ahead) next
  | counted ws r p n d grants memory pub order next =>
    exact .openFlag ws r p n d grants memory pub (.trans (.sequenced order) ahead) next

end NeuralReversi.PvsParallel.WorkerFrameMir
