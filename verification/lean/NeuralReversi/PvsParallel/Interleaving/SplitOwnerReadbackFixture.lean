import NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback

namespace NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback.Regression

open Threading
open AtomicWord (Event Write SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SlotFrameMir (RawMask OwnerExit)
open SlotFrameMir.RawProjection (OwnerSetup PhysicalOwner)
open SplitWorkerMir
open SplitSchedulerMir (Clock Guard)

/-- An owner exhausts its iterator and returns its negative seed and move.
    The slot was used earlier; a subsequent initializer reuses both values. -/
private def clock : Clock := ⟨0,fun i => 10+10*i,by omega⟩
private def stamp (n : Nat) : Event := ⟨0,n⟩
private def writes : List Write := [
  ⟨stamp 10,0,false,false⟩, ⟨stamp 20,1,false,true⟩,
  ⟨stamp 85,0,true,true⟩, ⟨stamp 150,0,false,false⟩]
local instance : NeZero writes.length := ⟨by decide⟩
def gen : RawMask writes := ⟨0,[.finish 0 (stamp 85)],1,by decide,by decide,by decide⟩
def hist : History := ⟨[],fun _ => [],[],[],
  fun a b => a = stamp 85 ∧ b = stamp 100⟩

private theorem localOrder (h : HappensBefore hist a b) : SequencedBefore a b := by
  induction h with
  | sequenced sb => exact sb
  | synchronized n => exact Fin.elim0 n
  | other h => obtain ⟨rfl,rfl⟩ := h; exact ⟨rfl,by decide⟩
  | trans _ _ left right => exact ⟨left.1.trans right.1,Nat.lt_trans left.2 right.2⟩

private def raw : List OwnerMaskGuards.Operand := [
  .shutdown (stamp 70) false, .mask (stamp 80) (stamp 20) 1 true,
  .shutdown (stamp 90) false, .mask (stamp 100) (stamp 85) 0 true]
private def log : List (Guard writes) := [.shutdown false,.mask 1,.shutdown false,.mask 2]
private def path : List (Fin 80) := [
  1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,
  29,30,32,34,35,36,38,39,19,20,21,22,23,40,41,42,43,44,76,77,4,5,6,7,9,
  10,11,12,13,14,15,16,78,79]
private def maskRank : Guard writes → Option Nat
  | .mask i => some i.val
  | _ => none

private theorem maskCoherent : OwnerMaskGuards.Coherent hist writes raw := by
  intro row member i decoded
  obtain ⟨k,rfl⟩ := List.get_of_mem member
  have verify : ∀ k : Fin raw.length, ∀ i j : Fin writes.length,
      ((raw.get k).guard writes).bind maskRank = some i.val →
      (SequencedBefore (writes.get j).event (raw.get k).site → j.val ≤ i.val) ∧
      (SequencedBefore (raw.get k).site (writes.get j).event → i.val < j.val) := by
    unfold SequencedBefore; decide
  have got : ((raw.get k).guard writes).bind maskRank = some i.val := by rw [decoded]; rfl
  exact ⟨fun j edge => (verify k i j got).1 (localOrder edge),
    fun j edge => (verify k i j got).2 (localOrder edge)⟩

private def exit_ : OwnerExit hist gen :=
  OwnerMaskGuards.ownerExit hist gen clock raw (log := log) rfl path (by decide) (by decide)
    rfl (by decide) maskCoherent

private def scoreWord := (-1536 : BitVec 32)
private def object (n : Nat) (v : BitVec w) : Write := ⟨stamp n,v.setWidth 64,false,false⟩
private def tested : MoveList.ConcurrentTrace.State 64 :=
  ⟨0,[],MoveList.ConcurrentTrace.replace (fun _ => .idle) 0 (.testing 81 0)⟩
private def drained : MoveList.ConcurrentTrace.State 64 :=
  { tested with localAt := MoveList.ConcurrentTrace.replace tested.localAt 0 (.drained 81 0) }
private theorem exhausted : MoveList.ConcurrentTrace.Execution 0 0 drained :=
  .decline (worker := 0) (opening := 81) (prev := 0)
    (.load (worker := 0) (source := 0) (seen := 0) .init rfl rfl 81) rfl rfl

def tickets : NativeSplit.Tickets [] true 100 where
  width := 64
  start := 0
  state := { drained with localAt := MoveList.ConcurrentTrace.replace drained.localAt 0 .idle }
  run := .returnedNone (worker := 0) (opening := 81) (seen := 0) exhausted rfl
  fits := by decide
  offset := by decide
  journal := []
  stream := []
  workers := [0]
  distinct := by decide
  covers := by simp [drained,tested]
  dispatches := []
  traced := rfl
  initial := ⟨scoreWord,scoreWord,some 37⟩
  a := object 50 scoreWord
  b := object 30 scoreWord
  m := object 40 (37#8)
  seedAlpha := rfl
  seedBest := rfl
  seedMove := rfl
  final := FieldStores.initialLog ⟨scoreWord,scoreWord,some 37⟩ (object 50 scoreWord) (object 30 scoreWord) (object 40 (37#8))
  replay := by
    have done_ : IteratorFields.chronological ([] : List (MoveList.ConcurrentTrace.Write 64)) [] [] [0] = [] := by
      change ([] : List Iterator.LockedCall).mergeSort _ = []
      simp
    unfold IteratorFields.replay
    dsimp only
    rw [show drained.writes = [] by rfl, done_]
    rfl

private theorem sync {pre s post} (part : log = pre ++ Guard.mask s :: post) (i : Fin writes.length)
    (seq : AtomicWord.InReleaseSequence writes i s) :
    hist.otherSync (writes.get i).event (exit_.clock.atStep (4+pre.length)) := by
  have loc : log[pre.length]? = some (.mask s) := by rw [part]; simp
  have small : pre.length < log.length := by simp [part]
  let k : Fin log.length := ⟨pre.length,small⟩
  have selected : (log.get k |> maskRank) = some s.val := by
    have hit := (List.getElem?_eq_some_iff.mp loc).2
    change maskRank log[pre.length] = _
    rw [hit]
    rfl
  have check : ∀ k : Fin log.length, ∀ s i : Fin writes.length,
      maskRank (log.get k) = some s.val → (writes.get i).release = true → i.val ≤ s.val →
      (writes.get i).event = stamp 85 ∧ exit_.clock.atStep (4+k.val) = stamp 100 := by decide
  exact check k s i selected seq.1 seq.2.1

private def setup : OwnerSetup gen hist Event.step (0,0) tickets where
  exit := exit_
  gates := ⟨0,by simp,by
    intro j _ w e book
    have zero : j = ⟨0,by decide⟩ := Fin.ext (by change j.val = 0; have := j.isLt; change j.val < 1 at this; omega)
    subst j
    contradiction⟩
  causal := fun _ _ edge => (localOrder edge).2
  thread := rfl
  mark := rfl
  bestSeed := rfl
  alphaSeed := rfl
  moveSeed := rfl
  sync := fun _ _ _ => sync
  bPre := [object 1 scoreWord]
  mPre := [object 2 (37#8)]
  later := [⟨0,⟨6,by decide⟩,scoreWord,37,scoreWord,[],[],[]⟩]
  resets := []
  completed := rfl
  check := ⟨⟨stamp 108,[],[],.empty,by simp⟩,
    some ⟨[object 3 (1#64)],⟨stamp 109,0,true⟩⟩,by simp,false⟩
  running := rfl

private def score : Load := ⟨(0,0),stamp 120,.score scoreWord,false,stamp 30⟩
private def picked : Load := ⟨(0,0),stamp 130,.move 37,false,stamp 40⟩
private def rows := [score,picked]

private def sbCoherent (ws : List Write) (r : Load) : Bool :=
  ws.all fun w =>
    let ids := ws.map Write.event
    let i := ids.idxOf w.event
    let s := ids.idxOf r.fromWrite
    (w.event.worker != r.site.worker || w.event.step ≥ r.site.step || i ≤ s) &&
    (r.site.worker != w.event.worker || r.site.step ≥ w.event.step || s < i)

private theorem shared (good : sbCoherent ws r) : Coherent hist ws r := by
  constructor
  · intro w mem h
    obtain ⟨eq,lt⟩ := localOrder h
    have yes := (Bool.and_eq_true_iff.mp (List.all_eq_true.mp good w mem)).1
    simpa [eq, Nat.not_le_of_gt lt] using yes
  · intro w mem h
    obtain ⟨eq,lt⟩ := localOrder h
    have yes := (Bool.and_eq_true_iff.mp (List.all_eq_true.mp good w mem)).2
    simpa [eq, Nat.not_le_of_gt lt] using yes

def full : PhysicalOwner gen hist Event.step (0,0) tickets :=
  physical setup rows (by decide) ⟨shared (by decide),shared (by decide)⟩

example : full.source.val + 1 = full.bPre.length + tickets.final.best.length ∧
    full.moveSource.val + 1 = full.mPre.length + tickets.final.moves.length := full.read_sources

example : full.source.val = 1 ∧ full.moveSource.val = 1 ∧
    AtomicWord.signed32At (bestWords setup) full.source = -1536 ∧
    ((moveWords setup).get full.moveSource).value.setWidth 8 = 37 := by decide

example : ([{score with slot := (0,1)},picked] :: [{score with site := ⟨1,120⟩},picked] ::
    [{score with site := stamp 119},picked] :: [{score with acquire := true},picked] ::
    [{score with word := .alpha scoreWord},picked] :: [{score with word := .score 1536},picked] ::
    [score,{picked with word := .move 36}] :: [score,score,picked] ::
    [{score with fromWrite := stamp 40},picked] :: []).all
    (fun forged => (readback setup forged).isNone) := by decide

example : (sample rows (0,0) (stamp 120) (bestWords setup ++ [object 30 scoreWord]) Operand.score?).isNone := by decide

example : (readback {setup with mPre := [object 30 (37#8)]} rows).isNone := by decide

/-- Prior and future uses wrote exactly the same i32 value. Both pass numeric
    and RF decoding, but neither event can be the source of this owner load. -/
example : (readback setup [{score with fromWrite := stamp 1},picked]).isSome ∧
    ¬ Coherent hist (bestWords setup) {score with fromWrite := stamp 1} := by
  refine ⟨by decide,fun c => ?_⟩
  have impossible := c.writeRead (object 30 scoreWord) (List.mem_cons_of_mem _ (List.mem_cons_self ..)) (.sequenced (by unfold SequencedBefore; decide))
  exact (by decide : ¬ (1 : Nat) ≤ 0) impossible

example : (readback setup [{score with fromWrite := stamp 170},picked]).isSome ∧
    ¬ Coherent hist (bestWords setup) {score with fromWrite := stamp 170} := by
  refine ⟨by decide,fun c => ?_⟩
  have impossible := c.readWrite (object 170 scoreWord) (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self ..))) (.sequenced (by unfold SequencedBefore; decide))
  exact (Nat.lt_irrefl 2) impossible

end NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback.Regression
