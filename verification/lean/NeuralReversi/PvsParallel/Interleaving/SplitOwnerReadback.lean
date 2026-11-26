import NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards
import NeuralReversi.PvsParallel.SlotFrameMir.RawNative

namespace NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback

open Threading
open AtomicWord (Event Read Write)
open CutoffVisibility (History HappensBefore)
open SlotFrameMir (Slot RawMask)
open SlotFrameMir.RawProjection (OwnerSetup PhysicalOwner subsequent)
open SplitSchedulerMir (Clock)
open SplitSchedulerMir.OwnerFields (returnIndex)

/-- Words read by relaxed atomics on SplitPointState. Their native widths
    distinguish an i32 score or alpha from a u8 move. -/
inductive Operand where
  | score (raw : BitVec 32)
  | move (raw : BitVec 8)
  | alpha (raw : BitVec 32)

def Operand.score? : Operand → Option (BitVec 32)
  | .score w => some w
  | _ => none

def Operand.move? : Operand → Option (BitVec 8)
  | .move w => some w
  | _ => none

structure Load where
  slot : Slot
  site : Event
  word : Operand
  acquire : Bool
  fromWrite : Event

structure Sample (rows : List Load) (slot : Slot) (site : Event) (ws : List Write)
    (field : Operand → Option (BitVec width)) where
  load : Load
  member : load ∈ rows
  address : load.slot = slot
  atSite : load.site = site
  relaxed : load.acquire = false
  word : BitVec width
  register : field load.word = some word
  rank : Fin ws.length
  writes : (ws.map Write.event).Nodup
  origin : (ws.get rank).event = load.fromWrite
  value : (ws.get rank).value.setWidth width = word

/-- Reconstruct both the physical instruction and its actual reads-from event.
    Equal stored words are allowed. Ambiguous instruction or modification IDs fail. -/
def sample (rows : List Load) (slot : Slot) (site : Event) (ws : List Write)
    (field : Operand → Option (BitVec width)) : Option (Sample rows slot site ws field) :=
  match chosen : rows.filter (fun r => r.site == site) with
  | [load] =>
    if shape : load.slot = slot ∧ load.acquire = false ∧ (ws.map Write.event).Nodup then
      match raw : field load.word, rf : OwnerMaskGuards.source ws load.fromWrite with
      | some word, some i => if bits : (ws.get i).value.setWidth width = word then
        have member : load ∈ rows.filter (fun r => r.site == site) := by rw [chosen]; simp
        let here := List.mem_filter.mp member
        some ⟨load,here.1,shape.1,beq_iff_eq.mp here.2,shape.2.1,word,raw,i,shape.2.2,
          OwnerMaskGuards.source_event rf,bits⟩
        else none
      | _, _ => none
    else none
  | _ => none

/-- The ordinary C++20 write-read and read-write laws, stated for modifying
    event IDs on this object. The capture supplies no load ordinal. -/
structure Coherent (hist : History) (ws : List Write) (r : Load) : Prop where
  writeRead : ∀ w ∈ ws, HappensBefore hist w.event r.site →
    (ws.map Write.event).idxOf w.event ≤ (ws.map Write.event).idxOf r.fromWrite
  readWrite : ∀ w ∈ ws, HappensBefore hist r.site w.event →
    (ws.map Write.event).idxOf r.fromWrite < (ws.map Write.event).idxOf w.event

private theorem ordinal {ws : List Write} (unique : (ws.map Write.event).Nodup) (i : Fin ws.length) :
    (ws.map Write.event).idxOf (ws.get i).event = i.val := by
  have bounded : i.val < (ws.map Write.event).length := by simp
  simpa using unique.idxOf_getElem i.val bounded

theorem Sample.coherence (s : Sample rows slot site ws field) (c : Coherent hist ws s.load) :
    SplitRebookingMir.Coherence hist ws ⟨site,s.rank,false⟩ := by
  refine ⟨fun j edge => ?_,fun j edge => ?_⟩
  · have low := c.writeRead _ (List.get_mem ws j) (by rw [s.atSite]; exact edge)
    rwa [←s.origin,ordinal s.writes j,ordinal s.writes s.rank] at low
  · have high := c.readWrite _ (List.get_mem ws j) (by rw [s.atSite]; exact edge)
    rwa [←s.origin,ordinal s.writes j,ordinal s.writes s.rank] at high

variable {t : NativeSplit.Tickets rc pv beta} {g : RawMask mask}

/-- Existing guard/retirement/seed evidence is shared by the constructor and
    consumers. An OwnerMaskGuards.ownerExit can supply the setup's exit. -/
abbrev bestWords (b : OwnerSetup g hist time slot t) :=
  b.bPre ++ t.final.best ++ subsequent b.exit slot .score b.later

abbrev moveWords (b : OwnerSetup g hist time slot t) :=
  b.mPre ++ t.final.moves ++ subsequent b.exit slot .move b.later

abbrev bestSite (b : OwnerSetup g hist time slot t) :=
  b.exit.clock.atStep (b.exit.log.length+5)

abbrev moveSite (b : OwnerSetup g hist time slot t) :=
  b.exit.timeline.atStep (returnIndex b.exit.log.length .picked)

abbrev Result (b : OwnerSetup g hist time slot t) (rows : List Load) :=
  Sample rows slot (bestSite b) (bestWords b) Operand.score? ×
    Sample rows slot (moveSite b) (moveWords b) Operand.move?

def readback (b : OwnerSetup g hist time slot t) (rows : List Load) : Option (Result b rows) := do
  let score ← sample rows slot (bestSite b) (bestWords b) Operand.score?
  let picked ← sample rows slot (moveSite b) (moveWords b) Operand.move?
  if (moveWords b).any (fun w => w.event == score.load.fromWrite) ||
      (bestWords b).any (fun w => w.event == picked.load.fromWrite) then none
  else some (score,picked)

def physical (b : OwnerSetup g hist time slot t) (rows : List Load)
    (matched : (readback b rows).isSome)
    (coherence : let r := (readback b rows).get matched
      Coherent hist (bestWords b) r.1.load ∧ Coherent hist (moveWords b) r.2.load) :
    PhysicalOwner g hist time slot t :=
  let r := (readback b rows).get matched
  { toOwnerSetup := b, source := r.1.rank, moveSource := r.2.rank,
    coherence := r.1.coherence coherence.1, moveCoherence := r.2.coherence coherence.2 }

/-- Both physical loads keep their own source event and raw register word
    through conversion to the existing split-return proof. -/
theorem registers (b : OwnerSetup g hist time slot t) (rows : List Load)
    (ok : (readback b rows).isSome) (coherence) :
    let r := (readback b rows).get ok
    let p := physical b rows ok coherence
    ((bestWords b).get p.source).event = r.1.load.fromWrite ∧
    ((moveWords b).get p.moveSource).event = r.2.load.fromWrite ∧
    AtomicWord.signed32At (bestWords b) p.source = r.1.word.toInt ∧
    ((moveWords b).get p.moveSource).value.setWidth 8 = r.2.word := by
  let r := (readback b rows).get ok
  exact ⟨r.1.origin,r.2.origin,congrArg BitVec.toInt r.1.value,r.2.value⟩

end NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback
