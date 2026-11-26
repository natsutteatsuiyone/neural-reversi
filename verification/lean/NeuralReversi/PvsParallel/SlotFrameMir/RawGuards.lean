import NeuralReversi.PvsParallel.SlotFrameMir.RawActivation

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Write Read InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitSchedulerMir (Guard Returned OwnerMemory Clock scoreLoad)

def coordinate (g : RawMask ws) (i : Fin (words g).length) : Fin ((hits g).length+1) :=
  ⟨i.val, Nat.lt_of_lt_of_eq i.isLt (length_words g)⟩

def back (g : RawMask ws) (i : Fin (words g).length) := phys g (coordinate g i)

theorem at_back (g : RawMask ws) (i : Fin (words g).length) :
    (words g).get i=ws.get (back g i) :=
  word g i.val (by have:=(coordinate g i).isLt;change i.val<_ at this;omega)

def rf (g : RawMask ws) (s : Fin ws.length) : Fin (words g).length :=
  if h:g.mark.val≤s.val then index g (cut g s h).rank.val (by have:=(cut g s h).rank.isLt;omega)
  else index g 0 (by omega)

theorem rf_val (g : RawMask ws) (s : Fin ws.length) (within : g.mark.val≤s.val) :
    (rf g s).val=(cut g s within).rank.val := by simp [rf,within,index]

theorem rf_value (g : RawMask ws) (s : Fin ws.length)
    (range : g.mark.val≤s.val ∧ s.val≤g.mark.val+g.actions.length) :
    ((words g).get (rf g s)).value=(ws.get s).value := by
  rw [at_back]
  have congr : coordinate g (rf g s)=(cut g s range.1).rank := Fin.ext (rf_val g s range.1)
  simpa only [back,congr] using (cut g s range.1).value range.2

theorem rf_lower (g : RawMask ws) (s : Fin ws.length) (h : g.mark.val≤s.val) (i : Fin (words g).length) :
    i.val≤(rf g s).val ↔ (back g i).val≤s.val := by
  rw [rf_val g s h]
  constructor
  · intro le
    by_cases eq : coordinate g i=(cut g s h).rank
    · simpa only [back,eq] using (cut g s h).lower
    · have ne:=Fin.val_ne_of_ne eq
      exact Nat.le_trans (Nat.le_of_lt (ascending g (i:=coordinate g i) (j:=(cut g s h).rank)
        (by change i.val≠_ at ne;change i.val<_;omega))) (cut g s h).lower
  · intro le
    apply Classical.byContradiction
    intro bad
    have gt : (cut g s h).rank.val < (coordinate g i).val := by change _<i.val;omega
    exact (Nat.not_lt_of_ge le) ((cut g s h).later _ gt)

theorem lift_sequence (g : RawMask ws) (s : Fin ws.length)
    (range : g.mark.val≤s.val ∧ s.val≤g.mark.val+g.actions.length) (i : Fin (words g).length)
    (seq : InReleaseSequence (words g) i (rf g s)) : InReleaseSequence ws (back g i) s := by
  have head := seq.1
  rw [at_back] at head
  refine ⟨head,(rf_lower g s range.1 i).1 seq.2.1,?_⟩
  intro k after upto
  have base : g.mark.val≤(back g i).val := by
    by_cases z : (coordinate g i).val=0
    · have eq : coordinate g i=⟨0,by omega⟩ := Fin.ext z
      simp only [back,eq,phys,Nat.le_refl]
    · exact Nat.le_of_lt (ascending g (i:=⟨0,by omega⟩) (j:=coordinate g i) (by change 0<_ ;omega))
  let j : Fin g.actions.length := ⟨k.val-g.mark.val-1,by omega⟩
  have eq : g.at j=k := Fin.ext (by change g.mark.val+(k.val-g.mark.val-1)+1=k.val;omega)
  exact eq ▸ (g.rmw j).2.2.1

def guard (g : RawMask ws) : Guard ws → Guard (words g)
  | .shutdown b => .shutdown b
  | .mask s => .mask (rf g s)

theorem returned (g : RawMask ws) (run : Returned code ws pc log)
    (range : ∀ s, Guard.mask s∈log → g.mark.val≤s.val ∧ s.val≤g.mark.val+g.actions.length) :
    Returned code (words g) pc (log.map (guard g)) := by
  induction run with
  | done h => exact .done h
  | next h mem _ ih => exact .next h mem (ih range)
  | shutdown h _ ih => exact .shutdown h (ih (fun s mem => range s (List.mem_cons_of_mem _ mem)))
  | mask src h _ ih =>
    apply Returned.mask _ h
    rw [rf_value g src (range src (List.mem_cons_self ..))]
    exact ih (fun s mem => range s (List.mem_cons_of_mem _ mem))

theorem open_pool (g : RawMask ws) {log : List (Guard ws)} (live : Guard.shutdown true ∉ log) :
    Guard.shutdown true ∉ log.map (guard g) := by
  intro bad
  obtain ⟨v,member,eq⟩:=List.mem_map.mp bad
  cases v with
  | mask => cases eq
  | shutdown b =>
    change (Guard.shutdown b : Guard (words g))=Guard.shutdown true at eq
    cases eq;exact live member

theorem guard_origin (g : RawMask ws) (raw : List (Guard ws)) {pre post : List (Guard (words g))} {src : Fin (words g).length}
    (site : raw.map (guard g)=pre++Guard.mask src::post) :
    ∃ left s right, raw=left++Guard.mask s::right ∧ left.length=pre.length ∧ rf g s=src := by
  obtain ⟨left,rest,eq,matched,suffix⟩:=List.map_eq_append_iff.mp site
  obtain ⟨v,right,restEq,first,_⟩:=List.map_eq_cons_iff.mp suffix
  cases v with
  | shutdown b => cases first
  | mask s =>
    exact ⟨left,s,right,by rw [eq,restEq],by rw [←matched,List.length_map],Guard.mask.inj first⟩

theorem exit_returned {g : RawMask ws} (owner : OwnerExit hist g) :
    Returned SplitSchedulerMir.idle (words g) 0 (owner.log.map (guard g)) := by
  apply returned g owner.returned
  intro src here
  obtain ⟨l,r,eq⟩:=List.append_of_mem here
  exact (owner.sampled l src r eq).bounds

/-- The owner's program-order events are unchanged; only the reads-from index
    within the mask history and the list of physical mask modifications change. -/
theorem memory {g : RawMask ws} (owner : OwnerExit hist g) (closed : ClosedAfterFirst g)
    (mark : (ws.get g.mark).event=owner.clock.atStep 1)
    (sync : ∀ pre s post, owner.log=pre++ .mask s::post → ∀ i,
      InReleaseSequence ws i s → hist.otherSync (ws.get i).event (owner.clock.atStep (4+pre.length)))
    (scores : List Write) (source : Fin scores.length)
    (score : SplitRebookingMir.Coherence hist scores (scoreLoad scores source owner.clock owner.log)) :
    OwnerMemory hist (words g) scores (activation g closed) owner.clock (owner.log.map (guard g)) source where
  ownerBit := (congrArg Write.event (word g 0 (by omega))).trans mark
  maskBeforeReuse _ _ _ _ i h := by
    have bound := i.isLt
    change i.val=0+(actions g).length+1 at h
    have length := length_words g
    rw [length_actions] at h
    omega
  maskWR pre src post eq i hb := by
    obtain ⟨l,s,r,site,len,mapped⟩:=guard_origin g owner.log eq
    subst src
    have w:=owner.sampled l s r site
    apply (rf_lower g s w.bounds.1 i).2
    apply w.writeRead (back g i)
    simpa only [at_back,len] using hb
  maskRW pre src post eq i hb := by
    obtain ⟨l,s,r,site,len,mapped⟩:=guard_origin g owner.log eq
    subst src
    have w:=owner.sampled l s r site
    have le := w.readWrite (back g i) (by simpa only [at_back,len] using hb)
    apply Classical.byContradiction
    intro bad
    exact (Nat.not_le_of_gt le) ((rf_lower g s w.bounds.1 i).1 (by omega))
  sync pre src post eq i seq := by
    obtain ⟨l,s,r,site,len,mapped⟩:=guard_origin g owner.log eq
    subst src
    simpa only [at_back,len] using sync l s r site (back g i)
      (lift_sequence g s (owner.sampled l s r site).bounds i seq)
  scoreWR i hb := score.writeRead i (by simpa only [scoreLoad,List.length_map] using hb)
  scoreRW i hb := score.readWrite i (by simpa only [scoreLoad,List.length_map] using hb)

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection
