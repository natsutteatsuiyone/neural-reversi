import NeuralReversi.PvsParallel.Interleaving.TaskAncestry

namespace NeuralReversi.PvsParallel.Interleaving.FlagSegment

open Threading
open AtomicWord (Event)
open CutoffVisibility (History FlagRead FlagWrite HappensBefore)
open StopSchedule (Flags)
open TaskAncestry (Journal Generation)

deriving instance DecidableEq for FlagWrite

private instance : Decidable (AtomicWord.SequencedBefore a b) :=
  inferInstanceAs (Decidable (a.worker = b.worker ∧ a.step < b.step))

/-- Cross-worker edges in execution order. Locks contribute their actual
    handoffs; an epoch acquire contributes its release-sequence head. -/
inductive Transfer (h : History) where
  | handoff (release acquire : Event) (sync : h.otherSync release acquire)
  | epoch (head : Fin h.epochs.length) (r : AtomicWord.Read h.epochs) (logged : r ∈ h.epochReads)
      (acquire : r.acquire = true) (sequence : CutoffVisibility.InReleaseSequence h head r.source)

def Transfer.edge : Transfer h → {p : Event × Event // HappensBefore h p.1 p.2}
  | .handoff a b sync => ⟨(a,b),.other sync⟩
  | .epoch i r mem acq seq => ⟨(_,r.event),.synchronized i r mem acq seq⟩

private abbrev Reached (h : History) (a : Event) := List {e : Event // HappensBefore h a e}

private def finish (z : {e : Event // HappensBefore h a e}) (b : Event) :
    Option (PLift (HappensBefore h a b)) :=
  if eq : z.val = b then some ⟨eq ▸ z.property⟩ else
  if sb : AtomicWord.SequencedBefore z.val b then some ⟨z.property.trans (.sequenced sb)⟩ else none

private def reaches (past : Reached h a) (b : Event) : Option (PLift (HappensBefore h a b)) :=
  if sb : AtomicWord.SequencedBefore a b then some ⟨.sequenced sb⟩ else past.findSome? (finish · b)

private def extend (past : Reached h a) (t : Transfer h) : Reached h a :=
  let ⟨(x,y),edge⟩ := t.edge
  if eq : a = x then ⟨y,eq ▸ edge⟩ :: past
  else match reaches past x with
    | some path => ⟨y,path.down.trans edge⟩ :: past
    | none => past

/-- Local sequencing or a chain of observed handoffs publishes the flag's
    initialization to the reader. Numeric order across workers supplies no edge. -/
def published (transfers : List (Transfer h)) (a b : Event) : Option (PLift (HappensBefore h a b)) :=
  reaches (transfers.foldl extend []) b

private abbrev Local (a b : Event) := a = b ∨ AtomicWord.SequencedBefore a b

private abbrev Arrived (past : Reached h a) (b : Event) :=
  AtomicWord.SequencedBefore a b ∨ ∃ z ∈ past, Local z.val b

private theorem found (past : Reached h a) (ok : Arrived past b) : (reaches past b).isSome = true := by
  unfold reaches
  split
  · rfl
  · rename_i notLocal
    obtain ⟨z,mem,local_⟩ := ok.resolve_left notLocal
    cases eq : past.findSome? (finish · b) with
    | some => rfl
    | none =>
      have no := List.findSome?_eq_none_iff.mp eq z mem
      rcases local_ with same | later
      · simp [finish,same] at no
      · by_cases same : z.val = b <;> simp [finish,same,later] at no

private theorem through (t : Transfer h) (past : Reached h a)
    (ok : a = t.edge.val.1 ∨ Arrived past t.edge.val.1) : Arrived (extend past t) t.edge.val.2 := by
  unfold extend
  generalize t.edge = edge at *
  obtain ⟨⟨x,y⟩,sync⟩ := edge
  dsimp only at *
  split
  · exact Or.inr ⟨_,List.mem_cons_self,Or.inl rfl⟩
  · rename_i different
    have yes := found past (ok.resolve_left different)
    split
    · exact Or.inr ⟨_,List.mem_cons_self,Or.inl rfl⟩
    · simp_all

private theorem traverse {h : History} {x y : Event} (hb : HappensBefore h x y) :
    ∀ {a} (past : Reached h a), a = x ∨ Arrived past x →
      ∃ ts : List (Transfer h), Arrived (ts.foldl extend past) y := by
  induction hb with
  | sequenced sb =>
    intro a past start
    refine ⟨[],?_⟩
    rcases start with rfl | earlier | ⟨z,mem,rfl | earlier⟩
    · exact Or.inl sb
    · exact Or.inl ⟨earlier.1.trans sb.1,Nat.lt_trans earlier.2 sb.2⟩
    · exact Or.inr ⟨z,mem,Or.inr sb⟩
    · exact Or.inr ⟨z,mem,Or.inr ⟨earlier.1.trans sb.1,Nat.lt_trans earlier.2 sb.2⟩⟩
  | other sync =>
    intro a past start
    let t := Transfer.handoff _ _ sync
    exact ⟨[t],through t past start⟩
  | synchronized i r mem acq seq =>
    intro a past start
    let t := Transfer.epoch i r mem acq seq
    exact ⟨[t],through t past start⟩
  | trans _ _ left right =>
    intro a past start
    obtain ⟨first,reached⟩ := left past start
    obtain ⟨second,done_⟩ := right _ (Or.inr reached)
    exact ⟨first ++ second,by simpa [List.foldl_append] using done_⟩

private theorem retains (t : Transfer h) (past : Reached h a) : past ⊆ extend past t := by
  unfold extend
  generalize t.edge = edge
  obtain ⟨⟨x,y⟩,sync⟩ := edge
  dsimp only
  split
  · exact fun _ mem => List.mem_cons_of_mem _ mem
  · split
    · exact fun _ mem => List.mem_cons_of_mem _ mem
    · exact List.Subset.refl _

private theorem survives (ts : List (Transfer h)) (past : Reached h a) (ok : Arrived past b) :
    Arrived (ts.foldl extend past) b := by
  induction ts generalizing past with
  | nil => exact ok
  | cons t tail ih =>
    apply ih
    rcases ok with sb | ⟨z,mem,local_⟩
    · exact Or.inl sb
    · exact Or.inr ⟨z,retains t past mem,local_⟩

private def InOrder (clock : Event → Nat) (ts : List (Transfer h)) :=
  ts.Pairwise fun t u => clock t.edge.val.2 ≤ clock u.edge.val.2

private theorem divide (clock : Event → Nat) (ts : List (Transfer h)) (pivot : Nat) (sorted : InOrder clock ts) :
    ts = ts.filter (fun t => clock t.edge.val.2 ≤ pivot) ++ ts.filter (fun t => pivot < clock t.edge.val.2) := by
  induction ts with
  | nil => rfl
  | cons t ts ih =>
    obtain ⟨prior,rest⟩ := List.pairwise_cons.mp sorted
    by_cases early : clock t.edge.val.2 ≤ pivot
    · simpa only [List.filter_cons,decide_eq_true early,decide_eq_false (Nat.not_lt.mpr early),
        Bool.false_eq_true,ite_false,ite_true,List.cons_append,List.cons.injEq,true_and] using congrArg (List.cons t) (ih rest)
    · have none_ : ts.filter (fun u => decide (clock u.edge.val.2 ≤ pivot)) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro u mem
        have := prior u mem
        simp; omega
      have late : pivot < clock t.edge.val.2 := Nat.lt_of_not_ge early
      have all_ := ih rest
      rw [none_,List.nil_append] at all_
      simp only [List.filter_cons,decide_eq_false early,decide_eq_true late,Bool.false_eq_true,ite_false,
        ite_true,none_,List.nil_append,List.cons.injEq,true_and]
      exact all_

private theorem useTransfer (t : Transfer h) (past : Reached h a) (ts : List (Transfer h))
    (member : t ∈ ts) (start : a = t.edge.val.1 ∨ Arrived past t.edge.val.1) :
    Arrived (ts.foldl extend past) t.edge.val.2 := by
  obtain ⟨pre,post,rfl⟩ := List.append_of_mem member
  rw [List.foldl_append,List.foldl_cons]
  exact survives post _ (through t _ (start.imp_right (survives pre past)))

/-- A single pass over the actual synchronization edges in acquire order
    covers every path. A transfer can be interleaved with unrelated handoffs. -/
private theorem traverseOrdered {h : History} (clock : Event → Nat)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y) {x y : Event} (hb : HappensBefore h x y) :
    ∀ {a} (past : Reached h a), a = x ∨ Arrived past x → ∀ ts : List (Transfer h), InOrder clock ts →
      (∀ t : Transfer h, clock x < clock t.edge.val.2 → clock t.edge.val.2 ≤ clock y → t ∈ ts) →
      Arrived (ts.foldl extend past) y := by
  induction hb with
  | sequenced sb =>
    intro a past start ts _ _
    apply survives ts past
    rcases start with rfl | earlier | ⟨z,mem,rfl | earlier⟩
    · exact Or.inl sb
    · exact Or.inl ⟨earlier.1.trans sb.1,Nat.lt_trans earlier.2 sb.2⟩
    · exact Or.inr ⟨z,mem,Or.inr sb⟩
    · exact Or.inr ⟨z,mem,Or.inr ⟨earlier.1.trans sb.1,Nat.lt_trans earlier.2 sb.2⟩⟩
  | other sync =>
    intro a past start ts _ cover
    let t := Transfer.handoff _ _ sync
    exact useTransfer t past ts (cover t (causal _ _ t.edge.property) (Nat.le_refl _)) start
  | synchronized i r mem acq seq =>
    intro a past start ts _ cover
    let t := Transfer.epoch i r mem acq seq
    exact useTransfer t past ts (cover t (causal _ _ t.edge.property) (Nat.le_refl _)) start
  | @trans x mid y first second left right =>
    intro a past start ts sorted cover
    let early (t : Transfer h) := decide (clock t.edge.val.2 ≤ clock mid)
    let late (t : Transfer h) := decide (clock mid < clock t.edge.val.2)
    have split_ : ts = ts.filter early ++ ts.filter late := divide clock ts (clock mid) sorted
    have leftReached := left past start (ts.filter early) (List.Pairwise.filter _ sorted)
      (fun t lower upper => List.mem_filter.mpr ⟨cover t lower (Nat.le_trans upper (Nat.le_of_lt (causal _ _ second))),decide_eq_true upper⟩)
    rw [split_,List.foldl_append]
    exact right _ (Or.inr leftReached) _ (List.Pairwise.filter _ sorted)
      (fun t lower upper => List.mem_filter.mpr ⟨cover t (Nat.lt_trans (causal _ _ first) lower) upper,decide_eq_true lower⟩)

theorem complete (clock : Event → Nat) (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (ts : List (Transfer h)) (sorted : ts.Pairwise fun t u => clock t.edge.val.2 ≤ clock u.edge.val.2)
    (covers : ∀ t : Transfer h, t ∈ ts) : HappensBefore h x y ↔ (published ts x y).isSome = true :=
  ⟨fun hb => found _ (traverseOrdered clock causal hb [] (Or.inl rfl) ts sorted (fun t _ _ => covers t)),
   fun ok => ((published ts x y).get ok).down⟩

private theorem batch (h : History) (pairs : List (Event × Event))
    (all_ : ∀ p ∈ pairs, HappensBefore h p.1 p.2) :
    ∃ ts : List (Transfer h), ∀ p ∈ pairs, Arrived (ts.foldl extend ([] : Reached h p.1)) p.2 := by
  induction pairs with
  | nil => exact ⟨[],by simp⟩
  | cons p tail ih =>
    obtain ⟨first,reached⟩ := ih (fun q mem => all_ q (List.mem_cons_of_mem _ mem))
    obtain ⟨last,done_⟩ := traverse (all_ p List.mem_cons_self)
      (first.foldl extend ([] : Reached h p.1)) (Or.inl rfl)
    refine ⟨first ++ last,?_⟩
    intro q mem
    rw [List.foldl_append]
    obtain rfl | inside := List.mem_cons.mp mem
    · exact done_
    · exact survives last _ (reached q inside)

def PublicationContract := ∀ (h : History) (pairs : List (Event × Event)),
  (∀ p ∈ pairs, HappensBefore h p.1 p.2) ↔
    ∃ ts : List (Transfer h), ∀ p ∈ pairs, (published ts p.1 p.2).isSome = true

/-- One finite certificate can serve every indicated read. Its references to
    actual synchronizations may repeat; local execution order is reconstructed. -/
theorem publicationPath : PublicationContract := by
  intro h pairs
  constructor
  · intro hb
    obtain ⟨ts,path⟩ := batch h pairs hb
    exact ⟨ts,fun p mem => found _ (path p mem)⟩
  · rintro ⟨ts,ok⟩ p mem
    exact ((published ts p.1 p.2).get (ok p mem)).down

structure Era (h : History) (j : Journal) (g : Generation) extends Flags h g.slot where
  initialized : j.stores clear = some (g,false)
  swaps : ∀ e ∈ marks, j.stores e = some (g,true)

/-- Locate the initialization of this generation in physical modification order.
    Every current true swap must have been emitted against that same installed task. -/
def recover (h : History) (j : Journal) (g : Generation) : Option (Era h j g) := do
  let ws := h.flags g.slot
  let [k] := (List.finRange ws.length).filter (fun i =>
    let x := ws.get i
    !x.value && j.stores x.event == some (g,false)) | none
  let pre := ws.take k.val
  let clear := (ws.get k).event
  let more := ws.drop (k.val+1)
  let current := more.takeWhile (fun x => x.value && j.stores x.event == some (g,true))
  let future := more.drop current.length
  let outside := fun (x : FlagWrite) => (j.stores x.event).all (fun p => p.1 != g)
  guard (pre.all outside && future.all outside && future.head?.all (fun x => !x.value))
  let marks := current.map FlagWrite.event
  if ok : ws = pre ++ ⟨clear,false⟩ :: (marks.map (fun e => ⟨e,true⟩) ++ future) then
    if sources : j.stores clear = some (g,false) ∧ ∀ e ∈ marks, j.stores e = some (g,true) then
      return ⟨⟨pre,clear,marks,future,ok⟩,sources.1,sources.2⟩
  none

def segment (f : Flags h slot) : List FlagWrite := ⟨f.clear,false⟩ :: f.marks.map (fun e => ⟨e,true⟩)

/-- The only lifetime edges needed are initialization before the read and
    read before the first later modification outside the protected segment. -/
structure Bracket (h : History) (r : FlagRead h.flags) (f : Flags h r.slot) : Prop where
  began : HappensBefore h f.clear r.event
  next : ∀ i : Fin (h.flags r.slot).length, i.val = f.pre.length+(segment f).length →
    HappensBefore h r.event ((h.flags r.slot).get i).event

private theorem split (f : Flags h slot) : h.flags slot = f.pre ++ segment f ++ f.future := by
  rw [f.words]
  simp [segment,List.append_assoc]

private theorem word (r : FlagRead h.flags) (f : Flags h r.slot) (co : CutoffVisibility.Coherent h)
    (logged : r ∈ h.flagReads) (b : Bracket h r f) :
    ∃ i : Fin (segment f).length, r.source.val = f.pre.length+i.val ∧
      (h.flags r.slot).get r.source = (segment f).get i := by
  have atClear : (h.flags r.slot)[f.pre.length]? = some ⟨f.clear,false⟩ := by rw [f.words]; simp
  obtain ⟨bound,head⟩ := List.getElem?_eq_some_iff.mp atClear
  let c : Fin (h.flags r.slot).length := ⟨f.pre.length,bound⟩
  have low : f.pre.length ≤ r.source.val := co.flagWriteRead r logged c (by simpa [c,head] using b.began)
  have hi : r.source.val < f.pre.length+(segment f).length := by
    apply Nat.lt_of_not_ge
    intro late
    let k : Fin (h.flags r.slot).length := ⟨_,Nat.lt_of_le_of_lt late r.source.isLt⟩
    exact (Nat.not_lt_of_ge late) (co.flagReadWrite r logged k (b.next k rfl))
  let i : Fin (segment f).length := ⟨r.source.val - f.pre.length,by omega⟩
  have pos : r.source.val = f.pre.length+i.val := by dsimp [i]; omega
  have get := (congrArg (fun ws : List FlagWrite => ws[r.source.val]?) (split f)).symm.trans
    (List.getElem?_eq_getElem r.source.isLt)
  rw [List.append_assoc,List.getElem?_append_right low] at get
  change (segment f ++ f.future)[i.val]? = _ at get
  rw [List.getElem?_append_left i.isLt,List.getElem?_eq_getElem i.isLt] at get
  exact ⟨i,pos,(Option.some.inj get).symm⟩

def Contract := ∀ (h : History) (r : FlagRead h.flags) (f : Flags h r.slot),
    CutoffVisibility.Coherent h → r ∈ h.flagReads → Bracket h r f →
    (r.value = false → r.source.val = f.pre.length ∧ ((h.flags r.slot).get r.source).event = f.clear) ∧
    (r.value = true → ∃ k : Fin f.marks.length, r.source.val = f.pre.length+1+k.val ∧
      ((h.flags r.slot).get r.source).event = f.marks.get k)

theorem identifies : Contract := by
  intro h r f co logged b
  obtain ⟨i,rank,wrote⟩ := word r f co logged b
  cases i with | mk n inside =>
    cases n with
    | zero =>
      have init : (h.flags r.slot).get r.source = ⟨f.clear,false⟩ := wrote
      exact ⟨fun _ => ⟨by simpa using rank,congrArg FlagWrite.event init⟩,
        fun v => by rw [FlagRead.value,init] at v; contradiction⟩
    | succ k =>
      have ok : k < f.marks.length := by simpa [segment] using inside
      have actual : (h.flags r.slot).get r.source = ⟨f.marks[k],true⟩ := by simpa [segment] using wrote
      exact ⟨fun v => by rw [FlagRead.value,actual] at v; contradiction,
        fun _ => ⟨⟨k,ok⟩,by simpa [Nat.add_assoc,Nat.add_comm 1 k] using rank,congrArg FlagWrite.event actual⟩⟩

def SourceContract := ∀ (h : History) (j : Journal) (r : FlagRead h.flags) (mark : Event)
    (era : Era h j ⟨r.slot,mark⟩), CutoffVisibility.Coherent h → r ∈ h.flagReads → Bracket h r era.toFlags →
    j.stores ((h.flags r.slot).get r.source).event = some (⟨r.slot,mark⟩,r.value)

/-- Coherence and the retained lifetime determine the writer generation even
    if an earlier incarnation wrote the same value at the same address. -/
theorem sourceGeneration : SourceContract := by
  intro h j r mark era co logged bounds
  have ⟨no,yes⟩ := identifies h r era.toFlags co logged bounds
  cases bit : r.value with
  | false => rw [(no bit).2]; exact era.initialized
  | true =>
    obtain ⟨i,_,site⟩ := yes bit
    rw [site]
    exact era.swaps _ (List.get_mem ..)

end NeuralReversi.PvsParallel.Interleaving.FlagSegment
