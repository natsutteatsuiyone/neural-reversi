import NeuralReversi.PvsParallel.Interleaving.ReadPrefix
import NeuralReversi.PvsParallel.Interleaving.ReadPrefixLitmus
import NeuralReversi.PvsParallel.Interleaving.PhysicalReadback
import NeuralReversi.PvsParallel.Interleaving.SnapshotReadback

namespace NeuralReversi.PvsParallel.Interleaving.CutClosure

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site Pc mods)
open JournalAtomicGraph (Graph Rules SB HB RF WriterAt Reads Release Acquire WriterOK)
open ModificationIndex (event word decode)

/-- A cut `k ≤ es.length` is downward-closed under happens-before. -/
def HBClosed (g : Graph) (es : Joint.Tape Op) (k : Nat) : Prop :=
  ∀ u v, HB g es u v → v < k → u < k

/-- A cut `k ≤ es.length` retains every real write observed by any prefix read `< k`.
    Initial synthetic tokens (`≥ es.length`) are remapped by `projectGraph`. -/
def RFClosed (g : Graph) (c : Calls) (rs : List Record) (es : Joint.Tape Op) (k : Nat) : Prop :=
  ∀ a w r, RF g c rs es a w r → r < k → w < k ∨ es.length ≤ w

/-- Map real write stamps `< k` identically, remap initial sentinel tokens
    `es.length + tag l ↦ k + tag l`, and reject out-of-cut real writes `[k, es.length)`. -/
def remapOpt (n k w : Nat) : Option Nat :=
  if w < k then some w
  else if n ≤ w then some (k + (w - n))
  else none

/-- Restrict a physical reads-from graph to the prefix `es.take k`. -/
def projectGraph (g : Graph) (es : Joint.Tape Op) (k : Nat) : Graph where
  initial := g.initial
  source a r := (g.source a r).bind (remapOpt es.length k)
  hb := g.hb

private theorem take_worker {es : Joint.Tape Op} (hk : u < k) :
    Joint.worker (es.take k) u = Joint.worker es u := by
  unfold Joint.worker
  rw [List.getElem?_take_of_lt hk]

private theorem take_opcode {es : Joint.Tape Op} (hk : r < k) :
    Joint.opcode (es.take k) r = Joint.opcode es r := by
  unfold Joint.opcode
  rw [List.getElem?_take_of_lt hk]

private theorem take_word {es : Joint.Tape Op} (hk : r < k) :
    Joint.word (es.take k) r = Joint.word es r := by
  unfold Joint.word
  rw [List.getElem?_take_of_lt hk]

private theorem take_sb_iff {es : Joint.Tape Op} (hn : k ≤ es.length) :
    SB (es.take k) u v ↔ v < k ∧ SB es u v := by
  have len : (es.take k).length = k := List.length_take_of_le hn
  constructor
  · rintro ⟨lt,vk,same⟩
    rw [len] at vk
    have uk : u < k := Nat.lt_trans lt vk
    rw [take_worker uk,take_worker vk] at same
    exact ⟨vk,lt,Nat.lt_of_lt_of_le vk hn,same⟩
  · rintro ⟨vk,lt,_,same⟩
    have uk : u < k := Nat.lt_trans lt vk
    refine ⟨lt,by rw [len]; exact vk,?_⟩
    rw [take_worker uk,take_worker vk]
    exact same

private theorem take_hb_iff {es : Joint.Tape Op} (hn : k ≤ es.length) :
    HB (projectGraph g es k) (es.take k) u v ↔ u < k ∧ v < k ∧ HB g es u v := by
  have len : (es.take k).length = k := List.length_take_of_le hn
  simp only [HB,projectGraph,len]
  constructor
  · rintro ⟨uk,vk,h⟩
    exact ⟨uk,vk,Nat.lt_of_lt_of_le uk hn,Nat.lt_of_lt_of_le vk hn,h⟩
  · rintro ⟨uk,vk,_,_,h⟩
    exact ⟨uk,vk,h⟩

private theorem mods_append (c : Calls) (pre post : List Record) (a : Address) (l : Location) :
    mods c (pre ++ post) a l = mods c pre a l ++ mods c post a l :=
  List.filterMap_append

private theorem timed_of_facts (h : WriterSteps.Facts c es rs) :
    ModificationIndex.Timed c rs es.length :=
  ⟨fun _ mem => h.bound mem,fun _ _ m n same => h.injective m n same,h.distinct⟩

private theorem event_eq_of_hit {c : Calls} {rs : List Record} {n : Nat} {a : Address} {l : Location}
    {j : Nat} {s : Site} (t : ModificationIndex.Timed c rs n) (hit : (mods c rs a l)[j]? = some s) :
    event c rs n a l (j + 1) = s.event c :=
  ((ModificationIndex.decode_eq_some_iff t).mp (ModificationIndex.decode_modification t hit)).2

private theorem decode_forward {es : Joint.Tape Op} (hn : k ≤ es.length)
    (hFull : WriterSteps.Facts c es (pre ++ post))
    (hPre : WriterSteps.Facts c (es.take k) pre)
    (postAfter : ∀ s, Record.writer s ∈ post → k ≤ s.event c)
    {a : Address} {w : Nat} {l : Location} {i : Nat}
    (inCut : w < k ∨ es.length ≤ w)
    (dec : decode c (pre ++ post) es.length a w = some (l,i)) :
    ∃ w', remapOpt es.length k w = some w' ∧
      decode c pre (es.take k).length a w' = some (l,i) ∧
      i ≤ (mods c pre a l).length := by
  have len : (es.take k).length = k := List.length_take_of_le hn
  have tFull := timed_of_facts hFull
  have tPre := timed_of_facts hPre
  obtain ⟨insideFull,stamp⟩ := (ModificationIndex.decode_eq_some_iff tFull).mp dec
  cases i with
  | zero =>
    change es.length + ModificationIndex.tag l = w at stamp
    subst w
    have geN : es.length ≤ es.length + ModificationIndex.tag l := Nat.le_add_right ..
    have notLtK : ¬ es.length + ModificationIndex.tag l < k := by omega
    refine ⟨k + ModificationIndex.tag l,?_,?_,Nat.zero_le _⟩
    · simp [remapOpt,notLtK,geN]
    · rw [len]
      exact ModificationIndex.decode_initial (by rw [←len]; exact tPre) a l
  | succ j =>
    have hitFull := List.getElem?_eq_getElem (show j < (mods c (pre ++ post) a l).length from insideFull)
    have evFull := event_eq_of_hit tFull hitFull
    rw [evFull] at stamp
    have memMod := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hitFull)
    generalize (mods c (pre ++ post) a l)[j] = s at hitFull memMod stamp
    have wLtN : w < es.length := stamp ▸ hFull.bound memMod.1
    have wLtK : w < k := by rcases inCut with h | h; exact h; omega
    have inPre : j < (mods c pre a l).length := by
      apply Classical.byContradiction
      intro hGe
      have geLen : (mods c pre a l).length ≤ j := Nat.le_of_not_gt hGe
      rw [mods_append,List.getElem?_append_right geLen] at hitFull
      have memPost := (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hitFull)).1
      have afterK := postAfter _ memPost
      omega
    have hitPre : (mods c pre a l)[j]? = some s := by
      rw [mods_append,List.getElem?_append_left inPre] at hitFull; exact hitFull
    refine ⟨w,by simp [remapOpt,wLtK],?_,inPre⟩
    rw [ModificationIndex.decode_eq_some_iff tPre]
    exact ⟨inPre,(event_eq_of_hit tPre hitPre).trans stamp⟩

private theorem decode_backward {es : Joint.Tape Op} (hn : k ≤ es.length)
    (hFull : WriterSteps.Facts c es (pre ++ post))
    (hPre : WriterSteps.Facts c (es.take k) pre)
    {a : Address} {w' : Nat} {l : Location} {i : Nat}
    (decPre : decode c pre (es.take k).length a w' = some (l,i)) :
    i ≤ (mods c pre a l).length ∧
    (∀ w, remapOpt es.length k w = some w' →
      decode c (pre ++ post) es.length a w = some (l,i)) ∧
    decode c (pre ++ post) es.length a (if w' < k then w' else es.length + (w' - k)) = some (l,i) := by
  have len : (es.take k).length = k := List.length_take_of_le hn
  have tFull := timed_of_facts hFull
  have tPre := timed_of_facts hPre
  obtain ⟨insidePre,stamp⟩ := (ModificationIndex.decode_eq_some_iff tPre).mp decPre
  have insideFull : i ≤ (mods c (pre ++ post) a l).length := by
    rw [mods_append,List.length_append]; omega
  cases i with
  | zero =>
    rw [len] at stamp
    change k + ModificationIndex.tag l = w' at stamp
    have notLtK : ¬ w' < k := by omega
    have decAtN : decode c (pre ++ post) es.length a (es.length + ModificationIndex.tag l) = some (l,0) :=
      ModificationIndex.decode_initial tFull a l
    refine ⟨insidePre,fun w hw => ?_,?_⟩
    · unfold remapOpt at hw
      split at hw
      · cases hw; omega
      · split at hw
        · rename_i geN
          cases hw
          have eqW : w = es.length + ModificationIndex.tag l := by omega
          rw [eqW]; exact decAtN
        · cases hw
    · simp only [notLtK,↓reduceIte]
      have eqW : es.length + (w' - k) = es.length + ModificationIndex.tag l := by omega
      rw [eqW]; exact decAtN
  | succ j =>
    have hitPre := List.getElem?_eq_getElem (show j < (mods c pre a l).length from insidePre)
    have evPre := event_eq_of_hit tPre hitPre
    rw [evPre] at stamp
    have w'LtK : w' < k := by
      have b := hPre.bound (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hitPre)).1
      rw [len] at b
      exact stamp ▸ b
    have hitFull : (mods c (pre ++ post) a l)[j]? = some ((mods c pre a l)[j]) := by
      rw [mods_append,List.getElem?_append_left insidePre]; exact hitPre
    have decFull : decode c (pre ++ post) es.length a w' = some (l,j + 1) :=
      (ModificationIndex.decode_eq_some_iff tFull).mpr ⟨insideFull,(event_eq_of_hit tFull hitFull).trans stamp⟩
    refine ⟨insidePre,fun w hw => ?_,by simp [w'LtK,decFull]⟩
    unfold remapOpt at hw
    split at hw
    · cases hw; exact decFull
    · split at hw
      · cases hw; omega
      · cases hw

private theorem prefix_word {c : Calls} {endian : Bool} {pre post : List Record}
    {init : Address → Location → Word} {a : Address} {l : Location} {i : Nat}
    (inside : i ≤ (mods c pre a l).length) :
    word c endian pre init a l i = word c endian (pre ++ post) init a l i := by
  unfold word
  cases i with
  | zero => rfl
  | succ j =>
    have lt : j < (mods c pre a l).length := inside
    have ltMap : j < ((mods c pre a l).map (fun s => s.word c endian)).length := by simpa using lt
    simp only [List.getElem?_cons_succ,mods_append,List.map_append,List.getElem?_append_left ltMap]

private theorem writerAt_pre_iff {es : Joint.Tape Op} (hn : k ≤ es.length)
    (hPre : WriterSteps.Facts c (es.take k) pre)
    (postAfter : ∀ s, Record.writer s ∈ post → k ≤ s.event c) :
    WriterAt c pre a pc e succ ↔ e < k ∧ WriterAt c (pre ++ post) a pc e succ := by
  have len : (es.take k).length = k := List.length_take_of_le hn
  constructor
  · rintro ⟨inv,mem,addr,ev,ok⟩
    have lt : e < k := by
      have b := hPre.bound mem
      rw [len] at b
      exact ev ▸ b
    exact ⟨lt,inv,List.mem_append_left _ mem,addr,ev,ok⟩
  · rintro ⟨ek,inv,mem,addr,ev,ok⟩
    rcases List.mem_append.mp mem with inPre | inPost
    · exact ⟨inv,inPre,addr,ev,ok⟩
    · have ge := postAfter ⟨inv,pc⟩ inPost
      change k ≤ (c inv).event pc.index at ge
      omega

private theorem rf_pre_to_full {es : Joint.Tape Op} (hn : k ≤ es.length)
    (hFull : WriterSteps.Facts c es (pre ++ post))
    (hPre : WriterSteps.Facts c (es.take k) pre)
    (postAfter : ∀ s, Record.writer s ∈ post → k ≤ s.event c)
    {a : Address} {w' r : Nat}
    (rfPre : RF (projectGraph g es k) c pre (es.take k) a w' r) :
    ∃ w, remapOpt es.length k w = some w' ∧ RF g c (pre ++ post) es a w r ∧ r < k ∧
      (w' < k → w = w') := by
  obtain ⟨srcPre,l,i,decPre,readsPre⟩ := rfPre
  obtain ⟨w,srcFull,remEq⟩ := Option.bind_eq_some_iff.mp srcPre
  have rk : r < k := by
    rcases readsPre with ⟨mode,op⟩ | ⟨_,wAt⟩
    · apply Classical.byContradiction
      intro hGe
      have len : (es.take k).length = k := List.length_take_of_le hn
      have noneOp : Joint.opcode (es.take k) r = none := by
        unfold Joint.opcode
        rw [List.getElem?_eq_none (by rw [len]; omega)]
        rfl
      rw [noneOp] at op; cases op
    · exact ((writerAt_pre_iff hn hPre postAfter).mp wAt).1
  have readsFull : Reads c (pre ++ post) es a l r := by
    rcases readsPre with ⟨mode,op⟩ | ⟨hl,wAt⟩
    · rw [take_opcode rk] at op; exact Or.inl ⟨mode,op⟩
    · exact Or.inr ⟨hl,((writerAt_pre_iff hn hPre postAfter).mp wAt).2⟩
  have decFull := (decode_backward hn hFull hPre decPre).2.1 w remEq
  refine ⟨w,remEq,⟨srcFull,l,i,decFull,readsFull⟩,rk,fun w'LtK => ?_⟩
  unfold remapOpt at remEq
  split at remEq
  · cases remEq; rfl
  · split at remEq
    · cases remEq; omega
    · cases remEq

/-- Main Prefix Projection Theorem (Task 3):
    Any cut `k ≤ es.length` of a lawful concurrent TT execution that is BOTH
    happens-before closed (`HBClosed`) and reads-from closed (`RFClosed`)
    restricts to a lawful `JournalAtomicGraph.Rules` on `es.take k`. -/
def PrefixProjectionContract : Prop :=
  ∀ {Op : Type} (g : Graph) (c : Calls) (endian : Bool) (pre post : List Record)
    (es : Joint.Tape Op) (k : Nat),
    k ≤ es.length →
    WriterSteps.Facts c es (pre ++ post) →
    WriterSteps.Facts c (es.take k) pre →
    (∀ s, Record.writer s ∈ post → k ≤ s.event c) →
    Rules g c endian (pre ++ post) es →
    HBClosed g es k →
    RFClosed g c (pre ++ post) es k →
    Rules (projectGraph g es k) c endian pre (es.take k)

theorem prefix_rules : PrefixProjectionContract := by
  intro _ g c endian pre post es k hn hFull hPre postAfter laws hbClosed rfClosed
  have len : (es.take k).length = k := List.length_take_of_le hn
  refine {
    trans := fun h1 h2 => by
      have ⟨u1,v1,hb1⟩ := (take_hb_iff hn).mp h1
      have ⟨_,w2,hb2⟩ := (take_hb_iff hn).mp h2
      exact (take_hb_iff hn).mpr ⟨u1,w2,laws.trans hb1 hb2⟩
    irrefl := fun u h => laws.irrefl u ((take_hb_iff hn).mp h).2.2
    program := fun sb => by
      have ⟨vk,sbFull⟩ := (take_sb_iff hn).mp sb
      have uk : _ < k := Nat.lt_trans sbFull.1 vk
      exact (take_hb_iff hn).mpr ⟨uk,vk,laws.program sbFull⟩
    sync := fun {a w' r} rel acq rf => by
      obtain ⟨w,_,rfFull,rk,sameW⟩ := rf_pre_to_full hn hFull hPre postAfter rf
      have ⟨w'k,relFull⟩ : w' < k ∧ Release c (pre ++ post) a w' := by
        rcases rel with fin | cas
        · have ⟨wk,f⟩ := (writerAt_pre_iff hn hPre postAfter).mp fin
          exact ⟨wk,Or.inl f⟩
        · have ⟨wk,cs⟩ := (writerAt_pre_iff hn hPre postAfter).mp cas
          exact ⟨wk,Or.inr cs⟩
      have acqFull : Acquire c (pre ++ post) es a r := by
        rcases acq with ⟨l,op⟩ | cas
        · rw [take_opcode rk] at op; exact Or.inl ⟨l,op⟩
        · exact Or.inr ((writerAt_pre_iff hn hPre postAfter).mp cas).2
      have eqW := sameW w'k
      subst eqW
      exact (take_hb_iff hn).mpr ⟨w'k,rk,laws.sync relFull acqFull rfFull⟩
    fences := fun fence acq sb1 rf sb2 => by
      have ⟨fk,fenceFull⟩ := (writerAt_pre_iff hn hPre postAfter).mp fence
      have ⟨wk,sb1Full⟩ := (take_sb_iff hn).mp sb1
      have ⟨bk,sb2Full⟩ := (take_sb_iff hn).mp sb2
      rw [take_opcode bk] at acq
      obtain ⟨w,_,rfFull,_,sameW⟩ := rf_pre_to_full hn hFull hPre postAfter rf
      have eqW := sameW wk
      subst eqW
      exact (take_hb_iff hn).mpr ⟨fk,bk,laws.fences fenceFull acq sb1Full rfFull sb2Full⟩
    ww := fun du dv hb => by
      have ⟨uk,vk,hbFull⟩ := (take_hb_iff hn).mp hb
      have duFull := (decode_backward hn hFull hPre du).2.2
      have dvFull := (decode_backward hn hFull hPre dv).2.2
      simp only [uk,vk,↓reduceIte] at duFull dvFull
      exact laws.ww duFull dvFull hbFull
    wr := fun du dw hb rf => by
      have ⟨uk,rk,hbFull⟩ := (take_hb_iff hn).mp hb
      have duFull := (decode_backward hn hFull hPre du).2.2
      simp only [uk,↓reduceIte] at duFull
      obtain ⟨w,remEq,rfFull,_,_⟩ := rf_pre_to_full hn hFull hPre postAfter rf
      have dwFull := (decode_backward hn hFull hPre dw).2.1 w remEq
      exact laws.wr duFull dwFull hbFull rfFull
    rr := fun du dv hb rf1 rf2 => by
      have ⟨_,_,hbFull⟩ := (take_hb_iff hn).mp hb
      obtain ⟨w1,rem1,rf1Full,_,_⟩ := rf_pre_to_full hn hFull hPre postAfter rf1
      obtain ⟨w2,rem2,rf2Full,_,_⟩ := rf_pre_to_full hn hFull hPre postAfter rf2
      have duFull := (decode_backward hn hFull hPre du).2.1 w1 rem1
      have dvFull := (decode_backward hn hFull hPre dv).2.1 w2 rem2
      exact laws.rr duFull dvFull hbFull rf1Full rf2Full
    rw := fun du dv rf hb => by
      have ⟨_,vk,hbFull⟩ := (take_hb_iff hn).mp hb
      have dvFull := (decode_backward hn hFull hPre dv).2.2
      simp only [vk,↓reduceIte] at dvFull
      obtain ⟨w,remEq,rfFull,_,_⟩ := rf_pre_to_full hn hFull hPre postAfter rf
      have duFull := (decode_backward hn hFull hPre du).2.1 w remEq
      exact laws.rw duFull dvFull rfFull hbFull
    writer := fun s mem => by
      have okFull := laws.writer s (List.mem_append_left _ mem)
      cases s with
      | mk inv pc =>
        cases pc with
        | cas =>
          obtain ⟨w,i,srcFull,decFull,cmp,nextMod⟩ := okFull
          have rk : (c inv).event 0 < k := by
            have b := hPre.bound mem; rwa [len] at b
          have rfFull : RF g c (pre ++ post) es (c inv).address w ((c inv).event 0) :=
            ⟨srcFull,.sequence,i,decFull,Or.inr ⟨rfl,inv,List.mem_append_left _ mem,rfl,rfl,by simp⟩⟩
          have inCut := rfClosed (c inv).address w ((c inv).event 0) rfFull rk
          obtain ⟨w',remEq,decPre,inPre⟩ := decode_forward hn hFull hPre postAfter inCut decFull
          refine ⟨w',i,by simp [projectGraph,srcFull,remEq],decPre,?_,?_⟩
          · rw [prefix_word inPre]; exact cmp
          · intro yes
            have hitFull := nextMod yes
            have modEq := mods_append c pre post (c inv).address .sequence
            rw [modEq] at hitFull
            have ltPre : i < (mods c pre (c inv).address .sequence).length := by
              apply Classical.byContradiction
              intro hGe
              have gePre := Nat.le_of_not_gt hGe
              rw [List.getElem?_append_right gePre] at hitFull
              have memPost := (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hitFull)).1
              have afterK := postAfter _ memPost
              change k ≤ (c inv).event 0 at afterK
              omega
            rwa [List.getElem?_append_left ltPre] at hitFull
        | fence => exact okFull
        | field _ => exact okFull
        | finish => exact okFull
    load := fun r a l mode op => by
      have rk : r < k := by
        apply Classical.byContradiction
        intro hGe
        have noneOp : Joint.opcode (es.take k) r = none := by
          unfold Joint.opcode
          rw [List.getElem?_eq_none (by rw [len]; omega)]
          rfl
        rw [noneOp] at op; cases op
      rw [take_opcode rk] at op
      obtain ⟨w,i,srcFull,decFull,wordEq⟩ := laws.load r a l mode op
      have rfFull : RF g c (pre ++ post) es a w r := ⟨srcFull,l,i,decFull,Or.inl ⟨mode,op⟩⟩
      have inCut := rfClosed a w r rfFull rk
      obtain ⟨w',remEq,decPre,inPre⟩ := decode_forward hn hFull hPre postAfter inCut decFull
      refine ⟨w',i,by simp [projectGraph,srcFull,remEq],decPre,?_⟩
      rw [prefix_word inPre,take_word rk]
      exact wordEq }

/-- Synchronized separation (`ReadPrefix.Separated`) implies `RFClosed`!
    This connects `ReadPrefix.afterRetirement` directly to `prefix_rules`. -/
def SeparatedClosureContract : Prop :=
  ∀ {Op : Type} (g : Graph) (c : Calls) (endian : Bool) (pre post : List Record)
    (es : Joint.Tape Op) (k : Nat)
    (hFull : WriterSteps.Facts c es (pre ++ post))
    (_ : WriterSteps.Facts c (es.take k) pre)
    (laws : Rules g c endian (pre ++ post) es),
    k ≤ es.length →
    (∀ a l r, r < k → Reads c (pre ++ post) es a l r →
      ReadPrefix.Separated (JournalAtomicGraph.execution hFull laws) a l r) →
    RFClosed g c (pre ++ post) es k

theorem separated_rfClosed : SeparatedClosureContract := by
  intro _ g c endian pre post es k hFull hPre laws hn sep a w r rf rk
  obtain ⟨src,l,i,dec,reads⟩ := rf
  have tFull := timed_of_facts hFull
  obtain ⟨_,stamp⟩ := (ModificationIndex.decode_eq_some_iff tFull).mp dec
  have rwCoh := JournalAtomicGraph.readWrite hFull laws
  have rfExec : ((JournalAtomicGraph.execution hFull laws).memory a).rf
      (((JournalAtomicGraph.execution hFull laws).memory a).write l i) r := by
    change RF g c (pre ++ post) es a (event c (pre ++ post) es.length a l i) r
    rw [stamp]
    exact ⟨src,l,i,dec,reads⟩
  have closed := ReadPrefix.closed (JournalAtomicGraph.execution hFull laws) a l r i rwCoh
    (sep a l r rk reads) rfExec
  rcases closed.2 with rfl | ⟨s,atPre,evEq⟩
  · change es.length + ModificationIndex.tag l = w at stamp
    exact Or.inr (by omega)
  · have memPre := (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? atPre)).1
    have bPre := hPre.bound memPre
    rw [List.length_take_of_le hn] at bPre
    have eqW : w = s.event c := stamp.symm.trans evEq
    exact Or.inl (eqW ▸ bPre)

/-- Any ADOPTED TT lookup hit (not a fast key-mismatch return) from epoch `m > 0`
    has its writer's `.finish` release happen-before the reader's opening `.sequence`
    acquire load — so an HB-closed cut containing the hit ALREADY includes the
    committed epoch's publication event! -/
theorem adopted_epoch_before_hit {Op : Type} {g : Graph} {c : Calls} {endian : Bool} {rs : List Record}
    {es : Joint.Tape Op} (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es)
    (init : NativeTableWrites.Journal.Even ((JournalAtomicGraph.execution h laws).memory a))
    {b : PvsBoardTree.Position} {ev source : Nat → Nat} {e : EntryData}
    (steps : ReaderMir.Realizes ((JournalAtomicGraph.execution h laws).memory a)
      (ReaderMir.lookupRun endian b.val.player b.val.opponent
        (((JournalAtomicGraph.execution h laws).memory a).readValue ∘ ev)).1 ev source)
    (hit : (ReaderMir.lookupRun endian b.val.player b.val.opponent
      (((JournalAtomicGraph.execution h laws).memory a).readValue ∘ ev)).2 = some e)
    (noWrap : source 5 < source 0 + SeqLock.modulus)
    (nonzero : 0 < source 0 / 2) :
    let m := source 0 / 2
    ∃ s, (mods c rs a .sequence)[2 * m - 1]? = some s ∧
      s.pc = .finish ∧
      HB g es (s.event c) (ev 0) := by
  let x := JournalAtomicGraph.execution h laws
  have seen := Table.seen_lookup_sources x a init b ev source steps hit noWrap
  obtain ⟨_,_,_,_,even0⟩ := seen.2.indices
  let m := source 0 / 2
  have rf0 : (x.memory a).rf ((x.memory a).write .sequence (source 0)) (ev 0) := seen.2.first
  have ltMods : 2 * m - 1 < (mods c rs a .sequence).length := by
    have bound := x.source a .sequence (source 0) (ev 0) rf0
    omega
  have atFin := List.getElem?_eq_getElem ltMods
  let s := (mods c rs a .sequence)[2 * m - 1]
  have modEv := (x.modified a .sequence (2 * m - 1) s atFin).1
  let slot := x.slot a init
  have histLen : slot.history.length = (mods c rs a .sequence).length := by
    simp [slot,NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.Execution.toSequence,
      NativeTableWrites.Journal.sequence]
  have pubIdx : 2 * m - 1 + 1 = source 0 := by omega
  obtain ⟨call,pubAct,_⟩ := slot.publishedPayload m nonzero (by rw [histLen]; omega)
  have pubMap : ((mods c rs a .sequence).map (NativeTableWrites.Journal.Site.action c endian))[2 * m - 1]? =
      some (.publish call) := by
    simpa [slot,NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.Execution.toSequence,
      NativeTableWrites.Journal.sequence] using pubAct
  rw [List.getElem?_map,atFin] at pubMap
  have memS := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? atFin)
  have locS : s.pc.location = some .sequence := memS.2.2.2
  cases hPc : s.pc with
  | cas =>
    simp [NativeTableWrites.Journal.Site.action,s,hPc] at pubMap
  | fence | field _ =>
    rw [hPc] at locS; cases locS
  | finish =>
    have relS : Release c rs a (s.event c) :=
      Or.inl ⟨s.invocation,hPc ▸ memS.1,memS.2.1,by simp [NativeTableWrites.Journal.Site.event,hPc],fun _ => memS.2.2.1⟩
    have firstAcq : (ReaderMir.lookupRun endian b.val.player b.val.opponent
        (((JournalAtomicGraph.execution h laws).memory a).readValue ∘ ev)).1[0]? =
        some (.load .sequence .acquire) := by
      unfold ReaderMir.lookupRun
      split <;> try rfl
      split <;> try rfl
      split <;> rfl
    have acq0 : Acquire c rs es a (ev 0) := steps.acquire 0 .sequence firstAcq
    have rfS : RF g c rs es a (s.event c) (ev 0) := by
      rw [←modEv,pubIdx]; exact rf0
    exact ⟨s,atFin,hPc,laws.sync relS acq0 rfS⟩

/-- Compute the minimal cut `k' ≥ k` (bounded by `n`) that is closed under BOTH
    `hbEdges` and `rfEdges` (excluding initial sentinel sources `≥ n`). -/
def stepCut (edges : List (Nat × Nat)) (n k : Nat) : Nat :=
  edges.foldl (fun acc e => if e.2 < acc && e.1 < n then max acc (e.1 + 1) else acc) k

private def iterateCut (edges : List (Nat × Nat)) (n : Nat) : Nat → Nat → Nat
  | 0, k => min k n
  | fuel + 1, k =>
    let next := min (stepCut edges n k) n
    if next = k then k else iterateCut edges n fuel next

def minimalCut (hbEdges rfEdges : List (Nat × Nat)) (n k : Nat) : Nat :=
  iterateCut (hbEdges ++ rfEdges) n n (min k n)

private theorem stepCut_ge (edges : List (Nat × Nat)) (n k : Nat) : k ≤ stepCut edges n k := by
  unfold stepCut
  induction edges generalizing k with
  | nil => exact Nat.le_refl _
  | cons e rest ih =>
    simp only [List.foldl_cons]
    refine Nat.le_trans ?_ (ih _)
    split <;> omega

private theorem stepCut_le (edges : List (Nat × Nat)) (n k : Nat) (hk : k ≤ n) :
    stepCut edges n k ≤ n := by
  unfold stepCut
  induction edges generalizing k with
  | nil => exact hk
  | cons e rest ih =>
    simp only [List.foldl_cons]
    apply ih
    split
    · rename_i cond
      simp only [Bool.and_eq_true,decide_eq_true_eq] at cond
      omega
    · exact hk

private theorem stepCut_mem (edges : List (Nat × Nat)) (n k : Nat) {u v : Nat}
    (mem : (u,v) ∈ edges) (vk : v < k) (un : u < n) : u < stepCut edges n k := by
  induction edges generalizing k with
  | nil => cases mem
  | cons e rest ih =>
    change u < stepCut rest n (if e.2 < k && e.1 < n then max k (e.1 + 1) else k)
    rcases List.mem_cons.mp mem with rfl | tail
    · have cond : (v < k && u < n) = true := by simp [vk,un]
      simp only [cond,↓reduceIte]
      have ge := stepCut_ge rest n (max k (u + 1))
      omega
    · have geStep : k ≤ (if e.2 < k && e.1 < n then max k (e.1 + 1) else k) := by split <;> omega
      exact ih _ tail (Nat.lt_of_lt_of_le vk geStep)

private theorem iterateCut_props (edges : List (Nat × Nat)) (n fuel k : Nat)
    (hk : k ≤ n) (enough : n - k ≤ fuel) :
    let out := iterateCut edges n fuel k
    k ≤ out ∧ out ≤ n ∧ ∀ u v, (u,v) ∈ edges → v < out → u < n → u < out := by
  induction fuel generalizing k with
  | zero =>
    have eqN : k = n := by omega
    subst k
    simp only [iterateCut,Nat.min_self]
    exact ⟨Nat.le_refl _,Nat.le_refl _,fun _ _ _ _ un => un⟩
  | succ f ih =>
    simp only [iterateCut]
    have leN : stepCut edges n k ≤ n := stepCut_le edges n k hk
    have minEq : min (stepCut edges n k) n = stepCut edges n k := Nat.min_eq_left leN
    rw [minEq]
    have geK : k ≤ stepCut edges n k := stepCut_ge edges n k
    by_cases fixed : stepCut edges n k = k
    · simp only [fixed,↓reduceIte]
      refine ⟨Nat.le_refl _,hk,fun u v mem vk un => ?_⟩
      rw [←fixed]
      exact stepCut_mem edges n k mem vk un
    · simp only [fixed,↓reduceIte]
      have gtK : k < stepCut edges n k := Nat.lt_of_le_of_ne geK (Ne.symm fixed)
      have ⟨r1,r2,r3⟩ := ih (stepCut edges n k) leN (by omega)
      exact ⟨Nat.le_trans geK r1,r2,r3⟩

/-- `minimalCut` always returns a cut `k'` with `min k n ≤ k' ≤ n` that is closed
    under every HB and RF edge whose source lies inside `< n`. -/
def MinimalCutContract : Prop :=
  ∀ (hbEdges rfEdges : List (Nat × Nat)) (n k : Nat),
    let k' := minimalCut hbEdges rfEdges n k
    min k n ≤ k' ∧ k' ≤ n ∧
    (∀ u v, (u,v) ∈ hbEdges → v < k' → u < n → u < k') ∧
    (∀ u v, (u,v) ∈ rfEdges → v < k' → u < n → u < k')

theorem minimalCut_sound : MinimalCutContract := by
  intro hbEdges rfEdges n k
  have hk : min k n ≤ n := Nat.min_le_right k n
  have ⟨r1,r2,r3⟩ := iterateCut_props (hbEdges ++ rfEdges) n n (min k n) hk (by omega)
  exact ⟨r1,r2,
    fun u v m => r3 u v (List.mem_append_left _ m),
    fun u v m => r3 u v (List.mem_append_right _ m)⟩

namespace LitmusBoundary

-- The 7-event key-mismatch litmus trace from `ReadPrefixLitmus`:
-- Worker 0: events 0, 1, 2, 6 (HB: 0->1->2->6)
-- Worker 1: events 3, 4, 5   (HB: 3->4->5)
-- RF edges with real writes < 7: event 4 reads event 6 (player payload), incomparable in HB.
private def litmusHB : List (Nat × Nat) := [(0,1),(1,2),(2,6),(3,4),(4,5)]
private def litmusRF : List (Nat × Nat) := [(6,4)]

-- At k = 6: HB-closed alone leaves k = 6 unchanged (missing event 6!).
-- Closing under BOTH HB and RF expands k = 6 to k' = 7, retaining event 6.
-- At k = 3: events 0..2 have no RF into 6, so k = 3 stays k' = 3.
example :
    minimalCut litmusHB [] 7 6 = 6 ∧
    minimalCut litmusHB litmusRF 7 6 = 7 ∧
    minimalCut litmusHB litmusRF 7 4 = 4 ∧
    minimalCut litmusHB litmusRF 7 5 = 7 ∧
    minimalCut litmusHB litmusRF 7 3 = 3 := by
  decide

end LitmusBoundary

end NeuralReversi.PvsParallel.Interleaving.CutClosure
