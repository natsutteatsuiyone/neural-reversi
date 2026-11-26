import NeuralReversi.PvsParallel.Interleaving.IteratorEventOrder
import NeuralReversi.PvsParallel.Interleaving.Joint
import NeuralReversi.PvsParallel.Interleaving.BestPublisher
import NeuralReversi.PvsParallel.Interleaving.SchedulerReplay

namespace NeuralReversi.PvsParallel.Interleaving.SchedulerTrace

open Threading
open AtomicWord (Event Write)
open SplitCompletion (Action)
open SchedulerReplay (Instruction)

private def once (es : Joint.Tape Op) (at_ : Event) (test : Joint.SplitWord → Bool) : Option Nat :=
  match es.zipIdx.filterMap fun (e,k) => match e with
    | .splitWord w pc op => if w == at_.worker && pc == at_.step then some (k,op) else none
    | _ => none with
  | [(k,op)] => if test op then some k else none
  | _ => none

private def isMark (bank : Nat) : Joint.SplitWord → Bool
  | .mark b => bank == b
  | _ => false

private def isResult (bank : Nat) (expected : BitVec 32) : Joint.SplitWord → Bool
  | .result b word => bank == b && word == expected
  | _ => false

private def target : Action → Nat
  | .book w _ | .finish w _ => w.val

/-- Verify the actual pre/post mask words and emitting instruction before projecting
    onto effective credits. A redundant finish still consumes a real physical RMW. -/
private def mask (g : SlotFrameMir.RawMask ws) (pos : Nat) (site : Event) (participant : Nat)
    (booking : Bool) (old stored : BitVec 64) : Option Bool := do
  if bound : pos < g.actions.length then
    let i := (⟨pos,bound⟩ : Fin g.actions.length)
    let change := g.actions.get i
    guard (change.event = site ∧ target change = participant ∧ !change.release = booking)
    guard ((ws.get (g.before i)).value = old ∧ (ws.get (g.at i)).value = stored)
    return SlotFrameMir.RawProjection.keeps g i
  else none

structure Row where
  site : Event
  op : Joint.SplitWord

private def within (owner bank worker : Nat) : Joint.SplitWord → Bool
  | .mark k | .result k _ => worker == owner && k == bank
  | .maskRmw o k .. | .claim _ o k .. | .exhausted o k .. | .lock o k | .unlock o k =>
    o == owner && k == bank

private def scan {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t)
    (acc : Nat × List (Instruction t.width)) (row : Row) : Option (Nat × List (Instruction t.width)) := do
  match row.op with
  | .mark .. | .result .. => none
  | .maskRmw _ _ who booking old stored =>
    let effective ← mask b.generation acc.1 row.site who booking old stored
    return (acc.1+1, if effective then .mask row.site :: acc.2 else acc.2)
  | .claim opening _ _ old stored =>
    guard (old < 2^t.width ∧ stored < 2^t.width)
    return (acc.1,.claim ⟨row.site.worker,opening,BitVec.ofNat _ old,BitVec.ofNat _ stored⟩ :: acc.2)
  | .exhausted _ _ load failures => return (acc.1,.exhausted row.site.worker load failures :: acc.2)
  | .lock .. => return (acc.1,.lock row.site :: acc.2)
  | .unlock .. => return acc

/-- Only the interval from this owner's initialization to its result load can
    contribute. The raw mask modifications, including no-op resets, are all required. -/
def interval {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) (es : Joint.Tape Op) :
    Option (List Row) := do
  let start := (b.mask.get b.generation.mark).event
  let finish := b.owner.exit.clock.atStep (b.owner.exit.log.length+5)
  let a ← once es start (isMark b.bankIndex)
  let winner ← BestPublisher.publisher t
  let z ← once es finish (isResult b.bankIndex winner.1.word)
  guard (a < z)
  return ((es.take z).drop (a+1)).filterMap fun e => match e with
    | .splitWord w pc op =>
      if within b.generation.owner.val b.bankIndex w op then some ⟨⟨w,pc⟩,op⟩ else none
    | _ => none

def decode {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) (rows : List Row) :
    Option (List (Instruction t.width)) := do
  let (used,reverse) ← rows.foldlM (scan b) (0,[])
  guard (used = b.generation.actions.length)
  return reverse.reverse

def Row.matches (r : Row) (op : MoveList.ConcurrentTrace.Write width) : Bool :=
  match r.op with
  | .claim opening _ _ old stored =>
    r.site.worker == op.worker && opening == op.opening && old == op.old.toNat && stored == op.stored.toNat
  | _ => false

def Row.exhausted (r : Row) : Bool :=
  match r.op with | .exhausted .. => true | _ => false

private def occurrence (rs : List Row) (op : MoveList.ConcurrentTrace.Write width) :
    Option (Fin rs.length) :=
  match (List.finRange rs.length).filter (fun i => (rs.get i).matches op) with
  | [i] => some i
  | _ => none

abbrev OrderedRows (rs : List Row) (before : Event → Event → Bool) :=
  EventOrder.Extension rs.length fun i j => before (rs.get i).site (rs.get j).site

private def ranks {width : Nat} (ops : List (MoveList.ConcurrentTrace.Write width)) (rs : List Row)
    (before : Event → Event → Bool) : Option (OrderedRows rs before) := do
  let locate := fun i : Fin ops.length => occurrence rs (ops.get i)
  if allFound : ∀ i, (locate i).isSome then
    let ids := fun i => (locate i).get (allFound i)
    guard (List.ofFn ids).Nodup
    let hb := fun i j : Fin rs.length => before (rs.get i).site (rs.get j).site = true
    let indices := IteratorEventOrder.withExhausted hb ids (fun i => (rs.get i).exhausted)
    if edges : ∀ i j, hb i j → indices.idxOf i < indices.idxOf j then
      return ⟨indices,EventOrder.permutation _,edges⟩
    else none
  else none

/-- Put this generation's successful iterator RMWs in atomic modification order.
    All original row identities and supplied HB edges survive an accepted replay. -/
def order {width : Nat} (ops : List (MoveList.ConcurrentTrace.Write width)) (rs : List Row)
    (before : Event → Event → Bool) : Option (List Row) :=
  (ranks ops rs before).map fun r => r.indices.map rs.get

def OrderContract := ∀ {width} (ops : List (MoveList.ConcurrentTrace.Write width)) (rs sorted : List Row)
    (before : Event → Event → Bool), order ops rs before = some sorted →
    ∃ r : OrderedRows rs before, sorted = r.indices.map rs.get

theorem keepsRows : OrderContract := by
  intro _ ops rs sorted before ok
  obtain ⟨r,_,exactRows⟩ := Option.map_eq_some_iff.mp ok
  exact ⟨r,exactRows.symm⟩

private theorem occurrence_at (rs : List Row) (op : MoveList.ConcurrentTrace.Write w) (j : Fin rs.length)
    (unique : ∀ i, (rs.get i).matches op = true ↔ i = j) : occurrence rs op = some j := by
  let chosen := (List.finRange rs.length).filter (fun i => (rs.get i).matches op)
  have single : chosen.Perm [j] := by
    apply (List.perm_ext_iff_of_nodup ((List.nodup_finRange _).filter _) (by simp)).mpr
    intro i
    simpa only [List.mem_filter,List.mem_finRange,true_and,List.mem_singleton] using unique i
  unfold occurrence
  have eq := List.perm_singleton.mp single
  dsimp only [chosen] at eq
  rw [eq]

private theorem distinct_claims (r : Row) (a b : MoveList.ConcurrentTrace.Write w)
    (one : r.matches a = true) (two : r.matches b = true) : a.old.toNat = b.old.toNat := by
  cases r with | mk site op =>
    cases op <;> simp [Row.matches] at one two
    exact one.1.2.symm.trans two.1.2

def PhysicalOrderContract := ∀ {w len} (start finalWord : BitVec w) (ops : List (MoveList.ConcurrentTrace.Write w))
    (rs : List Row) (before : Event → Event → Bool) (ticket : Fin ops.length → Fin rs.length),
    MoveList.ConcurrentTrace.replay len start ops = some finalWord → len < 2^w →
    (∀ i j, (rs.get j).matches (ops.get i) = true ↔ j = ticket i) →
    (∀ e, before e e = false) → (∀ a b c, before a b = true → before b c = true → before a c = true) →
    (∀ i j, before (rs.get (ticket i)).site (rs.get (ticket j)).site = true → i < j) →
    (∀ e : Fin rs.length, (rs.get e).exhausted = true →
      IteratorEventOrder.ExhaustedRead start len ops (fun i => (rs.get (ticket i)).site) before (rs.get e).site) →
    (order ops rs before).isSome

/-- For a real set of uniquely identified CAS instructions, coherence and the read
    operands of the exhausted calls suffice for the actual order decoder to succeed. -/
theorem physicalOrder : PhysicalOrderContract := by
  intro w len start last ops rs before ticket run fits match_ noLoop trans writeWrite reads
  have hit i := (match_ i (ticket i)).2 rfl
  have located i := occurrence_at rs (ops.get i) (ticket i) (match_ i)
  have distinct := (MoveList.ConcurrentTrace.consecutive ops run fits).2
  have unique : Function.Injective ticket := by
    intro i j eq
    have same := distinct_claims (rs.get (ticket j)) _ _ (eq ▸ hit i) (hit j)
    let old := ops.map (fun op => op.old.toNat)
    have at_ (k : Fin ops.length) : old[k.val]'(by simp [old]) = (ops.get k).old.toNat := by simp [old]
    have idx := congrArg old.idxOf same
    rw [←at_ i,←at_ j,distinct.idxOf_getElem,distinct.idxOf_getElem] at idx
    exact Fin.ext idx
  have nodup : (List.ofFn ticket).Nodup := by
    apply List.pairwise_iff_getElem.mpr
    intro i j hi hj lt
    simp only [List.getElem_ofFn]
    exact fun eq => (Nat.ne_of_lt lt) (congrArg Fin.val (unique eq))
  have separate i : (rs.get (ticket i)).exhausted = false := by
    have found := hit i
    generalize rs.get (ticket i) = r at found ⊢
    cases r with | mk site op => cases op <;> simp_all [Row.matches,Row.exhausted]
  have edges := (IteratorEventOrder.projectsReads start last ops (fun i => (rs.get i).site) before
    ticket (fun i => (rs.get i).exhausted) run unique noLoop trans writeWrite separate reads).2.1
  simp only [order,ranks,located,Option.isSome_some,implies_true,dite_eq_left,Option.get_some,nodup]
  rw [dite_eq_left edges]
  rfl

def instructions {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) (es : Joint.Tape Op)
    (before : Option (Event → Event → Bool) := none) : Option (List (Instruction t.width)) := do
  let raw ← interval b es
  let rs ← match before with
    | none => some raw
    | some hb => order t.state.writes raw hb
  decode b rs

def fromTape {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t)
    (root cut : Bool) (depth : Nat) (stop : SchedulerReplay.Stops b) (es : Joint.Tape Op)
    (before : Option (Event → Event → Bool) := none) :=
  (instructions b es before).bind (SchedulerReplay.capture b root cut depth stop)

end NeuralReversi.PvsParallel.Interleaving.SchedulerTrace
