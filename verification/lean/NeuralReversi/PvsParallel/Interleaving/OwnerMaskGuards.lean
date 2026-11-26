import NeuralReversi.PvsParallel.SlotFrameMir.Previous

namespace NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitSchedulerMir (Clock Guard Op Returned)
open SplitSchedulerMir.OwnerFields (maskClock)
open SlotFrameMir (RawMask OwnerExit)

/-- Physical reads-from names a modifying event, including when several events wrote zero. -/
def source (ws : List Write) (stamp : Event) : Option (Fin ws.length) :=
  match (List.finRange ws.length).filter (fun i => (ws.get i).event == stamp) with
  | [i] => some i
  | _ => none

theorem source_event (found : source ws stamp = some i) : (ws.get i).event = stamp := by
  unfold source at found
  split at found <;> try contradiction
  rename_i row one
  cases found
  have mem : i ∈ (List.finRange ws.length).filter (fun k => (ws.get k).event == stamp) := by
    rw [one]; simp
  exact beq_iff_eq.mp (List.mem_filter.mp mem).2

theorem source_of_distinct (distinct : (ws.map Write.event).Nodup) (i : Fin ws.length) :
    source ws (ws.get i).event = some i := by
  have ordinal (j : Fin ws.length) : (ws.map Write.event).idxOf (ws.get j).event = j.val := by
    simpa using distinct.idxOf_getElem j.val (by simp)
  unfold source
  let chosen := (List.finRange ws.length).filter (fun j => (ws.get j).event == (ws.get i).event)
  have single : chosen.Perm [i] := by
    apply (List.perm_ext_iff_of_nodup ((List.nodup_finRange _).filter _) (by simp)).mpr
    intro j
    simp only [List.mem_filter,List.mem_finRange,true_and,beq_iff_eq,List.mem_singleton]
    exact ⟨fun eq => Fin.ext ((ordinal j).symm.trans ((congrArg (ws.map Write.event).idxOf eq).trans (ordinal i))),
      fun eq => eq ▸ rfl⟩
  rw [show (List.finRange ws.length).filter _ = [i] from List.perm_singleton.mp single]

inductive Operand where
  | shutdown (site : Event) (closed : Bool)
  | mask (site stamp : Event) (word : BitVec 64) (acquire : Bool)

def Operand.site : Operand → Event
  | .shutdown e _ | .mask e .. => e

def Operand.guard (ws : List Write) : Operand → Option (Guard ws)
  | .shutdown _ b => some (.shutdown b)
  | .mask _ stamp word acquire => do
    let i ← source ws stamp
    if acquire && (ws.get i).value == word then some (.mask i) else none

def site (owner : Clock) (index : Nat) := (maskClock owner).atStep (4 + index)

def guards (ws : List Write) (owner : Clock) (index : Nat) : List Operand → Option (List (Guard ws))
  | [] => some []
  | row :: tail => do
    if row.site ≠ site owner index then none else
    let op ← row.guard ws
    return op :: (← guards ws owner (index + 1) tail)

private theorem located (done_ : guards ws owner n rows = some log)
    (part : log = pre ++ Guard.mask i :: post) :
    ∃ row ∈ rows, row.site = site owner (n + pre.length) ∧ row.guard ws = some (.mask i) := by
  induction rows generalizing n log pre with
  | nil => simp [guards] at done_; simp_all
  | cons head tail ih =>
    simp only [guards] at done_
    split at done_ <;> try contradiction
    rename_i right
    obtain ⟨op, decoded, rest⟩ := Option.bind_eq_some_iff.mp done_
    obtain ⟨xs, passed, emit⟩ := Option.bind_eq_some_iff.mp rest
    cases emit
    cases pre with
    | nil =>
      cases part
      exact ⟨head, List.mem_cons_self .., by simpa using right, decoded⟩
    | cons a pre =>
      obtain ⟨_, p⟩ := List.cons.inj part
      obtain ⟨row, mem, addr, read⟩ := ih passed p
      exact ⟨row, List.mem_cons_of_mem _ mem, by simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using addr, read⟩

/-- The first modification after all this generation's RMWs is the next clear of
    this physical slot, even if the next initialization stops before marking its owner. -/
def afterReturn (owner : Clock) (numGuards : Nat) (next : Option Write) : Bool :=
  match next with
  | none => true
  | some w =>
    !w.release && !w.rmw && w.value == 0 && w.event.worker == owner.worker &&
      (SplitSchedulerMir.OwnerFields.returned owner numGuards).step < w.event.step

private theorem beforeNext (g : RawMask ws) (owner : Clock) (n j : Nat) (inside : j < n)
    (ok : afterReturn owner n ws[g.mark.val+g.actions.length+1]?)
    (i : Fin ws.length) (pos : i.val = g.mark.val+g.actions.length+1) :
    SequencedBefore (site owner j) (ws.get i).event := by
  rw [←pos, List.getElem?_eq_getElem i.isLt] at ok
  obtain ⟨checks,later⟩ := Bool.and_eq_true_iff.mp ok
  have actor := beq_iff_eq.mp (Bool.and_eq_true_iff.mp checks).2
  refine ⟨actor.symm, Nat.lt_trans ?_ (of_decide_eq_true later)⟩
  change owner.tick (if 4+j<3 then 4+j else 4+j+2) <
    owner.tick (SplitSchedulerMir.OwnerFields.returnIndex n .returned)
  apply owner.advances
  change (if 4+j<3 then 4+j else 4+j+2) < 6+n+3
  split <;> omega

/-- Ordinary write-read and read-write coherence for physically observed loads.
    The specification imposes no common order on loads by different workers. -/
def Coherent (hist : History) (ws : List Write) (rows : List Operand) : Prop :=
  ∀ row ∈ rows, ∀ i, row.guard ws = some (.mask i) →
    SplitRebookingMir.Coherence hist ws ⟨row.site,i,true⟩

private def edge (ws : List Write) (op : Op n) (next : Fin n) (log : List (Guard ws)) :
    Option (List (Guard ws)) :=
  match op, log with
  | .any choices, _ => if next ∈ choices then some log else none
  | .shutdown live closed, .shutdown flag :: rest =>
    if next = (if flag then closed else live) then some rest else none
  | .mask busy empty, .mask i :: rest =>
    if next = (if (ws.get i).value = 0 then empty else busy) then some rest else none
  | _, _ => none

/-- The finite PC path consists of actual control-flow successors, including
    loop iterations. Only shutdown and mask load blocks consume a guard row. -/
def follows (code : Fin n → Op n) (ws : List Write) :
    List (Fin n) → Fin n → List (Guard ws) → Bool
  | [], pc, log => match code pc with | .done => log.isEmpty | _ => false
  | next :: rest, pc, log => (edge ws (code pc) next log).any (follows code ws rest next)

theorem follows_correct (ok : follows code ws path pc log) : Returned code ws pc log := by
  induction path generalizing pc log with
  | nil =>
    cases typed : code pc <;> simp [follows,typed] at ok
    subst log
    exact .done typed
  | cons next path ih =>
    obtain ⟨after,move,tail⟩ := (Option.any_eq_true _ _).mp ok
    have run := ih tail
    cases typed : code pc with
    | any choices =>
      simp [edge,typed] at move
      exact move.2 ▸ Returned.next typed move.1 (move.2 ▸ run)
    | done => simp [edge,typed] at move
    | shutdown live closed =>
      cases log with
      | nil => simp [edge,typed] at move
      | cons head log =>
        cases head with
        | mask => simp [edge,typed] at move
        | shutdown b =>
          simp [edge,typed] at move
          exact .shutdown typed (move.1 ▸ move.2 ▸ run)
    | mask busy empty =>
      cases log with
      | nil => simp [edge,typed] at move
      | cons head log =>
        cases head with
        | shutdown => simp [edge,typed] at move
        | mask i =>
          simp [edge,typed] at move
          exact .mask i typed (move.1 ▸ move.2 ▸ run)

theorem follows_complete (run : Returned code ws pc log) : ∃ path, follows code ws path pc log := by
  induction run with
  | done typed => exact ⟨[],by simp [follows,typed]⟩
  | @next pc choices next rest typed mem _ ih =>
    obtain ⟨tail,rest⟩ := ih
    exact ⟨next :: tail,by simpa [follows,edge,typed,mem] using rest⟩
  | @shutdown pc live closed b more typed _ ih =>
    obtain ⟨tail,rest⟩ := ih
    exact ⟨(if b then closed else live) :: tail,by simpa [follows,edge,typed] using rest⟩
  | @mask pc busy empty more i typed _ ih =>
    obtain ⟨tail,rest⟩ := ih
    exact ⟨(if (ws.get i).value = 0 then empty else busy) :: tail,by simpa [follows,edge,typed] using rest⟩

def Contract := ∀ (n : Nat) (code : Fin n → Op n) (ws : List Write) (pc : Fin n)
    (log : List (Guard ws)), Returned code ws pc log ↔ ∃ path, follows code ws path pc log = true

theorem idlePath : Contract := fun _ _ _ _ _ => ⟨follows_complete,fun ⟨_,h⟩ => follows_correct h⟩

def poolOpen (log : List (Guard ws)) :=
  log.all fun | .shutdown closed => !closed | .mask _ => true

private theorem notClosed (ok : poolOpen log) : Guard.shutdown true ∉ log := by
  intro closed
  exact Bool.noConfusion (List.all_eq_true.mp ok _ closed)

/-- Construct all generation bounds from the owner mark, the owner's next clear
    (if present), and the coherent raw loads used by the decoded idle-loop path. -/
def ownerExit (hist : History) (g : RawMask ws) (owner : Clock) (rows : List Operand)
    (decoded : guards ws owner 0 rows = some log)
    (path : List (Fin 80)) (idle : follows SplitSchedulerMir.idle ws path 0 log) (running : poolOpen log)
    (mark : (ws.get g.mark).event = owner.atStep 1)
    (next : afterReturn owner log.length ws[g.mark.val+g.actions.length+1]?)
    (physical : Coherent hist ws rows) : OwnerExit hist g where
  timeline := owner
  log := log
  returned := follows_correct idle
  poolOpen := notClosed running
  sampled pre i post part := by
    obtain ⟨row, mem, point, op⟩ := located decoded part
    simp only [Nat.zero_add] at point
    have coherent := physical row mem i op
    rw [point] at coherent
    refine ⟨.sequenced ?_, fun k pos => .sequenced (beforeNext g owner log.length pre.length
      (by simp [part]) next k pos), coherent.writeRead, coherent.readWrite⟩
    rw [mark]
    refine ⟨rfl, owner.advances _ _ ?_⟩
    change 1 < if 4+pre.length<3 then 4+pre.length else 4+pre.length+2
    split <;> omega

end NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards
