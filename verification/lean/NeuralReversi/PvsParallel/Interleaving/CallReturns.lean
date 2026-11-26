import NeuralReversi.PvsParallel.Interleaving.RootCausal
import NeuralReversi.PvsParallel.NativeFrames.EntryStream
import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Program

namespace NeuralReversi.PvsParallel.Interleaving.CallReturns

open PvsBoardTree (Position)
open NativeFrames (Args Event Record)
open NativeFrames.EntryStream (Sample)
open EndgameSearch.CacheOwner (Node)
open SerialCalls (Request Program)
open ScoreRefresh (Score)

structure Entry where
  request : Request
  endgame : Bool
  level : Nat
  rootLine : Option Nat
  deriving DecidableEq

def Entry.read (arg : Args) : Option Entry :=
  if wf : arg.board.player &&& arg.board.opponent = 0 then
    some ⟨⟨⟨arg.board,wf⟩,arg.depth,arg.pv,arg.cut,arg.alpha.raw.toInt,arg.beta.raw.toInt⟩,
      arg.endgame,arg.selectivity.val,arg.rootLine⟩
  else none

structure Activation where
  worker : Nat
  first : Nat
  site : Nat
  callee : Node
  parent : Option Nat
  input : Option Entry
  chosen : Option (Nat × Score) := none
  returned : Option Nat := none
  entered : Nat := 0

structure Thread where
  stack : List Nat := []
  nextStep : Nat := 0

structure State where
  workers : Nat → Thread := fun _ => {}
  calls : Array Activation := #[]
  tick : Nat := 0

private def State.setWorker (s : State) (w : Nat) (u : Thread) := {s with workers := fun k => if k = w then u else s.workers k}

private def enter (s : State) (worker opening site : Nat) (callee : Node) (args : Option Args) : Option State := do
  let thread := s.workers worker
  let input ← if callee = .pvs then args.bind Entry.read |>.map some else some none
  let a : Activation := ⟨worker,opening,site,callee,thread.stack.head?,input,none,none,s.tick⟩
  return {(s.setWorker worker ⟨s.calls.size :: thread.stack,opening+1⟩) with calls := s.calls.push a}

private def changeTop (s : State) (worker stamp : Nat) (pop : Bool)
    (update : Activation → Option Activation) : Option State := do
  let thread := s.workers worker
  let i :: rest := thread.stack | none
  let a ← s.calls[i]?
  let next ← update a
  return {(s.setWorker worker ⟨if pop then rest else thread.stack,stamp+1⟩) with calls := s.calls.setIfInBounds i next}

/-- Score selection, TT writing and delivery to the caller may happen at separate
    events. A discard or unfinished activation has no normal return. -/
def step (s : State) (sample : Sample) : Option State := do
  let next ← match sample.record with
    | .table _ => some s
    | .localCall worker ev => do
      guard ((s.workers worker).nextStep ≤ ev.step)
      match ev with
      | .entry t site callee .. => enter s worker t site callee sample.args
      | .scored t score => changeTop s worker t false fun a => do
        guard (a.callee = .pvs ∧ a.chosen.isNone)
        return {a with chosen := some (s.tick,score)}
      | .leave t first node _ word => changeTop s worker t true fun a => do
        guard (a.first = first ∧ a.callee = node)
        match word with
        | none => some a
        | some raw => do
          let (_,chosen) ← a.chosen
          guard (node = .pvs ∧ chosen.raw = raw.raw)
          return {a with returned := some s.tick}
      | .discarded t first _ => changeTop s worker t true fun a => do
        guard (a.first = first ∧ a.callee = .pvs ∧ a.chosen.isNone)
        return a
  return {next with tick := s.tick+1}

def capture (samples : List Sample) : Option State := samples.foldlM step {}

/-- Non-frame events occupy their real location in the common trace clock. -/
def timedStep (s : State) : Option Sample → Option State
  | some sample => step s sample
  | none => some {s with tick := s.tick+1}

def captureTimed (samples : List (Option Sample)) : Option State := samples.foldlM timedStep {}

theorem captureTimed_all (samples : List Sample) : captureTimed (samples.map some) = capture samples := by
  unfold captureTimed capture
  generalize ({} : State) = state
  induction samples generalizing state with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons,List.foldlM_cons]
    cases h : step state x
    · simp [timedStep,h]
    · simpa [timedStep,h] using ih _

/-- Invocation identity is the index allocated at entry. Each recorded frame
    gets its depth, cut flag and window from that same entry. -/
structure Row where
  id : Nat
  parent : Option Nat
  worker : Nat
  site : Nat
  entry : Entry
  chosen : Nat
  returned : Option Nat
  score : Int
  deriving DecidableEq

def Row.frame (r : Row) : CausalTable.Frame :=
  ⟨r.chosen,r.returned,r.entry.request.board,r.entry.request.pv,r.entry.request.alpha,r.entry.request.beta,r.score⟩

def Activation.row (a : Activation) (id : Nat) : Option Row := do
  let e ← a.input
  let (chosen,word) ← a.chosen
  return ⟨id,a.parent,a.worker,a.site,e,chosen,a.returned,word.raw.toInt⟩

def State.allRows (s : State) : List Row :=
  s.calls.toList.zipIdx.filterMap fun (a,i) => a.row i

def State.rows (s : State) (full : Bool) : List Row :=
  s.allRows.filter
    fun r => r.entry.endgame && (decide (r.entry.level = 5) == full) && (r.entry.rootLine.getD 0 == 0)

def State.frames (s : State) (full : Bool) := (s.rows full).map Row.frame

private theorem Activation.row_id (a : Activation) (found : a.row i = some r) : r.id = i := by
  cases h : a.input <;> cases k : a.chosen <;> simp [row,h,k] at found
  cases found
  rfl

/-- Two frames at different array positions cannot acquire the same invocation
    ID, even when their board, window, thread and returned word coincide. -/
theorem State.rows_ordered (s : State) (full : Bool) :
    (s.rows full).Pairwise (fun a b => a.id < b.id) := by
  have increasing (xs : List Activation) (k : Nat) : (xs.zipIdx k).Pairwise (fun a b => a.2 < b.2) := by
    induction xs generalizing k with
    | nil => exact .nil
    | cons a xs ih =>
      rw [List.zipIdx_cons,List.pairwise_cons]
      exact ⟨fun _ h => Nat.lt_of_succ_le (List.le_snd_of_mem_zipIdx h),ih _⟩
  apply List.Pairwise.filter
  apply List.Pairwise.filterMap _ _ (increasing _ 0)
  intro a b less r ra t tb
  rw [Activation.row_id _ ra,Activation.row_id _ tb]
  exact less

/-- The request labels and word can be recovered at the physical ID even if
    filtering removed many intervening helper or unfinished activations. -/
theorem State.source (s : State) (full : Bool) (present : r ∈ s.rows full) :
    ∃ a, s.calls[r.id]? = some a ∧ a.row r.id = some r := by
  obtain ⟨inside,_⟩ := List.mem_filter.mp present
  obtain ⟨⟨a,i⟩,index,found⟩ := List.mem_filterMap.mp inside
  have id := Activation.row_id a found
  exact ⟨a,by simpa [id] using List.mk_mem_zipIdx_iff_getElem?.mp index,by simpa [id] using found⟩

def Row.direct (r : Row) (parent consumer : Nat) :=
  decide (r.parent = some parent) && r.frame.returned.any (fun t => decide (t < consumer))

def direct (rows : List Row) (parent consumer : Nat) : List Row :=
  rows.filter (fun r => r.direct parent consumer)

structure Scope where
  rows : List Row
  parent : Nat
  consumer : Nat
  calls : Array Activation

structure Slot where
  id : Nat
  request : Option Request
  result : Option Row

/-- Every issued child occupies a slot, even if it was abandoned, is still
    running, or was selected but never returned normally. -/
def Scope.slots (s : Scope) : List Slot :=
  let calls := s.calls.toList.zipIdx.filter fun (a,_) =>
    a.callee == .pvs && a.parent == some s.parent && decide (a.entered < s.consumer)
  calls.map fun (a,id) =>
    ⟨id,a.input.map Entry.request,(a.row id).filter (fun r => r.direct s.parent s.consumer)⟩

def Scope.atOrdinal (s : Scope) (n : Nat) := (s.slots[n]?).bind Slot.result

private def normalPrefix : List Slot → List Row
  | [] => []
  | slot :: rest => match slot.result with
    | some r => r :: normalPrefix rest
    | none => []

/-- A sequential consumer can run up to the first child that never delivered a word. -/
def Scope.physical (s : Scope) := normalPrefix s.slots

def Scope.tape (s : Scope) := direct s.rows s.parent s.consumer

def scope (state : State) (full : Bool) (parent : Nat) : Option Scope := do
  let a ← state.calls[parent]?
  let (time,_) ← a.chosen
  return ⟨state.rows full,parent,time,state.calls⟩

/-- A worker inside a split helper can already consume a completed recursive call
    at the current prefix even if the activation at the top never returns. -/
def active (state : State) (full : Bool) (worker : Nat) : Option Scope := do
  let parent ← (state.workers worker).stack.head?
  return ⟨state.rows full,parent,state.tick,state.calls⟩

def earlier (i j : Nat) := decide (i < j)

theorem Scope.direct_return (s : Scope) {r : Row} (used : r ∈ s.tape) :
    r.parent = some s.parent ∧ CausalTable.returned (fun i j => earlier i j = true) (s.rows.map Row.frame)
      s.consumer r.entry.request.board r.entry.request.pv r.entry.request.alpha r.entry.request.beta r.score := by
  obtain ⟨mem,matched⟩ := List.mem_filter.mp used
  obtain ⟨p,t,exit,lt⟩ := (show r.parent = some s.parent ∧ ∃ t, r.returned = some t ∧ t < s.consumer by
    simpa [Row.direct,Row.frame,Option.any_eq_true] using matched)
  exact ⟨p,r.frame,List.mem_map_of_mem mem,t,exit,decide_eq_true lt,rfl,rfl,rfl,rfl,rfl⟩

/-- The ordinal interpreter consumes one return per actual request and checks all
    operands. Repeated equal requests can therefore use different fail-soft words. -/
def execute : Program Request A → List Row → Option (A × List Row)
  | .pure a, tape => some (a,tape)
  | .fail, _ => none
  | .call q k, r::rest => if r.entry.request = q then execute (k r.score) rest else none
  | .call .., [] => none

theorem executed (code : Program Request A) (ran : execute code tape = some (out,tail)) :
    ∃ used, tape = used ++ tail ∧ Program.Runs code (used.map fun r => (r.entry.request,r.score)) out := by
  induction code generalizing tape with
  | pure a =>
    cases ran
    exact ⟨[],rfl,.pure⟩
  | fail => contradiction
  | call q k ih =>
    cases tape with
    | nil => contradiction
    | cons r rest =>
      simp only [execute] at ran
      split at ran <;> try contradiction
      rename_i same
      obtain ⟨used,split,steps⟩ := ih _ ran
      refine ⟨r :: used,by simp [split],?_⟩
      simp only [List.map_cons,same]
      exact .call steps

theorem replay (code : Program Request A) (actual : Program.Runs code (used.map fun r : Row => (r.entry.request,r.score)) out) :
    execute code (used ++ more) = some (out,more) := by
  induction code generalizing used with
  | pure =>
    cases used <;> cases actual
    rfl
  | fail => cases actual
  | call q k ih =>
    cases used with
    | nil => cases actual
    | cons r rest =>
      cases actual with
      | call next => simpa [execute] using ih _ next

def Scope.ordinal (s : Scope) (n : Nat) (q : Request) : Option Row := do
  let r ← s.atOrdinal n
  if r.entry.request = q ∧ r ∈ s.tape then some r else none

theorem Scope.ordinal_hit (s : Scope) (received : s.ordinal n q = some r) :
    s.atOrdinal n = some r ∧ r.entry.request = q ∧ r ∈ s.tape ∧ r.parent = some s.parent ∧
      CausalTable.returned (fun i j => earlier i j = true) (s.rows.map Row.frame) s.consumer q.board q.pv q.alpha q.beta r.score := by
  obtain ⟨row,index,h⟩ := Option.bind_eq_some_iff.mp received
  split at h <;> try contradiction
  cases h
  rename_i same
  exact ⟨index,same.1,same.2,(s.direct_return same.2).1,same.1 ▸ (s.direct_return same.2).2⟩

def Scope.atSite (s : Scope) (n : Nat) (caller : Node) (site : Nat) (q : Request) : Option Row := do
  let r ← s.ordinal n q
  if r.site = site ∧ (s.calls[s.parent]?).map Activation.callee = some caller then some r else none

theorem Scope.atSite_hit (s : Scope) (found : s.atSite n caller site q = some r) :
    s.ordinal n q = some r ∧ r.site = site ∧ (s.calls[s.parent]?).map Activation.callee = some caller := by
  obtain ⟨row,received,label⟩ := Option.bind_eq_some_iff.mp found
  split at label <;> try contradiction
  cases label
  exact ⟨received,by assumption⟩

def moveSite (pv : Bool) := if pv then NativeFrames.OwnerSites.Role.wide.site else NativeFrames.OwnerSites.Role.scout.site

def Scope.matching (s : Scope) (q : Request) := (s.slots.filter (fun slot => slot.request == some q)).map Slot.result

inductive Access where
  | unique
  | occurrence (ordinal : Nat)
  | callSite (ordinal : Nat) (parent : Node) (site : Nat)

def Scope.selected (s : Scope) (access : Access) (q : Request) : Option Row :=
  match access with
  | .unique => match s.matching q with
    | [some r] => if r ∈ s.tape ∧ r.entry.request = q then some r else none
    | _ => none
  | .occurrence n => s.ordinal n q
  | .callSite n parent site => s.atSite n parent site q

private theorem Scope.selected_mem (s : Scope) (hit : s.selected access q = some r) :
    r ∈ s.tape ∧ r.entry.request = q := by
  cases access with
  | occurrence n =>
    obtain ⟨_,same,present,_,_⟩ := s.ordinal_hit hit
    exact ⟨present,same⟩
  | callSite n parent site =>
    obtain ⟨_,same,present,_,_⟩ := s.ordinal_hit (s.atSite_hit hit).1
    exact ⟨present,same⟩
  | unique =>
    change (match s.matching q with
      | [some r] => if r ∈ s.tape ∧ r.entry.request = q then some r else none | _ => none) = some r at hit
    split at hit <;> try contradiction
    split at hit <;> try contradiction
    rename_i eligible
    cases hit
    exact eligible

def Scope.sites (s : Scope) (access : Access) : RootCausal.Sites where
  depth i := ((s.rows[i]?).map fun r => r.entry.request.depth).getD 0
  cut i := ((s.rows[i]?).map fun r => r.entry.request.cut).getD false
  event _ _ _ _ := s.consumer
  select b d pv cut a z := (s.selected access ⟨b,d,pv,cut,a,z⟩).map s.rows.idxOf

def Scope.native (s : Scope) (access : Access) : NativeNodes.Returns :=
  ⟨s.rows.map Row.frame,(s.sites access).depth,(s.sites access).cut,earlier,s.consumer,(s.sites access).select⟩

theorem Scope.reply (s : Scope) (access : Access) (q : Request) :
    (s.native access).reply q.board q.depth q.pv q.cut q.alpha q.beta = (s.selected access q).map Row.score := by
  cases q with
  | mk b d pv cut a z =>
    cases h : s.selected access ⟨b,d,pv,cut,a,z⟩ with
    | none => simp [native,sites,NativeNodes.Returns.reply,h]
    | some r =>
      obtain ⟨used,req⟩ := s.selected_mem h
      obtain ⟨listed,directly⟩ := List.mem_filter.mp used
      have i := List.idxOf_lt_length_of_mem listed
      have idx : s.rows[s.rows.idxOf r]? = some r := by rw [List.getElem?_eq_getElem i,List.getElem_idxOf i]
      obtain ⟨_,t,returned,earlier'⟩ := (show r.parent = some s.parent ∧ ∃ t, r.returned = some t ∧ t < s.consumer by
        simpa [Row.direct,Row.frame,Option.any_eq_true] using directly)
      simp [native,sites,NativeNodes.Returns.reply,Row.frame,h,idx,req,returned,earlier,earlier',guard]

theorem Scope.root (s : Scope) (access : Access) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig)
    (b : Position) (m : PvsBoardTree.GeneratedSquare mc b.val)
    (frames : RootCausal.frames weak exact lvl = s.rows.map Row.frame) (p : Bool) (a z : Int) :
    (s.sites access).reply mc fc b earlier weak exact lvl m p a z =
      (s.selected access (SerialRootCalls.request mc fc b (Board.emptyCount b.val) ⟨m,p,a,z⟩)).map Row.score := by
  unfold RootCausal.Sites.reply
  rw [frames]
  exact s.reply access _


open RootOwnerMir (Replies Result)
open PvsPostpone (Queue)
open RootReplay (Snapshot)

abbrev Plan (Move : Type) := List (Move × (Nat × Bool))

structure Owner (Move : Type) where
  result : Result Move
  next : Nat
  plan : Plan Move

/-- Advance only for calls actually issued by the compiled header. In particular,
    postponing a move does not read a word and the first move skips the scout. -/
def ownerRun (square : Move → Bitboard.Square) (replyAt : Nat → Bool → Replies Move)
    (schedule : Nat → Bool × Bool) (depth minSplit : Nat) (beta : Int) :
    Nat → Nat → Nat → Nat → Queue Move → Snapshot → Option (Owner Move)
  | 0, _, _, _, _, _ => none
  | fuel+1, i, n, raw, q, s =>
    let free := (schedule i).1
    let busy := (schedule i).2
    let alpha := s.state.alpha
    let replies := replyAt n (decide (q.idx = 0))
    match RootOwnerMir.issued replies q raw depth minSplit free busy alpha beta with
    | .finish => some ⟨⟨[],q,s,false⟩,n,[]⟩
    | .split => some ⟨⟨[],q,s,true⟩,n,[]⟩
    | .postpone _ fresh =>
      match PvsPostpone.step busy q with
      | some (none,next) => ownerRun square replyAt schedule depth minSplit beta fuel (i+1) n fresh next s
      | _ => none
    | .child nextIdx scout wide score =>
      match PvsPostpone.step busy q with
      | some (some m,next) =>
        if (!scout || (replies m false (-(alpha+1)) (-alpha)).isSome) &&
            (!wide || (replies m true (-beta) (-alpha)).isSome) then
          let nextOrdinal := n + scout.toNat + wide.toNat
          let d := RootUpdateMir.decision (decide (nextIdx = 1)) alpha beta s.state.best score
          let updated := RootUpdateMir.after s (square m) score d
          let call := (m,n,decide (q.idx = 0))
          if d.cuts then some ⟨⟨[m],next,updated,false⟩,nextOrdinal,[call]⟩ else
            (ownerRun square replyAt schedule depth minSplit beta fuel (i+1) nextOrdinal raw next updated).map
              (fun r => {r with result := r.result.visit m, plan := call :: r.plan})
        else none
      | _ => none

/-- Forgetting the cursor recovers the existing native owner for any ordinary
    pointwise reply function, including failure on a missing return. -/
theorem ownerRun_agrees (square : Move → Bitboard.Square) (replies : Replies Move)
    (schedule : Nat → Bool × Bool) (depth minSplit : Nat) (beta : Int)
    (fuel i n raw : Nat) (q : Queue Move) (s : Snapshot) :
    (ownerRun square (fun _ _ => replies) schedule depth minSplit beta fuel i n raw q s).map Owner.result =
      RootOwnerMir.run square replies schedule depth minSplit beta fuel i raw q s := by
  induction fuel generalizing i n raw q s with
  | zero => rfl
  | succ fuel ih =>
    simp only [ownerRun,RootOwnerMir.run]
    cases action : RootOwnerMir.issued replies q raw depth minSplit (schedule i).1 (schedule i).2 s.state.alpha beta with
    | finish | split => rfl
    | postpone _ fresh =>
      cases result : PvsPostpone.step (schedule i).2 q with
      | none => rfl
      | some pair =>
        obtain ⟨m,next⟩ := pair
        cases m with
        | none => exact ih ..
        | some => rfl
    | child nextIdx scout wide score =>
      cases result : PvsPostpone.step (schedule i).2 q with
      | none => rfl
      | some pair =>
        obtain ⟨m,next⟩ := pair
        cases m with
        | none => rfl
        | some m =>
          dsimp only
          split <;> rename_i branch <;> try rw [if_pos branch] <;> try rw [if_neg branch]
          · split <;> rename_i cutoff <;> try rw [if_pos cutoff] <;> try rw [if_neg cutoff]
            · rfl
            · rw [← ih (i+1) (n+scout.toNat+wide.toNat)]
              simp only [Option.map_map]
              rfl
          · rfl

/-- For a later move a wide retry follows its own scout. The first move goes
    directly to its wide child. The request constructor consumes the actual
    move, mode, and signed window generated by the header. -/
def Scope.ownerReplies (s : Scope) (request : Move → Bool → Int → Int → Request)
    (n : Nat) (first : Bool) : Replies Move := fun m pv a z =>
  (s.atSite (if pv && !first then n+1 else n) .pvs (moveSite pv) (request m pv a z)).map Row.score

def Scope.rootRun (s : Scope) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig)
    (b : Position) (schedule : Nat → Bool × Bool) (minSplit : Nat) (alpha beta : Int)
    (moves : List (PvsBoardTree.GeneratedSquare mc b.val)) :
    Option (Owner (PvsBoardTree.GeneratedSquare mc b.val)) :=
  let depth := Board.emptyCount b.val
  let q : Queue _ := ⟨0,moves,[]⟩
  let request := fun m p a z => SerialRootCalls.request mc fc b depth ⟨m,p,a,z⟩
  ownerRun (fun m => m.val) (s.ownerReplies request) schedule depth.toNat minSplit beta
    (q.budget+1) 0 0 moves.length q (RootReplay.start alpha)

private def administrative : HeaderAction → Bool
  | .child .. => false
  | _ => true

private theorem control (r u : Replies M) (q : Queue M) (raw depth limit : Nat) (free busy : Bool) (a z : Int)
    (ordinary : administrative (RootOwnerMir.issued r q raw depth limit free busy a z)) :
    RootOwnerMir.issued u q raw depth limit free busy a z = RootOwnerMir.issued r q raw depth limit free busy a z := by
  unfold RootOwnerMir.issued at *
  simp only [headerAction_eq] at *
  by_cases run : q.idx < q.idx + q.pending.length <;> simp only [run,ite_true,ite_false] at * <;> try rfl
  by_cases handoff : canHandoff q.idx (q.idx + q.pending.length) depth limit free <;>
    simp only [handoff,ite_true,ite_false] at * <;> try rfl
  by_cases defer : PvsPostpone.shouldPostpone busy q.idx raw = true <;>
    simp [defer] at ordinary ⊢ <;> try rfl
  by_cases first : q.idx = 0 <;> simp only [first,ite_true,ite_false] at ordinary
  · contradiction
  · split at ordinary <;> contradiction

private theorem head_taken {q : Queue M} (chosen : PvsPostpone.step busy q = some (some m,next)) :
    q.pending.head? = some m := by
  obtain ⟨idx,fresh,rest⟩ := q
  cases fresh with
  | nil => cases rest <;> simp [PvsPostpone.step,Queue.pending] at chosen ⊢; exact chosen.1
  | cons first more =>
    simp only [PvsPostpone.step] at chosen
    split at chosen
    · cases chosen
    · exact congrArg Prod.fst (Option.some.inj chosen)

private theorem issued_at_head (r u : Replies M) (q : Queue M) (head : q.pending.head? = some m)
    (same : u m = r m) : RootOwnerMir.issued u q raw depth limit free busy a z =
      RootOwnerMir.issued r q raw depth limit free busy a z := by simp [RootOwnerMir.issued,head,same]

/-- All callbacks recorded for a visited move use its own starting ordinal.
    Pending moves can be postponed; their order is tracked as a permutation. -/
theorem ownerRun_recorded (square : M → Bitboard.Square) (atCall : Nat → Bool → Replies M)
    (schedule : Nat → Bool × Bool) (depth limit : Nat) (beta : Int)
    (fuel i n raw : Nat) (q : Queue M) (s : Snapshot)
    (recorded : ownerRun square atCall schedule depth limit beta fuel i n raw q s = some out) :
    ((out.plan.map Prod.fst) ++ out.result.remaining.pending).Perm q.pending ∧
    ∀ (replies : Replies M), (∀ item ∈ out.plan, replies item.1 = atCall item.2.1 item.2.2 item.1) →
      RootOwnerMir.run square replies schedule depth limit beta fuel i raw q s = some out.result := by
  induction fuel generalizing i n raw q s out with
  | zero => contradiction
  | succ fuel ih =>
    let r := atCall n (decide (q.idx = 0))
    let free := (schedule i).1
    let busy := (schedule i).2
    let alpha := s.state.alpha
    cases action : RootOwnerMir.issued r q raw depth limit free busy alpha beta with
    | finish =>
      rw [ownerRun,action] at recorded
      dsimp only at recorded
      cases recorded
      refine ⟨List.Perm.refl _,fun u _ => ?_⟩
      rw [RootOwnerMir.run,control r u q raw depth limit free busy alpha beta (by rw [action]; rfl),action]
    | split =>
      rw [ownerRun,action] at recorded
      dsimp only at recorded
      cases recorded
      refine ⟨List.Perm.refl _,fun u _ => ?_⟩
      rw [RootOwnerMir.run,control r u q raw depth limit free busy alpha beta (by rw [action]; rfl),action]
    | postpone j fresh =>
      rw [ownerRun,action] at recorded
      dsimp only at recorded
      split at recorded <;> try contradiction
      rename_i next moved
      have perm := ((PvsPostpone.stepContract_proven M).2.2 q busy none next moved).1
      obtain ⟨layout,go⟩ := ih _ _ _ _ _ recorded
      refine ⟨layout.trans perm,fun u available => ?_⟩
      rw [RootOwnerMir.run,control r u q raw depth limit free busy alpha beta (by rw [action]; rfl),action,moved]
      exact go u available
    | child nextIdx scout wide score =>
      rw [ownerRun,action] at recorded
      dsimp only at recorded
      split at recorded <;> try contradiction
      rename_i m next moved
      have perm := ((PvsPostpone.stepContract_proven M).2.2 q busy (some m) next moved).1
      split at recorded <;> try contradiction
      rename_i ready
      let call := (m,n,decide (q.idx = 0))
      have resume (u : Replies M) (same : u m = r m) :
          RootOwnerMir.issued u q raw depth limit free busy alpha beta = .child nextIdx scout wide score :=
        (issued_at_head r u q (head_taken moved) same).trans action
      split at recorded
      · rename_i cut
        cases recorded
        refine ⟨perm,fun u available => ?_⟩
        have same : u m = r m := available call List.mem_cons_self
        rw [RootOwnerMir.run,resume u same,moved]
        dsimp only
        rw [same,ite_eq_left ready,ite_eq_left cut]
      · rename_i more
        obtain ⟨tail,continued,eq⟩ := Option.map_eq_some_iff.mp recorded
        cases eq
        obtain ⟨layout,go⟩ := ih _ _ _ _ _ continued
        refine ⟨(layout.cons m).trans perm,fun u available => ?_⟩
        have same : u m = r m := available call List.mem_cons_self
        rw [RootOwnerMir.run,resume u same,moved]
        dsimp only
        rw [same,ite_eq_left ready,ite_eq_right more,
          go u (fun item mem => available item (List.mem_cons_of_mem _ mem))]
        rfl

def boardPlan (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (plan : Plan (PvsBoardTree.GeneratedSquare mc b.val)) : Plan Position :=
  plan.map fun item => ((PvsBoardTree.play mc fc b item.1).val,item.2)

private def planLookup [DecidableEq M] (plan : Plan M) (m : M) := plan.lookup m

private theorem mappedPlan_nodup (f : X → Y) (inj : ∀ {x z}, f x = f z → x = z) (xs : Plan X)
    (unique : (xs.map Prod.fst).Nodup) : ((xs.map (fun e => (f e.1,e.2))).map Prod.fst).Nodup := by
  rw [List.map_map]
  apply List.pairwise_map.mpr
  exact (List.pairwise_map.mp unique).imp fun diff equal => diff (inj equal)

private theorem boardPlan_at (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (plan : Plan (PvsBoardTree.GeneratedSquare mc b.val)) (unique : (plan.map Prod.fst).Nodup)
    {item : PvsBoardTree.GeneratedSquare mc b.val × (Nat × Bool)} (present : item ∈ plan) :
    planLookup (boardPlan mc fc b plan) (PvsBoardTree.play mc fc b item.1).val = some item.2 := by
  unfold planLookup
  apply Program.lookup_mem
  · exact mappedPlan_nodup (fun m => (PvsBoardTree.play mc fc b m).val)
      (fun h => PvsBoardTree.play_injective mc fc b (congrArg Subtype.val h)) plan unique
  · exact List.mem_map_of_mem present

/-- The board identifies a unique visited move within one root attempt. The
    matching plan entry determines both actual physical ordinals for this move. -/
def Scope.planSites (s : Scope) (plan : Plan Position) : RootCausal.Sites :=
  {s.sites .unique with select := fun b d pv cut a z => do
    let (n,first) ← planLookup plan b
    (s.sites (.callSite (if pv && !first then n+1 else n) .pvs (moveSite pv))).select b d pv cut a z}

private theorem Scope.planSites_at (s : Scope) (plan : Plan Position)
    (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (m : PvsBoardTree.GeneratedSquare mc b.val) (n : Nat) (first : Bool)
    (located : planLookup plan (PvsBoardTree.play mc fc b m).val = some (n,first))
    (frames : RootCausal.frames weak exact lvl = s.rows.map Row.frame) :
    (s.planSites plan).reply mc fc b earlier weak exact lvl m =
      s.ownerReplies (fun m p a z => SerialRootCalls.request mc fc b (Board.emptyCount b.val) ⟨m,p,a,z⟩) n first m := by
  funext p a z
  refine Eq.trans (b := (s.sites (.callSite (if p && !first then n+1 else n) .pvs (moveSite p))).reply mc fc b earlier weak exact lvl m p a z) ?_ ?_
  · unfold RootCausal.Sites.reply RootCausal.Sites.atCall NativeNodes.Returns.reply
    simp only [planSites,SerialRootCalls.request,located]
    rfl
  · exact s.root _ mc fc b m frames p a z

/-- A successful physical run supplies the exact callback consumed by the
    existing root publication proof. The queue's distinct legal moves suffice
    even if other attempts at identical operands returned different words. -/
theorem Scope.rootRun_fromMoves (s : Scope) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig)
    (b : Position) (schedule : Nat → Bool × Bool) (minSplit : Nat) (alpha beta : Int)
    (moves : List (PvsBoardTree.GeneratedSquare mc b.val)) (distinct : moves.Nodup)
    (frames : RootCausal.frames weak exact lvl = s.rows.map Row.frame)
    {out : Owner (PvsBoardTree.GeneratedSquare mc b.val)} (normal : s.rootRun mc fc b schedule minSplit alpha beta moves = some out) :
    RootOwnerMir.fromMoves Subtype.val
      ((s.planSites (boardPlan mc fc b out.plan)).reply mc fc b earlier weak exact lvl)
      schedule (Board.emptyCount b.val).toNat minSplit alpha beta moves = some out.result := by
  unfold rootRun at normal
  obtain ⟨queue,transport⟩ := ownerRun_recorded _ _ _ _ _ _ _ _ _ _ _ _ normal
  have unique : (out.plan.map Prod.fst).Nodup := (List.nodup_append.mp (queue.nodup_iff.mpr (by simpa [Queue.pending] using distinct))).1
  apply transport
  intro item present
  exact s.planSites_at _ mc fc b item.1 item.2.1 item.2.2
    (boardPlan_at mc fc b out.plan unique present) frames

def Scope.nativePlan (s : Scope) (plan : Plan Position) : NativeNodes.Returns :=
  ⟨s.rows.map Row.frame,(s.planSites plan).depth,(s.planSites plan).cut,earlier,s.consumer,(s.planSites plan).select⟩

def pvRequest (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (depth : ExactDepth.Depth) (cut : Bool) (m : PvsBoardTree.GeneratedSquare mc b.val)
    (p : Bool) (a z : Int) : Request :=
  ⟨(PvsBoardTree.play mc fc b m).val,depth-1,p,!p && !cut,a,z⟩

private theorem Scope.nativePlan_at (s : Scope) (plan : Plan Position) (b : Position) (depth : ExactDepth.Depth)
    (p cut : Bool) (a z : Int) (n : Nat) (first : Bool) (found : planLookup plan b = some (n,first)) :
    (s.nativePlan plan).reply b depth p cut a z =
      (s.atSite (if p && !first then n+1 else n) .pvs (moveSite p) ⟨b,depth,p,cut,a,z⟩).map Row.score := by
  refine Eq.trans (b := (s.native (.callSite (if p && !first then n+1 else n) .pvs (moveSite p))).reply b depth p cut a z) ?_ ?_
  · unfold nativePlan NativeNodes.Returns.reply native
    simp only [planSites,found]
    rfl
  · exact s.reply (.callSite (if p && !first then n+1 else n) .pvs (moveSite p)) ⟨b,depth,p,cut,a,z⟩

/-- Apply the same source proof to an ordinary recursive PV owner. Its scout
    inherits the negated parent cut flag and its wide child clears that flag. -/
theorem Scope.pvRun_fromMoves (s : Scope) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig)
    (b : Position) (depth : ExactDepth.Depth) (cut : Bool) (schedule : Nat → Bool × Bool)
    (minSplit : Nat) (alpha beta : Int) (moves : List (PvsBoardTree.GeneratedSquare mc b.val))
    (distinct : moves.Nodup) {out : Owner (PvsBoardTree.GeneratedSquare mc b.val)}
    (normal : ownerRun Subtype.val (s.ownerReplies (pvRequest mc fc b depth cut)) schedule depth.toNat minSplit beta
      ((⟨0,moves,[]⟩ : Queue _).budget+1) 0 0 moves.length ⟨0,moves,[]⟩ (RootReplay.start alpha) = some out) :
    RootOwnerMir.fromMoves Subtype.val
      ((s.nativePlan (boardPlan mc fc b out.plan)).pvReplies mc fc b depth cut)
      schedule depth.toNat minSplit alpha beta moves = some out.result := by
  obtain ⟨queue,transport⟩ := ownerRun_recorded _ _ _ _ _ _ _ _ _ _ _ _ normal
  have unique : (out.plan.map Prod.fst).Nodup :=
    (List.nodup_append.mp (queue.nodup_iff.mpr (by simpa [Queue.pending] using distinct))).1
  apply transport
  intro item present
  funext p a z
  exact s.nativePlan_at _ _ _ _ _ _ _ _ _ (boardPlan_at mc fc b out.plan unique present)

theorem scope_frames (state : State) (parent level : Nat)
    (opened : scope state (decide (level = 5)) parent = some s) :
    RootCausal.frames (state.frames false) (state.frames true) level = s.rows.map Row.frame := by
  obtain ⟨a,_,next⟩ := Option.bind_eq_some_iff.mp opened
  obtain ⟨⟨t,w⟩,_,eq⟩ := Option.bind_eq_some_iff.mp next
  cases eq
  by_cases full : level = 5 <;> simp [RootCausal.frames,State.frames,full]

/-- An aspiration retry is a PVS child at site zero of this solve's
    endgame-window activation. Allocate its ordinal at entry, so an incomplete
    or mismatched retry cannot shift subsequent roots. -/
def State.rootCallIds (s : State) (solve : Nat) : List Nat :=
  let candidates := s.calls.toList.zipIdx
  (candidates.filter fun (a,_) => (s.calls[solve]?).any (fun base => base.callee == .solve) &&
    a.callee == .pvs && a.site == 0 && (a.parent.bind fun id => s.calls[id]?).any
      (fun w => w.callee == .endWindow && w.site == 0 && w.parent == some solve)).map Prod.snd

structure Trial where
  id : Nat
  entered : Nat
  returned : Nat
  score : Score
  children : Scope

def rootEntry (board : Position) (row level : Nat) (alpha beta : Int) : Entry :=
  ⟨⟨board,Board.emptyCount board.val,true,false,alpha,beta⟩,true,level,some row⟩

def State.rootAt (s : State) (solve ordinal : Nat) (board : Position) (row level : Nat) (alpha beta : Int) :
    Option Trial := do
  let id ← (s.rootCallIds solve)[ordinal]?
  let a ← s.calls[id]?
  guard (a.input = some (rootEntry board row level alpha beta))
  let (chosen,score) ← a.chosen
  let returned ← a.returned
  return ⟨id,a.entered,returned,score,⟨s.rows (decide (level = 5)),id,chosen,s.calls⟩⟩

theorem State.rootAt_origin (s : State) (solve ordinal : Nat) (board : Position) (row level : Nat)
    (alpha beta : Int) (ok : s.rootAt solve ordinal board row level alpha beta = some trial) :
    (s.rootCallIds solve)[ordinal]? = some trial.id ∧
    ∃ a, s.calls[trial.id]? = some a ∧ a.input = some (rootEntry board row level alpha beta) ∧
      a.entered = trial.entered ∧ a.chosen = some (trial.children.consumer,trial.score) ∧
      a.returned = some trial.returned ∧
      scope s (decide (level = 5)) trial.id = some trial.children := by
  obtain ⟨id,number,next⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨a,found,cont⟩ := Option.bind_eq_some_iff.mp next
  simp only [guard] at cont
  split at cont <;> try contradiction
  rename_i label
  simp only [pure,Bind.bind,Option.bind] at cont
  obtain ⟨⟨chosen,score⟩,heard,next⟩ := Option.bind_eq_some_iff.mp cont
  obtain ⟨returned,back,done_⟩ := Option.bind_eq_some_iff.mp next
  cases done_
  exact ⟨number,a,found,label,rfl,heard,back,by simp [scope,found,heard]⟩

theorem State.rootCallIds_origin (s : State) {solve ordinal id : Nat} (call : (s.rootCallIds solve)[ordinal]? = some id) :
    ∃ (a base window : Activation) (parent : Nat), s.calls[id]? = some a ∧ s.calls[solve]? = some base ∧ base.callee = .solve ∧
      a.callee = .pvs ∧ a.site = 0 ∧ a.parent = some parent ∧ s.calls[parent]? = some window ∧
      window.callee = .endWindow ∧ window.site = 0 ∧ window.parent = some solve := by
  obtain ⟨⟨a,i⟩,inOrder,eq⟩ := List.mem_map.mp (List.mem_of_getElem? call)
  cases eq
  obtain ⟨source,valid⟩ := List.mem_filter.mp inOrder
  simp only [Bool.and_eq_true,Option.any_eq_true,beq_iff_eq] at valid
  obtain ⟨⟨⟨⟨base,top,kind⟩,node⟩,site⟩,window,win,⟨wk,ws⟩,parent⟩ := valid
  obtain ⟨pid,own,located⟩ := Option.bind_eq_some_iff.mp win
  exact ⟨a,base,window,pid,by simpa using List.mk_mem_zipIdx_iff_getElem?.mp source,top,kind,node,
    site,own,located,wk,ws,parent⟩

/-- A split worker's scout and wide call use the signed machine operands created
    by the native block, including the independent, later alpha load. -/
def queryRequest (next : Position) (q : SplitWorkerMir.Query) : Request :=
  ⟨next,BitVec.ofNat 32 q.depth,q.pv,q.cut,q.alpha.toInt,q.beta.toInt⟩

def Scope.word (s : Scope) (n : Nat) (request : Request) : Option SplitWorkerMir.Word := do
  let r ← s.atSite n .splitSearch (if request.pv then 1 else 0) request
  let w := BitVec.ofInt 32 r.score
  guard (w.toInt = r.score)
  return w

private theorem Scope.word_returned (s : Scope) (received : s.word n req = some w) :
    ∃ r, s.atOrdinal n = some r ∧ r.entry.request = req ∧ r.score = w.toInt ∧ r.parent = some s.parent ∧ r.site = (if req.pv then 1 else 0) ∧
      (s.calls[s.parent]?).map Activation.callee = some .splitSearch ∧
      CausalTable.returned (fun i j => earlier i j = true) (s.rows.map Row.frame)
        s.consumer req.board req.pv req.alpha req.beta w.toInt := by
  obtain ⟨r,hit,eq⟩ := Option.bind_eq_some_iff.mp received
  obtain ⟨ordinal,site,caller⟩ := s.atSite_hit hit
  obtain ⟨index,same,_,parent,returned⟩ := s.ordinal_hit ordinal
  simp only [guard] at eq
  split at eq <;> try contradiction
  rename_i fits
  cases eq
  exact ⟨r,index,same,fits.symm,parent,site,caller,fits.symm ▸ returned⟩

structure Dispatch where
  index : Nat
  depth : Nat
  pv : Bool
  root : Bool
  cut : Bool
  beta : SplitWorkerMir.Word
  alpha : SplitWorkerMir.Word
  fresh : SplitWorkerMir.Word

def Dispatch.input (d : Dispatch) (scout wide : SplitWorkerMir.Word) : SplitWorkerMir.Input :=
  ⟨d.index,d.depth,d.pv,d.root,d.cut,d.beta,d.alpha,d.fresh,scout,wide⟩

def Scope.dispatch (s : Scope) (next : Position) (d : Dispatch) (n : Nat) :
    Option (SplitWorkerMir.Input × Nat) := do
  guard (0 < d.depth ∧ (d.pv = true → 0 < d.index))
  let scout ← s.word n (queryRequest next (SplitWorkerMir.scoutQuery (d.input 0 0)))
  if SplitWorkerMir.retry (d.input scout 0) then
    let wide ← s.word (n+1) (queryRequest next (SplitWorkerMir.wideQuery (d.input scout 0)))
    return (d.input scout wide,n+2)
  else return (d.input scout 0,n+1)

theorem Scope.dispatch_returns (s : Scope) (next : Position) (d : Dispatch) (n : Nat)
    (received : s.dispatch next d n = some (input,after)) :
    SplitWorkerMir.search input = some (SplitWorkerMir.issued input) ∧
    after = n + (if SplitWorkerMir.retry input then 2 else 1) ∧
    ∀ q, .returned q ∈ (SplitWorkerMir.issued input).steps →
      let k := if q.pv then n+1 else n
      let req := queryRequest next q
      ∃ r, s.atOrdinal k = some r ∧ r.entry.request = req ∧ r.score = q.result.toInt ∧ r.parent = some s.parent ∧
        r.site = (if q.pv then 1 else 0) ∧ (s.calls[s.parent]?).map Activation.callee = some .splitSearch ∧
        CausalTable.returned (fun i j => earlier i j = true) (s.rows.map Row.frame)
          s.consumer req.board req.pv req.alpha req.beta q.result.toInt := by
  unfold dispatch at received
  simp only [guard] at received
  split at received <;> try contradiction
  rename_i valid
  simp only [pure,Bind.bind,Option.bind] at received
  obtain ⟨scout,scoutRead,cont⟩ := Option.bind_eq_some_iff.mp received
  split at cont
  · rename_i retried
    obtain ⟨wide,wideRead,eq⟩ := Option.bind_eq_some_iff.mp cont
    cases eq
    have retry : SplitWorkerMir.retry (d.input scout wide) = true := retried
    refine ⟨Worker.search_returns _ valid.1 valid.2,by simp [retry],?_⟩
    intro q used
    simp [SplitWorkerMir.issued,retry] at used
    rcases used with rfl | rfl
    · exact s.word_returned scoutRead
    · exact s.word_returned wideRead
  · rename_i no
    cases cont
    refine ⟨Worker.search_returns _ valid.1 valid.2,by rw [ite_eq_right no],?_⟩
    intro q used
    rw [SplitWorkerMir.issued,ite_eq_right no] at used
    simp at used
    cases used
    exact s.word_returned scoutRead

end NeuralReversi.PvsParallel.Interleaving.CallReturns
