module
public import NeuralReversi.PvsParallel.NativeFrameCache
public import NeuralReversi.PvsParallel.NativeTableWrites.Journal

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeTableWrites

open TranspositionTable
open SeqLock (Word)
open WriterMir (Request SourceStep publishRound saveRound storeRound)
open FrameTableReads (Address Table InvocationAt)
open FrameReturns (Frame Log WriteStep)
open PvsBoardTree (Position)
open ConcurrentReplay (Event)

abbrev Call := WriterTrace.Call Nat
abbrev Certified (log : Log mc fc) (endian : Bool) := {e : Event // WriteStep log endian e}
abbrev Prior (log : Log mc fc) (endian : Bool) (atEvent : Nat) := ConcurrentReplay.ReadAt log.before log.events atEvent endian

structure CallKey where
  expected : Word
  payload : SeqLock.Payload
  cas : Nat
  fence : Nat
  player : Nat
  opponent : Nat
  metadata : Nat
  finish : Nat
  deriving DecidableEq

def callKey (c : Call) : CallKey := ⟨c.expected,c.payload,c.cas,c.fence,c.store .player,c.store .opponent,c.store .metadata,c.finish⟩

private theorem call_eq {c d : Call} (same : callKey c = callKey d) : c = d := by
  cases c with | mk seq payload cas fence store finish =>
    cases d with | mk seq' payload' cas' fence' store' finish' =>
      obtain ⟨rfl,rfl,rfl,rfl,p,o,m,rfl⟩ := (by simpa [callKey] using same :
        seq = seq' ∧ payload = payload' ∧ cas = cas' ∧ fence = fence' ∧ store .player = store' .player ∧
        store .opponent = store' .opponent ∧ store .metadata = store' .metadata ∧ finish = finish')
      have eq : store = store' := by funext fld; cases fld <;> assumption
      rw [eq]

structure StoreKey where
  frame : Option Nat
  board : Board.Board
  entry : EntryData
  deriving DecidableEq

structure EventKey where
  id : Nat
  call : CallKey
  stored : Option StoreKey
  deriving DecidableEq

def signature (e : Event) : Option EventKey := match e.action with
  | .cleared c => some ⟨e.id,callKey c,none⟩
  | .stored frame c b entry => some ⟨e.id,callKey c,some ⟨frame,b.val,entry⟩⟩
  | _ => none

private theorem event_eq (e : Event) (valid : WriteStep log endian e) (other : Event)
    (same : signature e = signature other) : e = other := by
  cases e with | mk id act =>
    cases other with | mk id' act' =>
      cases act <;> cases act' <;> simp [signature,callKey] at same <;> (try cases valid)
      all_goals first
        | obtain ⟨rfl,eq⟩ := same; congr 2; apply call_eq; simpa only [callKey,CallKey.mk.injEq] using eq
        | obtain ⟨rfl,eq,rest⟩ := same; obtain ⟨rfl,b,rfl⟩ := rest; congr 2
          · apply call_eq; simpa only [callKey,CallKey.mk.injEq] using eq
          · exact Subtype.ext b

structure Stamp where
  address : Address
  worker : Nat
  claim : Nat
  fence : Nat
  player : Nat
  opponent : Nat
  metadata : Nat
  finish : Nat

def Stamp.step (w : Stamp) : Nat → Nat
  | 0 => w.claim | 1 => w.fence | 2 => w.player | 3 => w.opponent | 4 => w.metadata | _ => w.finish

def Stamp.ev (w : Stamp) (log : Log mc fc) (idx : Nat) := log.identify ⟨w.worker,w.step idx⟩

def Stamp.claimIndex {log : Log mc fc} (t : Table log) (w : Stamp) :=
  ((t.slot w.address).history.map WriterTrace.Action.event).idxOf (w.ev log 0)

def Stamp.current {log : Log mc fc} (t : Table log) (w : Stamp) := (t.memory w.address).writeValue .sequence (w.claimIndex t)

def actionMatches (claimed : Bool) (call : Call) : Option (WriterTrace.Action Nat) → Bool
  | some (.claim c) => claimed && decide (callKey c = callKey call)
  | some (.publish c) => !claimed && decide (callKey c = callKey call)
  | none => false

private theorem action_found {claimed call action} (ok : actionMatches claimed call action = true) :
    action = some (if claimed then .claim call else .publish call) := by
  cases action with
  | none => contradiction
  | some action =>
    cases action <;> cases claimed <;> simp [actionMatches] at ok ⊢
    all_goals exact call_eq ok

def Stamp.present {log : Log mc fc} (t : Table log) (w : Stamp) (call : Call) : Bool :=
  let atSlot := (t.slot w.address).history
  decide (w.claim < w.fence ∧ w.fence < w.player ∧ w.player < w.opponent ∧ w.opponent < w.metadata ∧ w.metadata < w.finish) &&
    actionMatches true call atSlot[w.claimIndex t]? && actionMatches false call atSlot[w.claimIndex t + 1]?

theorem Stamp.present_at {log : Log mc fc} {t : Table log} {w : Stamp} {call : Call} (ok : w.present t call = true) :
    (t.slot w.address).history[w.claimIndex t]? = some (.claim call) ∧
    (t.slot w.address).history[w.claimIndex t + 1]? = some (.publish call) := by
  obtain ⟨⟨_,claimed⟩,published⟩ := (by simpa only [Stamp.present,Bool.and_eq_true] using ok :
    (_ ∧ actionMatches true call _ = true) ∧ actionMatches false call _ = true)
  exact ⟨action_found claimed,action_found published⟩

theorem Stamp.present_of_release {log : Log mc fc} (t : Table log) (w : Stamp) (call : Call)
    (published : WriterTrace.Action.publish call ∈ (t.slot w.address).history)
    (cas : w.ev log 0 = call.cas)
    (time : w.claim < w.fence ∧ w.fence < w.player ∧ w.player < w.opponent ∧
      w.opponent < w.metadata ∧ w.metadata < w.finish) :
    w.present t call = true ∧ w.current t = call.expected := by
  have pair := (t.slot w.address).toSequence.publicationPair published
  simp only [Stamp.present,Stamp.current,Stamp.claimIndex,cas,pair.1,pair.2.1,pair.2.2,time,
    actionMatches,decide_true,Bool.and_self,Bool.not_false,true_and]

structure Load {log : Log mc fc} (t : Table log) (w : Stamp) (loc : SeqLock.Location) where
  id : Nat
  source : Nat
  loaded : (t.memory w.address).rf ((t.memory w.address).write loc source) id
  beforeClaim : (t.memory w.address).sb id (w.ev log 0)

def Load.word {log : Log mc fc} {t : Table log} {w : Stamp} (r : Load t w loc) := (t.memory w.address).readValue r.id

structure WriterRead {log : Log mc fc} (t : Table log) (w : Stamp) (addr : Address)
    (code : (Nat → Word) → List ReaderMir.AtomicAction) extends InvocationAt t addr (w.ev log 5) code where
  beforeClaim : (t.memory addr).sb (ev ((code ((t.memory addr).readValue ∘ ev)).length-1)) (w.ev log 0)

def WriterRead.last {log : Log mc fc} {t : Table log} {w : Stamp} {addr : Address}
    {code : (Nat → Word) → List ReaderMir.AtomicAction} (r : WriterRead t w addr code) :=
  r.ev ((code r.words).length-1)

inductive SaveRead {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) where
  | exact (read : Load t w .sequence)
  | snapshot (read : WriterRead t w w.address (FrameTableReads.snapCode endian))

def SaveRead.obs {log : Log mc fc} {t : Table log} {w : Stamp} : SaveRead t w endian → Save.Observation
  | .exact r => ⟨r.word,r.word,⟨0,0,0⟩,w.current t⟩
  | .snapshot r => FrameTableReads.saveObservation endian r.words (w.current t)

def SaveRead.matches {log : Log mc fc} {t : Table log} {w : Stamp} (d : EntryData) : SaveRead t w endian → Bool
  | .exact _ => decide (d.bound = Packing.boundByte .exact)
  | .snapshot _ => decide (d.bound ≠ Packing.boundByte .exact)

theorem SaveRead.checked {log : Log mc fc} {t : Table log} {w : Stamp} (r : SaveRead t w endian) (b : Position)
    (d : EntryData) (matched : r.matches d = true) :
    WriterMir.observations (Prior log endian (w.ev log 5)) endian r.obs
      (saveRound endian b.val.player b.val.opponent d r.obs) := by
  cases r with
  | exact load =>
    have bound : d.bound = Packing.boundByte .exact := of_decide_eq_true matched
    intro snapshot
    simp [saveRound,WriterMir.forward_exact bound,WriterMir.exactStep] at snapshot
    repeat' split at snapshot <;> simp_all
  | snapshot r =>
    intro _
    exact FrameTableReads.save_read t endian r.toInvocationAt _

def candidates (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :=
  (MoveList.Generation.records fc b.val (Board.getMoves mc b.val)).map Move.Move.sq

def validMove (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) : Option Bitboard.Square → Bool
  | none => true
  | some sq => decide (sq ∈ candidates mc fc b)

theorem generated_move (b : Position) (move : Option Bitboard.Square) (valid : validMove mc fc b move = true) :
    FrameReturns.MoveOrigin mc fc b source move := by
  cases move with
  | none => exact .absent
  | some sq =>
    obtain ⟨idx,selected⟩ := List.mem_iff_getElem?.mp (of_decide_eq_true valid : sq ∈ candidates mc fc b)
    exact .indexed _ _ idx sq (fun _ h => h) (List.Perm.refl _) selected

structure Catalog (p : EndgameSearch.CacheHistory.Pool) (log : Log mc fc) where
  out : NativeFrameCache.Checked p
  stream : List NativeFrames.Record
  rows : log.frames = (out.physical.rows.filter NativeFrames.whole).map NativeFrames.Row.frame

structure Arguments where
  address : Address
  board : Position
  score : Score
  bound : Bound
  depth : Depth
  move : Option Bitboard.Square
  selectivity : Selectivity
  generation : BitVec 8
  endgame : Bool
  pv : Bool

def checkedCall {log : Log mc fc} (t : Table log) (w : Stamp) (call : Call)
    (receipt : Certified log endian) : Option (Certified log endian) :=
  if w.present t call then some receipt else none

def clear {log : Log mc fc} (t : Table log) (w : Stamp) (endian more : Bool)
    (seq : Load t w .sequence) (metadata : Load t w (.payload .metadata)) (last : Word) :
    Option (Certified log endian) :=
  match hit : publishRound (w.ev log) endian (w.current t) (WriterMir.clearStep endian more
      (decide (w.current t = seq.word)) seq.word metadata.word last) with
  | none => none
  | some (req,call) => checkedCall t w call ⟨_,.clear _ _ _ _ _ _ req call hit⟩

def regular {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) (args : Arguments)
    (frame : Option Nat) (r : SaveRead t w endian) : Option (Certified log endian) := do
  if w.address ≠ args.address then none else
  let (_,_,entry) ← WriterMir.storeInput args.address args.board args.score args.bound args.depth args.move
    args.selectivity args.generation args.endgame args.pv
  letI : Decidable (Cutoffs.ExactEndgame entry) := inferInstanceAs (Decidable (_ ∧ _))
  if ok : ¬Cutoffs.ExactEndgame entry ∧ entry.bestMove = Packing.encodeBestMove args.move ∧
      validMove mc fc args.board args.move = true ∧ r.matches entry then
    match hit : publishRound (w.ev log) endian r.obs.current
        (saveRound endian args.board.val.player args.board.val.opponent entry r.obs) with
    | none => none
    | some (req,call) =>
      if id : call.finish = w.ev log 5 then
        let read := r.checked args.board entry ok.2.2.2
        checkedCall t w call ⟨_,.other _ frame _ entry args.move _ req call ok.1
          (generated_move _ _ ok.2.2.1) ok.2.1 (show WriterMir.observations (Prior log endian call.finish) _ _ _
            from id ▸ read) hit⟩
      else none
  else none

structure Site where
  opening : Nat
  frame : Nat
  scored : Nat

def identifyRow (w : Stamp) (site : Site) (r : NativeFrames.Row) := decide
  (r.opening = site.opening ∧ r.frame.id = site.frame ∧ r.frame.scoreStep = site.scored ∧
    NativeFrames.whole r = true ∧ r.frame.worker = w.worker ∧ r.frame.scoreStep < w.claim ∧
    w.claim < w.finish ∧ (r.frame.returnStep.map (fun ret => decide (w.finish < ret)) |>.getD true) = true)

def searched {log : Log mc fc} (t : Table log) (c : Catalog p log) (w : Stamp) (endian : Bool)
    (args : Arguments) (site : Site) (r : SaveRead t w endian) : Option (Certified log endian) := do
  match found : c.out.physical.rows.find? (identifyRow w site) with
  | none => none
  | some row =>
    let f := row.frame
    let save : CausalReplay.SearchSave :=
      ⟨⟨row.alpha.raw⟩,⟨row.beta.raw⟩,args.score,args.move,args.depth,args.generation,args.pv,r.obs,0⟩
    let bound := classify f.pv save.score save.alpha save.beta
    let entry := save.incoming bound
    if ok : args.address = w.address ∧ args.board.val = f.position.val ∧ args.depth = row.depth ∧
        args.score.raw = row.rawScore.raw ∧ args.endgame = true ∧ args.selectivity = Cutoffs.exactSelectivity ∧
        args.bound = bound ∧ validMove mc fc f.position save.move ∧ r.matches entry then
      match hit : storeRound (w.ev log) endian args.address.index f.position save bound with
      | none => none
      | some (req,call) =>
        if id : call.finish = w.ev log 5 then
          have valid : row.opening = site.opening ∧ row.frame.id = site.frame ∧ row.frame.scoreStep = site.scored ∧
              NativeFrames.whole row = true ∧ row.frame.worker = w.worker ∧ row.frame.scoreStep < w.claim ∧
              w.claim < w.finish ∧ (row.frame.returnStep.map (fun ret => decide (w.finish < ret)) |>.getD true) = true := by
            simpa only [identifyRow,decide_eq_true_eq] using List.find?_some found
          have member : f ∈ log.frames := by
            rw [c.rows]
            exact List.mem_map.mpr ⟨row,List.mem_filter.mpr ⟨List.mem_of_find?_eq_some found,valid.2.2.2.1⟩,rfl⟩
          have prior : log.before (log.identify row.frame.use) call.finish := by
            rw [id]
            exact log.sequenced _ _ ⟨valid.2.2.2.2.1,Nat.lt_trans valid.2.2.2.2.2.1 valid.2.2.2.2.2.2.1⟩
          have score : f.score = args.score.raw.toInt := row.value.trans (congrArg BitVec.toInt ok.2.2.2.1.symm)
          let saved : FrameReturns.SavedScore log call f save := ⟨member,prior,row.lo,row.hi,score⟩
          checkedCall t w call ⟨_,.searched f save _ _ req call saved row.window row.narrow
            (generated_move _ _ ok.2.2.2.2.2.2.2.1) (show WriterMir.observations (Prior log endian call.finish) _ _ _
              from id ▸ r.checked f.position _ ok.2.2.2.2.2.2.2.2) hit⟩
        else none
    else none

open EndgameSearch.CacheHistory (Pool Input Timeline)
open EndgameSearch.CacheHistory.NativeTrace (Raw Cursor Operands workerAt workerEvents expectedSites)

/-- The 12-empty strategy writes after the NWS child returns and before the enclosing PVS
    returns. A selective caller can reach this store without producing an exact PVS row. -/
def boundaryAt (p : Pool) (stream : List NativeFrames.Record) (w : Stamp) : Option (Nat × Operands) := do
  let owner ← workerAt p w.worker
  let events := (workerEvents w.worker (NativeFrameCache.project stream)).takeWhile (fun e => decide (e.step < w.claim))
  let cursor ← events.foldlM Cursor.advance (Cursor.initial owner)
  let top :: _ := cursor.stack | none
  let args ← top.operands
  if top.node = .boundary ∧ expectedSites (.deep args.input.active) = some cursor.boundarySites.reverse then
    return (top.first,args)
  else none

def boundary {log : Log mc fc} (t : Table log) (c : Catalog p log) (w : Stamp) (endian : Bool)
    (args : Arguments) (nodesBefore nodesAtStore : Nat) (r : SaveRead t w endian) : Option (Certified log endian) := do
  let (opening,op) ← boundaryAt p c.stream w
  let i := op.input
  let save := FrameReturns.boundarySave i args.score args.generation args.pv r.obs
  let bound := classify false args.score ⟨i.alpha.raw⟩ ⟨i.beta.raw⟩
  if args.address ≠ w.address || args.board.val != i.board.val || args.depth != i.depth ||
      args.move.isSome || args.selectivity != Cutoffs.exactSelectivity || !args.endgame || args.pv || args.bound != bound ||
      2^64 ≤ nodesBefore || 2^64 ≤ nodesAtStore ||
      !DeepLeafMir.enoughNodes i.checked (BitVec.ofNat 64 nodesBefore) (BitVec.ofNat 64 nodesAtStore) then none else
  if guards : i.depth.toNat = 12 ∧ i.depth = i.empties.count ∧ i.parallel ∧ !i.aborted ∧
      -32767 ≤ i.alpha.raw.toInt ∧ i.alpha.raw.toInt < i.beta.raw.toInt ∧ i.beta.raw.toInt ≤ 32767 ∧
      i.beta.raw.toInt = i.alpha.raw.toInt+1 then
    match branch : i.arm with
    | .deep polling =>
      if use : polling = i.active then
        let prior := c.out.physical.timeline.readAt mc fc w.worker opening
        if scored : (i.run mc fc p.address prior.history).score = some args.score.raw.toInt ∧ r.matches (save.incoming bound) then
          match hit : storeRound (w.ev log) endian w.address.index i.board save bound with
          | none => none
          | some (req,call) =>
            if id : call.finish = w.ev log 5 then
              checkedCall t w call ⟨_,.boundary c.out.physical.timeline i w.worker opening op.frame args.score
                args.generation args.pv r.obs _ _ req call guards.1 guards.2.1 guards.2.2.1
                (by simpa using guards.2.2.2.1) (use ▸ branch)
                ⟨guards.2.2.2.2.1,guards.2.2.2.2.2.1,guards.2.2.2.2.2.2.1⟩
                guards.2.2.2.2.2.2.2 scored.1 (show WriterMir.observations (Prior log endian call.finish) _ _ _
                  from id ▸ r.checked i.board _ scored.2) hit⟩
            else none
        else none
      else none
    | _ => none
  else none

theorem prior_lookup {b : Position} (seen : TableReads.SnapshotReadFrom source (some (Cutoffs.key b,expected)))
    (occupied : expected.bound ≠ 0) : TableReads.SlotLookupFrom source b (some expected) := by
  generalize obs : some (Cutoffs.key b,expected) = o at seen
  cases seen with
  | unavailable => contradiction
  | vacant d empty =>
    have data := congrArg Prod.snd (Option.some.inj obs)
    exact (occupied ((show expected = d from data) ▸ empty)).elim
  | sampled raw snap trusted accept =>
    have data : expected = Packing.unpack raw.bigEndian snap.metadata := congrArg Prod.snd (Option.some.inj obs)
    have bits : Cutoffs.key b = (snap.player,snap.opponent) := congrArg Prod.fst (Option.some.inj obs)
    obtain ⟨even,stable,metadata,_,keys⟩ := (Snapshot.readContract_proven _ _ _ _ _).mp accept
    have live : Snapshot.empty raw.bigEndian raw.payload.metadata = false := by
      simpa [Snapshot.empty,← metadata,← data] using occupied
    simp only [live,Bool.false_eq_true,ite_false] at keys
    apply TableReads.SlotLookupFrom.hit raw expected trusted
    exact (LookupCut.readContract_proven _ _ _ _ _ _ _).mpr ⟨even,
      keys.1.symm.trans (congrArg Prod.fst bits).symm,keys.2.symm.trans (congrArg Prod.snd bits).symm,
      stable,metadata ▸ data.symm,occupied⟩

def penalty {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) (b : Position) (frame : Option Nat)
    (expected : EntryData) (earlier current : WriterRead t w w.address (FrameTableReads.snapCode endian)) :
    Option (Certified log endian) := do
  if matched : FrameTableReads.snapshot endian earlier.words = some (Cutoffs.key b,expected) ∧ expected.bound ≠ 0 then
    let snap := (ReaderMir.run endian current.words).returned
    let old := snap.getD WriterMir.dummy
    match hit : publishRound (w.ev log) endian (w.current t) (WriterMir.penaltyStep snap.isSome
      b.val.player b.val.opponent expected old (Packing.unpack endian old.metadata)) with
    | none => none
    | some (req,call) =>
      if id : call.finish = w.ev log 5 then
        have previous : TableReads.SlotLookupFrom (Prior log endian (w.ev log 5)) b (some expected) :=
          prior_lookup (matched.1 ▸ FrameTableReads.snapshot_read_at t endian earlier.toInvocationAt) matched.2
        checkedCall t w call ⟨_,.penalty b frame _ _ expected snap req call
          (show TableReads.SlotLookupFrom (Prior log endian call.finish) b (some expected) from id ▸ previous) hit⟩
      else none
  else none

structure Child {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) where
  sq : Bitboard.Square
  position : Position
  allocation : Nat
  clusters : Word
  key : Word
  front : WriterRead t w (FrameTableReads.slotAt allocation clusters key 0)
    (FrameTableReads.lookupCode endian position)
  back : Option (WriterRead t w (FrameTableReads.slotAt allocation clusters key 1)
    (FrameTableReads.lookupCode endian position))
  frontBeforeBack : ∀ seen, back = some seen → log.before front.last (seen.ev 0)

def Child.decode {log : Log mc fc} {t : Table log} {w : Stamp} (c : Child t w endian) (b : Position) :
    Option (TableReads.ChildLookupFrom mc fc b (Prior log endian (w.ev log 5))) := do
  if generated : (Board.getMoves mc b.val).getLsbD c.sq.val then
    let move : PvsBoardTree.GeneratedSquare mc b.val := ⟨c.sq,generated⟩
    if same : c.position = (PvsBoardTree.play mc fc b move).val then
      let front := FrameTableReads.lookupValue endian c.position c.front.words
      have left := FrameTableReads.lookup_read_at t endian c.position c.front.toInvocationAt
      match c.back with
      | none =>
        if front.isSome then return ⟨move,front,none,same ▸ left,.miss⟩ else none
      | some seen =>
        let back := FrameTableReads.lookupValue endian c.position seen.words
        have right : TableReads.SlotLookupFrom (Prior log endian (w.ev log 5)) c.position back :=
          FrameTableReads.lookup_read_at t endian c.position seen.toInvocationAt
        if front.isNone then return ⟨move,front,back,same ▸ left,same ▸ right⟩
        else none
    else none
  else none

def enhanced {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) (args : Arguments)
    (frame : Option Nat) (alpha beta : Score) (children : List (Child t w endian)) (r : SaveRead t w endian) :
    Option (Certified log endian) := do
  if args.address ≠ w.address || args.bound != .lower then none else
  let reads ← children.mapM (fun child => child.decode args.board)
  let q : PvsEnhancedCutoff.Query := ⟨args.depth,args.selectivity,args.endgame,alpha⟩
  match scan : PvsEnhancedCutoff.scan q (reads.map TableReads.ChildLookupFrom.observation) with
  | none => none
  | some result =>
    let save : CausalReplay.SearchSave := ⟨alpha,beta,args.score,args.move,args.depth,args.generation,args.pv,r.obs,0⟩
    if flags : q.selectivity = Cutoffs.exactSelectivity ∧ q.endgame = true ∧
        -32767 ≤ alpha.raw.toInt ∧ alpha.raw.toInt < beta.raw.toInt ∧ beta.raw.toInt ≤ 32767 ∧
        beta.raw.toInt = alpha.raw.toInt+1 ∧ args.score.raw.toInt = result.2 ∧ args.move = some result.1 ∧
        r.matches (save.incoming .lower) then
      match hit : storeRound (w.ev log) endian w.address.index args.board save .lower with
      | none => none
      | some (req,call) =>
        if id : call.finish = w.ev log 5 then
          have castRead : ∀ s, Prior log endian (w.ev log 5) s → Prior log endian call.finish s := by rw [id]; simp
          let selected := reads.map (fun read => read.mapEvidence castRead)
          have observed : PvsEnhancedCutoff.scan q (selected.map TableReads.ChildLookupFrom.observation) = some result := by
            rw [show selected.map TableReads.ChildLookupFrom.observation = reads.map TableReads.ChildLookupFrom.observation
              from TableReads.ChildLookupFrom.observations_mapEvidence reads castRead]
            exact scan
          checkedCall t w call ⟨_,.enhanced _ frame _ _ save q result req call selected flags.1 flags.2.1 observed
            ⟨flags.2.2.1,flags.2.2.2.1,flags.2.2.2.2.1⟩ flags.2.2.2.2.2.1 flags.2.2.2.2.2.2.1
            rfl flags.2.2.2.2.2.2.2.1 hit⟩
        else none
    else none

inductive Operation {log : Log mc fc} (t : Table log) (w : Stamp) (endian : Bool) where
  | clear (more : Bool) (seq : Load t w .sequence) (metadata : Load t w (.payload .metadata)) (last : Word)
      (ordered : (t.memory w.address).sb seq.id metadata.id)
  | regular (args : Arguments) (frame : Option Nat) (read : SaveRead t w endian)
  | searched (args : Arguments) (site : Site) (read : SaveRead t w endian)
  | boundary (args : Arguments) (before after : Nat) (read : SaveRead t w endian)
  | penalty (board : Position) (frame : Option Nat) (expected : EntryData)
      (earlier current : WriterRead t w w.address (FrameTableReads.snapCode endian))
      (ordered : (t.memory w.address).sb earlier.last (current.ev 0))
  | enhanced (args : Arguments) (frame : Option Nat) (alpha beta : Score) (probes : List (Child t w endian))
      (read : SaveRead t w endian)

structure Packet {log : Log mc fc} (t : Table log) (endian : Bool) where
  stamp : Stamp
  operation : Operation t stamp endian

def Packet.decode {log : Log mc fc} {t : Table log} (c : Catalog p log) (packet : Packet t endian) :
    Option (Certified log endian) :=
  let w := packet.stamp
  match packet.operation with
  | .clear more seq metadata last _ => clear t w endian more seq metadata last
  | .regular args frame rd => regular t w endian args frame rd
  | .searched args site rd => searched t c w endian args site rd
  | .boundary args start finish rd => boundary t c w endian args start finish rd
  | .penalty b frame exp earlier current _ => penalty t w endian b frame exp earlier current
  | .enhanced args frame a z probes rd => enhanced t w endian args frame a z probes rd

def consume {log : Log mc fc} {t : Table log} (c : Catalog p log) :
    List FrameTableReads.Mark → List (Packet t endian) → Bool
  | [], packets => packets.isEmpty
  | .claim .. :: marks, packets => consume c marks packets
  | .finish addr e :: marks, packet :: packets =>
    if packet.stamp.address ≠ addr then false else
    match packet.decode c with
    | none => false
    | some actual => decide (signature actual.val = signature e) && consume c marks packets
  | .finish .. :: _, [] | .other _ :: _, _ => false

private theorem consumed {log : Log mc fc} {t : Table log} (c : Catalog p log) (marks : List FrameTableReads.Mark)
    (packets : List (Packet t endian)) (ok : consume c marks packets = true) :
    ∀ e ∈ marks.filterMap FrameTableReads.Mark.event?, WriteStep log endian e := by
  induction marks generalizing packets with
  | nil => simp
  | cons head tail ih =>
    cases head with
    | other e => contradiction
    | claim => exact ih packets ok
    | finish a native =>
      cases packets with
      | nil => contradiction
      | cons packet rest =>
        unfold consume at ok
        split at ok <;> try contradiction
        cases h : packet.decode c with
        | none => simp [h] at ok
        | some actual =>
          obtain ⟨same,later⟩ := (by simpa [h] using ok :
            signature actual.val = signature native ∧ consume c tail rest = true)
          intro e mem
          rcases List.mem_cons.mp mem with rfl | member
          · exact event_eq actual.val actual.property e same ▸ actual.property
          · exact ih rest later e member

def check {log : Log mc fc} (t : Table log) (c : Catalog p log) (packets : List (Packet t endian)) :=
  consume c t.trace packets

theorem stores {log : Log mc fc} (t : Table log) (c : Catalog p log) (packets : List (Packet t endian))
    (ok : check t c packets = true) : ∀ e ∈ log.writes, WriteStep log endian e := by
  rw [← t.recorded]
  exact consumed c t.trace packets ok

def releasePacket {t : Table log} (calls : Journal.Calls) (packet : Nat → Option (Packet t endian)) :
    Journal.Record → Option (Packet t endian)
  | .writer ⟨k,.finish⟩ => if (calls k).success then packet k else none
  | _ => none

/-- A single decoded store at each executed release suffices for the whole
    writer check, regardless of contention or unfinished writes in the prefix. -/
theorem check_of_releases {log : Log mc fc} (t : Table log) (c : Catalog p log)
    (calls : Journal.Calls) (rs : List Journal.Record) (packet : Nat → Option (Packet t endian))
    (trace : t.trace = rs.filterMap (Journal.mark calls endian))
    (source : ∀ k, .writer ⟨k,.finish⟩ ∈ rs → (calls k).success = true →
      ∃ pkt e, packet k = some pkt ∧ pkt.stamp.address = (calls k).address ∧
        pkt.decode c = some e ∧ e.val = (calls k).published endian) :
    check t c (rs.filterMap (releasePacket calls packet)) = true := by
  unfold check
  rw [trace]
  clear trace
  induction rs with
  | nil => rfl
  | cons head tail ih =>
    have rest := ih (fun k mem => source k (List.mem_cons_of_mem _ mem))
    cases head with
    | localCall => exact rest
    | writer site =>
      obtain ⟨k,pc⟩ := site
      cases pc <;> cases success : (calls k).success <;>
        simp [Journal.mark,releasePacket,success,consume,rest]
      all_goals try exact rest
      obtain ⟨pkt,e,found,addr,decoded,event⟩ := source k (List.mem_cons_self) success
      simp [releasePacket,success,found,consume,addr,decoded,event,rest]

end NeuralReversi.PvsParallel.NativeTableWrites
