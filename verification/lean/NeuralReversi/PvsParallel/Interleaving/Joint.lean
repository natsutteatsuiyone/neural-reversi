module
public import NeuralReversi.PvsParallel.Interleaving.CallJournal
public import NeuralReversi.PvsParallel.Interleaving.ReaderWindow
public import NeuralReversi.PvsParallel.NativeFrames.EntryStream

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.Joint

open TranspositionTable
open FrameTableReads.Operations.Execution (Instruction)
open NativeFrames.EntryStream (Sample)

deriving instance DecidableEq for RootMoveStorage.Row
deriving instance DecidableEq for PublicationMir.Iteration

/-- The machine operands and the root context at the entry to solve_root. -/
structure SolveArgs where
  board : PvsBoardTree.Position
  base : BitVec 32
  rows : List RootMoveStorage.Row
  saved : Option PublicationMir.Iteration
  multi : Bool
  minimum : Nat
  last : Nat
  deriving DecidableEq

/-- Slot generation, successful move-iterator CAS, and lock acquire/release.
    The address is the owner worker and its bank index. -/
inductive SplitWord where
  | mark (bank : Nat)
  | claim (opening owner bank old stored : Nat)
  | exhausted (owner bank load : Nat) (failures : List Nat)
  | maskRmw (owner bank participant : Nat) (booking : Bool) (old stored : BitVec 64)
  | lock (owner bank : Nat)
  | unlock (owner bank : Nat)
  | result (bank : Nat) (word : BitVec 32)
  deriving DecidableEq

/-- Pointer operands of split-task construction, assignment, and restoration.
    Slot numbers name the physical cutoff flag of the pointed-to allocation. -/
inductive TaskWord where
  | copyParent (slot : Nat) (active : Option Nat)
  | install (slot : Nat) (storedParent : Option Nat)
  | ownerActive (slot : Nat)
  | book (helper slot : Nat)
  | detach (old : Nat)
  | restore (slot : Nat) (loadedParent : Option Nat)
  | clearTask (slot : Nat)
  | resetCache (loaded source : Threading.AtomicWord.Event) (epoch : BitVec 64)
  | poll
  | afterSplit (active : Option Nat)
  | beginWalk (active : Option Nat)
  | endWalk (stopped : Bool)
  | storeFlag (slot : Nat) (word : Bool)
  | loadFlag (slot : Nat) (word : Bool) (source : Threading.AtomicWord.Event)
  | latch (word : Bool)
  | epoch (loaded source : Threading.AtomicWord.Event) (word seen : BitVec 64)
  | checkStop (latchStep : Nat)
  | cloneActive (active : Option Nat)
  | detachEmpty
  deriving DecidableEq

/-- Owner-side mask initialization and the guards of its idle-loop invocation.
    The bank belongs to the worker emitting the instruction. -/
inductive MaskWord where
  | clear (bank : Nat)
  | shutdown (bank : Nat) (closed : Bool)
  | load (bank : Nat) (source : Threading.AtomicWord.Event) (word : BitVec 64) (acquire : Bool)
  | returned (bank : Nat)
  | pc (bank : Nat) (block : Fin 80)
  deriving DecidableEq

/-- Relaxed i32 field accesses. Loads retain the physical modifying event even
    when a report or a different slot generation wrote the same score. -/
inductive AlphaWord where
  | store (value : BitVec 32)
  | load (source : Threading.AtomicWord.Event) (value : BitVec 32)
  deriving DecidableEq

inductive OwnerWord where
  | best (score : BitVec 32)
  | picked (square : BitVec 8)
  deriving DecidableEq

/-- Physical writes to the global cutoff_epoch counter. -/
inductive EpochWord where
  | init (value : BitVec 64)
  | bump (value : BitVec 64)
  deriving DecidableEq

/-- Physical writes and loads on the shared 64-bit AbortState word
    (high 32 bits = generation, low 32 bits = requested generation). -/
inductive AbortWord where
  | init (value : BitVec 64)
  | begin (value : BitVec 64)
  | request (value : BitVec 64)
  | load (source : Threading.AtomicWord.Event) (value : BitVec 64) (acquire : Bool)
  deriving DecidableEq

/-- Physical writes and loads on per-worker helper_job_aborted flags. -/
inductive HelperAbortWord where
  | store (target : Nat) (value : BitVec 64) (release : Bool)
  | load (source : Threading.AtomicWord.Event) (value : BitVec 64) (acquire : Bool)
  deriving DecidableEq

/-- Registers at the calls to evaluate, endgame_probcut_window and
    evaluate_depth2 inside endgame::probcut_decision. -/
inductive ProbWord where
  | estimate (board : Board.Board) (main : Bool) (score : BitVec 32)
  | window (depth : Depth) (level : Selectivity) (cut : Bool) (beta evaluation pcBeta screenBeta : BitVec 32)
  | verify (board : Board.Board) (main : Bool) (alpha beta score : BitVec 32)
  deriving DecidableEq

inductive Event (Operand : Type) where
  /-- Earlier worker calls from this pool, numbered over its lifetime. A new search
      may start its shared TT journal from an already validated resident image. -/
  | poolHistory (base size : Nat) (earlier : List Sample)
  | rootEnter (worker opening site : Nat) (receiver : EndgameSearch.CacheOwner.Resource)
      (args : NativeFrames.Args) (rows : List RootMoveStorage.Row)
  | probWord (worker : Nat) (registers : ProbWord)
  | splitWord (worker step : Nat) (instruction : SplitWord)
  | engine (event : CallJournal.Event Operand)
  | iterateEnd (worker step opening : Nat) (board : PvsBoardTree.Position) (base : BitVec 32)
      (saved : PublicationMir.Iteration) (rows : List RootMoveStorage.Row)
  | solveEnter (worker opening site : Nat) (receiver : EndgameSearch.CacheOwner.Resource) (args : SolveArgs)
  | enter (worker : Nat) (args : ReaderWindow.Args)
  | leave (worker : Nat) (answer : ReaderWindow.Answer)
  | seedRegister (worker : Nat) (b : PvsBoardTree.Position) (raw : ProbeRetry.Result)
  | fallbackRegister (worker : Nat) (b : PvsBoardTree.Position) (data : Option EntryData)
  | issue (worker : Nat) (opcode : Instruction) (source : Nat) (word : SeqLock.Word)

  | taskWord (worker step : Nat) (instruction : TaskWord)
  | maskWord (worker step : Nat) (instruction : MaskWord)
  /-- Successful CAS or release store on the target thread's mutex_for_state. -/
  | workerLock (worker step target : Nat) (acquired : Bool)
  | alphaWord (worker step owner bank : Nat) (op : AlphaWord)
  | ownerStore (worker step owner bank : Nat) (op : OwnerWord)
  | ownerLoad (worker step owner bank : Nat) (source : Threading.AtomicWord.Event) (op : OwnerWord)
  | epochWord (worker step : Nat) (op : EpochWord)
  | abortWord (worker step : Nat) (op : AbortWord)
  | helperAbortWord (worker step : Nat) (op : HelperAbortWord)

def Event.engineEvent : Event Operand → CallJournal.Event Operand
  | .poolHistory .. => .other 0
  | .engine e => e
  | .rootEnter w t site r args _ => .localCall w (.entry t site .pvs r none) (some args)
  | .iterateEnd w t first .. => .localCall w (.leave t first .iterate none none) none
  | .solveEnter w t site r _ => .localCall w (.entry t site .solve r none) none
  | .helperAbortWord w .. | .abortWord w .. | .epochWord w .. | .ownerStore w .. | .ownerLoad w .. | .alphaWord w .. | .workerLock w .. | .maskWord w .. | .taskWord w .. | .probWord w _ | .splitWord w .. | .enter w _ | .leave w _ | .issue w .. | .seedRegister w .. | .fallbackRegister w .. => .other w

def Event.frameSample : Event Operand → Option Sample
  | .rootEnter w t site r args _ => some ⟨.localCall w (.entry t site .pvs r none),some args⟩
  | .iterateEnd w t first .. => some ⟨.localCall w (.leave t first .iterate none none),none⟩
  | .solveEnter w t site r _ => some ⟨.localCall w (.entry t site .solve r none),none⟩
  | .engine (.localCall w e args) => some ⟨.localCall w e,args⟩
  | _ => none

def Event.worker : Event Operand → Nat
  | .poolHistory .. => 0
  | .rootEnter w .. | .solveEnter w .. | .iterateEnd w .. | .helperAbortWord w .. | .abortWord w .. | .epochWord w .. | .ownerStore w .. | .ownerLoad w .. | .alphaWord w .. | .workerLock w .. | .maskWord w .. | .taskWord w .. | .splitWord w .. | .probWord w _ => w
  | .enter w _ | .leave w _ | .issue w .. | .seedRegister w .. | .fallbackRegister w .. => w
  | .engine (.other w) | .engine (.localCall w ..) | .engine (.beginWrite w ..)
  | .engine (.atomic w _) | .engine (.boundaryScore w ..) => w

def Event.marker : Event Operand → ReaderWindow.Mark
  | .enter _ args => .enter args | .leave _ answer => .leave answer | _ => .other

def Event.op : Event Operand → Option Instruction
  | .issue _ op .. => some op | _ => none

abbrev Tape (Operand : Type) := List (Event Operand)

def marker (events : Tape Operand) (i : Nat) := (events[i]?.map Event.marker).getD .other
def opcode (events : Tape Operand) (i : Nat) := events[i]?.bind Event.op
def word (events : Tape Operand) (i : Nat) : SeqLock.Word := match events[i]? with
  | some (.issue _ _ _ v) => v | _ => 0
def origin (events : Tape Operand) (i : Nat) : Nat := match events[i]? with
  | some (.issue _ _ src _) => src | _ => 0

/-- Each out-of-prefix position has a distinct unused worker label. Thus a
    sequenced-before edge can only connect two real events of the same worker. -/
def worker (events : Tape Operand) (i : Nat) := match events[i]? with
  | some e => 2*e.worker | none => 2*i+1

/-- Follow the result register on one worker. Starting a new reader, consuming a
    result, or crossing a frame boundary ends its lifetime. -/
def lastRead (es : Tape Operand) (w : Nat) : Nat → Option (Nat × ReaderWindow.Answer)
  | 0 => none
  | i+1 =>
    match es[i]? with
    | none => none
    | some ev => if ev.worker = w then
        match ev with
        | .leave _ value => some (i,value)
        | .engine (.other _) => lastRead es w i
        | _ => none
      else lastRead es w i

def receiverAt (es : Tape Operand) (i : Nat) (b : PvsBoardTree.Position) : Option (Nat × ReaderWindow.Answer) := do
  let event ← es[i]?
  let (board,raw) ← (match event with
    | .seedRegister _ board raw => some (board,ReaderWindow.Answer.probe raw)
    | .fallbackRegister _ board data => some (board,ReaderWindow.Answer.lookup data)
    | _ => none)
  guard (board = b)
  let (finish,value) ← lastRead es event.worker i
  guard (value = raw)
  return (finish,value)

private theorem last_returned (es : Tape Operand) (w : Nat) (end_ : Nat)
    (hit : lastRead es w end_ = some (finish,value)) :
    marker es finish = .leave value := by
  induction end_ with
  | zero => contradiction
  | succ i ih =>
    unfold lastRead at hit
    split at hit <;> try contradiction
    rename_i ev present
    split at hit
    · split at hit
      · cases hit; simp [marker,present,Event.marker]
      · exact ih hit
      · contradiction
    · exact ih hit

theorem received (es : Tape Operand) (ok : receiverAt es i b = some (finish,raw)) :
    marker es finish = .leave raw := by
  unfold receiverAt at ok
  obtain ⟨event,_,tail⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨⟨_,_⟩,_,tail⟩ := Option.bind_eq_some_iff.mp tail
  simp only [guard] at tail
  split at tail <;> try contradiction
  simp only [pure,Bind.bind,Option.bind] at tail
  obtain ⟨⟨_,_⟩,fromWorker,tail⟩ := Option.bind_eq_some_iff.mp tail
  split at tail <;> try contradiction
  cases tail
  exact last_returned es _ _ fromWorker

def seedAt (es : Tape Operand) (i : Nat) (b : PvsBoardTree.Position) : Option (Nat × ProbeRetry.Result) := do
  let (finish,.probe raw) ← receiverAt es i b | none
  return (finish,raw)

theorem seed_returned (es : Tape Operand) {word : ProbeRetry.Result} (ok : seedAt es i b = some (finish,word)) :
    marker es finish = .leave (.probe word) := by
  obtain ⟨⟨j,v⟩,hit,out⟩ := Option.bind_eq_some_iff.mp ok
  cases v <;> cases out
  exact received es hit

namespace Regression

private def board : PvsBoardTree.Position := ⟨⟨0,0⟩,rfl⟩
private def lookup (middle : Event Unit) := receiverAt
    [.leave 0 (.lookup none),middle,.fallbackRegister 0 board none] 2 board

example : lookup (.leave 1 (.lookup none)) = some (0,.lookup none) := by rfl
example : lookup (.leave 0 (.lookup none)) = some (1,.lookup none) := by rfl
example : lookup (.fallbackRegister 0 board none) = none := by rfl
example : lookup (.seedRegister 0 board (.miss 0)) = none := by rfl

end Regression

variable {x : NativeTableWrites.Journal.Execution c endian rs}

/-- These are primitive memory-model facts at the executed opcode. Reads-from
    is a modification-order index, including when two writes store the same word. -/
structure Atomics (x : NativeTableWrites.Journal.Execution c endian rs) (es : Tape Operand) : Prop where
  reads : ∀ i a loc mode, opcode es i = some (.load a loc mode) →
    (x.memory a).rf ((x.memory a).write loc (origin es i)) i ∧ (x.memory a).readValue i = word es i
  acquire : ∀ i a loc, opcode es i = some (.load a loc .acquire) → (x.memory a).acquire i
  fence : ∀ i, opcode es i = some (.fence .acquire) → ∀ a, (x.memory a).acquireFence i
  sb : ∀ a i j, i < j → j < es.length → worker es i = worker es j → (x.memory a).sb i j

def bus (es : Tape Operand) (m : Atomics x es) : ReaderJournal.Accesses x where
  instruction := opcode es
  source := origin es
  generation _ i := (word es i).setWidth 8
  worker := worker es
  tick := id
  read i a loc mode issued := (m.reads i a loc mode issued).1
  acquire := m.acquire
  fence := m.fence
  programOrder a i j same earlier := by
    change i < j at earlier
    change worker es i = worker es j at same
    apply m.sb a i j earlier _ same
    by_cases inside : j < es.length
    · exact inside
    · have absent := List.getElem?_eq_none (by omega : es.length ≤ j)
      unfold worker at same
      rw [absent] at same
      cases seen : es[i]? with
      | none =>
        rw [seen] at same
        change 2*i+1 = 2*j+1 at same
        omega
      | some other =>
        rw [seen] at same
        change 2*other.worker = 2*j+1 at same
        omega

def stream (es : Tape Operand) (m : Atomics x es) : ReaderWindow.Stream (bus es m) where
  marker := marker es
  ordered lt _ := lt

/-- The native trace records the single word returned by each load. Replaying
    the reader with that word gives the same effects and result as reading the
    addressed memory-model register at every executed opcode. -/
theorem checked (es : Tape Operand) (mem : Atomics x es) (args : ReaderWindow.Args)
    (finish consumer opening : Nat)
    (accepted : ReaderWindow.check (worker es) (marker es) (opcode es) (fun _ => word es)
      (fun _ i => (word es i).setWidth 8) endian finish consumer = some (opening,args)) :
    ReaderWindow.check (worker es) (marker es) (opcode es) (fun a => (x.memory a).readValue)
      (fun _ i => (word es i).setWidth 8) endian finish consumer = some (opening,args) := by
  let ids := ReaderWindow.positions (worker es) (opcode es) opening finish
  let gen := fun (_ i : Nat) => (word es i).setWidth 8
  let raw := ReaderWindow.registers ids consumer (fun _ => word es) gen
  let model := ReaderWindow.registers ids consumer (fun a => (x.memory a).readValue) gen
  let emitted := args.ops raw endian
  obtain ⟨entered,returned,ops⟩ := (ReaderWindow.check_iff ..).mp accepted
  change ids.map (opcode es) = emitted.map _ at ops
  have agree : FrameTableReads.Operations.Runs (FrameTableReads.Operations.Execution.agrees raw model) 0 emitted := by
    intro i op atIndex
    have obs := congrArg (fun xs : List (Option Instruction) => xs[i]?) ops
    simp only [List.getElem?_map,atIndex,Option.map_some] at obs
    obtain ⟨id,get,how⟩ := Option.map_eq_some_iff.mp obs
    cases op with
    | generation => rfl
    | spin => trivial
    | slot site act =>
      cases act with
      | fence => trivial
      | load loc mode =>
        have actual := (mem.reads id site.address loc mode how).2.symm
        simpa only [FrameTableReads.Operations.Execution.agrees,raw,model,ReaderWindow.registers,Nat.zero_add,get,Option.getD_some] using actual
  have congruent : emitted = args.ops model endian ∧ args.answer raw endian = args.answer model endian := by
    cases args with
    | mk method allocation clusters key =>
      cases method with
      | probe q =>
        obtain ⟨steps,val⟩ := FrameTableReads.Operations.Execution.probe_congr raw model allocation clusters key endian q agree
        exact ⟨steps,congrArg ReaderWindow.Answer.probe val⟩
      | lookup b =>
        obtain ⟨steps,val⟩ := FrameTableReads.Operations.Execution.lookup_congr raw model allocation clusters key endian b agree
        exact ⟨steps,congrArg (fun x : Option EntryData × Option EntryData => ReaderWindow.Answer.lookup (x.1.or x.2)) val⟩
  apply (ReaderWindow.check_iff ..).mpr
  exact ⟨entered,returned.trans (congrArg ReaderWindow.Mark.leave congruent.2),
    ops.trans (congrArg (List.map _) congruent.1)⟩

end NeuralReversi.PvsParallel.Interleaving.Joint
