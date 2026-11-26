import NeuralReversi.PvsParallel.Interleaving.ReaderJournal
import NeuralReversi.Board.Hash

namespace NeuralReversi.PvsParallel.Interleaving.ReaderWindow

open TranspositionTable
open FrameTableReads.Operations
open FrameTableReads.Operations.Execution (Method Instruction)
open ReaderJournal (Accesses Invocation)
open PvsBoardTree (Position)

variable {x : NativeTableWrites.Journal.Execution c endian rs}

structure Args where
  method : Method
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word

/-- Search computes this argument with Board::hash before calling the TT. The
    public table methods can also be tested with a caller-supplied raw key. -/
def engineKey (endian : Bool) : Method → SeqLock.Word
  | .probe (player,opponent) => Board.Hash.hash endian ⟨player,opponent⟩
  | .lookup b => Board.Hash.hash endian b.val

def Args.inTable (args : Args) (endian : Bool) (allocation : Nat) (clusters : SeqLock.Word) : Bool :=
  args.allocation == allocation && args.clusters == clusters && args.key == engineKey endian args.method

deriving instance DecidableEq for ProbeRetry.Result

inductive Answer where
  | probe (result : ProbeRetry.Result)
  | lookup (result : Option EntryData)
  deriving DecidableEq

inductive Mark where
  | enter (args : Args)
  | leave (answer : Answer)
  | other

/-- Global positions enumerate a finite observation prefix. Only the order of
    positions on the same worker constrains sequenced-before in the atomic model. -/
structure Stream (bus : Accesses x) where
  marker : Nat → Mark
  ordered : ∀ {i j}, i < j → bus.worker i = bus.worker j → bus.tick i < bus.tick j

variable {bus : Accesses x}

/-- Scan to the closest method boundary on this worker. A previous return ends
    that invocation even when a later request reuses the exact same arguments. -/
def activeAt (thread : Nat → Nat) (marker : Nat → Mark) (worker : Nat) : Nat → Option (Nat × Args)
  | 0 => none
  | n+1 =>
    if thread n = worker then
      match marker n with
      | .enter args => some (n,args)
      | .leave _ => none
      | .other => activeAt thread marker worker n
    else activeAt thread marker worker n

abbrev Stream.active (s : Stream bus) := activeAt bus.worker s.marker

def positions (thread : Nat → Nat) (opcode : Nat → Option Instruction) (opening finish : Nat) : List Nat :=
  (List.range finish).filter fun i =>
    decide (opening < i ∧ thread i = thread finish) && (opcode i).isSome

abbrev Stream.events (_ : Stream bus) := positions bus.worker bus.instruction

private theorem Stream.member (s : Stream bus) (h : i ∈ s.events opening finish) :
    i < finish ∧ opening < i ∧ bus.worker i = bus.worker finish := by
  obtain ⟨before,inside,_⟩ : i < finish ∧ (opening < i ∧ bus.worker i = bus.worker finish) ∧
    (bus.instruction i).isSome = true := by simpa [events,positions] using h
  exact ⟨before,inside⟩

structure Use (bus : Accesses x) (finish : Nat) where
  event : Nat
  later : finish ≤ event
  worker : bus.worker finish = bus.worker event

def Stream.call (s : Stream bus) (opening finish : Nat) (use : Use bus finish) : Invocation bus where
  consumer := use.event
  events := s.events opening finish
  sameThread _ inside := (s.member inside).2.2.trans use.worker
  ordered := by
    apply (List.pairwise_lt_range.filter _).imp_of_mem
    exact fun ha hb lt => s.ordered lt ((s.member ha).2.2.trans (s.member hb).2.2.symm)
  used := fun _ member => s.ordered (Nat.lt_of_lt_of_le (s.member member).1 use.later) ((s.member member).2.2.trans use.worker)

def Args.ops (args : Args) (s : Supply) (endian : Bool) : List Op := match args.method with
  | .probe q => (probe s args.allocation args.clusters args.key endian q 0).emit s endian q
  | .lookup b => (lookup s args.allocation args.clusters args.key endian b 0).emit s endian b

def Args.answer (args : Args) (s : Supply) (endian : Bool) : Answer := match args.method with
  | .probe q => .probe (probe s args.allocation args.clusters args.key endian q 0).result
  | .lookup b =>
    let pair := (lookup s args.allocation args.clusters args.key endian b 0).pair s endian b
    .lookup (pair.1.or pair.2)

def registers (events : List Nat) (consumer : Nat) (word : FrameTableReads.Address → Nat → SeqLock.Word)
    (generation : Nat → Nat → BitVec 8) : Supply :=
  ⟨fun a i => word a (events[i]?.getD consumer),fun a i => generation a (events[i]?.getD consumer)⟩

/-- Decode the method and its entire instruction window from the observed worker,
    address, ordering, returned register values, and method boundaries. -/
def check (thread : Nat → Nat) (marker : Nat → Mark) (opcode : Nat → Option Instruction)
    (word : FrameTableReads.Address → Nat → SeqLock.Word) (generation : Nat → Nat → BitVec 8)
    (endian : Bool) (finish consumer : Nat) : Option (Nat × Args) := do
  let (opening,args) ← activeAt thread marker (thread finish) finish
  let .leave raw := marker finish | none
  let events := positions thread opcode opening finish
  let s := registers events consumer word generation
  guard (events.map opcode = (args.ops s endian).map (fun op => some (Execution.instruction op)))
  guard (raw = args.answer s endian)
  return (opening,args)

theorem check_iff (thread marker opcode word generation endian finish consumer opening args) :
    let es := positions thread opcode opening finish
    let s := registers es consumer word generation
    check thread marker opcode word generation endian finish consumer = some (opening,args) ↔
      activeAt thread marker (thread finish) finish = some (opening,args) ∧
      marker finish = .leave (args.answer s endian) ∧
      es.map opcode = (args.ops s endian).map (fun op => some (Execution.instruction op)) := by
  intro es words
  constructor
  · intro accepted
    unfold check at accepted
    obtain ⟨⟨first,a⟩,entered,h⟩ := Option.bind_eq_some_iff.mp accepted
    dsimp only at h
    cases m : marker finish with
    | enter _ | other => simp [m] at h
    | leave raw =>
      simp only [m,guard,Option.pure_def] at h
      split at h <;> try contradiction
      rename_i ops
      split at h <;> try contradiction
      rename_i result
      obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj h)
      exact ⟨entered,congrArg Mark.leave result,ops⟩
  · rintro ⟨entered,returned,ops⟩
    dsimp only [words,es] at ops returned
    simp [check,entered,returned,guard,ops]

/-- Both checks have finite, observed operands. Neither event membership nor
    a separate instruction sequence for any slot is supplied by the caller. -/
structure Finished (s : Stream bus) (args : Args) (use : Use bus finish) where
  opening : Nat
  entered : s.active (bus.worker finish) finish = some (opening,args)
  left : s.marker finish = .leave (args.answer (((s.call opening finish use).reader bus).supply x) endian)
  instructions : let call := s.call opening finish use
    call.events.map bus.instruction =
      (args.ops ((call.reader bus).supply x) endian).map (fun op => some (Execution.instruction op))

abbrev Finished.call {s : Stream bus} {use : Use bus finish} (f : Finished s args use) : Invocation bus :=
  s.call f.opening finish use

def Stream.finished (s : Stream bus) (use : Use bus finish)
    (accepted : check bus.worker s.marker bus.instruction (fun a => (x.memory a).readValue) bus.generation
      endian finish use.event = some (opening,args)) : Finished s args use := by
  let ok := (check_iff ..).mp accepted
  exact ⟨opening,ok.1,ok.2.1,ok.2.2⟩

theorem Finished.runs {s : Stream bus} (f : Finished s args use) :
    Runs (f.call.Issues bus) 0 (args.ops ((f.call.reader bus).supply x) endian) := by
  intro i op atIndex
  have atEvent := congrArg (fun xs : List (Option Instruction) => xs[i]?) f.instructions
  simp only [List.getElem?_map,atIndex,Option.map_some] at atEvent
  obtain ⟨e,selected,issued⟩ := Option.map_eq_some_iff.mp atEvent
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  exact ⟨by simpa using inside,by simpa [Finished.call,Invocation.event,selected] using issued⟩

/-- All occupied hits read by this actual invocation come from the writer journal. -/
theorem lookup_hit (stream : Stream bus) (use : Use bus finish) (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    (b : Position) (f : Finished stream ⟨.lookup b,allocation,clusters,key⟩ use)
    (nowrap : f.call.NoWrap bus ((Args.mk (.lookup b) allocation clusters key).ops ((f.call.reader bus).supply x) endian))
    (returned : stream.marker finish = .leave (.lookup (some entry))) : CausalTable.lookup x init use.event b entry := by
  obtain ⟨first,later⟩ := ReaderJournal.lookup_sites bus f.call allocation clusters key b f.runs nowrap
  have value := f.left.symm.trans returned
  injection value with out
  injection out with selected
  exact CausalReads.cluster_hit x init (f.call.reader bus) use.event allocation clusters key b 0 first later selected

def rootProbe {h : RootNative.History mc fc} {bus : Accesses h.native} (stream : Stream bus) (use : Use bus finish)
    (b : Position) (f : Finished stream ⟨.probe (Cutoffs.key b),allocation,clusters,key⟩ use)
    (nowrap : f.call.NoWrap bus ((Args.mk (.probe (Cutoffs.key b)) allocation clusters key).ops
        ((f.call.reader bus).supply h.native) h.endian)) : Entry.Probe h b :=
  ⟨f.call.reader bus,use.event,allocation,clusters,key,0,ReaderJournal.probe_sites bus f.call _ _ _ b f.runs nowrap⟩

theorem probe_answer (stream : Stream bus) (use : Use bus finish) (b : Position)
    (f : Finished stream ⟨.probe (Cutoffs.key b),allocation,clusters,key⟩ use)
    (returned : stream.marker finish = .leave (.probe result)) :
    (probe ((f.call.reader bus).supply x) allocation clusters key endian (Cutoffs.key b) 0).result = result := by
  have same := f.left.symm.trans returned
  injection same with out
  exact Answer.probe.inj out

end NeuralReversi.PvsParallel.Interleaving.ReaderWindow
