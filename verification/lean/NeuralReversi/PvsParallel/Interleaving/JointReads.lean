import NeuralReversi.PvsParallel.Interleaving.Joint

namespace NeuralReversi.PvsParallel.Interleaving.JointReads

open TranspositionTable
open FrameTableReads.Operations
open FrameTableReads.Operations.Execution (Method)
open PvsBoardTree (Position GeneratedSquare play)

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape OpType}
variable {m : Joint.Atomics x es}

structure Call (m : Joint.Atomics x es) (consumer : Nat) (method : Method) where
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  opening : Nat
  finish : Nat
  later : finish ≤ consumer
  same : Joint.worker es finish = Joint.worker es consumer
  decoded : ReaderWindow.check (Joint.worker es) (Joint.marker es) (Joint.opcode es) (fun _ => Joint.word es)
    (fun _ i => (Joint.word es i).setWidth 8) endian finish consumer = some (opening,⟨method,allocation,clusters,key⟩)
  nowrap : let call := (Joint.stream es m).call opening finish ⟨consumer,later,same⟩
    call.NoWrap (Joint.bus es m) ((ReaderWindow.Args.mk method allocation clusters key).ops
      ((call.reader (Joint.bus es m)).supply x) endian)

def Call.invocation (r : Call m consumer method) :=
  (Joint.stream es m).call r.opening r.finish ⟨consumer,r.later,r.same⟩

def Call.reader (r : Call m consumer method) := r.invocation.reader (Joint.bus es m)
def Call.supply (r : Call m consumer method) := r.reader.supply x

theorem Call.issued (r : Call m consumer method) :
    Runs (r.invocation.Issues (Joint.bus es m)) 0 ((ReaderWindow.Args.mk method r.allocation r.clusters r.key).ops r.supply endian) :=
  ((Joint.stream es m).finished ⟨consumer,r.later,r.same⟩
    (Joint.checked es m _ r.finish consumer r.opening r.decoded)).runs

def attempts (b : Position) (p : Option (Call m consumer (.probe (Cutoffs.key b)))) : List ProbeRetry.RawAttempt :=
  p.toList.flatMap fun r =>
    (probe r.supply r.allocation r.clusters r.key endian (Cutoffs.key b) 0).rounds.map (fun t => t.read r.supply endian)

theorem parent (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    (b : Position) (p : Option (Call m consumer (.probe (Cutoffs.key b))))
    (hit : ProbeRetry.probe ((attempts b p).map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e) :
    CausalTable.lookup x init consumer b e := by
  cases p with
  | none => simp [attempts,ProbeRetry.probe,ProbeRetry.scan] at hit
  | some r =>
    simp only [attempts,Option.toList_some,List.flatMap_cons,List.flatMap_nil,List.append_nil,List.map_map,Function.comp_def] at hit
    exact CausalReads.probe_hit x init r.reader consumer r.allocation r.clusters r.key b 0
      (ReaderJournal.probe_sites _ r.invocation _ _ _ b r.issued r.nowrap)
      ((probe_result r.supply _ _ _ _ _ 0).symm.trans hit)

abbrev Child (m : Joint.Atomics x es) (consumer : Nat) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :=
  (move : GeneratedSquare mc b.val) × Call m consumer (.lookup (play mc fc b move).val)

private theorem operand (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (reader : CausalReads.Reader)
    (b : Position) (site : Site) (actual : CausalReads.Issued x reader consumer site (FrameTableReads.lookupCode endian b)) :
    TableReads.SlotLookupFrom (fun _ => True) b (FrameTableReads.lookupValue endian b (site.words (reader.supply x))) :=
  FrameTableReads.lookup_operand (x.slot site.address (init _)) endian b (fun i => reader.event (site.opening+i))
    (fun i => reader.source (site.opening+i)) actual.steps actual.noWrap

def Child.row (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (child : Child m consumer mc fc b) :
    TableReads.ChildLookupFrom mc fc b (fun _ => True) := by
  let r := child.2
  let next := (play mc fc b child.1).val
  let out := lookup r.supply r.allocation r.clusters r.key endian next 0
  obtain ⟨first,rest⟩ := ReaderJournal.lookup_sites (Joint.bus es m) r.invocation _ _ _ next r.issued r.nowrap
  refine ⟨child.1,(out.pair r.supply endian next).1,(out.pair r.supply endian next).2,
    operand init r.reader next out.front first,?_⟩
  change TableReads.SlotLookupFrom _ _ (out.back.bind _)
  cases h : out.back with
  | none => exact .miss
  | some back => exact operand init r.reader next back (rest back h)

/-- The parent probe and each tested child board are replayed from their own
    instruction windows in the same writer memory. A skipped probe has no hits. -/
theorem physical (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (b : Position)
    (p : Option (Call m consumer (.probe (Cutoffs.key b)))) (children : List (Child m consumer mc fc b)) :
    NativeNodes.Probed mc fc b (CausalTable.lookup x init consumer) (attempts b p) (children.map (Child.row init)) := by
  constructor
  · exact fun _ _ hit => parent init b p hit
  · intro row seen e hit
    obtain ⟨child,_,rfl⟩ := List.mem_map.mp seen
    let r := child.2
    let next := (play mc fc b child.1).val
    obtain ⟨front,back⟩ := ReaderJournal.lookup_sites (Joint.bus es m) r.invocation _ _ _ next r.issued r.nowrap
    exact CausalReads.cluster_hit x init r.reader consumer r.allocation r.clusters r.key next 0 front back hit

open EndgameOrderMir (Heuristic framePriority)
open ScoreRefresh (Score)

/-- Internal null-window nodes use the generated parent and ETC operands for
    both prefix cutoffs and the later owner loop. -/
theorem scout (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (b : Position)
    (p : Option (Call m consumer (.probe (Cutoffs.key b)))) (children : List (Child m consumer mc fc b))
    (depth : Depth) (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (storage : MoveList.Initialization.Memory Move.Move) (returns : NativeNodes.Returns) (inside : returns.InPrefix prior)
    (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
    (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val))
    (splits : Nested.Splits (BoardFrame.Children mc fc b returns.prior) false cut depth.toNat width beta.raw.toInt)
    (full : ExactDepth.Full b depth) (wide : 7 ≤ width)
    (normal : GeneralOwnerMir.scheduled mc fc b depth alpha beta paired checked cut storage (attempts b p)
      (children.map (Child.row init)) (framePriority mc fc b mode tie) returns.scout schedule (Nested.scoutSplit splits) = some v) :
    Nested.Body mc fc prior (CausalTable.lookup x init consumer) b false alpha.raw.toInt beta.raw.toInt v :=
  .scout depth alpha beta paired checked cut storage _ _ returns inside mode tie schedule splits full wide (physical init b p children) normal

end NeuralReversi.PvsParallel.Interleaving.JointReads
