import NeuralReversi.PvsParallel.Interleaving.Entry
import NeuralReversi.PvsParallel.FrameTableReads.ProbeRegisters

namespace NeuralReversi.PvsParallel.Interleaving.ReaderJournal

open TranspositionTable
open FrameTableReads (Address snapCode lookupCode lookupValue slotAt)
open FrameTableReads.Operations
open FrameTableReads.Operations.Execution (Caller Atomic Instruction)
open PvsBoardTree (Position)

variable {x : NativeTableWrites.Journal.Execution c endian rs}

/-- Each instruction in the per-thread trace has a unique position supplied by the
    execution. Both loads of an equal value keep their individual reads-from indices. -/
structure Accesses (x : NativeTableWrites.Journal.Execution c endian rs) where
  instruction : Nat → Option Instruction
  source : Nat → Nat
  generation : Nat → Nat → BitVec 8
  worker : Nat → Nat
  tick : Nat → Nat
  read : ∀ ev addr loc mode, instruction ev = some (.load addr loc mode) →
    (x.memory addr).rf ((x.memory addr).write loc (source ev)) ev
  acquire : ∀ ev addr loc, instruction ev = some (.load addr loc .acquire) → (x.memory addr).acquire ev
  fence : ∀ ev, instruction ev = some (.fence .acquire) → ∀ addr, (x.memory addr).acquireFence ev
  programOrder : ∀ addr i j, worker i = worker j → tick i < tick j → (x.memory addr).sb i j

/-- The calling worker can be suspended between any two positions. The consumer
    lies after the complete reader call on the same worker. -/
structure Invocation (bus : Accesses x) where
  consumer : Nat
  events : List Nat
  sameThread : ∀ e ∈ events, bus.worker e = bus.worker consumer
  ordered : events.Pairwise (fun a b => bus.tick a < bus.tick b)
  used : ∀ e ∈ events, bus.tick e < bus.tick consumer

variable {bus : Accesses x}

def Invocation.event (call : Invocation bus) (i : Nat) : Nat := call.events[i]?.getD call.consumer

def Invocation.reader (bus : Accesses x) (call : Invocation bus) : CausalReads.Reader :=
  ⟨call.event, fun i => bus.source (call.event i),fun a i => bus.generation a (call.event i)⟩

def Invocation.Issues (bus : Accesses x) (call : Invocation bus) (i : Nat) (op : Op) : Prop :=
  i < call.events.length ∧ bus.instruction (call.event i) = some (Execution.instruction op)

/-- The seq nonwrap condition is local to the two sequence loads in one occupied-size
    read. Other slot generations, and the total lifetime of the table, are unrestricted. -/
def Invocation.NoWrap (bus : Accesses x) (call : Invocation bus) (ops : List Op) : Prop :=
  ∀ r : Site, ops[r.opening]? = some (.slot r (.load .sequence .acquire)) →
    ops[r.opening+5]? = some (.slot r (.load .sequence .relaxed)) →
    bus.source (call.event (r.opening+5)) < bus.source (call.event r.opening) + SeqLock.modulus

private theorem Invocation.member (call : Invocation bus) (i : Nat) (h : i < call.events.length) :
    call.event i ∈ call.events := by simp [event,List.getElem?_eq_getElem h]

private theorem Invocation.before (bus : Accesses x) (call : Invocation bus) (a : Address)
    {i j : Nat} (less : i < j) (end_ : j < call.events.length) :
    (x.memory a).sb (call.event i) (call.event j) := by
  have inside : i < call.events.length := Nat.lt_trans less end_
  apply bus.programOrder
  · exact (call.sameThread _ (call.member i inside)).trans (call.sameThread _ (call.member j end_)).symm
  · simpa only [event,List.getElem?_eq_getElem inside,List.getElem?_eq_getElem end_,Option.getD_some] using
      (List.pairwise_iff_getElem.mp call.ordered i j inside end_ less)

private theorem Invocation.site (bus : Accesses x) (call : Invocation bus) (r : Site)
    {code : Code} (path : ReadPath code value) (whole : List Op)
    (ran : Runs (fun i op => whole[i]? = some op ∧ call.Issues bus i op) r.opening
      (r.emit ((call.reader bus).supply x) code))
    (nowrap : call.NoWrap bus whole) :
    CausalReads.Issued x (call.reader bus) call.consumer r code := by
  let s := (call.reader bus).supply x
  have first := r.issued_at ran (path.first (r.words s))
  have localRun : ReaderMir.Realizes (x.memory r.address) (code (r.words s))
      (fun i => call.event (r.opening+i)) (fun i => bus.source (call.event (r.opening+i))) := by
    constructor
    · intro i hi
      have second := r.issued_at ran (List.getElem?_eq_getElem hi)
      exact call.before bus _ (i := r.opening+i) (j := r.opening+(i+1)) (by omega) second.2.1
    · intro i loc mode load
      exact bus.read _ _ _ _ (r.issued_at ran load).2.2
    · intro i loc load
      exact bus.acquire _ _ _ (r.issued_at ran load).2.2
    · intro i fence
      exact bus.fence _ (r.issued_at ran fence).2.2 _
  refine ⟨localRun,?_,?_⟩
  · have atRead := call.member _ first.2.1
    exact bus.programOrder _ _ _ (call.sameThread _ atRead) (call.used _ atRead)
  · intro n
    exact nowrap r first.1 (r.issued_at ran (path.final _ n)).1

private theorem round_sites (s : Supply) (q : ProbeRetry.Key) (e : Bool) (r : Round)
    (atStart : Placed s allocation clusters key e q start r) (run : Runs issued start (r.emit s e q)) :
    Runs issued r.front.opening (r.front.emit s (snapCode e)) ∧
      ∀ b ∈ r.back, Runs issued b.opening (b.emit s (snapCode e)) := by
  rw [r.emit_slots s e q atStart.branch] at run
  obtain ⟨front,back⟩ := run.append.1.append.1.append
  refine ⟨atStart.opening ▸ front,?_⟩
  intro b present
  have eq : r.back = some b := present
  simpa [eq,(atStart.back b eq).2,Site.next,atStart.opening] using back

private theorem sweep_sites (s : Supply) (allocation : Nat) (clusters key : SeqLock.Word) (e : Bool)
    (q : ProbeRetry.Key) (fuel start : Nat) (last : Fin 2) :
    let out := sweep s allocation clusters key e q fuel start last
    Runs issued start (out.emit s e q) → ∀ r ∈ out.rounds,
      Runs issued r.front.opening (r.front.emit s (snapCode e)) ∧
      ∀ b ∈ r.back, Runs issued b.opening (b.emit s (snapCode e)) := by
  induction fuel generalizing start last with
  | zero => simp [sweep]
  | succ n ih =>
    let r := roundAt s allocation clusters key e q start
    simp only [sweep,←show r = roundAt s allocation clusters key e q start from rfl]
    cases branch : (r.pass s e q).exit with
    | done _ =>
      intro run row member
      obtain rfl : row = r := List.mem_singleton.mp member
      exact round_sites _ _ _ _ (placed ..) (by simpa [Probed.emit] using run)
    | retry fallback =>
      intro run row member
      let tail := sweep s allocation clusters key e q n (start+(r.emit s e q).length) (fallback.getD last)
      have parts : Runs issued start (r.emit s e q ++ tail.emit s e q) := run
      rcases List.mem_cons.mp member with head | later
      · exact head ▸ round_sites _ _ _ _ (placed ..) parts.append.1
      · exact ih _ _ parts.append.2 _ later

/-- One native instruction tape covers all retries and both slots. -/
theorem probe_sites (bus : Accesses x) (call : Invocation bus) (allocation : Nat)
    (clusters key : SeqLock.Word) (b : Position) :
    let reader := call.reader bus
    let s := reader.supply x
    let p := probe s allocation clusters key endian (Cutoffs.key b) 0
    let ops := p.emit s endian (Cutoffs.key b)
    Runs (call.Issues bus) 0 ops → call.NoWrap bus ops → ∀ r ∈ p.rounds,
      CausalReads.Issued x reader call.consumer r.front (snapCode endian) ∧
        ∀ back ∈ r.back, CausalReads.Issued x reader call.consumer back (snapCode endian) := by
  intro reader s p ops ran nowrap r member
  have annotated : Runs (fun i op => ops[i]? = some op ∧ call.Issues bus i op) 0 ops := by
    exact fun i op hit => ⟨by simpa,ran i op hit⟩
  obtain ⟨front,back⟩ := sweep_sites s _ _ _ _ _ _ 0 0 annotated r member
  exact ⟨call.site bus _ (snapshot_path endian) ops front nowrap,fun k h =>
    call.site bus _ (snapshot_path endian) ops (back k h) nowrap⟩

/-- A lookup returns after the first hit or tries the second entry after a miss. -/
theorem lookup_sites (bus : Accesses x) (call : Invocation bus) (allocation : Nat)
    (clusters key : SeqLock.Word) (b : Position) :
    let reader := call.reader bus
    let s := reader.supply x
    let out := lookup s allocation clusters key endian b 0
    let ops := out.emit s endian b
    Runs (call.Issues bus) 0 ops → call.NoWrap bus ops →
      CausalReads.Issued x reader call.consumer out.front (lookupCode endian b) ∧
      ∀ back ∈ out.back, CausalReads.Issued x reader call.consumer back (lookupCode endian b) := by
  intro reader s out ops ran nowrap
  let front : Site := ⟨slotAt allocation clusters key 0,0⟩
  let back : Site := ⟨slotAt allocation clusters key 1,front.next s (lookupCode endian b)⟩
  have annotated : Runs (fun i op => ops[i]? = some op ∧ call.Issues bus i op) 0 (out.emit s endian b) :=
    fun i op hit => ⟨by simpa using hit,ran i op hit⟩
  cases hit : lookupValue endian b (front.words s) with
  | some v =>
    have shape : out = ⟨front,none⟩ := by change (match lookupValue endian b (front.words s) with | some _ => _ | none => _) = _; rw [hit]
    rw [shape] at annotated ⊢
    exact ⟨call.site bus _ (lookup_path endian b) ops (by simpa [Looked.emit,front] using annotated) nowrap,by simp⟩
  | none =>
    have shape : out = ⟨front,some back⟩ := by change (match lookupValue endian b (front.words s) with | some _ => _ | none => _) = _; rw [hit]
    rw [shape] at annotated ⊢
    obtain ⟨f,rest⟩ := annotated.append
    refine ⟨call.site bus _ (lookup_path endian b) ops f nowrap,?_⟩
    intro later member
    obtain rfl : back = later := by simpa using member
    exact call.site bus _ (lookup_path endian b) ops (by simpa [back,Site.next,front] using rest) nowrap

end NeuralReversi.PvsParallel.Interleaving.ReaderJournal
