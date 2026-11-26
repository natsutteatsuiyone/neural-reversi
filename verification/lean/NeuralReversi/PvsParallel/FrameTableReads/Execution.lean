import NeuralReversi.PvsParallel.FrameTableReads.Instructions
import NeuralReversi.PvsParallel.NativeTableWrites.Journal

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution

open TranspositionTable
open SeqLock (Word Location)
open FrameReturns (Log Frame)

/-- Raw atomic instructions carry their actual address. The acquire fence has
    no address. The reader's opening index is reconstructed by the MIR. -/
inductive Instruction where
  | load (addr : Address) (loc : Location) (order : ReaderMir.Ordering)
  | fence (order : ReaderMir.Ordering)
  | generation (allocation : Nat)
  | spin
  deriving DecidableEq

def instruction : Op → Instruction
  | .slot site (.load loc order) => .load site.address loc order
  | .slot _ (.fence order) => .fence order
  | .generation a => .generation a
  | .spin => .spin

structure Atomic where
  id : Nat
  instruction : Instruction
  /-- Modification-order index of the write supplying a load. -/
  source : Nat
  word : Word

structure Caller where
  worker : Nat
  opening : Nat
  deriving DecidableEq

inductive Record where
  | search (r : NativeTableWrites.Journal.Record)
  | reader (caller : Caller) (op : Atomic)

def selected (caller : Caller) (rs : List Record) := rs.filterMap fun
  | .reader c r => if c = caller then some r else none
  | .search _ => none

def journal (rs : List Record) := rs.filterMap fun | .search r => some r | .reader .. => none

def supply (rs : List Atomic) : Supply where
  word _ i := (rs[i]?.map Atomic.word).getD 0
  generationByte _ i := ((rs[i]?.map Atomic.word).getD 0).setWidth 8

/-- Replay the probe from the words returned by its loads. Check the executed
    addresses, fences, spins and generation reads, including the end of the call. -/
def readProbe (allocation : Nat) (clusters key : Word) (endian : Bool) (q : ProbeRetry.Key)
    (registers : List Atomic) : Option ProbeRetry.Result :=
  let s := supply registers
  let p := probe s allocation clusters key endian q 0
  if registers.map Atomic.instruction = (p.emit s endian q).map instruction then some p.result else none

structure ProbeCapture (q : ProbeRetry.Key) where
  allocation : Nat
  clusters : Word
  key : Word
  endian : Bool
  registers : List Atomic
  result : ProbeRetry.Result
  completed : readProbe allocation clusters key endian q registers = some result

/-- Rust atomic semantics, independent of the branch taken by probe or lookup.
    Load values are constrained by the actual write at their modification-order
    index. Sequenced-before uses the projected program order of a single caller. -/
structure Atomics {log : Log mc fc} (t : Table log) (consumer : Nat) (rs : List Atomic) : Prop where
  read : ∀ a ∈ rs, ∀ addr loc mode, a.instruction = .load addr loc mode →
    (t.memory addr).rf ((t.memory addr).write loc a.source) a.id ∧ a.word = (t.memory addr).writeValue loc a.source
  acquire : ∀ a ∈ rs, ∀ addr loc, a.instruction = .load addr loc .acquire → (t.memory addr).acquire a.id
  fence : ∀ a ∈ rs, a.instruction = .fence .acquire → ∀ addr, (t.memory addr).acquireFence a.id
  next : ∀ i a b, rs[i]? = some a → rs[i+1]? = some b → ∀ addr, (t.memory addr).sb a.id b.id
  ready : ∀ a ∈ rs, ∀ addr, a.instruction = .load addr .sequence .acquire → log.before (t.ready addr) a.id
  used : ∀ a ∈ rs, ∀ addr, a.instruction = .load addr .sequence .relaxed → (t.memory addr).sb a.id consumer

def issued (ops : List Op) (rs : List Atomic) (i : Nat) (op : Op) : Prop :=
  ops[i]? = some op ∧ ∃ a, rs[i]? = some a ∧ a.instruction = instruction op

def NoWrap (ops : List Op) (rs : List Atomic) : Prop :=
  ∀ r a b, ops[r.opening]? = some (.slot r (.load .sequence .acquire)) →
    ops[r.opening+5]? = some (.slot r (.load .sequence .relaxed)) →
    rs[r.opening]? = some a → rs[r.opening+5]? = some b → b.source < a.source + SeqLock.modulus

private theorem Atomics.before_register {t : Table log} (actual : Atomics t consumer rs)
    (addr : Address) (before : ∀ a, rs[0]? = some a → (t.memory addr).hb w a.id)
    {i : Nat} {load : Atomic} (atLoad : rs[i]? = some load) : (t.memory addr).hb w load.id := by
  induction i generalizing load with
  | zero => exact before _ atLoad
  | succ i ih =>
    have valid := (List.getElem?_eq_some_iff.mp atLoad).1
    have prior := List.getElem?_eq_getElem (show i < rs.length by omega)
    exact (t.memory addr).hb_trans (ih prior) ((t.memory addr).sb_hb (actual.next _ _ _ prior atLoad addr))

/-- The last modification before the interval is visible at entry. If more
    writes follow the interval, the consumer precedes the next successful CAS. -/
structure Boundaries {m : SeqLock.Memory Nat} (slot : WriterTrace.Sequence m) (consumer : Nat)
    (rs : List Atomic) (left right : Nat) : Prop where
  start : 0 < left → ∀ first, rs[0]? = some first → m.hb (m.write .sequence left) first.id
  finish : right < slot.history.length → ∃ c, slot.history[right]? = some (.claim c) ∧ m.hb consumer c.cas

theorem Atomics.noWrap_of_window {t : Table log} (actual : Atomics t consumer rs)
    (same : rs.map Atomic.instruction = ops.map instruction)
    (window : ∀ r, Op.slot r (.load .sequence .relaxed) ∈ ops → ∃ left right,
      Boundaries (t.slot r.address).toSequence consumer rs left right ∧ right < left + SeqLock.modulus) : NoWrap ops rs := by
  intro r a b first final initial got
  obtain ⟨left,right,edges,small⟩ := window r (List.mem_of_getElem? final)
  have opAt (i : Nat) (x : Atomic) (op : Op) (here : rs[i]? = some x) (issued : ops[i]? = some op) :
      x.instruction = instruction op := by
    have projected := congrArg (fun xs : List Instruction => xs[i]?) same
    simpa only [List.getElem?_map,here,issued,Option.map_some,Option.some.injEq] using projected
  have lower : left ≤ a.source := by
    by_cases positive : 0 < left
    · exact (t.memory r.address).write_read (actual.before_register _ (edges.start positive) initial)
        (actual.read _ (List.mem_of_getElem? initial) _ _ _ (opAt _ _ _ initial first)).1
    · omega
  have lastOp := opAt _ _ _ got final
  have member := List.mem_of_getElem? got
  have upper := (t.slot r.address).read_before_claim b.source right b.id (actual.read _ member _ _ _ lastOp).1
    (fun more => by
      obtain ⟨c,claim,ahead⟩ := edges.finish more
      exact ⟨c,claim,(t.memory r.address).hb_trans ((t.memory r.address).sb_hb (actual.used _ member _ lastOp)) ahead⟩)
  omega

/-- Taking the whole history requires no synchronization facts at either end. -/
theorem Atomics.noWrap_of_slot_lengths {t : Table log} (actual : Atomics t consumer rs)
    (same : rs.map Atomic.instruction = ops.map instruction)
    (short : ∀ r, Op.slot r (.load .sequence .relaxed) ∈ ops →
      (t.slot r.address).history.length < SeqLock.modulus) : NoWrap ops rs :=
  actual.noWrap_of_window same (fun r valid => ⟨0,(t.slot r.address).history.length,⟨by simp,by simp⟩,by simpa using short r valid⟩)

theorem run_of_instructions (ops : List Op) (rs : List Atomic) (same : rs.map Atomic.instruction = ops.map instruction) :
    Runs (issued ops rs) 0 ops := by
  intro i op atIndex
  have actual := congrArg (fun xs : List Instruction => xs[i]?) same
  simp only [List.getElem?_map,atIndex,Option.map_some] at actual
  obtain ⟨a,found,what⟩ := Option.map_eq_some_iff.mp actual
  simpa only [Nat.zero_add] using (show issued ops rs i op from ⟨atIndex,⟨a,found,what⟩⟩)

def machine {log : Log mc fc} {t : Table log} (ops : List Op) (rs : List Atomic)
    (actual : Atomics t consumer rs) (noWrap : NoWrap ops rs) (s : Supply)
    (reported : ∀ r i loc mode a, ops[i]? = some (.slot r (.load loc mode)) → rs[i]? = some a → s.word r.address i = a.word) :
    Machine t consumer (issued ops rs) s where
  event i := (rs[i]?.map Atomic.id).getD 0
  origin i := (rs[i]?.map Atomic.source).getD 0
  word := by
    rintro r i loc mode ⟨effect,a,atIndex,eq⟩
    obtain ⟨rf,value⟩ := actual.read a (List.mem_of_getElem? atIndex) r.address loc mode eq
    simpa only [atIndex,Option.map_some,Option.getD_some] using (reported _ _ _ _ _ effect atIndex).trans (value.trans ((t.memory r.address).rf_value rf).symm)
  reads := by
    rintro r i loc mode ⟨_,a,atIndex,eq⟩
    simpa [atIndex] using (actual.read a (List.mem_of_getElem? atIndex) r.address loc mode eq).1
  acquire := by
    rintro r i loc ⟨_,a,atIndex,eq⟩
    simpa [atIndex] using actual.acquire a (List.mem_of_getElem? atIndex) r.address loc eq
  fence := by
    rintro r i ⟨_,a,atIndex,eq⟩
    simpa [atIndex] using actual.fence a (List.mem_of_getElem? atIndex) eq r.address
  adjacent := by
    rintro r i x y ⟨_,a,one,_⟩ ⟨_,b,two,_⟩
    simpa [one,two] using actual.next i a b one two r.address
  allocated := by
    rintro r ⟨_,a,atIndex,eq⟩
    simpa [atIndex] using actual.ready a (List.mem_of_getElem? atIndex) r.address eq
  consumed := by
    rintro r ⟨_,a,atIndex,eq⟩
    simpa [atIndex] using actual.used a (List.mem_of_getElem? atIndex) r.address eq
  noWrap := by
    rintro r ⟨op1,a,at1,_⟩ ⟨op2,b,at2,_⟩
    simpa [at1,at2] using noWrap r a b op1 op2 at1 at2

def fromRegisters {log : Log mc fc} {t : Table log} (ops : List Op) (rs : List Atomic)
    (actual : Atomics t consumer rs) (noWrap : NoWrap ops rs) : Machine t consumer (issued ops rs) (supply rs) :=
  machine ops rs actual noWrap _ (fun _ _ _ _ _ _ observed => by simp [supply,observed])

end NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution
