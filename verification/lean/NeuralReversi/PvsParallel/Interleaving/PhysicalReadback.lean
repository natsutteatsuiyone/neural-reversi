import NeuralReversi.PvsParallel.Interleaving.JournalAtomicGraph
import NeuralReversi.PvsParallel.Interleaving.PhysicalCalls
import NeuralReversi.PvsParallel.Interleaving.WriterErasure

namespace NeuralReversi.PvsParallel.Interleaving.PhysicalReadback

open TranspositionTable
open NativeTableWrites.Journal (Calls Record)
open JournalAtomicGraph (Graph Rules execution)

/-- Forget the proof operands attached to a TT store and the ghost source rank
    on a load. Keep the physical addresses, clocks, words and all frame events. -/
def rawEvent : Joint.Event Operand → Joint.Event Unit
  | .engine (.other w) => .engine (.other w)
  | .engine (.beginWrite w k _) => .engine (.beginWrite w k ())
  | .engine (.atomic w k) => .engine (.atomic w k)
  | .engine (.boundaryScore w p t b a v) => .engine (.boundaryScore w p t b a v)
  | .engine (.localCall w e args) => .engine (.localCall w e args)
  | .poolHistory b n xs => .poolHistory b n xs
  | .rootEnter w t s r a rows => .rootEnter w t s r a rows
  | .iterateEnd w t o b v it rows => .iterateEnd w t o b v it rows
  | .solveEnter w o s r args => .solveEnter w o s r args
  | .probWord w r => .probWord w r
  | .splitWord w t op => .splitWord w t op
  | .taskWord w t op => .taskWord w t op
  | .maskWord w t op => .maskWord w t op
  | .workerLock w t p acquired => .workerLock w t p acquired
  | .alphaWord w t p b op => .alphaWord w t p b op
  | .ownerStore w t p b op => .ownerStore w t p b op
  | .ownerLoad w t p b src op => .ownerLoad w t p b src op
  | .epochWord w t op => .epochWord w t op
  | .abortWord w t op => .abortWord w t op
  | .helperAbortWord w t op => .helperAbortWord w t op
  | .enter w a => .enter w a
  | .leave w v => .leave w v
  | .seedRegister w b r => .seedRegister w b r
  | .fallbackRegister w b r => .fallbackRegister w b r
  | .issue w op _ v => .issue w op 0 v

private theorem engineEvent (e : Joint.Event Op) :
    (rawEvent e).engineEvent = e.engineEvent.erase := by
  cases e <;> try rfl
  rename_i e
  cases e <;> rfl

private theorem views (es : Joint.Tape Op) (r : Nat) :
    Joint.opcode (es.map rawEvent) r = Joint.opcode es r ∧
    Joint.word (es.map rawEvent) r = Joint.word es r ∧ Joint.worker (es.map rawEvent) r = Joint.worker es r := by
  unfold Joint.opcode Joint.word Joint.worker
  rw [List.getElem?_map]
  cases es[r]? with
  | none => exact ⟨rfl,rfl,rfl⟩
  | some e =>
    cases e <;> try exact ⟨rfl,rfl,rfl⟩
    rename_i e
    cases e <;> exact ⟨rfl,rfl,rfl⟩

/-- The physical writer prefix can be decoded before any score certificate is
    attached to its save operands. -/
structure Prefix (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  calls : Calls
  endian : Bool
  events : Joint.Tape Unit
  state : CallJournal.State Unit
  ran : CallJournal.run mc fc calls endian (events.map Joint.Event.engineEvent) = some state
  graph : Graph
  rules : Rules graph calls endian state.records events

theorem Prefix.facts (p : Prefix mc fc) : WriterSteps.Facts p.calls p.events p.state.records :=
  WriterSteps.run p.ran
def Prefix.memory (p : Prefix mc fc) := JournalAtomicGraph.execution p.facts p.rules

/-- Resolve a Rust semantic source event to its location's modification rank.
    An invalid or missing source is rejected by the load clause of Rules. -/
def Prefix.rank (p : Prefix mc fc) (r : Nat) : Nat :=
  (do
    let some (.load a _ _) := Joint.opcode p.events r | none
    let w ← p.graph.source a r
    let (_,i) ← ModificationIndex.decode p.calls p.state.records p.events.length a w
    return i).getD 0

def Prefix.attach (p : Prefix mc fc) (es : Joint.Tape Op) := AtomicReadback.annotate es p.rank

theorem Prefix.atomics (p : Prefix mc fc) (es : Joint.Tape Op) (same : es.map rawEvent = p.events) :
    Joint.Atomics p.memory (p.attach es) := by
  have phys := JournalAtomicGraph.observed p.facts p.rules
  have close r := same ▸ views es r
  have obs : AtomicReadback.Observed p.memory es := {
    reads := fun r a l m load => by
      obtain ⟨rf,value⟩ := phys.reads r a l m ((close r).1.trans load)
      exact ⟨rf,value.trans (close r).2.1⟩
    acquire := fun r a l load => phys.acquire r a l ((close r).1.trans load)
    fence := fun r f => phys.fence r ((close r).1.trans f)
    sb := fun a i j lt inside worker => phys.sb a i j lt (by simpa [←same] using inside)
      ((close i).2.2.trans (worker.trans (close j).2.2.symm)) }
  apply AtomicReadback.restored obs
  intro r a l m load
  have code := (close r).1.trans load
  obtain ⟨w,i,physical,decoded,_⟩ := p.rules.load _ _ _ _ code
  have timed : ModificationIndex.Timed p.calls p.state.records p.events.length :=
    ⟨fun _ => p.facts.bound,fun _ _ => p.facts.injective,p.facts.distinct⟩
  obtain ⟨_,stamp⟩ := (ModificationIndex.decode_eq_some_iff timed).mp decoded
  have rank : p.rank r = i := by simp [Prefix.rank,code,physical,decoded]
  rw [rank]
  subst w
  exact ⟨physical,l,i,decoded,Or.inl ⟨m,code⟩⟩

theorem Prefix.rank_le (p : Prefix mc fc) (r : Nat) : p.rank r ≤ p.state.records.length := by
  let bound := p.state.records.length
  unfold Prefix.rank
  cases issued : Joint.opcode p.events r with
  | none => exact Nat.zero_le _
  | some op =>
    cases op with
    | load a loc mode =>
      cases src : p.graph.source a r with
      | none => simp [src]
      | some w =>
        cases dec : ModificationIndex.decode p.calls p.state.records p.events.length a w with
        | none => simp [src,dec]
        | some value =>
          obtain ⟨l,i⟩ := value
          have timed : ModificationIndex.Timed p.calls p.state.records p.events.length :=
            ⟨fun _ => p.facts.bound,fun _ _ => p.facts.injective,p.facts.distinct⟩
          have inside := ((ModificationIndex.decode_eq_some_iff timed).mp dec).1
          have out : (NativeTableWrites.Journal.mods p.calls p.state.records a l).length ≤ bound :=
            List.length_filterMap_le ..
          simpa [issued,src,dec] using Nat.le_trans inside out
    | _ => exact Nat.zero_le _

/-- A short journal discharges the local no-wrap requirement even if the
    initial sequence word was close to its maximum. Longer histories can use
    the usual per-read condition directly. -/
theorem Prefix.noWrap_of_length (p : Prefix mc fc) (es : Joint.Tape Op) (same : es.map rawEvent = p.events)
    (short : p.state.records.length < SeqLock.modulus) : PhysicalCalls.NoWrap (p.atomics es same) := by
  intro finish consumer opening args use _ _ call site _ _
  change Joint.origin (p.attach es) _ < Joint.origin (p.attach es) _ + SeqLock.modulus
  apply Nat.lt_of_lt_of_le _ (Nat.le_add_left _ _)
  exact Nat.lt_of_le_of_lt (Nat.le_trans (AtomicReadback.origin_le _ _ _) (p.rank_le _)) short

def Contract := ∀ mc fc (p : Prefix mc fc) (Op : Type) (es : Joint.Tape Op), es.map rawEvent = p.events →
  Joint.Atomics p.memory (p.attach es) ∧
    (p.attach es).map Joint.Event.engineEvent = es.map Joint.Event.engineEvent

theorem fromPhysicalEvents : Contract := fun _ _ p _ es same =>
  ⟨p.atomics es same,AtomicReadback.engineEvents es p.rank⟩

/-- Construct the actual public-replay input using writer history and semantic
    read identities decoded from the same physical tape as the scored frames. -/
def Prefix.toRaw (p : Prefix mc fc) (init : ∀ a, NativeTableWrites.Journal.Even (p.memory.memory a))
    (resident : ∀ a, Entry.Resident mc fc p.endian ((p.memory.slot a (init a)).payloadAt 0))
    (es : Joint.Tape (CallJournal.Operand p.calls p.endian (CallJournal.observed p.memory init)))
    (same : es.map rawEvent = p.events) (s : CallJournal.State _)
    (ran : CallJournal.decode mc fc p.calls p.endian (CallJournal.observed p.memory init)
      (es.map Joint.Event.engineEvent) = some s)
    (ordered : ∀ i j, JournalAtomicGraph.HB p.graph p.events i j → i < j)
    (nowrap : PhysicalCalls.NoWrap (p.atomics es same)) (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool)
    (owners : JournalCoverage.Owners mc fc (p.attach es) s) (root : CausalTable.Frame → JournalRootStore.Options)
    (splits : (f : CausalTable.Frame) → JournalRootStore.Splits (p.attach es) mc fc f.board) : PhysicalCalls.Raw mc fc where
  calls := p.calls
  endian := p.endian
  records := p.state.records
  memory := p.memory
  init := init
  resident := resident
  events := p.attach es
  state := s
  replay := by rw [Prefix.attach,AtomicReadback.engineEvents]; exact ran
  recorded := by
    apply WriterErasure.sameRecords ran
    have h := p.ran
    rw [←same,List.map_map] at h
    simpa only [List.map_map,Function.comp_def,engineEvent] using h
  order := fun _ => ordered
  atomics := p.atomics es same
  nowrap := nowrap
  pool := pool
  checked := checked
  owners := owners
  root := root
  rootSplits := splits

end NeuralReversi.PvsParallel.Interleaving.PhysicalReadback
