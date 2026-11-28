module
public import NeuralReversi.PvsParallel.Interleaving.CallJournal
public import NeuralReversi.PvsParallel.SerialTable.MidgameNative

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.CallJournal

open NativeTableWrites.Journal (Calls Even)
open PvsBoardTree (Position)
open TranspositionTable
open SeqLock (Payload)
open FrameTableReads (saveObservation)

variable {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}

/-- The load site for a writer's old snapshot. The memory event addresses and
    read-from indices belong to the same physical slot as its subsequent CAS. -/
structure SnapshotSite (x : NativeTableWrites.Journal.Execution c endian rs) (k : Nat) where
  event : Nat → Nat
  source : Nat → Nat
  steps : let m := x.memory (c k).address
    ReaderMir.Realizes m (ReaderMir.run endian (m.readValue ∘ event)).actions event source
  beforeClaim : let m := x.memory (c k).address
    m.sb (event ((ReaderMir.run endian (m.readValue ∘ event)).actions.length-1)) ((c k).event 0)
  earlier : (x.memory (c k).address).sb (event 0) ((c k).event 5)
  noWrap : let m := x.memory (c k).address
    (ReaderMir.run endian (m.readValue ∘ event)).actions.length = 6 → source 5 < source 0 + SeqLock.modulus

def SnapshotSite.words (r : SnapshotSite x k) := (x.memory (c k).address).readValue ∘ r.event

theorem SnapshotSite.older (r : SnapshotSite x k) (s : SeqLock.Snapshot)
    (got : (ReaderMir.run endian r.words).returned = some s)
    (occupied : (Packing.unpack endian s.metadata).bound ≠ 0) : observed x init k ⟨s.player,s.opponent,s.metadata⟩ := by
  have length : (ReaderMir.run endian r.words).actions.length = 6 := by
    unfold ReaderMir.run at *
    repeat' split at got <;> first | contradiction | skip
    all_goals simp_all
    subst s
    contradiction
  have hit := Table.seen_snapshot x (c k).address (init _) r.event r.source r.steps got
    (by simpa [Snapshot.empty] using occupied) (r.noWrap length)
  exact hit.later ((x.memory _).sb_hb r.earlier)

theorem sameSnapshot (endian : Bool) (words : Nat → SeqLock.Word) (current : SeqLock.Word)
    (got : let obs := saveObservation endian words current
      Snapshot.read endian obs.first obs.last obs.sample = some s) (occupied : (Packing.unpack endian s.metadata).bound ≠ 0) :
    (ReaderMir.run endian words).returned = some s := by
  obtain ⟨even,same,metadata,sequence,keys⟩ := (Snapshot.readContract_proven ..).mp got
  by_cases empty : (Packing.unpack endian (words 1)).bound = 0
  · simp [saveObservation,empty] at metadata
    exact (occupied (metadata ▸ empty)).elim
  · simp only [saveObservation,ite_eq_right empty] at even same metadata sequence keys
    have full : Snapshot.empty endian (words 1) = false := by simpa [Snapshot.empty] using empty
    simp only [full,Bool.false_eq_true,ite_false] at keys
    obtain ⟨player,opponent⟩ := keys
    cases s
    simp_all [ReaderMir.run]

def SnapshotSite.save (r : SnapshotSite x k) (a : StoreArgs) : Operand c endian (observed x init) :=
  .save k a (saveObservation endian r.words (c k).expected.val) fun s _ _ got _ occupied =>
    r.older s (sameSnapshot endian r.words (c k).expected.val got occupied) occupied

def SnapshotSite.penalty (r : SnapshotSite x k) (b : Position) (e : EntryData) : Operand c endian (observed x init) :=
  .penalty k b e (ReaderMir.run endian r.words).returned (c k).expected.val (fun s got occupied => r.older s got occupied)

structure SequenceSite (x : NativeTableWrites.Journal.Execution c endian rs) (k : Nat) where
  event : Nat
  source : Nat
  loaded : let m := x.memory (c k).address
    m.rf (m.write .sequence source) event
  acquire : (x.memory (c k).address).acquire event
  beforeClaim : (x.memory (c k).address).sb event ((c k).event 0)

/-- On an exact save the emitted code needs only the sequence and does not inherit
    any move byte. -/
def exactSave (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a)) (k : Nat) (a : StoreArgs)
    (exactTag : a.bound = .exact) (site : SequenceSite x k) : Operand c endian (observed x init) :=
  let seq := (x.memory (c k).address).readValue site.event
  .save k a ⟨seq,seq,⟨0,0,0⟩,(c k).expected.val⟩ fun _ notExact => (notExact (by
    change Packing.boundByte a.bound = Packing.boundByte .exact
    rw [exactTag])).elim

/-- Every real exact-bound word returned by the serial native midgame interpreter
    satisfies the seed-range check, including words from its selective PV calls. -/
theorem midgame_exact_range (p : EndgameSearch.CacheHistory.Pool) (ops : SerialTable.MidgameNative.Inputs)
    (root : Position) (q : SerialCalls.Request) (m : SerialTable.Memory l) (window : Aspiration.Window q.alpha q.beta)
    (ready : AspirationMir.SerialTable.Repeated.ReadyMemory p mc fc m)
    (run : SerialTable.MidgameNative.runFromRoot (mc := mc) (fc := fc) p ops root q m = some (v,next))
    (tag : classify q.pv (SerialTable.Writes.word v) (SerialTable.Writes.word q.alpha) (SerialTable.Writes.word q.beta) = .exact) :
    SeedTable.Range v := by
  obtain ⟨word,after,executed,_,finite,range⟩ := SerialTable.MidgameNative.returnsFromRoot p ops root q m window ready
  obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj (run.symm.trans executed))
  exact SerialTable.MidgameMemory.classified q _ window finite range tag

/-- The presearch and ETC explicit lower stores use this same classifier result
    for every successful cutoff on a null window. -/
theorem scout_lower (a z score : Score) (null : z.raw.toInt = a.raw.toInt+1) (hit : a.raw.toInt < score.raw.toInt) :
    classify false score a z = .lower :=
  ((classify_contract ..).1).2 (by omega)

namespace Regression

private def board : Position := ⟨⟨0,0⟩,rfl⟩
private def word (v : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 v⟩
private def args : NativeFrames.Args := ⟨board.val,8,word (-32767),word 32767,false,Cutoffs.exactSelectivity,true,false,none⟩
private def saveArgs (v : Int) : StoreArgs := ⟨board,⟨(word v).raw⟩,.exact,8,none,Cutoffs.exactSelectivity,0,false,true⟩

private def calls (v : Int) : Calls := fun _ =>
  ⟨⟨0,0⟩,⟨0,rfl⟩,.store none board (saveArgs v).data,true,fun pc => if pc = 0 then 3 else pc+6⟩

private def seen (_ : Nat) (_ : Payload) : Prop := False
private def raw (v : Int) : Operand (calls v) false seen :=
  .save 0 (saveArgs v) ⟨0,0,⟨0,0,0⟩,0⟩ (fun _ different => (different rfl).elim)

/-- The scoring worker enters a save, a second worker returns, and the first
    worker publishes its original word before abandoning its own return. -/
private def interleaved (saved owner other : Int) : List (Event (Operand (calls saved) false seen)) := [
  .localCall 0 (.entry 0 0 .pvs .none none) (some args), .localCall 0 (.scored 1 (word owner)) none,
  .beginWrite 0 0 (raw saved), .atomic 0 0,
  .localCall 1 (.entry 0 0 .pvs .none none) (some args), .localCall 1 (.scored 1 (word other)) none,
  .localCall 1 (.leave 2 0 .pvs none (some (word other))) none,
  .atomic 0 0, .atomic 0 0, .atomic 0 0, .atomic 0 0, .atomic 0 0, .other 2,
  .localCall 0 (.leave 2 0 .pvs none none) none]

private def finish (i : Issued α) : Bool := match i with | .atomic _ .finish => true | _ => false

private def outcome mc fc (saved owner other : Int) (length : Nat) :=
  (decode mc fc (calls saved) false seen ((interleaved saved owner other).take length)).map
    (fun s => (s.issued.any finish,s.closed.length,s.frames.length))

example (mc fc) : outcome mc fc 27 27 30 14 = some (true,1,0) := by rfl
example (mc fc) : outcome mc fc 30 27 30 14 = none := by rfl
example (mc fc) : outcome mc fc 16385 16385 30 14 = none := by rfl
example (mc fc) : outcome mc fc 27 27 30 4 = some (false,0,0) := by rfl

end Regression

end NeuralReversi.PvsParallel.Interleaving.CallJournal
