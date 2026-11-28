module
public import NeuralReversi.PvsParallel.Interleaving.CausalTable
public import NeuralReversi.PvsParallel.SerialTable.Finite

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.FiniteTable

open TranspositionTable
open NativeTableWrites.Journal
open CausalTable (Values Lookups Frame)
open PvsBoardTree (Position)
open FrameTableReads (Address)
open SeqLock (Payload)

abbrev Finite := SerialCalls.ScoreWords.Finite

/-- Native TT::store rejects both sentinels, for every bound, generation and search phase. -/
def Passed (b : Position) (e : EntryData) : Prop :=
  ∃ (slot : Nat) (score : Score) (bound : Bound) (depth : Depth) (move : Option Bitboard.Square)
    (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool),
    WriterMir.storeInput slot b score bound depth move selectivity generation endgame pv = some (slot,b,e)

theorem Passed.finite (passed : Passed b e) : Finite e.score.toInt := by
  obtain ⟨slot,score,bound,d,m,sel,g,phase,pv,run⟩ := passed
  rw [WriterMir.store_input] at run
  split at run
  next ok =>
    cases Option.some.inj run
    have h : Finite score.raw.toInt := by unfold Finite SerialCalls.ScoreWords.Finite; omega
    change Finite (score.raw.setWidth 16).toInt
    rwa [StoredBound.narrow_score score h.1 h.2]
  next => contradiction

/-- Save may inherit a move, and penalize may change depth. Neither changes the
    bound byte and signed score supplied by their respective source. -/
def FieldWords (test : BitVec 8 → BitVec 16 → Prop) (endian : Bool) (p : Payload) : Prop :=
  let e := Packing.unpack endian p.metadata
  e.bound ≠ 0 → test e.bound e.score

private theorem saveFields (wrote : Save.attempt endian key incoming obs = .written seq e) :
    e.bound = incoming.bound ∧ e.score = incoming.score := by
  by_cases tag : incoming.bound = Packing.boundByte .exact
  · rw [Save.exactAttemptContract_proven _ _ _ _ tag] at wrote
    split at wrote <;> cases wrote
    exact ⟨rfl,rfl⟩
  · have ⟨_,_,_,_,made⟩ := ((Save.otherAttemptContract_proven endian _ _ obs tag).2 _ _).mp wrote
    rw [(Save.contract_proven ..).1] at made
    split at made
    · contradiction
    · cases Option.some.inj made
      unfold Save.payload
      split <;> exact ⟨rfl,rfl⟩

theorem fieldWritten {w : Attempt} (operands : Table.Operands endian supplied seen w.origin)
    (incoming : ∀ b e, supplied b e → test e.bound e.score)
    (prior : ∀ p, seen p → FieldWords test endian p) : FieldWords test endian (w.call endian).payload := by
  cases origin : w.origin with
  | clear =>
    intro used
    simp only [Attempt.call,Attempt.args,origin,Packing.unpack_pack] at used
    cases endian <;> contradiction
  | store frame b e =>
    rw [origin] at operands
    simp only [FieldWords,Attempt.call,Attempt.args,origin,Packing.unpack_pack]
    intro used
    cases operands with
    | save _ _ data admitted obs seq v wrote _ =>
      obtain ⟨same,word⟩ := saveFields wrote
      rw [same,word]
      exact incoming _ _ admitted
    | penalty _ _ prev s cur v wrote sampled =>
      obtain ⟨_,snap,atSnap,_,_,data,_,rfl⟩ := (Penalty.writeContract_proven ..).mp wrote
      have ok := prior _ (sampled snap atSnap (by rw [data]; exact used))
      simp only [FieldWords,data] at ok
      exact ok used

structure Releases (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (supplied : Position → EntryData → Prop) : Prop where
  operands : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).success = true →
    Table.Operands endian supplied (Table.Seen x (c k).address (init _) ((c k).event 5)) (c k).origin

theorem fieldLookup {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : Releases x init supplied) (seed : ∀ a, FieldWords test endian ((x.slot a (init a)).payloadAt 0))
    (incoming : ∀ b e, supplied b e → test e.bound e.score) (seen : CausalTable.lookup x init consumer b e) :
    test e.bound e.score := by
  obtain ⟨occupied,a,origin⟩ := seen
  have invariant (a : Address) (p : Payload) (source : Table.Seen x a (init a) consumer p) : FieldWords test endian p := by
    cases source with
    | initial => exact seed _
    | stored n k atIndex _ =>
      apply Table.publication_invariant x a (init a) (FieldWords test endian) (seed a) _ n k atIndex
      intro writer listed location success previous
      subst a
      exact fieldWritten (stores.operands writer listed success) incoming previous
  simpa only [FieldWords,SlotJournal.packed,Packing.unpack_pack] using invariant a _ origin (by
    simpa only [SlotJournal.packed,Packing.unpack_pack] using occupied)

abbrev Words := FieldWords (fun _ score => Finite score.toInt)
abbrev Stores (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a)) := Releases x init Passed

theorem warm (m : SerialTable.Memory l) (good : m.Valid p mc fc) (i : l.Index) :
    Words l.bigEndian (SerialTable.payload (m.tt.image.get i)) := by
  intro occupied
  obtain ⟨b,member,_,_⟩ := good.1.1.supported.resident i occupied
  exact good.2 b _ member

theorem zero : Words endian ⟨0,0,0⟩ := by
  intro occupied
  cases endian <;> exact False.elim (occupied rfl)

theorem written {w : Attempt} (operands : Table.Operands endian Passed seen w.origin)
    (prior : ∀ p, seen p → Words endian p) : Words endian (w.call endian).payload :=
  fieldWritten operands (fun _ _ source => source.finite) prior

/-- No returned-frame facts enter the physical slot invariant. The only source for
    a score on a fresh save is the compiled store-input gate; penalize copies the
    score observed at its matching, occupied snapshot. -/
theorem lookup {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : Stores x init) (seed : ∀ a, Words endian ((x.slot a (init a)).payloadAt 0))
    (seen : CausalTable.lookup x init consumer b e) : Finite e.score.toInt :=
  fieldLookup stores seed (fun _ _ source => source.finite) seen

def Preserves (rule : CausalTable.Rule) : Prop :=
  ∀ prior reads, (∀ b p a z v, prior b p a z v → Finite v) →
    (∀ b e, reads b e → Finite e.score.toInt) → ∀ b p a z v, rule prior reads b p a z v → Finite v

structure Prefix (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (frames : List Frame) (before : Nat → Nat → Prop) (rule : CausalTable.Rule) : Prop where
  irrefl : ∀ t, ¬before t t
  trans : ∀ {i j k}, before i j → before j k → before i k
  left : ∀ f ∈ frames, ∀ t, f.returned = some t → before f.chosen t
  body : ∀ f ∈ frames, rule (CausalTable.returned before frames f.chosen) (CausalTable.lookup x init f.chosen)
    f.board f.pv f.alpha f.beta f.score

/-- Weak or exact frame results may be consumed by another worker after return.
    Only recursive return dependencies need a temporal induction: any coherent
    engine TT read is finite independently of whether its producer eventually returns. -/
theorem certified {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : Stores x init) (seed : ∀ a, Words endian ((x.slot a (init a)).payloadAt 0))
    (run : Prefix x init frames before rule) (kernel : Preserves rule) :
    (∀ f ∈ frames, Finite f.score) ∧ (∀ t b e, CausalTable.lookup x init t b e → Finite e.score.toInt) := by
  have words := fun t b e (use : CausalTable.lookup x init t b e) => lookup stores seed use
  refine ⟨?_,words⟩
  let prior (t : Nat) := ∀ f ∈ frames, f.chosen = t → Finite f.score
  have certifiedAt : ∀ t ∈ frames.map Frame.chosen, prior t := by
    apply CausalTable.finiteInduction (frames.map Frame.chosen) before prior run.irrefl run.trans
    intro t _ ih f listed same
    subst t
    apply kernel _ _ _ (words _) _ _ _ _ _ (run.body f listed)
    intro b p a z v used
    obtain ⟨old,mem,ret,finished,earlier,board,kind,lo,hi,score⟩ := used
    rw [←score]
    exact ih _ (List.mem_map.mpr ⟨old,mem,rfl⟩) (run.trans (run.left old mem _ finished) earlier) old mem rfl
  exact fun f mem => certifiedAt _ (List.mem_map.mpr ⟨f,mem,rfl⟩) f mem rfl

end NeuralReversi.PvsParallel.Interleaving.FiniteTable
