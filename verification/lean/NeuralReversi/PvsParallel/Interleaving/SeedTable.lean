module
public import NeuralReversi.AspirationMir.SerialTable.InitialEstimate
public import NeuralReversi.PvsParallel.Interleaving.CausalReads
public import NeuralReversi.PvsParallel.Interleaving.FiniteTable

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.SeedTable

open PvsMoveLoop (scoreMax)
open PvsBoardTree (Position)
open TranspositionTable
open NativeTableWrites.Journal (Even)
open AspirationMir.SerialTable.InitialEstimate (fallback fallback_bounded)

def Range (score : Int) := -scoreMax ≤ score ∧ score ≤ scoreMax

def Incoming (_ : Position) (e : EntryData) : Prop := e.bound = Packing.boundByte .exact → Range e.score.toInt

abbrev Words := FiniteTable.FieldWords (fun bound score => bound = Packing.boundByte .exact → Range score.toInt)
abbrev Stores (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a)) :=
  FiniteTable.Releases x init Incoming

/-- The native score classifier supplies the narrower range only for an exact bound.
    A fail-low or fail-high need only pass the store's ordinary finite-word guard. -/
theorem ofStore {slot : Nat} (selected : bound = .exact → Range score.raw.toInt)
    (accepts : WriterMir.storeInput slot b score bound depth move selectivity generation endgame pv = some (slot,b,e)) :
    Incoming b e := by
  rw [WriterMir.store_input] at accepts
  split at accepts
  next finite =>
    cases Option.some.inj accepts
    intro byte
    change Packing.boundByte bound = Packing.boundByte .exact at byte
    have tag : bound = .exact := by cases bound <;> first | rfl | contradiction
    have range := selected tag
    change Range (score.raw.setWidth 16).toInt
    rwa [StoredBound.narrow_score score (by omega) (by omega)]
  next => contradiction

theorem zero : Words endian ⟨0,0,0⟩ := by
  intro occupied
  cases endian <;> exact False.elim (occupied rfl)

theorem warm (t : SerialTable.Table l) (supported : t.image.Supported t.history) (wide : SerialTable.ExactScores t.history)
    (i : l.Index) : Words l.bigEndian (SerialTable.payload (t.image.get i)) := by
  intro occupied
  obtain ⟨b,member,_,_⟩ := supported.resident i occupied
  exact wide b _ member

theorem lookup {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : Stores x init) (seed : ∀ a, Words endian ((x.slot a (init a)).payloadAt 0))
    (hit : CausalTable.lookup x init consumer b e) (full : e.bound = Packing.boundByte .exact) : Range e.score.toInt :=
  FiniteTable.fieldLookup stores seed (fun _ _ source => source) hit full

/-- The direct endgame aspiration seed uses a matching exact-bound word at any
    selectivity or phase. A miss or other bound requests the board evaluator. -/
def estimate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (raw : Position → Int)
    (b : Position) (probe : ProbeRetry.Result) : BitVec 32 :=
  let data := match probe with | .hit _ e => some e | .miss _ => none
  match data.filter (fun e => e.bound == Packing.boundByte .exact) with
  | some e => BitVec.ofInt 32 e.score.toInt
  | none => BitVec.ofInt 32 (fallback mc fc raw b)

theorem estimate_bounded {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : Stores x init) (seed : ∀ a, Words endian ((x.slot a (init a)).payloadAt 0))
    (r : CausalReads.Reader) (consumer allocation : Nat) (clusters key : SeqLock.Word) (start : Nat)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (raw : Position → Int) (b : Position) :
    let out := FrameTableReads.Operations.probe (r.supply x) allocation clusters key endian (Cutoffs.key b) start
    (∀ round ∈ out.rounds, CausalReads.Issued x r consumer round.front (FrameTableReads.snapCode endian) ∧
      ∀ back ∈ round.back, CausalReads.Issued x r consumer back (FrameTableReads.snapCode endian)) →
    Range (estimate mc fc raw b out.result).toInt := by
  intro out sites
  have repr (v : Int) (h : Range v) : Range (BitVec.ofInt 32 v).toInt := by
    have ⟨lo,hi⟩ := h
    rwa [BitVec.toInt_ofInt_eq_self (by decide : 0 < 32) (by unfold scoreMax at lo; omega)
      (by unfold scoreMax at hi; omega)]
  have miss := repr _ (fallback_bounded mc fc raw b)
  cases result : out.result with
  | miss slot => exact miss
  | hit slot e =>
    by_cases exactBound : e.bound = Packing.boundByte .exact
    · have matched := CausalReads.probe_hit x init r consumer allocation clusters key b start sites result
      simpa [estimate,Option.filter,exactBound] using repr _ (lookup stores seed matched exactBound)
    · simpa [estimate,Option.filter,exactBound] using miss

end NeuralReversi.PvsParallel.Interleaving.SeedTable
