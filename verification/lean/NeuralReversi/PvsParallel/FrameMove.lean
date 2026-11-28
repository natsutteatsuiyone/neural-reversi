module
public import NeuralReversi.PvsParallel.CausalReplay
public import NeuralReversi.PvsParallel.GeneralNonPvMir
public import NeuralReversi.MoveList.Generation

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameReturns

open PvsBoardTree (Position GeneratedSquare)
open TranspositionTable
open TableReads (LegalChoice LegalHints ProbeReadsFrom)
open MoveList.Generation (records IsSubset)

/-- Origins of the square operand of a TT store. A split read can choose the owner's
    initial square or an indexed move from a worker report. A presearch can reuse a TT hit. -/
inductive MoveOrigin (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (source : Cutoffs.RawSample → Prop) : Option Bitboard.Square → Prop
  | absent : MoveOrigin mc fc b source none
  | indexed (mask : Bitboard.Bits) (order : List Bitboard.Square) (idx : Nat) (sq : Bitboard.Square) :
      IsSubset mask (Board.getMoves mc b.val) →
      order.Perm ((records fc b.val mask).map Move.Move.sq) → order[idx]? = some sq →
      MoveOrigin mc fc b source (some sq)
  | tt (attempts : List ProbeRetry.RawAttempt) (slot : Fin 2) (entry : EntryData) :
      ProbeReadsFrom source attempts →
      ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot entry →
      MoveOrigin mc fc b source (Packing.decodeBestMove entry.bestMove)
  | enhanced (query : PvsEnhancedCutoff.Query) (reads : List (TableReads.ChildLookupFrom mc fc b source)) :
      PvsEnhancedCutoff.scan query (reads.map TableReads.ChildLookupFrom.observation) = some hit →
      MoveOrigin mc fc b source (some hit.1)
  | split (division : SplitExecution len pv a z score) (moveAt : Nat → Option Bitboard.Square)
      (seed picked : Option Bitboard.Square)
      (read : division.MoveRead Packing.encodeBestMove moveAt seed picked) :
      MoveOrigin mc fc b source seed → (∀ r ∈ division.reports, MoveOrigin mc fc b source (moveAt r.index)) →
      MoveOrigin mc fc b source (Packing.decodeBestMove ((read.writes.get read.load.source).value.setWidth 8))

theorem MoveOrigin.filled (mask : Bitboard.Bits) (skip paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (idx count : Nat) (mv : Move.Move)
    (subset : IsSubset mask (Board.getMoves mc b.val)) (nonempty : skip = true → mask ≠ 0)
    (len : (MoveList.Generation.fill skip paired fc b.val mask mem).len = some count)
    (live : idx < count) (found : (MoveList.Generation.fill skip paired fc b.val mask mem).slots idx = some mv) :
    MoveOrigin mc fc b source (some mv.sq) := by
  obtain ⟨length,slot,_⟩ := MoveList.Generation.generationContract_proven mc fc b.val mask skip paired mem
    b.property subset nonempty
  have bound : idx < (records fc b.val mask).length := Option.some.inj (len.symm.trans length) ▸ live
  rw [slot, ite_eq_left bound] at found
  apply MoveOrigin.indexed mask _ idx mv.sq subset (List.Perm.refl _)
  simp [List.getElem?_map,found]

theorem MoveOrigin.presearch (attempts : List ProbeRetry.RawAttempt) (first : Bitboard.Square)
    (read : ProbeReadsFrom source attempts) (firstSlot : MoveOrigin mc fc b source (some first)) :
    MoveOrigin mc fc b source (some ((GeneralNonPvMir.probeHint b attempts).getD first)) := by
  unfold GeneralNonPvMir.probeHint
  cases observed : ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) with
  | miss => exact firstSlot
  | hit slot entry =>
    have fromTable := MoveOrigin.tt (mc := mc) (fc := fc) (b := b) attempts slot entry read observed
    change MoveOrigin mc fc b source (some ((Packing.decodeBestMove entry.bestMove).getD first))
    cases choice : Packing.decodeBestMove entry.bestMove with
    | none => exact firstSlot
    | some sq => exact choice ▸ fromTable

theorem MoveOrigin.legal (origin : MoveOrigin mc fc b source move) (prior : LegalHints hist)
    (used : ∀ s, source s → s.publication hist) : LegalChoice b move := by
  induction origin with
  | absent => intro sq h; contradiction
  | indexed mask order idx sq subset perm atIndex =>
    have present := perm.mem_iff.mp (List.mem_of_getElem? atIndex)
    obtain ⟨mv, member, same⟩ := List.mem_map.mp present
    obtain ⟨s, allowed, rfl⟩ := List.mem_map.mp member
    cases same
    intro other equal
    cases equal
    exact (Board.getMoves_exact mc b.val s).mp (subset s.val (List.mem_filter.mp allowed).2)
  | tt attempts slot entry sampled hit =>
    intro sq decoded
    have legal := TableReads.ProbeReads.legalHint (sampled.mapEvidence used) prior hit
    obtain ⟨m, _, rfl⟩ := legal.generated (mc := mc) decoded
    exact (Board.getMoves_exact mc b.val _).mp m.property
  | enhanced query reads scanned => exact TableReads.enhancedHint query reads scanned
  | split division moveAt seed picked read _ _ seedLegal fromReport =>
    rw [read.value, Packing.decodeBestMove_encodeBestMove]
    rcases read.stores.member with rfl | ⟨r,mem,rfl⟩
    · exact seedLegal
    · exact fromReport r mem

theorem MoveOrigin.hint (origin : MoveOrigin mc fc b source move) (prior : LegalHints hist)
    (used : ∀ s, source s → s.publication hist) : CausalReplay.SearchHint mc b pv a z score move := by
  cases move with
  | none => exact .absent
  | some sq =>
    have legal := origin.legal prior used sq rfl
    exact .chosen ⟨sq, (Board.getMoves_exact mc b.val sq).mpr legal⟩

theorem MoveOrigin.otherHint (origin : MoveOrigin mc fc b source move) (prior : LegalHints hist)
    (used : ∀ s, source s → s.publication hist) : CausalReplay.OtherHint mc fc b move := by
  cases move with
  | none => exact .fromBoard none
  | some sq =>
    have legal := origin.legal prior used sq rfl
    exact .fromBoard (some ⟨sq, (Board.getMoves_exact mc b.val sq).mpr legal⟩)

end NeuralReversi.PvsParallel.FrameReturns
