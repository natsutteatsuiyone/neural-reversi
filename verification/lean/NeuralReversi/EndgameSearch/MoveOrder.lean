import NeuralReversi.EndgameSearch.SourceFour
import NeuralReversi.EmptyListParity
import NeuralReversi.MoveList.BestFirst
import NeuralReversi.MoveList.SmallGenerate
import NeuralReversi.ShallowMoveOrder
import NeuralReversi.MoveList.Ordering

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open Bitboard.Moves (X86Config)
open EndgameTreeCache (Address Store Table soundHistory)

open MoveList.OrderWeights

/-- The fast endgame ordering key, computed with `i32` operations on the extracted weights. -/
def fastMoveValue (sq : Bitboard.Square) (next : Board.Board) (moves : Bitboard.Bits) : Int32 :=
  let stability := Int32.ofInt (Bitboard.cornerStability next.opponent).toNat
  let mobility := Int32.ofInt (Bitboard.cornerWeightedCount moves).toNat
  Int32.ofInt squareValue[sq] * Int32.ofInt squareWeight + stability * Int32.ofInt stabilityWeight +
    (36 - mobility) * Int32.ofInt mobilityWeight

theorem fastMoveValue_exact (sq : Bitboard.Square) (next : Board.Board) (moves : Bitboard.Bits) :
    (fastMoveValue sq next moves).toInt =
      squareValue[sq] * squareWeight + ((Bitboard.cornerStability next.opponent).toNat : Int) * stabilityWeight +
        (36 - (Bitboard.cornerWeightedCount moves).toNat) * mobilityWeight := by
  have ⟨svMin, svMax⟩ := MoveList.Ordering.squareValue_range sq
  have stabilityMax := (Bitboard.cornerStabilityContract_proven next.opponent).2
  have mobilityMax := (Bitboard.cornerWeightedCountContract_proven moves).2.1
  simp only [MoveList.OrderWeights.squareMax] at svMax
  have base : (36 : Int32).toInt = 36 := by decide
  unfold fastMoveValue
  simp only [squareWeight, stabilityWeight, mobilityWeight, Int32.toInt_add, Int32.toInt_mul, Int32.toInt_sub,
    Int32.toInt_ofInt, Int32.size, base]
  simp (disch := omega) only [Int.bmod_eq_of_le]

structure EvaluatedMove where
  square : Bitboard.Square
  next : Board.Board
  nextMoves : Bitboard.Bits
  value : Int32
  deriving DecidableEq

private def evaluateMove (mc : X86Config) (fc : FlipConfig) (b : Board.Board)
    (sq : Bitboard.Square) : EvaluatedMove :=
  let next := Board.place b (Flip.Dispatch.resultValue fc sq b.player b.opponent) sq
  let nextMoves := Board.getMoves mc next
  ⟨sq, next, nextMoves, fastMoveValue sq next nextMoves⟩

private def evaluated (mc : X86Config) (fc : FlipConfig) (b : Position)
    (sq : GeneratedSquare mc b.val) : EvaluatedMove :=
  evaluateMove mc fc b.val sq.val

def deepOrder (mc : X86Config) (fc : FlipConfig) (b : Position) : List (GeneratedSquare mc b.val) :=
  MoveList.BestFirst.ordered (EvaluatedMove.value ∘ evaluated mc fc b) (legalSquares mc b.val)

theorem deepOrder_members (mc : X86Config) (fc : FlipConfig) (b : Position)
    (sq : GeneratedSquare mc b.val) :
    sq ∈ deepOrder mc fc b ↔ sq ∈ legalSquares mc b.val :=
  ((MoveList.BestFirst.orderContract_proven _ _).1 (legalSquares mc b.val)).1.mem_iff

def DeepOrderContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (history : List Store)
    (address : Address) (alpha : Int) (paired : Bool),
    (Board.emptyCount b.val).toNat ≤ 12 →
    EndgameCutoffs.beforeMoves fc b.val history address .deep alpha
      ((legalSquares mc b.val).map Subtype.val) = none →
    let flips := fun sq => Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent
    let generated := MoveList.SmallGenerate.generate paired (fun sq => flips sq == b.val.opponent)
      (MoveList.SmallGenerate.candidates mc b.val)
    generated.1 = true ∧ generated.2.length ≤ 12 ∧
      (deepOrder mc fc b).map (evaluated mc fc b) =
        MoveList.BestFirst.ordered EvaluatedMove.value (generated.2.map (evaluateMove mc fc b.val)) ∧
      ∀ sq ∈ deepOrder mc fc b,
        let next := (play mc fc b sq).val.val
        (evaluated mc fc b sq).next = next ∧ (evaluated mc fc b sq).nextMoves = Board.getMoves mc next

theorem deepOrderContract_proven : DeepOrderContract := by
  intro mc fc b history address alpha paired depth ready
  let choices := legalSquares mc b.val
  have same : choices.map Subtype.val = MoveList.SmallGenerate.candidates mc b.val :=
    (PvsBoardTree.legalSquares_values mc b.val).trans
      (MoveList.SmallGenerate.candidates_eq_filter mc b.val).symm
  obtain ⟨_, capacity, _, _, _, clear, complete, _⟩ :=
    MoveList.SmallGenerate.generationContract_proven mc fc paired true true b.val b.property depth
  have success := clear.mpr (fun sq found => EndgameCutoffs.no_wipeout_after_cutoffs
    fc b.val history address alpha _ ready sq (same ▸ found))
  refine ⟨success, by simpa only [List.length_map] using capacity, ?_, fun _ _ => ⟨rfl, rfl⟩⟩
  rw [deepOrder, MoveList.BestFirst.ordered_map, complete success, ← same, List.map_map]
  rfl

private def groups (mc : X86Config) (b : Position) (parity : Fin 16) : List LegalMoves.Raw :=
  ShallowMoveOrder.groups ⟨Board.getMoves mc b.val⟩ (Board.emptySquares b.val) parity

def shallowOrder (mc : X86Config) (b : Position) (parity : Fin 16) : List (GeneratedSquare mc b.val) :=
  (groups mc b parity).flatMap fun group =>
    (legalSquares mc b.val).filter fun sq => group.bits.getLsbD sq.val.val

theorem shallowOrder_members (mc : X86Config) (b : Position) (parity : Fin 16)
    (sq : GeneratedSquare mc b.val) :
    sq ∈ shallowOrder mc b parity ↔ sq ∈ legalSquares mc b.val := by
  constructor
  · intro found
    obtain ⟨group, _, member⟩ := List.mem_flatMap.mp found
    exact (List.mem_filter.mp member).1
  · intro member
    obtain ⟨group, found, bit⟩ := (ShallowMoveOrder.mem_groups_iff _ _ parity sq.val).mpr sq.property
    exact List.mem_flatMap.mpr ⟨group, found, List.mem_filter.mpr ⟨member, bit⟩⟩

theorem shallowOrder_values (mc : X86Config) (b : Position) (parity : Fin 16) :
    (shallowOrder mc b parity).map Subtype.val =
      (groups mc b parity).flatMap fun g => (List.finRange 64).filter fun sq => g.bits.getLsbD sq.val := by
  rw [shallowOrder, List.map_flatMap]
  simp only [List.flatMap_def]
  congr 1
  apply List.map_congr_left
  intro g included
  change ((legalSquares mc b.val).filter ((fun (sq : Bitboard.Square) => g.bits.getLsbD sq.val) ∘
    Subtype.val)).map Subtype.val = _
  rw [← List.filter_map, PvsBoardTree.legalSquares_values, List.filter_filter]
  apply List.filter_congr
  intro sq _
  cases h : g.bits.getLsbD sq.val
  · simp
  · have legal := (ShallowMoveOrder.mem_groups_iff _ _ parity sq).mp ⟨g, included, h⟩
    simpa [h] using legal

def boardParity (b : Board.Board) : Fin 16 :=
  ⟨(EmptyListParity.fromEmpties (Board.emptySquares b)).toNat,
    (EmptyListParity.parityContract_proven _ _ (.initial (Board.emptySquares b))).1⟩

theorem boardParity_eq_reachable (b : Board.Board) (cells : List Bitboard.Square) (byte : BitVec 8)
    (state : EmptyListParity.Reachable cells byte)
    (same : cells.Perm (List.finRange 64 |>.filter (fun sq => (Board.emptySquares b).getLsbD sq.val))) :
    byte.toNat = (boardParity b).val := by
  unfold boardParity
  congr 1
  rw [EmptyListParity.parity_of_perm state same,
    EmptyListParity.parity_of_perm (.initial (Board.emptySquares b)) (List.Perm.refl _)]

def orderedMoves (mc : X86Config) (fc : FlipConfig) (b : Position) :
    Table → List (GeneratedSquare mc b.val)
  | .deep => deepOrder mc fc b
  | .shallow => shallowOrder mc b (boardParity b.val)

theorem orderedMoves_members (mc : X86Config) (fc : FlipConfig) (b : Position) (table : Table)
    (sq : GeneratedSquare mc b.val) :
    sq ∈ orderedMoves mc fc b table ↔ sq ∈ legalSquares mc b.val := by
  cases table
  · exact deepOrder_members mc fc b sq
  · exact shallowOrder_members mc b _ sq

/-- A concrete deep order: the generated moves with their successor move sets, ranked by
    decreasing fast key. The keys and their order depend on the extracted weights. -/
example :
    let b : Position := ⟨⟨0x4022060737000080, 0x14d9b8f8c87efc7f⟩, by unfold Board.wellFormed; decide⟩
    let mc : X86Config := ⟨false, false⟩
    let fc : FlipConfig := ⟨false, false, false⟩
    let records := (deepOrder mc fc b).map (evaluated mc fc b)
    (records.map fun record => (record.square.val, record.nextMoves)).isPerm
        [(50, 0x2b00410000000000), (8, 0x2104410000010000), (23, 0x2304410000010000),
          (61, 0x8b04410000000000), (46, 0xa304010000010000), (9, 0x2304410000010100)] ∧
      (records.map fun record => record.value.toInt).Pairwise (· ≥ ·) := by decide +kernel

end NeuralReversi.EndgameSearch
