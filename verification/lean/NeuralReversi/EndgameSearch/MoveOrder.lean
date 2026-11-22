import NeuralReversi.EndgameSearch.SourceFour
import NeuralReversi.EmptyListParity
import NeuralReversi.MoveList.BestFirst
import NeuralReversi.MoveList.SmallGenerate
import NeuralReversi.ShallowMoveOrder

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open Bitboard.Moves (X86Config)
open EndgameTreeCache (Address Store Table soundHistory)

private def squareValues : Vector Int32 64 := #v[
  18, 4, 16, 12, 12, 16, 4, 18,
  4, 2, 6, 8, 8, 6, 2, 4,
  16, 6, 14, 10, 10, 14, 6, 16,
  12, 8, 10, 0, 0, 10, 8, 12,
  12, 8, 10, 0, 0, 10, 8, 12,
  16, 6, 14, 10, 10, 14, 6, 16,
  4, 2, 6, 8, 8, 6, 2, 4,
  18, 4, 16, 12, 12, 16, 4, 18]

def fastMoveValue (sq : Bitboard.Square) (next : Board.Board) (moves : Bitboard.Bits) : Int32 :=
  let stability := Int32.ofInt (Bitboard.cornerStability next.opponent).toNat
  let mobility := Int32.ofInt (Bitboard.cornerWeightedCount moves).toNat
  squareValues[sq] * 128 + stability * 2048 + (36 - mobility) * 16384

private theorem squareValues_bounded : ∀ sq : Fin 64,
    0 ≤ squareValues[sq].toInt ∧ squareValues[sq].toInt ≤ 18 := by decide

theorem fastMoveValue_exact (sq : Bitboard.Square) (next : Board.Board) (moves : Bitboard.Bits) :
    (fastMoveValue sq next moves).toInt =
      squareValues[sq].toInt * 128 + ((Bitboard.cornerStability next.opponent).toNat : Int) * 2048 +
        (36 - (Bitboard.cornerWeightedCount moves).toNat) * 16384 := by
  have ⟨squareMin, squareMax⟩ := squareValues_bounded sq
  have stabilityMax := (Bitboard.cornerStabilityContract_proven next.opponent).2
  have mobilityMax := (Bitboard.cornerWeightedCountContract_proven moves).2.1
  let st : Int := (Bitboard.cornerStability next.opponent).toNat
  let mo : Int := (Bitboard.cornerWeightedCount moves).toNat
  have stCast : (Int32.ofInt st).toInt = st :=
    Int32.toInt_ofInt_of_le (by dsimp [st]; omega) (by dsimp [st]; omega)
  have moCast : (Int32.ofInt mo).toInt = mo :=
    Int32.toInt_ofInt_of_le (by dsimp [mo]; omega) (by dsimp [mo]; omega)
  unfold fastMoveValue
  change (squareValues[sq] * 128 + Int32.ofInt st * 2048 + (36 - Int32.ofInt mo) * 16384).toInt = _
  simp only [Int32.toInt_add, Int32.toInt_mul, Int32.toInt_sub, stCast, moCast]
  change (((squareValues[sq].toInt * 128).bmod 4294967296 +
    (st * 2048).bmod 4294967296).bmod 4294967296 +
    ((36 - mo).bmod 4294967296 * 16384).bmod 4294967296).bmod 4294967296 = _
  have ⟨stMin, stMax⟩ : 0 ≤ st ∧ st ≤ 12 := by dsimp [st]; omega
  have ⟨moMin, moMax⟩ : 0 ≤ mo ∧ mo ≤ 68 := by dsimp [mo]; omega
  rw [Int.bmod_eq_of_le (n := squareValues[sq].toInt * 128) (by omega) (by omega),
    Int.bmod_eq_of_le (n := st * 2048) (by omega) (by omega),
    Int.bmod_eq_of_le (n := squareValues[sq].toInt * 128 + st * 2048) (by omega) (by omega),
    Int.bmod_eq_of_le (n := 36 - mo) (by omega) (by omega),
    Int.bmod_eq_of_le (n := (36 - mo) * 16384) (by omega) (by omega),
    Int.bmod_eq_of_le (by omega) (by omega)]

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

example :
    let b : Position := ⟨⟨0x4022060737000080, 0x14d9b8f8c87efc7f⟩, by unfold Board.wellFormed; decide⟩
    let mc : X86Config := ⟨false, false⟩
    let fc : FlipConfig := ⟨false, false, false⟩
    (deepOrder mc fc b).map (fun m =>
      let record := evaluated mc fc b m
      (record.square.val, record.value.toInt, record.nextMoves)) =
        [(50, 477952, 0x2b00410000000000), (8, 477696, 0x2104410000010000),
          (23, 464896, 0x2304410000010000), (61, 446464, 0x8b04410000000000),
          (46, 445184, 0xa304010000010000), (9, 444672, 0x2304410000010100)] := by decide +kernel

end NeuralReversi.EndgameSearch
