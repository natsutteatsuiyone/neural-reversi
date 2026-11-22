import NeuralReversi.Board.SetPlacement
import NeuralReversi.Bitboard.SquareProperties
import NeuralReversi.Solve2.BoardBridge

namespace NeuralReversi.ScoreRefresh.MoveBridge
abbrev Board := NeuralReversi.Board.Board
abbrev Square := Fin 64
abbrev Word := BitVec 64
abbrev Backend := Square → Word → Word → Except String Word

def isEmptySquare (b : Board) (sq : Square) : Bool :=
  (NeuralReversi.Board.emptySquares b &&& Bitboard.squareMask sq) != 0

def makeMove (flip : Backend) (checked : Bool) (b : Board) (sq : Square) : Except String Board :=
  if checked && !isEmptySquare b sq then
    .error "sq must be an empty square"
  else match flip sq b.player b.opponent with
    | .error reason => .error reason
    | .ok flipped =>
      if checked && flipped == 0 then .error "assertion failed: !flipped.is_empty()"
      else .ok (NeuralReversi.Board.SetPlacement.place b flipped sq)

theorem occupied_rejected_first (flip : Backend) (b : Board) (sq : Square)
    (occupied : isEmptySquare b sq = false) :
    makeMove flip true b sq = .error "sq must be an empty square" := by
  simp [makeMove, occupied]

theorem empty_flip_failure (flip : Backend) (checked : Bool) (b : Board) (sq : Square)
    (empty : isEmptySquare b sq = true) (err : String)
    (failed : flip sq b.player b.opponent = .error err) :
    makeMove flip checked b sq = .error err := by
  simp [makeMove, empty, failed]

theorem empty_zero_rejected (flip : Backend) (b : Board) (sq : Square)
    (empty : isEmptySquare b sq = true)
    (zero : flip sq b.player b.opponent = .ok 0) :
    makeMove flip true b sq = .error "assertion failed: !flipped.is_empty()" := by
  simp [makeMove, empty, zero]

theorem success_setPlacement (flip : Backend) (checked : Bool) (b : Board) (sq : Square)
    (empty : isEmptySquare b sq = true) (f : Word) (nonzero : f ≠ 0)
    (result : flip sq b.player b.opponent = .ok f) :
    makeMove flip checked b sq = .ok (NeuralReversi.Board.SetPlacement.place b f sq) := by
  have hn : f ≠ 0#64 := nonzero
  simp [makeMove, empty, result, hn]

def scalarBackend : Backend := fun sq p o => .ok (Flip.value ⟨sq.val, by omega⟩ p o)
def scalarMove (checked : Bool) (b : Board) (sq : Square) : Except String Board :=
  makeMove scalarBackend checked b sq

theorem scalar_success (checked : Bool) (b : Board) (sq : Square)
    (empty : isEmptySquare b sq = true)
    (nonzero : Flip.value ⟨sq.val, by omega⟩ b.player b.opponent ≠ 0) :
    scalarMove checked b sq = .ok (NeuralReversi.Board.SetPlacement.place b
      (Flip.value ⟨sq.val, by omega⟩ b.player b.opponent) sq) :=
  success_setPlacement scalarBackend checked b sq empty _ nonzero rfl

theorem occupied_over_backend_failure (b : Board) (sq : Square)
    (occupied : isEmptySquare b sq = false) (err : String) :
    makeMove (fun _ _ _ => .error err) true b sq = .error "sq must be an empty square" :=
  occupied_rejected_first _ b sq occupied

theorem scalar_success_disjoint (checked : Bool) (b : Board) (sq : Square) (next : Board)
    (wf : NeuralReversi.Board.wellFormed b) (success : scalarMove checked b sq = .ok next) :
    NeuralReversi.Board.wellFormed next := by
  change (if checked && !isEmptySquare b sq then Except.error "sq must be an empty square"
    else if checked && Flip.value ⟨sq.val, by omega⟩ b.player b.opponent == 0 then
      Except.error "assertion failed: !flipped.is_empty()"
    else Except.ok (NeuralReversi.Board.SetPlacement.place b
      (Flip.value ⟨sq.val, by omega⟩ b.player b.opponent) sq)) = Except.ok next at success
  split at success
  · contradiction
  · split at success
    · contradiction
    · cases success
      exact NeuralReversi.Board.SetPlacement.wellFormed_preserved _ _ _ wf

end NeuralReversi.ScoreRefresh.MoveBridge
