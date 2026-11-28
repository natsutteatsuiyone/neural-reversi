module
public import NeuralReversi.Board.Model
public import NeuralReversi.Bitboard.GeometryModel

@[expose] public section

namespace NeuralReversi.Board

def transformPure (f : Bitboard.Bits → Bitboard.Bits) (b : Board) : Board :=
  ⟨f b.player, f b.opponent⟩

def rotate90 (b : Board) : Board := transformPure Bitboard.rotate90 b
def rotate180 (b : Board) : Board := transformPure Bitboard.rotate180 b
def rotate270 (b : Board) : Board := transformPure Bitboard.rotate270 b
def flipVertical (b : Board) : Board := transformPure Bitboard.flipVertical b
def flipHorizontal (b : Board) : Board := transformPure Bitboard.flipHorizontal b
def flipDiagA1H8 (b : Board) : Board := transformPure Bitboard.flipDiagA1H8 b
def flipDiagA8H1 (b : Board) : Board := transformPure Bitboard.flipDiagA8H1 b

def symmetryCandidates (b : Board) : List Board :=
  [rotate90 b, rotate180 b, rotate270 b, flipHorizontal b, flipVertical b,
    flipDiagA1H8 b, flipDiagA8H1 b]

def tupleLess (a b : Board) : Prop :=
  a.player.toNat < b.player.toNat ∨
    a.player.toNat = b.player.toNat ∧ a.opponent.toNat < b.opponent.toNat

instance (a b : Board) : Decidable (tupleLess a b) := inferInstanceAs (Decidable (_ ∨ _))

def selectCandidate (result candidate : Board) : Board :=
  if tupleLess candidate result then candidate else result

def unique (b : Board) : Board := (symmetryCandidates b).foldl selectCandidate b

end NeuralReversi.Board
