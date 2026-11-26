import NeuralReversi.Board.Model
import NeuralReversi.MulHi

namespace NeuralReversi.Board.Hash

abbrev Word := BitVec 64
abbrev Byte := BitVec 8

def littleBytes (word : Word) : List Byte :=
  [word.extractLsb' 0 8, word.extractLsb' 8 8, word.extractLsb' 16 8, word.extractLsb' 24 8,
    word.extractLsb' 32 8, word.extractLsb' 40 8, word.extractLsb' 48 8, word.extractLsb' 56 8]

def nativeBytes (bigEndian : Bool) (word : Word) : List Byte :=
  if bigEndian then (littleBytes word).reverse else littleBytes word

def readLittle (bytes : List Byte) (offset : Nat) : Word :=
  let byte := fun i => bytes[offset + i]?.getD 0
  byte 7 ++ byte 6 ++ byte 5 ++ byte 4 ++ byte 3 ++ byte 2 ++ byte 1 ++ byte 0

def swapped (word : Word) : Word := readLittle (littleBytes word).reverse 0

def serialize (bigEndian : Bool) (board : Board) : List Byte :=
  nativeBytes bigEndian board.player ++ nativeBytes bigEndian board.opponent

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem readLittle_exact (word : Word) : readLittle (littleBytes word) 0 = word := by
  simp only [readLittle, littleBytes, Nat.zero_add, List.getElem?_cons_zero,
    List.getElem?_cons_succ, Option.getD_some]
  apply BitVec.eq_of_getLsbD_eq
  simp only [Nat.forall_lt_succ_right]
  simp only [BitVec.getLsbD_append, BitVec.getLsbD_extractLsb']
  simp

def InputContract : Prop :=
  ∀ bigEndian board,
    let bytes := serialize bigEndian board
    bytes.length = 16 ∧
      (∀ offset, offset = 0 ∨ offset = 8 → offset + 8 ≤ bytes.length) ∧
      readLittle bytes 0 = (if bigEndian then swapped board.player else board.player) ∧
      readLittle bytes 8 = (if bigEndian then swapped board.opponent else board.opponent)

theorem inputContract_proven : InputContract := by
  intro bigEndian board bytes
  have len : bytes.length = 16 := by cases bigEndian <;> simp [bytes, serialize, nativeBytes, littleBytes]
  refine ⟨len, ?_, ?_, ?_⟩
  · intro offset selected
    rw [len]
    omega
  · cases bigEndian <;> simp [bytes, serialize, nativeBytes, readLittle, littleBytes, swapped]
    · exact readLittle_exact board.player
  · cases bigEndian <;> simp [bytes, serialize, nativeBytes, readLittle, littleBytes, swapped]
    · exact readLittle_exact board.opponent

def foldedProduct (a b : Word) : Word :=
  ((a.setWidth 128 * b.setWidth 128).setWidth 64) ^^^ MulHi.evaluate a b

def nano16 (player opponent : Word) : Word :=
  let secret1 : Word := 0x8bb84b93962eacc9
  let seed : Word := foldedProduct 0x4b33a62ed433d4a3 secret1
  foldedProduct (player ^^^ secret1) (opponent ^^^ seed ^^^ 16)

def hash (bigEndian : Bool) (board : Board) : Word :=
  let bytes := serialize bigEndian board
  nano16 (readLittle bytes 0) (readLittle bytes 8)

def HashContract : Prop :=
  ∀ board, hash false board = nano16 board.player board.opponent ∧
    hash true board = nano16 (swapped board.player) (swapped board.opponent)

theorem hashContract_proven : HashContract := by
  intro board
  constructor <;> (unfold hash; dsimp only)
  · have ⟨_, _, player, opponent⟩ := inputContract_proven false board
    simp [player, opponent]
  · have ⟨_, _, player, opponent⟩ := inputContract_proven true board
    simp [player, opponent]

end NeuralReversi.Board.Hash
