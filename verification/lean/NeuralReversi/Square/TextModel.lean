import NeuralReversi.Square.Model

namespace NeuralReversi.Square

def rustWhitespace (c : Char) : Bool :=
  let n := c.toNat
  decide (9 ≤ n ∧ n ≤ 13 ∨ n = 32 ∨ n = 133 ∨ n = 160 ∨ n = 5760 ∨
    8192 ≤ n ∧ n ≤ 8202 ∨ n = 8232 ∨ n = 8233 ∨ n = 8239 ∨ n = 8287 ∨ n = 12288)

def unicodeWhitespaceScalars : List Nat :=
  [9, 10, 11, 12, 13, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196,
    8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288]

def trimWhitespace (chars : List Char) : List Char :=
  ((chars.dropWhile rustWhitespace).reverse.dropWhile rustWhitespace).reverse

def trimmedBytes (text : String) : List Byte :=
  ((String.ofList (trimWhitespace text.toList)).toUTF8.data.toList).map fun b =>
    BitVec.ofNat 8 b.toNat

def asciiLowerByte (value : Byte) : Byte :=
  if 65 ≤ value.toNat ∧ value.toNat ≤ 90 then value + 32 else value

inductive ParseError where
  | invalidFormat
  | invalidFile (originalByte : Byte)
  | invalidRank (byte : Byte)
  deriving DecidableEq, Repr

inductive ParseFailure where
  | error (reason : ParseError)
  | panic (reason : Panic)
  deriving DecidableEq, Repr

def parseBytes (bytes : List Byte) : Except ParseFailure Square :=
  match bytes with
  | [originalFile, rankByte] =>
    if ¬ (originalFile.toNat < 128 ∧ rankByte.toNat < 128) then
      .error (.error .invalidFormat)
    else
      let fileByte := asciiLowerByte originalFile
      if ¬ (97 ≤ fileByte.toNat ∧ fileByte.toNat ≤ 104) then
        .error (.error (.invalidFile originalFile))
      else if ¬ (49 ≤ rankByte.toNat ∧ rankByte.toNat ≤ 56) then
        .error (.error (.invalidRank rankByte))
      else (fromFileRank (fileByte - 97) (rankByte - 49)).mapError ParseFailure.panic
  | _ => .error (.error .invalidFormat)

def fromStr (text : String) : Except ParseFailure Square := parseBytes (trimmedBytes text)

end NeuralReversi.Square
