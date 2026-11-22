import NeuralReversi.Square.TextModel

namespace NeuralReversi.Square

def byteArray (bytes : List Byte) : ByteArray :=
  ByteArray.mk (bytes.map UInt8.ofBitVec).toArray

def decodeByteList (bytes : List Byte) : Option String := String.fromUTF8? (byteArray bytes)

def parseChunk (bytes : List Byte) : Except ParseFailure Square :=
  match decodeByteList bytes with
  | none => .error (.error .invalidFormat)
  | some text => fromStr text

def chunks2 : List Byte → List (List Byte)
  | [] => []
  | [a] => [[a]]
  | a :: b :: rest => [a, b] :: chunks2 rest

structure SeqError where
  index : BitVec 64
  source : ParseError
  deriving DecidableEq, Repr

inductive SeqFailure where
  | error (reason : SeqError)
  | panic (reason : Panic)
  deriving DecidableEq, Repr

def parseChunks (index : BitVec 64) : List (List Byte) → Except SeqFailure (List Square)
  | [] => .ok []
  | bytes :: rest => match parseChunk bytes with
    | .ok sq => (parseChunks (index + 1#64) rest).map (sq :: ·)
    | .error (.error reason) => .error (.error ⟨index + 1#64, reason⟩)
    | .error (.panic reason) => .error (.panic reason)

def parseSequence (text : String) : Except SeqFailure (List Square) :=
  parseChunks 0#64 (chunks2 (trimmedBytes text))

def SequenceDomain (text : String) : Prop := text.toUTF8.size < 2 ^ 64

end NeuralReversi.Square
