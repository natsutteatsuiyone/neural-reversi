import NeuralReversi.TranspositionTable.ReaderMir.ProbeSyntax
import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.TranspositionTable.ReaderMir

def bestMove : ProbeRetry.Result → Option (Fin 64)
  | .hit _ data => Packing.decodeBestMove data.bestMove
  | .miss _ => none

def probePass (generation : BitVec 8) : ProbeRetry.Observation → ProbeRetry.Observation → Pass
  | .busy,.busy => ⟨[.read 0,.read 1,.spin],.retry none⟩
  | .busy,.empty => ⟨[.read 0,.read 1],.done (.miss 1)⟩
  | .busy,(.matching back) => ⟨[.read 0,.read 1],.done (.hit 1 back)⟩
  | .busy,(.other _) => ⟨[.read 0,.read 1,.spin],.retry (some 1)⟩
  | .empty,_ => ⟨[.read 0],.done (.miss 0)⟩
  | (.matching front),_ => ⟨[.read 0],.done (.hit 0 front)⟩
  | (.other _),.busy => ⟨[.read 0,.read 1,.spin],.retry (some 0)⟩
  | (.other _),.empty => ⟨[.read 0,.read 1],.done (.miss 1)⟩
  | (.other _),(.matching back) => ⟨[.read 0,.read 1],.done (.hit 1 back)⟩
  | (.other front),(.other back) => ⟨[.read 0,.read 1,.generation],.done (.miss (Replacement.victim generation front back))⟩

end NeuralReversi.TranspositionTable.ReaderMir
