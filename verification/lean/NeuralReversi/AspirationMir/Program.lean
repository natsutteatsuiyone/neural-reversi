module
public import NeuralReversi.AspirationMir.Syntax

@[expose] public section

namespace NeuralReversi.AspirationMir

def initialDelta : Int := 256
def infinity : Int := 32767

def driverCode : List DriverOp := [
  .prepare 1,
  .go 2,
  .query 3,
  .go 4,
  .poll 5,
  .branchAbort 7 6,
  .saveScore 12,
  .widen 8,
  .branchWiden 10 9,
  .double 11,
  .saveScore 12,
  .go 1,
  .returned,
  .unwind]

def widenCode : List WidenPath := [
  ([((.le .score .alpha), true)], (.max (.sub .score .delta) (.neg .inf)), .alpha, true),
  ([((.le .score .alpha), false), ((.ge .score .beta), true)], (.max (.sub .beta .delta) .alpha), (.min (.add .score .delta) .inf), true),
  ([((.le .score .alpha), false), ((.ge .score .beta), false)], .alpha, .beta, false)]

end NeuralReversi.AspirationMir
