import NeuralReversi.AspirationMir.Syntax

namespace NeuralReversi.AspirationMir

def initialDelta : Int := 256
def infinity : Int := 32767

def driverCode : List DriverOp := [
  .prepare 1,
  .go 2,
  .go 3,
  .query 4,
  .go 5,
  .poll 6,
  .branchAbort 8 7,
  .saveScore 13,
  .widen 9,
  .branchWiden 11 10,
  .double 12,
  .saveScore 13,
  .go 2,
  .returned,
  .unwind]

def widenCode : List WidenPath := [
  ([((.le .score .alpha), true)], (.max (.sub .score .delta) (.neg .inf)), .alpha, true),
  ([((.le .score .alpha), false), ((.ge .score .beta), true)], (.max (.sub .beta .delta) .alpha), (.min (.add .score .delta) .inf), true),
  ([((.le .score .alpha), false), ((.ge .score .beta), false)], .alpha, .beta, false)]

end NeuralReversi.AspirationMir
