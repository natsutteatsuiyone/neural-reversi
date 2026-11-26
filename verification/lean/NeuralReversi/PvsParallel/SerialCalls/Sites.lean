import NeuralReversi.PvsParallel.SerialCalls

namespace NeuralReversi.PvsParallel.SerialCalls.Sites

/-- The native loop suspends at each recursive call. A wide re-search consumes
    the scout's signed return before deciding whether to issue a second call. -/
def ownerWith (pv first : Bool) (alpha : Int) (scout wide : Req)
    (finish : Option Int → Option Int → Int → Program Req A) : Program Req A :=
  if pv && first then
    .call wide fun w => finish none (some w) (-w)
  else
    .call scout fun s =>
      if pv && decide (alpha < -s) then
        .call wide fun w => finish (some s) (some w) (-w)
      else finish (some s) none (-s)

def owner (pv first : Bool) (alpha : Int) (scout wide : Req) : Program Req Int :=
  ownerWith pv first alpha scout wide fun _ _ v => .pure v

def scoutRequest (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (cut : Bool) (alpha : Int) : Request :=
  ⟨next, depth - 1, false, !cut, -(alpha + 1), -alpha⟩

def wideRequest (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (alpha beta : Int) : Request :=
  ⟨next, depth - 1, true, false, -beta, -alpha⟩

def ownerMove (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (pv cut first : Bool) (alpha beta : Int) : Program Request Int :=
  owner pv first alpha (scoutRequest next depth cut alpha) (wideRequest next depth alpha beta)

def presearchWith (request : Req) (finish : Int → Program Req A) : Program Req A :=
  .call request fun w => finish (-w)

def presearch (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (cut : Bool) (alpha : Int) : Program Request Int :=
  presearchWith ⟨next, depth - 1, false, !cut, -(alpha + 1), -alpha⟩ .pure

def pass (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (pv cut : Bool) (alpha beta : Int) : Program Request Int :=
  .call ⟨next, depth, pv, (!pv && !cut), -beta, -alpha⟩ fun w => .pure (-w)

end NeuralReversi.PvsParallel.SerialCalls.Sites
