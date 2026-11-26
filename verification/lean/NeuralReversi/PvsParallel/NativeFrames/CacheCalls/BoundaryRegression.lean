import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Boundary
import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Reads
import NeuralReversi.Board.Hash

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.BoundaryRegression

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheHistory.Payloads (Op observeRun)
open EndgameSearch.Cancellation (EmptyState)
open PvsBoardTree (Position)

private def pool : Pool := ⟨4,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : EndgameSearch.FlipConfig := ⟨false,false,false⟩

private def empty (b : Position) : EmptyState :=
  let mask := Board.emptySquares b.val
  ⟨Empties.encode mask,Board.emptyCount b.val⟩

/-- Names each stored board by its key rather than its slot, which depends on the cache budget. -/
private def stored (o : Op) : Option (EndgameTreeCache.Table × BitVec 64 × Int × Int) := do
  let .store _ b a s .. := o.data | none
  return (o.table,pool.key b,a.toInt,s.toInt)

private def observations (code : Program node [] h) :=
  observeRun pool pool.empty (pool.fresh (mc:=mc) (fc:=fc)).1.shape (code.accesses.map Access.instruction)

private def shallow (b : Position) :=
  let code := Shallow.program pool .other false mc fc b (empty b) (-3) (by constructor <;> decide) []
  let result := EndgameSearch.ShallowMir.run .other false mc fc pool.address b (empty b) (-3) []
  (result.score,code.forest.direct, (observations code).filterMap stored)

/- The Rust run passed, then cut in the third parity group. The child and
    group wrote different slots, in that order, before the pass returned. -/
#guard shallow ⟨⟨17891997908486940929,49777857980438270⟩,rfl⟩ ==
  (-16,[(0,.shallow)],[(.shallow,600936225978133602,-3,-16),(.shallow,7122331998918454071,2,16)])

#guard shallow ⟨⟨432408240905326881,17942278238766247454⟩,rfl⟩ ==
  (-2,[(1,.shallowSlot),(2,.shallowProbe),(5,.shallowMoves)],[(.shallow,9481734207839425253,2,2),(.shallow,3903768271292950320,-3,-2)])

private def deep (b : Position) (limit : Nat) :=
  let poll : Threading.SplitCutoff.Polling := ⟨⟨0,false⟩,fun k => if k < limit then ⟨0,[]⟩ else ⟨1,[true]⟩,0⟩
  let s : EndgameSearch.Cancellation.State := ⟨empty b,[],poll⟩
  let code := Deep.program pool .other false false mc fc b (-3) (by constructor <;> decide) s
  let out := EndgameSearch.DeepRecursionMir.run .other false false mc fc pool.address b (-3) s
  (out.score,code.forest.direct,(observations code).filterMap stored)

/- The parent stopped after the child had completed three shallow stores. -/
#guard deep ⟨⟨17891715342651963649,49777849327060734⟩,rfl⟩ 2 ==
  (none,[(1,.deepSlot),(2,.deepProbe),(3,.deepChild)],
    [(.shallow,600936225978133602,-3,-16),(.shallow,7122331998918454071,2,16),(.shallow,3982915057040669784,-3,-16)])

/- All five enhanced probes ran before the first child polled for cutoff. -/
#guard deep ⟨⟨17891851967110922009,49641216278151204⟩,rfl⟩ 0 ==
  (none,[(1,.deepSlot),(2,.deepProbe)] ++ (List.replicate 5 [(5,.deepSlot),(6,.deepProbe)]).flatten ++ [(7,.deepChild)],[])

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.BoundaryRegression
