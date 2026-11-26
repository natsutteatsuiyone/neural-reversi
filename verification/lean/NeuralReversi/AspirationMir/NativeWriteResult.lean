import NeuralReversi.AspirationMir.NativeCacheResult
import NeuralReversi.PvsParallel.NativeTableWrites

namespace NeuralReversi.AspirationMir

open PvsParallel
open NativeFrames (Branch logOf)
open FrameReturns (RootRows)

theorem publicWriteScore {out : NativeFrameCache.Checked pool}
    (captured : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → Branch c log endian row)
    (t : FrameTableReads.Table (logOf mc fc out.physical.rows out.physical.table identify before sequenced))
    (packets : List (NativeTableWrites.Packet t endian))
    (writes : NativeTableWrites.check t ⟨out,stream,rfl⟩ packets = true)
    (b : PvsBoardTree.Position) (worker : Nat) (returnedAt : Nat → Nat)
    (ran : Returns (rootQuery out.physical b worker 0 returnedAt) (fun _ => false) m raw last) (entry : m.pc = 0)
    (found : rootReturn? out.physical b worker 0 (returnedAt last.index) last.alpha last.beta = some row)
    (initial scored : List RootMoveStorage.Row) (pv : Bitboard.Square → List Bitboard.Square)
    (covers : Root.Covers mc b (initial.map RootMoveStorage.Row.square)) (positive : 0 < initial.length)
    (root : RootRows (logOf mc fc out.physical.rows out.physical.table identify before sequenced)
      row.frame row.depth initial 0 row.frame.alpha row.frame.beta pv row.frame.score scored)
    (sort : PublicationMir.SliceSort) (minimum nodes : Nat) (reportable : minimum ≤ 5)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort scored row.depth.toNat nodes old fallback = some result ∧
      PublicationMir.score result = some raw.toInt ∧ raw.toInt = PvsBoardTree.minimax mc fc b :=
  publicCacheScore captured identify before sequenced irrefl trans scores
    (NativeTableWrites.stores t ⟨out,stream,rfl⟩ packets writes) b worker returnedAt ran entry found initial scored pv covers positive root sort minimum nodes reportable old fallback

theorem publicWriteLines {out : NativeFrameCache.Checked pool}
    (captured : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → Branch c log endian row)
    (t : FrameTableReads.Table (logOf mc fc out.physical.rows out.physical.table identify before sequenced))
    (packets : List (NativeTableWrites.Packet t endian))
    (writes : NativeTableWrites.check t ⟨out,stream,rfl⟩ packets = true)
    (lines : NativeLines out.physical (logOf mc fc out.physical.rows out.physical.table identify before sequenced)
      b (Board.emptyCount b.val) 0 rows final)
    (covers : Root.Covers mc b (rows.map RootMoveStorage.Row.square)) (nonempty : rows ≠ [])
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final (Board.emptyCount b.val).toNat nodes old fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square :=
  publicCacheLines captured identify before sequenced irrefl trans scores
    (NativeTableWrites.stores t ⟨out,stream,rfl⟩ packets writes) lines covers nonempty minimum reportable sort nodes old fallback

end NeuralReversi.AspirationMir
