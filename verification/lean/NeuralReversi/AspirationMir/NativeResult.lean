import NeuralReversi.AspirationMir.FrameResult
import NeuralReversi.PvsParallel.NativeFrames

namespace NeuralReversi.AspirationMir

open PvsParallel
open NativeFrames (Capture Row Record Branch logOf tableEvents)
open FrameReturns (RootRows WriteStep)

def rootReturn? (c : Capture pool) (board : PvsBoardTree.Position) (worker idx step : Nat) (a z : Word) : Option Row :=
  c.rows.find? fun r => decide (r.frame.worker = worker ∧ r.frame.returnStep = some step ∧ r.rootLine = some idx ∧
    r.frame.pv = true ∧ r.frame.cut = false ∧ r.frame.position = board ∧ r.alpha.raw = a ∧ r.beta.raw = z)

def rootQuery (c : Capture pool) (b : PvsBoardTree.Position) (worker idx : Nat) (returnedAt : Nat → Nat) : Query :=
  fun n a z => ((rootReturn? c b worker idx (returnedAt n) a z).map (·.rawScore.raw)).getD 0

theorem rootReturn_fields (found : rootReturn? c b w i t a z = some row) :
    row ∈ c.rows ∧ row.frame.worker = w ∧ row.frame.returnStep = some t ∧ row.rootLine = some i ∧
    row.frame.pv = true ∧ row.frame.cut = false ∧ row.frame.position = b ∧ row.alpha.raw = a ∧ row.beta.raw = z := by
  exact ⟨List.mem_of_find?_eq_some found,by simpa using List.find?_some found⟩

theorem query_score (found : rootReturn? c b w i (returnedAt n) a z = some row) :
    row.frame.score = (rootQuery c b w i returnedAt n a z).toInt := by
  simpa [rootQuery,found] using row.value

theorem publicNativeScore {c : Capture pool} (captured : NativeFrames.decode pool inputs stream = some c)
    (mc fc identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → Branch c log endian row)
    (stores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ e ∈ c.table, WriteStep log endian e)
    (b : PvsBoardTree.Position) (worker : Nat) (returnedAt : Nat → Nat)
    (ran : Returns (rootQuery c b worker 0 returnedAt) (fun _ => false) m raw last) (entry : m.pc = 0)
    (found : rootReturn? c b worker 0 (returnedAt last.index) last.alpha last.beta = some row)
    (initial scored : List RootMoveStorage.Row) (pv : Bitboard.Square → List Bitboard.Square)
    (covers : Root.Covers mc b (initial.map RootMoveStorage.Row.square)) (positive : 0 < initial.length)
    (root : RootRows (logOf mc fc c.rows (tableEvents stream) identify before sequenced)
      row.frame row.depth initial 0 row.frame.alpha row.frame.beta pv row.frame.score scored)
    (sort : PublicationMir.SliceSort) (minimum nodes : Nat) (reportable : minimum ≤ 5)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort scored row.depth.toNat nodes old fallback = some result ∧
      PublicationMir.score result = some raw.toInt ∧ raw.toInt = PvsBoardTree.minimax mc fc b := by
  have decoded := NativeFrames.decode_observations captured mc fc identify before sequenced irrefl trans scores stores
  obtain ⟨member,_,_,line,_,_,position,alpha,beta⟩ := rootReturn_fields found
  have through : row.frame ∈ (logOf mc fc c.rows (tableEvents stream) identify before sequenced).frames :=
    List.mem_map_of_mem (List.mem_filter.mpr ⟨member,by simp [NativeFrames.whole,line]⟩)
  have covered := position ▸ covers
  obtain ⟨snapshot,attempt,writes,best⟩ := root.checked covered
  obtain ⟨result,returned,same,truth⟩ := publicFrameScore decoded row.frame through ran entry
    (row.lo.trans (congrArg BitVec.toInt alpha)) (row.hi.trans (congrArg BitVec.toInt beta)) (query_score found)
    covered positive attempt best writes minimum reportable sort nodes old fallback
  exact ⟨result,returned,same,position ▸ truth⟩

inductive NativeLines (c : Capture pool) (log : FrameReturns.Log mc fc)
    (b : PvsBoardTree.Position) (depth : TranspositionTable.Depth) :
    Nat → List RootMoveStorage.Row → List RootMoveStorage.Row → Prop
  | done (rows : List RootMoveStorage.Row) : NativeLines c log b depth rows.length rows rows
  | next (worker : Nat) (returnedAt : Nat → Nat) (m : Machine) (raw : Word) (last : Call) (r : Row)
      (pv : Bitboard.Square → List Bitboard.Square) (prepared : List RootMoveStorage.Row) :
      m.pc = 0 → Returns (rootQuery c b worker idx returnedAt) (fun _ => false) m raw last →
      rootReturn? c b worker idx (returnedAt last.index) last.alpha last.beta = some r →
      Prepare idx before prepared → idx < prepared.length →
      RootRows log r.frame r.depth prepared idx r.frame.alpha r.frame.beta pv r.frame.score physical →
      NativeLines c log b depth (idx+1) (RootMoveStorage.sortRows physical idx) final →
      NativeLines c log b depth idx before final

theorem NativeLines.frames (runs : NativeLines c log b (Board.emptyCount b.val) idx rows final) :
    FrameLines log b (Board.emptyCount b.val) idx rows final := by
  induction runs with
  | done => exact .done _
  | next w rAt m raw last r pv prepared entry ran found prior inside step _ ih =>
    obtain ⟨_,_,_,_,pvTag,cut,pos,a,z⟩ := rootReturn_fields found
    have depth : r.depth = Board.emptyCount b.val := pos ▸ r.full
    exact .line r.frame _ m raw last pv prepared pos (r.atDepth.trans (congrArg BitVec.toNat depth)) pvTag cut
      entry ran (r.lo.trans (congrArg BitVec.toInt a)) (r.hi.trans (congrArg BitVec.toInt z)) (query_score found)
      r.window.lower r.window.upper prior inside (depth ▸ step) ih

theorem publicNativeLines {c : Capture pool} (captured : NativeFrames.decode pool inputs stream = some c)
    (mc fc identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → Branch c log endian row)
    (stores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ e ∈ c.table, WriteStep log endian e)
    (lines : NativeLines c (logOf mc fc c.rows (tableEvents stream) identify before sequenced)
      b (Board.emptyCount b.val) 0 rows final)
    (covers : Root.Covers mc b (rows.map RootMoveStorage.Row.square)) (nonempty : rows ≠ [])
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final (Board.emptyCount b.val).toNat nodes old fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square :=
  publicFrameLines (NativeFrames.decode_observations captured mc fc identify before sequenced irrefl trans scores stores)
    lines.frames covers nonempty minimum reportable sort nodes old fallback

end NeuralReversi.AspirationMir
