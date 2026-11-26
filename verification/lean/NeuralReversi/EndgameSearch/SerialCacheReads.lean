import NeuralReversi.EndgameSearch.SerialLeafCache
import NeuralReversi.EndgameSearch.SerialReadPrefix

namespace NeuralReversi.EndgameSearch.SerialCacheReads

open CacheHistory (Pool Local)
open CacheState (Cells)
open EndgameTreeCache (Store)
open SerialLeafCache (snapshot Reachable)
open SerialLeafRestore (Reader)
open PvsParallel.SerialScout (Config)
open Bitboard.Moves (X86Config)

/-- Read the incoming live arrays plus the saves issued since entry to this leaf. -/
def cellsAt (p : Pool) (prior : Local) (h : List Store) : Cells :=
  (h.drop prior.history.length).foldl (CacheState.write p.address) prior.cells

theorem cellsAt_append (p : Pool) (prior : Local) (ws : List Store) :
    cellsAt p prior (prior.history ++ ws) = ws.foldl (CacheState.write p.address) prior.cells := by
  simp [cellsAt]

theorem native_changes (p : Pool) (prior : Local) (boot : Reachable p mc fc prior) (ws : List Store) :
    ws.foldlM (CacheMir.writeCells p.width p.key) prior.cells = some (cellsAt p prior (prior.history ++ ws)) := by
  have start := Option.some.inj (boot.2.symm.trans (SerialLeafCache.materialize_eq p prior.history))
  rw [cellsAt_append,start]
  exact CacheMir.replay_eq ws (SerialLeafCache.snapshot_shape p prior.history) (CacheMir.budget_fits p.size p.width p.word)

theorem cellsAt_eq (p : Pool) (prior : Local) (boot : Reachable p mc fc prior) (h : List Store)
    (within : prior.history <+: h) : cellsAt p prior h = snapshot p h := by
  obtain ⟨ws,rfl⟩ := within
  have start := Option.some.inj (boot.2.symm.trans (SerialLeafCache.materialize_eq p prior.history))
  simp [cellsAt_append,start,snapshot]

def reader (p : Pool) (prior : Local) : Reader where
  shallow h b a := (SerialLeafCache.probe p (cellsAt p prior h) .shallow b.val a).join
  deep fc paired b s a first rest := (SerialLeafCache.deepEarly p (cellsAt p prior s.history) fc paired b s a first rest).join

/-- Every lookup issued by the leaf is within this relation. At those prefixes the
    outer Option of the checked native probe is `some`, including on a cache miss. -/
theorem reader_same (p : Pool) (prior : Local) (boot : Reachable p mc fc prior) :
    SerialLeafRestore.SameReads prior.history (reader p prior) (SerialLeafRestore.historyReader p.address) := by
  constructor
  · intro h within b a
    simp only [reader,SerialLeafRestore.historyReader]
    rw [cellsAt_eq p prior boot h within]
    exact congrArg Option.join (SerialLeafCache.physical_reads p h .shallow b.val a)
  · intro fc' paired b s within a first rest
    simp only [reader,SerialLeafRestore.historyReader]
    rw [cellsAt_eq p prior boot s.history within]
    exact congrArg Option.join (SerialLeafCache.deepEarly_eq p _ fc' paired b s (SerialLeafCache.physical_reads p _) a first rest)

def leaf (p : Pool) (c : Config) (mc : X86Config) (fc : FlipConfig) (b : PvsBoardTree.Position)
    (s : Cancellation.EmptyState) (a z : ScoreRefresh.Score) (prior : Local) :=
  SerialLeafRestore.leafWith (reader p prior) c mc fc b s a z prior.history

theorem leaf_eq (p : Pool) (c : Config) (address : c.address = p.address) (mc : X86Config) (fc : FlipConfig)
    (b : PvsBoardTree.Position) (s : Cancellation.EmptyState) (a z : ScoreRefresh.Score) (prior : Local)
    (boot : Reachable p mc fc prior) :
    leaf p c mc fc b s a z prior = SerialLeafRestore.leaf c mc fc b s a z prior.history := by
  unfold leaf
  apply SerialLeafRestore.leaf_prefix (l := reader p prior) (r := SerialLeafRestore.historyReader c.address) _ _ _ _ _ _ _ _ _ (List.prefix_refl _)
  rw [address]
  exact reader_same p prior boot

end NeuralReversi.EndgameSearch.SerialCacheReads
