module
public import NeuralReversi.EndgameSearch.SerialLeafCache
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace.Order
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads

@[expose] public section

namespace NeuralReversi.EndgameSearch.SerialWarmCache

open CacheHistory (Pool Local Input Good Visit Timeline)
open CacheHistory.NativeTrace (Worker Cursor Event)
open SerialLeafCache (materialize materialize_eq snapshot Reachable)
open Bitboard.Moves (X86Config)

/-- This is the general native boundary, including TT shortcut, midgame promotion,
    arbitrary outer selectivity, split polling and discarded results. -/
theorem after_boundary (p : Pool) (mc : X86Config) (fc : FlipConfig) (i : Input) (before : Local)
    (boot : Reachable p mc fc before) : Reachable p mc fc (i.after mc fc p.address before) := by
  have cells : before.cells = snapshot p before.history := Option.some.inj (boot.2.symm.trans (materialize_eq p _))
  have g : Good mc fc p.width p.sizes p.key before := ⟨boot.1,
    cells ▸ SerialLeafCache.snapshot_shape p before.history,cells ▸ SerialLeafCache.snapshot_arrays p before.history⟩
  obtain ⟨valid,ws,grows,_⟩ := CacheHistory.after_call i g (CacheMir.budget_fits p.size p.width p.word)
  refine ⟨valid.valid,?_⟩
  change before.history ++ ws = (i.run mc fc p.address before.history).history at grows
  change materialize p (i.run mc fc p.address before.history).history =
    some (((i.run mc fc p.address before.history).history.drop before.history.length).foldl
      (CacheState.write p.address) before.cells)
  rw [← grows,materialize_eq,cells]
  simp [snapshot]

theorem replay_reachable (p : Pool) (mc : X86Config) (fc : FlipConfig) (visits : List (Visit p)) :
    Reachable p mc fc (CacheHistory.replay mc fc p visits) := by
  have go (s : Local) (boot : Reachable p mc fc s) :
      Reachable p mc fc (visits.foldl (fun prior v => v.input.after mc fc p.address prior) s) := by
    induction visits generalizing s with
    | nil => exact boot
    | cons v vs ih => exact ih _ (after_boundary p mc fc v.input s boot)
  exact go p.empty (SerialLeafCache.cold p mc fc)

/-- Every earlier completed cache call of this worker contributes, even when
    its result was discarded or a different worker ran between two visits. -/
theorem native_closed (p : Pool) (mc : X86Config) (fc : FlipConfig) (w : Worker p)
    (stream : List (Event S)) (cursor : Cursor w) (recorded : CacheHistory.NativeTrace.run w stream = some cursor)
    (outside : cursor.busy = false) :
    cursor.visible mc fc = some (CacheHistory.replay mc fc p cursor.completed) ∧
      Reachable p mc fc (CacheHistory.replay mc fc p cursor.completed) ∧
      ∀ v ∈ cursor.completed, v.worker = w.ordinal := by
  obtain ⟨ordered,owner⟩ := CacheHistory.NativeTrace.run_ordered w stream recorded
  exact ⟨by simp [Cursor.visible,Cursor.timeline,outside,ordered],
    replay_reachable p mc fc cursor.completed,owner⟩

/-- A checked native packet sequence includes the loaded board and bounds, slot and stored
    board/threshold/return for each store. Its final pair is the actual-sized warm pair. -/
theorem physical_visit (p : Pool) (mc : X86Config) (fc : FlipConfig) (timeline : Timeline p)
    (values : CacheHistory.Payloads.Values) (stream : List (Event S)) (v : Visit p) (audit : CacheHistory.Payloads.Audit)
    (checked : CacheHistory.Payloads.completeVisit p timeline mc fc values stream v = some audit) :
    Reachable p mc fc audit.after := by
  have projected := CacheHistory.Payloads.completeVisit_state p timeline mc fc values stream v checked
  rw [projected.2.1]
  exact after_boundary p mc fc v.input _ (replay_reachable p mc fc _)

end NeuralReversi.EndgameSearch.SerialWarmCache
