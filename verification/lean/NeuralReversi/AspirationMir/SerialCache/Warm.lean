module
public import NeuralReversi.AspirationMir.SerialCache.Physical
public import NeuralReversi.EndgameSearch.SerialWarmCache
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Coverage

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialCache.Warm

open PvsParallel
open PvsBoardTree (Position)
open SerialExecution (Inputs)
open EndgameSearch.CacheHistory (Pool Visit)
open EndgameSearch.CacheHistory.NativeTrace (Cursor Event Worker)
open RootMoveStorage (Row)

/-- Rows can already have been sorted by earlier selective iterations. The
    worker retains all private cache stores from its earlier completed leaves. -/
theorem thirteen_warm (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots) (rows : List Row) (m : Machine)
    (visits : List (Visit p)) (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult)
    (small : (Board.emptyCount b.val).toNat ≤ 13)
    (covers : Root.Covers mc b (rows.map Row.square)) (mobile : 0 < rows.length)
    (pc : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    Physical.Published p c b inputs roots rows m (EndgameSearch.CacheHistory.replay mc fc p visits)
      sort nodes fallback :=
  Physical.publication mc fc p c b inputs roots rows m _ sort nodes fallback address
    (fun i => shared_thirteen (inputs i) _ small) (EndgameSearch.SerialWarmCache.replay_reachable p mc fc visits)
    covers mobile pc window

def Contract : Prop :=
  ∀ mc fc (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc)
    (roots : Roots) (rows : List Row) (m : Machine) (w : Worker p) (stream : List (Event Unit)) (cursor : Cursor w)
    sort nodes fallback, c.address = p.address →
    EndgameSearch.CacheHistory.NativeTrace.run w stream = some cursor → cursor.busy = false →
    (Board.emptyCount b.val).toNat ≤ 13 → Root.Covers mc b (rows.map Row.square) → 0 < rows.length →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ prior, cursor.visible mc fc = some prior ∧ Physical.Published p c b inputs roots rows m prior sort nodes fallback

theorem publication : Contract := by
  intro mc fc p c b inputs roots rows m w stream cursor sort nodes fallback address parsed outside small covers positive pc window
  obtain ⟨start,_,_⟩ := EndgameSearch.SerialWarmCache.native_closed p mc fc w stream cursor parsed outside
  exact ⟨_,start,thirteen_warm p c address b inputs roots rows m cursor.completed sort nodes fallback
    small covers positive pc window⟩

/-- Checked cache packets of prior workers establish the retained private state
    used by this root. No bound or provenance hypothesis is needed for an incoming cache cell. -/
theorem packets (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots) (rows : List Row) (m : Machine)
    (values : EndgameSearch.CacheHistory.Payloads.Values) (stream : List (Event S))
    (capture : EndgameSearch.CacheHistory.Payloads.Checked p S)
    (captured : EndgameSearch.CacheHistory.Payloads.decode p mc fc values stream = some capture)
    (worker : (w : Worker p) × Cursor w) (registered : worker ∈ capture.native.workers)
    (outside : worker.2.busy = false) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (fallback : Unit → PublicationMir.SearchResult) (small : (Board.emptyCount b.val).toNat ≤ 13)
    (covers : Root.Covers mc b (rows.map Row.square)) (mobile : 0 < rows.length)
    (pc : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    (∃ prior, worker.2.visible mc fc = some prior ∧ Physical.Published p c b inputs roots rows m prior sort nodes fallback) ∧
    ∀ v ∈ capture.native.timeline.visits, ∃ audit ∈ capture.finished,
      EndgameSearch.CacheHistory.Payloads.completeVisit p capture.native.timeline mc fc values stream v = some audit ∧
      EndgameSearch.SerialLeafCache.Reachable p mc fc audit.after := by
  have obs := EndgameSearch.CacheHistory.Payloads.observed captured
  have parsed := EndgameSearch.CacheHistory.Payloads.capture_workers p stream capture.native obs.1 worker registered
  obtain ⟨visible,_,_⟩ := EndgameSearch.SerialWarmCache.native_closed p mc fc worker.1 stream worker.2 parsed outside
  refine ⟨⟨_,visible,thirteen_warm p c address b inputs roots rows m worker.2.completed sort nodes fallback
    small covers mobile pc window⟩,?_⟩
  intro v belongs
  obtain ⟨audit,member,accepted⟩ := EndgameSearch.CacheHistory.Payloads.mapped obs.2.2.1 belongs
  exact ⟨audit,member,accepted,EndgameSearch.SerialWarmCache.physical_visit p mc fc _ values stream v audit accepted⟩

end NeuralReversi.AspirationMir.SerialCache.Warm
