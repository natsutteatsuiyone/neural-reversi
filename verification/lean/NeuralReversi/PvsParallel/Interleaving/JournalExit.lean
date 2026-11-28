module
public import NeuralReversi.PvsParallel.Interleaving.JournalRoot

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalExit

open TranspositionTable
open PvsBoardTree (Position)
open AspirationMir
open AspirationMir.SerialTable
open PublicationMir (SliceSort SearchResult)

/-- The midgame has completed one depth. The exact continuation draws each endgame
    trial from the same per-worker tape used by direct entry. -/
def handoff (e : JournalEntry.Execution mc fc) (b : Position) (started : MidgameIterate.Completed M)
    (entry : Nat) (feed : JournalRoot.Feed e b) (multi : Bool) (clock : MultiPublic.Clock)
    (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) := do
  let state ← CallReturns.captureTimed (e.events.map Joint.Event.frameSample)
  let base := BitVec.ofInt 32 started.base.score
  let solve ← SolveEntry.timed e.events state entry ⟨b,base,started.state.rows,some started.saved,multi,2,5⟩
  RootNative.solve e.capture.history b (JournalRoot.calls e b state solve feed) multi clock 5 2 (some started.saved)
    base started.state.rows 0 sort nodes fallback

theorem after_midgame (e : JournalEntry.Execution mc fc) (b : Position) (trial : MidgameIterate.Trial M)
    (works : MidgameIterate.Works mc b safe trial) (initial : MidgameIterate.State M)
    (valid : MidgameIterate.Valid mc b safe initial)
    (crossed : MidgameIterate.run trial midSort level depth midClock initial = some (.endgame phase))
    (feed : JournalRoot.Feed e b) (multi : Bool) (clock : MultiPublic.Clock)
    (time : clock.ordered ((if multi then phase.state.rows.length else 1)-1))
    (normal : handoff e b phase entry feed multi clock sort nodes fallback = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report) := by
  obtain ⟨out,ran,ready⟩ := MidgameIterate.returns works initial valid midSort level depth midClock
  rw [crossed] at ran
  cases ran
  obtain ⟨state,_,rest⟩ := Option.bind_eq_some_iff.mp normal
  obtain ⟨solve,_,result⟩ := Option.bind_eq_some_iff.mp rest
  exact Entry.handoff (JournalRoot.calls e b state solve feed) phase ready multi clock time result tag full

structure CacheRead (e : JournalEntry.Execution mc fc) (b : Position) (data : Option EntryData) where
  consumer : Nat
  call : JointReads.Call e.atomics consumer (.lookup b)
  consumed : Joint.receiverAt e.events consumer b = some (call.finish,.lookup data)

open PublicationMir.ExactScore (Ensures returned)

def FallbackContract : Prop := ∀ mc fc (e : JournalEntry.Execution mc fc) (b : Position) evaluate
    (sort : SliceSort) (data : Option EntryData) nodes (exit : PublicationMir.ExactScore.Exit) result,
    returned mc fc b evaluate sort none data nodes exit = some result →
    (exit = .search → CacheRead e b data) → Ensures mc fc b result

/-- When no iteration was saved, classify the actual public return. A flagged
    cache fallback consumes an occupied, exact endgame entry on this same board. -/
theorem fallback : FallbackContract := by
  intro mc fc e b evaluate sort data nodes exit result published lookup v tagged
  obtain ⟨searched,snapshot | cached⟩ := PublicationMir.ExactScore.origin published tagged
  · obtain ⟨_,absent,_⟩ := snapshot
    contradiction
  obtain ⟨consumer,r,consumed⟩ := lookup searched
  obtain ⟨_,entry,hit,exactTag,ending,level,rfl,_⟩ := cached
  have checked := Joint.checked e.events e.atomics ⟨.lookup b,r.allocation,r.clusters,r.key⟩
    r.finish consumer r.opening r.decoded
  let seen := e.readers.finished ⟨consumer,r.later,r.same⟩ checked
  have observed := ReaderWindow.lookup_hit e.readers ⟨consumer,r.later,r.same⟩ e.init b seen r.nowrap (hit ▸ Joint.received e.events consumed)
  let h := e.capture.history
  have safe := (CausalTable.certified h.native h.init h.exactRun RootCausal.preserves h.safe).2 consumer b entry observed
  rw [selectivityRank_eq_min] at level
  have eligible : Cutoffs.ExactEndgame entry := ⟨(byte_masks entry.flags).1 ▸ ending,by omega⟩
  have bounds := (safe.1 eligible).1
  rw [EndgameTreeCache.scaled_discValue mc fc b]
  exact Int.le_antisymm (bounds.1 (by rw [exactTag]; decide)) (bounds.2 (by rw [exactTag]; decide))

end NeuralReversi.PvsParallel.Interleaving.JournalExit
