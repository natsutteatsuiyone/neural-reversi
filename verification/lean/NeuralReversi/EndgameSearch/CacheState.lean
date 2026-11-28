module
public import NeuralReversi.EndgameSearch.Cancellation

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheState

open EndgameTreeCache (Address Table Store entryFor slotWrites soundHistory)
open EndgameCache (Entry)
open PvsBoardTree (Position)

abbrev Cells := Table → Array Entry

def empty (length : Table → Nat) : Cells :=
  fun t => Array.replicate (length t) (EndgameCache.unset ⟨0, 0⟩)

def write (address : Address) (cells : Cells) (event : Store) : Cells := fun t =>
  if t = event.table then
    let index := address t event.write.board
    (cells t).modify index (fun prior => EndgameCache.store prior event.write)
  else cells t

def represents (address : Address) (history : List Store) (cells : Cells) : Prop :=
  ∀ t b, (cells t)[address t b]? = some (entryFor address history t b)

theorem empty_represents (address : Address) (length : Table → Nat)
    (fit : ∀ t b, address t b < length t) : represents address [] (empty length) := by
  intro t b
  simp [empty, fit, entryFor, slotWrites]

theorem write_represents (event : Store) (prior : represents address history cells) :
    represents address (history ++ [event]) (write address cells event) := by
  intro t b
  have before := prior t b
  have atSlot (a : Address) (h : List Store) (t : Table) (b : Board.Board) :
      entryFor a (h ++ [event]) t b =
        if event.table = t ∧ a t event.write.board = a t b then
          EndgameCache.store (entryFor a h t b) event.write else entryFor a h t b := by
    by_cases matchesSlot : event.table = t ∧ a t event.write.board = a t b <;>
      simp [entryFor, slotWrites, matchesSlot]
  rw [atSlot]
  by_cases sameTable : t = event.table
  · rw [write, ite_eq_left sameTable, Array.getElem?_modify]
    by_cases sameSlot : address t event.write.board = address t b <;>
      simp only [sameTable.symm, true_and, sameSlot, ite_true, ite_false, before, Option.map_some]
  · simpa [write, sameTable, Ne.symm sameTable] using before

theorem replay_represents (prior : represents address history cells) (writes : List Store) :
    represents address (history ++ writes) (writes.foldl (write address) cells) := by
  induction writes generalizing history cells with
  | nil => simpa using prior
  | cons e rest ih => simpa using ih (write_represents e prior)

theorem probe_bound (prior : represents address history cells) (valid : soundHistory mc fc history)
    (t : Table) (b : Position) (alpha score : Int)
    (returned : ((cells t)[address t b.val]?).bind (fun e => EndgameCache.probe e b.val alpha) = some score) :
    EndgameCache.CacheRange score ∧ Solve2.NWS alpha (EndgameTreeCache.discValue mc fc b.val) score := by
  rw [prior t b.val] at returned
  exact EndgameTreeCache.probe_sound valid address t b.val alpha score returned

/-- A deep call can stop at a split cutoff after completing some of its children. -/
inductive Call where
  | shallow (b : Position) (alpha : ScoreRefresh.Score)
  | deep (b : Position) (alpha : ScoreRefresh.Score) (empties : Cancellation.EmptyState)
      (polls : Threading.SplitCutoff.Polling) (aligned : empties.matches b)

def Call.historyAfter (mc : Bitboard.Moves.X86Config) (fc : FlipConfig)
    (options : Options mc) (history : List Store) : Call → List Store
  | .shallow b alpha =>
    (run mc fc options .shallow b (ScoreRefresh.toDiscDiff alpha).raw.toInt history).history
  | .deep b alpha empties polls _ =>
    (Cancellation.run mc fc options b (ScoreRefresh.toDiscDiff alpha).raw.toInt ⟨empties, history, polls⟩).state.history

theorem Call.preserves (call : Call) (correct : CorrectOptions mc fc opts) (valid : soundHistory mc fc history) :
    let after := call.historyAfter mc fc opts history
    soundHistory mc fc after ∧ history <+: after := by
  have safe (s : ScoreRefresh.Score) : SafeThreshold (ScoreRefresh.toDiscDiff s).raw.toInt := by
    rw [ScoreRefresh.disc_floor]
    have lower := BitVec.le_toInt s.raw
    have upper := BitVec.toInt_lt (x := s.raw)
    change -2147483648 < _ ∧ _ < 2147483647
    constructor <;> omega
  cases call with
  | shallow b alpha =>
    exact (contract_proven mc fc opts .shallow b _ history correct valid (fun _ _ => safe alpha)).2.2
  | deep b alpha empties polls aligned =>
    have kept := Cancellation.contract_proven mc fc opts b _ ⟨empties, history, polls⟩ correct aligned valid (safe alpha)
    exact ⟨kept.2.1, kept.2.2.1⟩

theorem after_calls (calls : List Call) (correct : CorrectOptions mc fc opts)
    (valid : soundHistory mc fc history) (prior : represents opts.address history cells) :
    let after := calls.foldl (fun h c => c.historyAfter mc fc opts h) history
    soundHistory mc fc after ∧
      represents opts.address after ((after.drop history.length).foldl (write opts.address) cells) := by
  have grows (h : List Store) (sound : soundHistory mc fc h) :
      let after := calls.foldl (fun past c => c.historyAfter mc fc opts past) h
      soundHistory mc fc after ∧ h <+: after := by
    induction calls generalizing h with
    | nil => exact ⟨sound, List.prefix_refl _⟩
    | cons first rest ih =>
      obtain ⟨next, grew⟩ := first.preserves correct sound
      obtain ⟨last, more⟩ := ih _ next
      exact ⟨last, grew.trans more⟩
  obtain ⟨sound, appended, eq⟩ := grows history valid
  refine ⟨sound, ?_⟩
  rw [← eq]
  simpa using replay_represents prior appended

end NeuralReversi.EndgameSearch.CacheState
