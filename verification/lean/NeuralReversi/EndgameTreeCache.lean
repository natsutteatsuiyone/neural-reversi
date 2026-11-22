import NeuralReversi.PvsBoardTree
import NeuralReversi.EndgameCache

namespace NeuralReversi.EndgameTreeCache

open Board (Board)
open PvsBoardTree (Position GeneratedSquare minimax legalSquares play pass)
open Bitboard.Moves (X86Config)
open EndgameCache (Entry Write CacheRange InRange StoreRange SearchRange)
private abbrev FlipConfig := Flip.Dispatch.X86Config

private theorem bestValue_dvd {Move : Type} (value : Move → Int) (moves : List Move)
    (seed : Int) (seedDvd : 256 ∣ seed) (all : ∀ m ∈ moves, 256 ∣ value m) :
    256 ∣ PvsMoveLoop.bestValue value moves seed := by
  induction moves generalizing seed with
  | nil => exact seedDvd
  | cons first tail ih =>
    rw [PvsMoveLoop.bestValue]
    apply ih
    · simp only [Int.max_def]
      split
      · exact all first (by simp)
      · exact seedDvd
    · intro move later
      exact all move (List.mem_cons_of_mem _ later)

theorem minimax_disc_multiple (mc : X86Config) (fc : FlipConfig) (b : Position) :
    256 ∣ minimax mc fc b := by
  cases moves : legalSquares mc b.val with
  | nil =>
    cases other : Board.hasLegalMoves mc (pass b).val
    · rw [PvsBoardTree.terminal_value mc fc b moves other]
      exact ⟨_, rfl⟩
    · have forced : Board.ForcedPass mc b.val := ⟨PvsBoardTree.stuck_of_nil moves, other⟩
      rw [PvsBoardTree.minimax_pass mc fc b moves other]
      exact Int.dvd_neg.mpr (minimax_disc_multiple mc fc (pass b))
  | cons first rest =>
    let value : PvsBoardTree.GeneratedSquare mc b.val → Int :=
      fun sq => -minimax mc fc (play mc fc b sq).val
    have all (sq : PvsBoardTree.GeneratedSquare mc b.val) : 256 ∣ value sq :=
      Int.dvd_neg.mpr (minimax_disc_multiple mc fc (play mc fc b sq).val)
    have bounded := (PvsMoveLoop.minimax_range
      (PvsBoardTree.build mc fc (play mc fc b first).val)).2
    change minimax mc fc (play mc fc b first).val ≤ PvsMoveLoop.scoreMax at bounded
    have above : -PvsMoveLoop.scoreInf ≤ value first := by
      unfold PvsMoveLoop.scoreInf PvsMoveLoop.scoreMax at *
      dsimp only [value]
      omega
    rw [PvsBoardTree.minimax_moves mc fc b first rest moves]
    change 256 ∣ PvsMoveLoop.bestValue value (first :: rest) (-PvsMoveLoop.scoreInf)
    rw [PvsMoveLoop.bestValue, Int.max_eq_right above]
    exact bestValue_dvd value rest _ (all first) (fun sq _ => all sq)
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass forced) b.property
  · exact (play mc fc b sq).property

def discValue (mc : X86Config) (fc : FlipConfig) (b : Board) : Int :=
  if separated : b.player &&& b.opponent = 0#64 then minimax mc fc ⟨b, separated⟩ / 256
  else 0

theorem scaled_discValue (mc : X86Config) (fc : FlipConfig) (b : Position) :
    256 * discValue mc fc b.val = minimax mc fc b := by
  have separated : b.val.player &&& b.val.opponent = 0#64 := b.property
  simp only [discValue, separated, dite_true]
  exact Int.mul_ediv_cancel' (minimax_disc_multiple mc fc b)

theorem discValue_range (mc : X86Config) (fc : FlipConfig) (b : Board) :
    InRange (discValue mc fc b) := by
  unfold discValue InRange
  split
  · rename_i wf
    have range := PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc ⟨b, wf⟩)
    change -PvsMoveLoop.scoreMax ≤ minimax mc fc ⟨b, wf⟩ ∧
      minimax mc fc ⟨b, wf⟩ ≤ PvsMoveLoop.scoreMax at range
    unfold PvsMoveLoop.scoreMax at range
    omega
  · decide

private theorem bestValue_scale {Move : Type} (value : Move → Int)
    (moves : List Move) (seed : Int) :
    256 * PvsMoveLoop.bestValue value moves seed =
      PvsMoveLoop.bestValue (fun move => 256 * value move) moves (256 * seed) := by
  induction moves generalizing seed with
  | nil => rfl
  | cons move rest ih =>
    simp only [PvsMoveLoop.bestValue, ih]
    have scale : 256 * max seed (value move) = max (256 * seed) (256 * value move) := by omega
    rw [scale]

theorem discValue_moves (mc : X86Config) (fc : FlipConfig) (b : Position)
    (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
    (moves : legalSquares mc b.val = first :: rest) :
    discValue mc fc b.val = PvsMoveLoop.bestValue
      (fun sq => -discValue mc fc (play mc fc b sq).val.val) (first :: rest) (-127) := by
  let value : GeneratedSquare mc b.val → Int := fun sq => -discValue mc fc (play mc fc b sq).val.val
  have lo := (discValue_range mc fc (play mc fc b first).val.val).2
  have above : -127 ≤ value first := by dsimp only [value]; omega
  have scaledFirst : -PvsMoveLoop.scoreInf ≤ 256 * value first := by
    unfold PvsMoveLoop.scoreInf
    omega
  have f (sq : GeneratedSquare mc b.val) :
      -minimax mc fc (play mc fc b sq).val = 256 * value sq := by
    rw [← scaled_discValue]
    dsimp only [value]
    omega
  have eq := scaled_discValue mc fc b
  rw [PvsBoardTree.minimax_moves mc fc b first rest moves] at eq
  simp only [f] at eq
  have right := bestValue_scale value (first :: rest) (-127)
  rw [PvsMoveLoop.bestValue, Int.max_eq_right scaledFirst] at eq
  rw [PvsMoveLoop.bestValue, Int.max_eq_right above, PvsMoveLoop.bestValue,
    Int.max_eq_right (show (256 : Int) * -127 ≤ 256 * value first by omega)] at right
  change discValue mc fc b.val = PvsMoveLoop.bestValue value (first :: rest) (-127)
  rw [PvsMoveLoop.bestValue, Int.max_eq_right above]
  omega

inductive Table where
  | deep | shallow
  deriving DecidableEq

abbrev Address := Table → Board → Nat

structure Store where
  table : Table
  write : Write

def soundHistory (mc : X86Config) (fc : FlipConfig) (history : List Store) : Prop :=
  ∀ event ∈ history, StoreRange event.write.score ∧
    Solve2.NWS event.write.alpha (discValue mc fc event.write.board) event.write.score

def slotWrites (address : Address) (history : List Store) (table : Table) (board : Board) : List Write :=
  history.filterMap fun event =>
    if event.table = table ∧ address table event.write.board = address table board then some event.write
    else none

def entryFor (address : Address) (history : List Store) (table : Table) (board : Board) : Entry :=
  (slotWrites address history table board).foldl EndgameCache.store (EndgameCache.unset ⟨0, 0⟩)

private theorem slot_sound {mc : X86Config} {fc : FlipConfig} {history : List Store}
    (sound : soundHistory mc fc history) (address : Address) (table : Table) (board : Board) :
    ∀ w ∈ slotWrites address history table board,
      StoreRange w.score ∧ Solve2.NWS w.alpha (discValue mc fc w.board) w.score := by
  intro w mem
  obtain ⟨event, found, selected⟩ := List.mem_filterMap.mp mem
  split at selected
  · cases selected
    exact sound event found
  · contradiction

theorem append_sound (mc : X86Config) (fc : FlipConfig) (history : List Store)
    (sound : soundHistory mc fc history) (event : Store) (range : StoreRange event.write.score)
    (result : Solve2.NWS event.write.alpha (discValue mc fc event.write.board) event.write.score) :
    soundHistory mc fc (history ++ [event]) := by
  intro w found
  rcases List.mem_append.mp found with old | new
  · exact sound w old
  · rw [List.mem_singleton] at new
    subst w
    exact ⟨range, result⟩

theorem probe_sound {mc : X86Config} {fc : FlipConfig} {history : List Store}
    (sound : soundHistory mc fc history) (address : Address) (table : Table) (board : Board) (alpha result : Int)
    (hit : EndgameCache.probe (entryFor address history table board) board alpha = some result) :
    CacheRange result ∧ Solve2.NWS alpha (discValue mc fc board) result :=
  EndgameCache.anyWindowContract_proven (discValue mc fc) (discValue_range mc fc)
    (slotWrites address history table board) (slot_sound sound address table board) board alpha result hit

theorem rawWindow_sound (mc : X86Config) (fc : FlipConfig) (b : Position) (alpha result : Int)
    (range : SearchRange result)
    (bounds : Solve2.NWS (alpha / 256) (discValue mc fc b.val) result) :
    Aspiration.SoundResult (minimax mc fc b) alpha (alpha + 1) (256 * result) := by
  have scaled := scaled_discValue mc fc b
  unfold SearchRange at range
  refine ⟨by omega, by omega, ?_, ?_, fun h₁ h₂ => by omega⟩
  · intro low
    have := bounds.1 (by omega)
    omega
  · intro high
    have := bounds.2 (by omega)
    omega

def ProbeContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (history : List Store)
    (address : Address) (table : Table) (alpha result : Int),
    soundHistory mc fc history →
    EndgameCache.probe (entryFor address history table b.val) b.val (alpha / 256) = some result →
    CacheRange result ∧ Aspiration.SoundResult (minimax mc fc b) alpha (alpha + 1) (256 * result) ∧
      soundHistory mc fc (history ++ [⟨table, ⟨b.val, alpha / 256, result⟩⟩])

theorem probeContract_proven : ProbeContract := by
  intro mc fc b history address table alpha result sound hit
  obtain ⟨range, bounds⟩ := probe_sound sound address table b.val (alpha / 256) result hit
  exact ⟨range, rawWindow_sound mc fc b alpha result (EndgameCache.cacheRange_search range) bounds,
    append_sound mc fc history sound _
      (EndgameCache.searchRange_store (EndgameCache.cacheRange_search range)) bounds⟩

def EnhancedContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (history : List Store)
    (address : Address) (alpha result : Int) (moves : List (PvsBoardTree.GeneratedSquare mc b.val)),
    soundHistory mc fc history →
    (∀ sq ∈ moves, sq ∈ legalSquares mc b.val) →
    EndgameCache.enhancedCutoff (entryFor address history .deep) alpha
      (moves.map fun sq => (play mc fc b sq).val.val) = some result →
    CacheRange result ∧ alpha < result ∧ 256 * result ≤ minimax mc fc b

theorem enhancedContract_proven : EnhancedContract := by
  intro mc fc b history address alpha result moves sound legal hit
  obtain ⟨child, member, candidate⟩ := List.exists_of_findSome?_eq_some hit
  obtain ⟨sq, found, same⟩ := List.mem_map.mp member
  obtain ⟨upper, probe, selected⟩ := Option.bind_eq_some_iff.mp candidate
  by_cases qualifies : upper ≤ -(alpha + 1)
  · simp [qualifies] at selected
    obtain ⟨range, bound⟩ := probe_sound sound address .deep child (-(alpha + 1)) upper probe
    have upperBound := bound.1 qualifies
    have scaled := scaled_discValue mc fc (play mc fc b sq).val
    change (play mc fc b sq).val.val = child at same
    rw [same] at scaled
    have member := legal sq found
    cases h : legalSquares mc b.val with
    | nil => rw [h] at member; exact False.elim (List.not_mem_nil member)
    | cons first rest =>
      rw [h] at member
      rw [PvsBoardTree.minimax_moves mc fc b first rest h]
      have below := PvsMoveLoop.bestValue_ge_move (fun m => -minimax mc fc (play mc fc b m).val)
        (first :: rest) (-PvsMoveLoop.scoreInf) member
      unfold CacheRange at range ⊢
      omega
  · simp [qualifies] at selected

end NeuralReversi.EndgameTreeCache
