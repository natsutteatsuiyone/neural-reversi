import NeuralReversi.PvsParallel.RootReplay.Split

namespace NeuralReversi.PvsParallel.RootAspiration

open Aspiration (Accepted Trials Window)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)
open PvsMoveLoop (scoreInf scoreMax)
open RootOrder (SoundAttempt sortFrom)

abbrev Search := Trials (Square → Int)

structure Pick where
  square : Square
  score : Int

structure Line where
  pick : Pick
  order : List Square

def finish (roots : List Square) (idx : Nat) (accepted : Accepted (Square → Int)) : Option Line :=
  let sorted := sortFrom roots idx accepted.trial.payload
  match sorted.drop idx with
  | [] => none
  | head :: _ => some ⟨⟨head, accepted.trial.score⟩, sorted⟩

def runLine (search : Search) (roots : List Square) (idx : Nat) (alpha beta : Int) : Option Line :=
  (Aspiration.runWith search 11 0 alpha beta 256).bind (finish roots idx)

def initialWindow (idx : Nat) (base : Int) (previous : Option Int) : Int × Int :=
  if idx = 0 then (base - 256, base + 256)
  else (-scoreInf, (previous.map fun score => min (score + 1) scoreInf).getD scoreInf)

def nextLevelWindow (score : Int) : Int × Int :=
  (max (score - 256) (-scoreInf), min (score + 256) scoreInf)

def SeedWindowsContract : Prop :=
  (∀ (idx : Nat) (base : Int) (previous : Option Int),
    -scoreMax ≤ base → base ≤ scoreMax → (∀ score ∈ previous, -scoreMax ≤ score) →
    Window (initialWindow idx base previous).1 (initialWindow idx base previous).2) ∧
  (∀ (score : Int), -scoreMax ≤ score → score ≤ scoreMax →
    Window (nextLevelWindow score).1 (nextLevelWindow score).2 ∧
      (nextLevelWindow score).1 < score ∧ score < (nextLevelWindow score).2)

theorem seedWindowsContract_proven : SeedWindowsContract := by
  constructor
  · intro idx base previous lower upper previousLower
    by_cases first : idx = 0
    · rw [initialWindow, ite_eq_left first]
      unfold scoreMax at lower upper
      exact ⟨by omega, by omega, by omega⟩
    · rw [initialWindow, ite_eq_right first]
      cases previous with
      | none => exact ⟨by decide, by decide, by decide⟩
      | some score =>
        have lower := previousLower score (by simp)
        unfold scoreMax at lower
        exact ⟨Int.le_refl _, by change -32767 < min (score + 1) 32767; omega,
          Int.min_le_right _ _⟩
  · intro score lower upper
    unfold nextLevelWindow scoreInf
    unfold scoreMax at lower upper
    exact ⟨⟨by omega, by omega, by omega⟩, by omega, by omega⟩

def SoundSearch (mc : X86Config) (fc : Root.FlipConfig) (board : Position) (roots : List Square)
    (idx : Nat) (search : Search) : Prop :=
  ∀ i a b, Window a b → (search i a b).aborted = false →
    SoundAttempt mc fc board roots idx a b (search i a b).score (search i a b).payload

structure LineSound (values : Square → Int) (roots : List Square) (idx : Nat) (line : Line) : Prop where
  order : line.order.Perm roots
  fixed : line.order.take idx = roots.take idx
  chosen : (line.order.drop idx).head? = some line.pick.square
  legal : line.pick.square ∈ roots.drop idx
  exactScore : line.pick.score = values line.pick.square
  maximal : ∀ other ∈ roots.drop idx, values other ≤ line.pick.score
  remaining : (line.order.drop (idx + 1)).Perm ((roots.drop idx).erase line.pick.square)

theorem line_of_interior {mc : X86Config} {fc : Root.FlipConfig} {board : Position}
    (inside : idx < roots.length) {accepted : Accepted (Square → Int)}
    (interior : accepted.alpha < accepted.trial.score ∧ accepted.trial.score < accepted.beta)
    (report : SoundAttempt mc fc board roots idx accepted.alpha accepted.beta accepted.trial.score
      accepted.trial.payload) :
    ∃ line, finish roots idx accepted = some line ∧
      line.pick.score = accepted.trial.score ∧
      accepted.trial.payload line.pick.square = line.pick.score ∧
      LineSound (Root.squareValue mc fc board) roots idx line := by
  have exactScore := report.1.exact interior.1 interior.2
  obtain ⟨head, tail, sorted, selected⟩ := report.2 interior.1 interior.2
  let ordered := sortFrom roots idx accepted.trial.payload
  let line : Line := ⟨⟨head, accepted.trial.score⟩, ordered⟩
  obtain ⟨order, fixed, remaining, _, _⟩ :=
    RootOrder.sortContract_proven roots idx accepted.trial.payload (Nat.le_of_lt inside)
  refine ⟨line, ?_, rfl, selected.2.1.trans exactScore.symm, ?_⟩
  · simp only [finish, sorted]
    rfl
  · have exactPick := exactScore.trans selected.2.2.1.symm
    refine ⟨order, fixed, ?_, selected.1, exactPick, ?_, ?_⟩
    · change (ordered.drop idx).head? = some head
      rw [sorted]
      rfl
    · intro other member
      change Root.squareValue mc fc board other ≤ accepted.trial.score
      rw [exactPick]
      exact selected.2.2.2.1 other member
    · change (ordered.drop (idx + 1)).Perm ((roots.drop idx).erase head)
      rw [List.drop_add_one_eq_tail_drop, sorted]
      rw [sorted] at remaining
      simpa only [List.erase_cons_head, List.tail_cons] using remaining.erase head

theorem line_of_accepted {mc : X86Config} {fc : Root.FlipConfig} {board : Position}
    (inside : idx < roots.length) (witness : Aspiration.Witness search start fuel accepted)
    (report : SoundAttempt mc fc board roots idx accepted.alpha accepted.beta accepted.trial.score
      accepted.trial.payload) :
    ∃ line, finish roots idx accepted = some line ∧
      line.pick.score = accepted.trial.score ∧
      accepted.trial.payload line.pick.square = line.pick.score ∧
      LineSound (Root.squareValue mc fc board) roots idx line :=
  line_of_interior inside witness.inside report

def LineContract : Prop :=
  ∀ (mc : X86Config) (fc : Root.FlipConfig) (board : Position) (roots : List Square) (idx : Nat)
    (alpha beta : Int) (search : Search),
    Root.Covers mc board roots → idx < roots.length → Window alpha beta →
    SoundSearch mc fc board roots idx search →
    (∀ i a b, Window a b → (search i a b).aborted = false) →
    ∃ line accepted, Aspiration.runWith search 11 0 alpha beta 256 = some accepted ∧
      Aspiration.Witness search 0 11 accepted ∧
      runLine search roots idx alpha beta = some line ∧
      line.pick.score = accepted.trial.score ∧
      accepted.trial.payload line.pick.square = line.pick.score ∧
      LineSound (Root.squareValue mc fc board) roots idx line

theorem lineContract_proven : LineContract := by
  intro mc fc board roots idx alpha beta search covers inside window sound uninterrupted
  let value := Root.value mc fc board roots idx
  have numeric : Aspiration.TrialSound search value := fun i a b valid done =>
    (sound i a b valid done).1
  obtain ⟨low, high⟩ := Root.value_range covers (fc := fc) inside
  have lo : -32767 < value := by change -32767 < Root.value mc fc board roots idx; unfold PvsMoveLoop.scoreMax at low; omega
  have hi : value < 32767 := by change Root.value mc fc board roots idx < 32767; unfold PvsMoveLoop.scoreMax at high; omega
  obtain ⟨accepted, found, witness, _⟩ := Aspiration.acceptedContract_proven
    _ search value alpha beta numeric lo hi window uninterrupted
  have report := sound accepted.index accepted.alpha accepted.beta witness.window
    (witness.source ▸ witness.completed)
  rw [← witness.source] at report
  obtain ⟨line, finished, eq, score, chosen⟩ := line_of_accepted inside witness report
  exact ⟨line, accepted, found, witness, by simpa [runLine, found] using finished, eq, score, chosen⟩

theorem aborted_line (search : Search) (roots : List Square) (idx : Nat)
    (aborted : (search 0 alpha beta).aborted = true) :
    runLine search roots idx alpha beta = none := by
  rw [runLine, Aspiration.runWith_aborted search aborted]
  rfl

end NeuralReversi.PvsParallel.RootAspiration
