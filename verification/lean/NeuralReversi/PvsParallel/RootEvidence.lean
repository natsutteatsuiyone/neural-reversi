module
public import NeuralReversi.ExactEndgameScore
public import NeuralReversi.PvsParallel.Replay
public import NeuralReversi.PvsParallel.RootReplay.Split

@[expose] public section

namespace NeuralReversi.PvsParallel.RootEvidence

open Aspiration (Window)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play build minimax)
open PvsMoveLoop (ChildSearch ScanCalls Tree scoreInf scoreMax)
open Threading.SplitWindow (reSearch)
open RootReplay (serial start)
open RootOrder (SoundAttempt)
open SearchPublication (Row capture)

abbrev FlipConfig := Flip.Dispatch.X86Config

abbrev child (mc : X86Config) (fc : FlipConfig) (b : Position)
    (move : GeneratedSquare mc b.val) : Tree :=
  build mc fc (play mc fc b move).val

abbrev squareAt (mc : X86Config) (b : Position) (first : GeneratedSquare mc b.val)
    (rest : List (GeneratedSquare mc b.val)) (i : Nat) : Square :=
  (((first :: rest)[i]?).getD first).val

inductive AttemptWithUses (completed : ChildResults) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (roots : List Square)
    (idx : Nat) (alpha beta : Int) : Int → (Square → Int) → Prop
  | sequential (moves : List (GeneratedSquare mc b.val))
      (observe : ChildSearch (GeneratedSquare mc b.val)) :
      moves.Perm (Root.choices mc b roots idx) →
      ScanCalls (fun m => completed .serial (child mc fc b m)) observe true beta moves true alpha (-scoreInf) →
      AttemptWithUses completed mc fc b roots idx alpha beta
        (serial Subtype.val observe beta moves (start alpha)).state.best
        (serial Subtype.val observe beta moves (start alpha)).scores
  | split (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
      (observe : ChildSearch Nat)
      (split : SplitExecution ((first :: rest).map (child mc fc b)).length true alpha beta score) :
      (first :: rest).Perm (Root.choices mc b roots idx) → 0 < split.start.toNat →
      Prefix.run true observe beta (List.range split.start.toNat) true ⟨alpha, -scoreInf⟩ =
        some split.entry →
      ScanCalls (fun i => completed .serial (childAt ((first :: rest).map (child mc fc b)) i))
        observe true beta (List.range split.start.toNat) true alpha (-scoreInf) →
      ReportCalls completed ((first :: rest).map (child mc fc b)) split →
      AttemptWithUses completed mc fc b roots idx alpha beta score
        (RootReplay.Split.replay (squareAt mc b first rest) beta split.reports
          (serial (squareAt mc b first rest) observe beta (List.range split.start.toNat)
            (start alpha))).scores
  | wipeout (move : GeneratedSquare mc b.val) :
      move ∈ Root.choices mc b roots idx → EndgameCutoffs.wipesOut fc b.val move.val = true →
      AttemptWithUses completed mc fc b roots idx alpha beta scoreMax
        ((start alpha).record beta ⟨move.val, alpha, scoreMax⟩).scores

abbrev AttemptWith (completed : Search) := AttemptWithUses (fun _ => completed)

abbrev Attempt := AttemptWith Replayed

theorem AttemptWithUses.boardExecution
    (attempt : AttemptWithUses children mc fc b roots 0 alpha beta score rows)
    (covers : Root.Covers mc b roots) (nonempty : 0 < roots.length)
    (wipes : ∀ m ∈ legalSquares mc b.val, EndgameCutoffs.wipesOut fc b.val m.val = true →
      exit mc fc b true alpha beta scoreMax) :
    ExecutionWithUses exit children (build mc fc b) true alpha beta score := by
  have selected := Root.choices_zero covers
  have turn : ∃ first rest, legalSquares mc b.val = first :: rest := by
    apply List.exists_cons_of_ne_nil
    intro nil
    have len : roots.length = 0 := by rw [covers.length_eq, nil]; rfl
    omega
  obtain ⟨first, rest, list⟩ := turn
  have board : build mc fc b = .turn (child mc fc b first) (rest.map (child mc fc b)) := by
    rw [build.eq_def]
    split
    next head tail found =>
      obtain ⟨rfl,rfl⟩ := List.cons.inj (list.symm.trans found)
      rfl
    next found => rw [list] at found; cases found
  cases attempt with
  | sequential moves see perm calls =>
    rw [board, RootReplay.serial_best]
    have permutation := perm.map (child mc fc b)
    rw [selected, list] at permutation
    exact .sequential moves (child mc fc b) permutation see calls
  | split head tail see division perm _ continued calls reports =>
    rw [board]
    exact .split _ (by simpa only [selected,list,List.map_cons] using perm.map (child mc fc b))
      see division continued calls reports
  | wipeout m legal hit =>
    exact .cutoff mc fc b (wipes m (selected ▸ legal) hit)

theorem AttemptWithUses.ofSerialRun {mc : X86Config} {b : Position}
    {q remaining : PvsPostpone.Queue (GeneratedSquare mc b.val)}
    {observe : ChildSearch (GeneratedSquare mc b.val)} {visited : List (GeneratedSquare mc b.val)}
    (order : q.pending.Perm (Root.choices mc b roots idx)) (first : q.idx = 0)
    (trace : Prefix.Serial.Run (fun move : GeneratedSquare mc b.val => completed .serial (child mc fc b move)) observe true beta q
      ⟨alpha, -scoreInf⟩ visited remaining (.returned score)) :
    AttemptWithUses completed mc fc b roots idx alpha beta score
      (serial Subtype.val observe beta (visited ++ remaining.pending) (start alpha)).scores := by
  obtain ⟨ordered, _, reproduced⟩ := trace.reproduced
  have scanEvidence : PvsMoveLoop.scan true observe beta (visited ++ remaining.pending) true alpha (-scoreInf) = score ∧
      ScanCalls (fun move => completed .serial (child mc fc b move)) observe true beta
        (visited ++ remaining.pending) true alpha (-scoreInf) := by
    simpa [Prefix.Serial.Reproduced, first] using reproduced
  have scoreEq := (RootReplay.serial_best Subtype.val observe beta
    (visited ++ remaining.pending) (start alpha)).trans scanEvidence.1
  rw [← scoreEq]
  exact .sequential _ observe (ordered.trans order) scanEvidence.2

theorem AttemptWithUses.ofSerialHandoff {mc : X86Config} {b : Position}
    {q remaining : PvsPostpone.Queue (GeneratedSquare mc b.val)}
    {observe : ChildSearch (GeneratedSquare mc b.val)} {visited : List (GeneratedSquare mc b.val)}
    (firstMove : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
    (order : q.pending.Perm (Root.choices mc b roots idx)) (first : q.idx = 0)
    (labels : firstMove :: rest = visited ++ remaining.pending)
    (split : SplitExecution ((firstMove :: rest).map (child mc fc b)).length true alpha beta score)
    (beginAt : split.start.toNat = remaining.idx) (started : 0 < split.start.toNat)
    (trace : Prefix.Serial.Run (fun m : GeneratedSquare mc b.val => completed .serial (child mc fc b m))
      observe true beta q ⟨alpha, -scoreInf⟩ visited remaining (.handoff split.entry))
    (reports : ReportCalls completed ((firstMove :: rest).map (child mc fc b)) split) :
    let scoreAt : ChildSearch Nat := fun i => observe (((firstMove :: rest)[i]?).getD firstMove)
    let square := squareAt mc b firstMove rest
    let before := serial square scoreAt beta (List.range split.start.toNat) (start alpha)
    AttemptWithUses completed mc fc b roots idx alpha beta score
      (RootReplay.Split.replay square beta split.reports before).scores := by
  have indexed := trace.handoffOrdered
    (target := fun i => completed .serial (childAt ((firstMove :: rest).map (child mc fc b)) i))
    first (firstMove :: rest) firstMove labels (by
      intro i inside p low high s used
      rwa [childAt_map (child mc fc b) (firstMove :: rest) firstMove inside])
  rw [← beginAt] at indexed
  have permutation : (firstMove :: rest).Perm (Root.choices mc b roots idx) :=
    labels ▸ trace.reproduced.1.trans order
  exact .split firstMove rest _ split permutation started indexed.1 indexed.2 reports

theorem AttemptWithUses.weaken {source target : ChildResults}
    (liftChild : ∀ stage tree pv alpha beta score, source stage tree pv alpha beta score →
      target stage tree pv alpha beta score)
    (run : AttemptWithUses source mc fc b roots idx alpha beta score rows) :
    AttemptWithUses target mc fc b roots idx alpha beta score rows := by
  cases run with
  | sequential moves observe order calls =>
    exact .sequential moves observe order (calls.weaken (fun _ => liftChild .serial _))
  | split first rest observe split order started continued headCalls reports =>
    exact .split first rest observe split order started continued
      (headCalls.weaken (fun _ => liftChild .serial _)) (reports.weaken liftChild)
  | wipeout move member wipes => exact .wipeout move member wipes

theorem AttemptWith.weaken {source target : Search}
    (liftChild : ∀ tree pv alpha beta score, source tree pv alpha beta score →
      target tree pv alpha beta score)
    (run : AttemptWith source mc fc b roots idx alpha beta score rows) :
    AttemptWith target mc fc b roots idx alpha beta score rows :=
  AttemptWithUses.weaken (fun _ => liftChild) run

theorem Attempt.sound (covers : Root.Covers mc b roots) (inside : idx < roots.length)
    (window : Window alpha beta) (run : Attempt mc fc b roots idx alpha beta score rows) :
    SoundAttempt mc fc b roots idx alpha beta score rows := by
  cases run with
  | sequential moves observe order calls =>
    exact (RootReplay.serialContract_proven Replayed replay_follows mc fc b roots idx moves observe
      alpha beta covers inside window order calls).2
  | split first rest observe split order started continued headCalls reportCalls =>
    exact (RootReplay.Split.rootContract_proven Replayed replay_follows mc fc b roots idx first rest
      observe alpha beta score split covers inside window order started continued headCalls reportCalls).2
  | wipeout move member wipes =>
    exact RootReplay.wipeoutContract_proven mc fc b roots idx alpha beta move covers inside
      window member wipes

def Checked (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square)
    (idx : Nat) (search : RootAspiration.Search) : Prop :=
  ∀ i alpha beta, Window alpha beta → (search i alpha beta).aborted = false →
    Attempt mc fc b roots idx alpha beta (search i alpha beta).score (search i alpha beta).payload

theorem Checked.sound (covers : Root.Covers mc b roots) (inside : idx < roots.length)
    (checked : Checked mc fc b roots idx search) :
    RootAspiration.SoundSearch mc fc b roots idx search :=
  fun i alpha beta window running => (checked i alpha beta window running).sound covers inside window

def SingleContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square)
    (alpha beta : Int) (search : RootAspiration.Search) (pv : Square → List Square) (depth nodes : Nat),
    Root.Covers mc b roots → 0 < roots.length → Window alpha beta →
    Checked mc fc b roots 0 search →
    (∀ i a z, Window a z → (search i a z).aborted = false) →
    ∃ accepted result,
      Aspiration.runWith search 11 0 alpha beta 256 = some accepted ∧
      let rows := roots.map (fun sq => (⟨sq, accepted.trial.payload sq, pv sq⟩ : Row))
      SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = minimax mc fc b

theorem singleContract_proven : SingleContract := by
  intro mc fc b roots alpha beta search pv depth nodes covers inside window checked running
  exact ExactEndgameScore.singleResultContract_proven mc fc b roots alpha beta search pv depth nodes
    covers inside window (checked.sound covers inside) running

def AllMovesContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Square)
    (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed)
    (pv : Square → List Square) (depth nodes : Nat),
    Root.Covers mc b roots → 0 < roots.length →
    (∀ order idx, Root.Covers mc b order → idx < order.length →
      Window (seed order idx).1 (seed order idx).2 ∧
      Checked mc fc b order idx (observe order idx) ∧
      ∀ i a z, Window a z → (observe order idx i a z).aborted = false) →
    ∃ batch result,
      RootMultiPV.run observe seed roots.length 0 roots = some batch ∧
      let rows := batch.picks.map (fun pick => (⟨pick.square, pick.score, pv pick.square⟩ : Row))
      SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = minimax mc fc b ∧ result.endgame = true

theorem allMovesContract_proven : AllMovesContract := by
  intro mc fc b roots observe seed pv depth nodes covers positive matched
  have attempts : RootMultiPV.Matched mc fc b observe seed := by
    intro order idx coverage inside
    obtain ⟨window, checked, running⟩ := matched order idx coverage inside
    exact ⟨window, checked.sound coverage inside, running⟩
  obtain ⟨batch, completed, saved⟩ := ExactEndgameScore.rootResultContract_proven mc fc b roots
    roots.length observe seed pv depth nodes covers positive (Nat.le_refl _) attempts
  obtain ⟨result, reported, exactScore, endgame, _⟩ := saved [] (by simp)
  exact ⟨batch, result, completed, by simpa using reported, exactScore, endgame⟩

end NeuralReversi.PvsParallel.RootEvidence
