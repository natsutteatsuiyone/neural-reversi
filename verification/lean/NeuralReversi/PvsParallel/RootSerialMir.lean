module
public import NeuralReversi.PvsParallel.RootHeaderMir
public import NeuralReversi.PvsParallel.RootUpdateMir

@[expose] public section

namespace NeuralReversi.PvsParallel.RootSerialMir

open RootReplay (Snapshot)
open PvsMoveLoop (ChildSearch MoveCalls ScanCalls moveScore)
open PvsBoardTree (Position GeneratedSquare build play)
open Bitboard (Square)

/-- The owner has no helpers available. Each candidate is claimed when first visited.
The square order is the order already produced by move ordering. -/
def run (square : Move → Square) (search : ChildSearch Move) (depth minSplit : Nat) (beta : Int) :
    List Move → Snapshot → Option Snapshot
  | [], s =>
    match headerAction s.seen.length s.seen.length s.seen.length depth minSplit false false s.state.alpha 0 0 with
    | .finish => some s
    | _ => none
  | m :: rest, s => do
    let idx := s.seen.length
    let n := idx + (m :: rest).length
    let alpha := s.state.alpha
    let scout := search m false (-(alpha + 1)) (-alpha)
    let pv := search m true (-beta) (-alpha)
    let .child next _ _ score := headerAction idx n n depth minSplit false false alpha scout pv | none
    let d := RootUpdateMir.decision (decide (next = 1)) alpha beta s.state.best score
    let updated := RootUpdateMir.after s (square m) score d
    if d.cuts then some updated else run square search depth minSplit beta rest updated

theorem ordinary_child (search : ChildSearch Move) (m : Move) (rest : List Move) (s : Snapshot)
    (depth minSplit : Nat) (beta : Int) :
    let n := s.seen.length + (m :: rest).length
    let a := s.state.alpha
    ∃ scout wide, headerAction s.seen.length n n depth minSplit false false a
      (search m false (-(a + 1)) (-a)) (search m true (-beta) (-a)) =
      .child (s.seen.length + 1) scout wide (moveScore true s.seen.isEmpty search m a beta) := by
  intro n a
  have more : s.seen.length < n := by dsimp [n]; omega
  rw [headerAction_eq, ite_eq_left more]
  simp only [canHandoff, Bool.false_eq_true, and_false, ite_false, PvsPostpone.shouldPostpone,
    Bool.false_and, Bool.false_eq_true, ite_false]
  cases s.seen with
  | nil => simp [moveScore]
  | cons x xs =>
    by_cases raises : a < -search m false (-(a + 1)) (-a) <;> simp [moveScore, raises]

theorem run_eq (square : Move → Square) (search : ChildSearch Move) (depth minSplit : Nat) (beta : Int)
    (moves : List Move) (s : Snapshot) :
    run square search depth minSplit beta moves s = some (RootReplay.serial square search beta moves s) := by
  induction moves generalizing s with
  | nil => simp [run, headerAction_eq, RootReplay.serial]
  | cons m rest ih =>
    obtain ⟨_, _, emitted⟩ := ordinary_child search m rest s depth minSplit beta
    let score := moveScore true s.seen.isEmpty search m s.state.alpha beta
    obtain ⟨updated, cutoff, _, _⟩ := RootUpdateMir.agrees s (square m) beta score
    have first : decide (s.seen.length + 1 = 1) = s.seen.isEmpty := by cases s.seen <;> simp
    rw [run, emitted]
    simp only [first, ← show score = moveScore true s.seen.isEmpty search m s.state.alpha beta from rfl,
      updated, cutoff, RootReplay.serial]
    split <;> simp_all

theorem rootAttempt (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (endian : Bool)
    (board : Position) (roots : List Square) (idx depth minSplit : Nat) (alpha beta : Int)
    (moves : List (GeneratedSquare mc board.val)) (search : ChildSearch (GeneratedSquare mc board.val))
    (o : ConcurrentReplay.Observations mc fc endian events) (consumer : Nat)
    (order : moves.Perm (Root.choices mc board roots idx))
    (calls : ScanCalls (fun m => ConcurrentReplay.Consumed o.before events consumer (build mc fc (play mc fc board m).val))
      search true beta moves true alpha (-PvsMoveLoop.scoreInf)) :
    ∃ final, run Subtype.val search depth minSplit beta moves (RootReplay.start alpha) = some final ∧
      ConcurrentReplay.RootAttempt mc fc endian board roots idx alpha beta final.state.best final.scores := by
  exact ⟨_, run_eq .., events, o, consumer, .sequential moves search order calls⟩

/-- Physical rows after the source owner's updates, including rows invalidated by a failed scout. -/
def rowsAfter (sq : Move → Square) (pv : Move → List Square) (search : ChildSearch Move) (beta : Int) :
    List Move → Snapshot → List RootMoveStorage.Row → List RootMoveStorage.Row
  | [], _, rows => rows
  | m :: rest, s, rows =>
    let score := moveScore true s.seen.isEmpty search m s.state.alpha beta
    let d := RootUpdateMir.decision s.seen.isEmpty s.state.alpha beta s.state.best score
    let updated := RootSuffix.writeFirst (sq m) (fun row => RootSuffix.updateRow row score d.visible (pv m)) rows
    if d.cuts then updated else rowsAfter sq pv search beta rest (RootUpdateMir.after s (sq m) score d) updated

theorem rowsAfter_writes (sq : Move → Square) (pv : Move → List Square) (search : ChildSearch Move)
    (moves : List Move) (s : Snapshot) (prior : RootSuffix.Writes initial idx alpha beta rows s)
    (unique : (initial.map RootMoveStorage.Row.square).Nodup)
    (members : ∀ m ∈ moves, sq m ∈ (initial.drop idx).map RootMoveStorage.Row.square) :
    RootSuffix.Writes initial idx alpha beta (rowsAfter sq pv search beta moves s rows)
      (RootReplay.serial sq search beta moves s) := by
  induction moves generalizing s rows with
  | nil => exact prior
  | cons m rest ih =>
    let score := moveScore true s.seen.isEmpty search m s.state.alpha beta
    let d := RootUpdateMir.decision s.seen.isEmpty s.state.alpha beta s.state.best score
    have same := (prior.scores unique).1
    have member := members m (by simp)
    rw [List.map_drop, ← same, ← List.map_drop] at member
    have written := RootUpdateMir.recordWrites prior (sq m) score (pv m) member
    obtain ⟨step, cut, _, _⟩ := RootUpdateMir.agrees s (sq m) beta score
    change d.cuts = _ at cut
    rw [rowsAfter, RootReplay.serial]
    change RootSuffix.Writes _ _ _ _ (if d.cuts then _ else _) (if _ then _ else _)
    rw [← cut]
    cases d.cuts <;> simp only [Bool.false_eq_true, ite_false, ite_true]
    · rw [← step]
      exact ih _ written (fun mv mem => members mv (List.mem_cons_of_mem _ mem))
    · exact step ▸ written

end NeuralReversi.PvsParallel.RootSerialMir
