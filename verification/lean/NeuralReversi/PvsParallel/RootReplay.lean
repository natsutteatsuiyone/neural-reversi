module
public import NeuralReversi.PvsParallel.RootOrder

@[expose] public section

namespace NeuralReversi.PvsParallel.RootReplay

open Aspiration (SoundResult Window)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (GeneratedSquare Position)
open PvsMoveLoop (ChildRelation ChildSearch ScanCalls Tree bestValue moveScore scan scoreInf scoreMax)
open RootOrder (Records Report SelectedLine SoundAttempt sortFrom visibleScore writeScore)
open Threading.SplitWindow (State marksCutoff publish)

structure Snapshot where
  state : State
  seen : List Square
  scores : Square → Int

def start (alpha : Int) : Snapshot :=
  ⟨⟨alpha, -scoreInf⟩, [], fun _ => -scoreInf⟩

def Snapshot.record (beta : Int) (before : Snapshot) (report : Report) : Snapshot :=
  ⟨publish true beta before.state report.score, report.square :: before.seen,
    writeScore before.scores report.square (visibleScore before.seen.isEmpty before.state.alpha report.score)⟩

def serial (square : Move → Square) (observe : ChildSearch Move) (beta : Int) :
    List Move → Snapshot → Snapshot
  | [], before => before
  | move :: rest, before =>
    let alpha := before.state.alpha
    let score := moveScore true before.seen.isEmpty observe move alpha beta
    let after := before.record beta ⟨square move, alpha, score⟩
    if marksCutoff true beta before.state score then after else serial square observe beta rest after

theorem serial_best (square : Move → Square) (observe : ChildSearch Move) (beta : Int)
    (moves : List Move) (before : Snapshot) :
    (serial square observe beta moves before).state.best =
      scan true observe beta moves before.seen.isEmpty before.state.alpha before.state.best := by
  induction moves generalizing before with
  | nil => rfl
  | cons move rest ih =>
    let score := moveScore true before.seen.isEmpty observe move before.state.alpha beta
    let after := before.record beta ⟨square move, before.state.alpha, score⟩
    rw [Prefix.scan_step, serial]
    change (if marksCutoff true beta before.state score then after
      else serial square observe beta rest after).state.best =
        if marksCutoff true beta before.state score then score else
          scan true observe beta rest false after.state.alpha after.state.best
    split
    · rename_i cut
      have improves := (of_decide_eq_true cut).1
      exact Int.max_eq_right (Int.le_of_lt improves)
    · exact ih after

theorem serial_prefix (square : Move → Square) (observe : ChildSearch Move) (beta : Int)
    (moves : List Move) (before : Snapshot)
    (continued : Prefix.run true observe beta moves before.seen.isEmpty before.state = some final) :
    (serial square observe beta moves before).state = final := by
  induction moves generalizing before with
  | nil => exact Option.some.inj continued
  | cons move rest ih =>
    let score := moveScore true before.seen.isEmpty observe move before.state.alpha beta
    let after := before.record beta ⟨square move, before.state.alpha, score⟩
    cases cut : marksCutoff true beta before.state score
    · have next : Prefix.run true observe beta rest false (publish true beta before.state score) = some final := by
        simpa [Prefix.run, score, cut] using continued
      simpa [serial, score, cut, after] using (ih after next)
    · simp [Prefix.run, score, cut] at continued

theorem serial_seen (square : Move → Square) (observe : ChildSearch Move) (beta : Int)
    (moves : List Move) (before : Snapshot) :
    let final := serial square observe beta moves before
    (∀ sq ∈ final.seen, sq ∈ before.seen ∨ sq ∈ moves.map square) ∧
      ((moves ≠ [] ∨ before.seen ≠ []) → final.seen ≠ []) := by
  induction moves generalizing before with
  | nil => simp [serial]
  | cons move rest ih =>
    rw [serial]
    split
    · refine ⟨fun sq member => ?_, fun _ => by simp [Snapshot.record]⟩
      rcases List.mem_cons.mp member with same | earlier
      · exact .inr (List.mem_cons.mpr (.inl same))
      · exact .inl earlier
    · obtain ⟨used, nonempty⟩ := ih (before.record beta
        ⟨square move, before.state.alpha, moveScore true before.seen.isEmpty observe move before.state.alpha beta⟩)
      refine ⟨fun sq member => ?_, fun _ => nonempty (.inr (by simp [Snapshot.record]))⟩
      rcases used sq member with earlier | later
      · rcases List.mem_cons.mp earlier with first | old
        · exact .inr (List.mem_cons.mpr (.inl first))
        · exact .inl old
      · exact .inr (List.mem_cons_of_mem _ later)

theorem serial_records (square : Move → Square) (observe : ChildSearch Move)
    (value : Square → Int) (moves : List Move) (before : Snapshot)
    (completed : ChildRelation Move)
    (window : Window before.state.alpha beta)
    (prior : Records candidates value original beta before.seen before.state before.scores)
    (distinct : (moves.map square).Nodup)
    (available : ∀ move ∈ moves, square move ∈ candidates ∧ square move ∉ before.seen)
    (calls : ScanCalls completed observe true beta moves before.seen.isEmpty before.state.alpha before.state.best)
    (children : ∀ move ∈ moves, ∀ cp a b s, Window a b → (cp = false → b = a + 1) →
      completed move cp a b s → SoundResult (-(value (square move))) a b s) :
    let final := serial square observe beta moves before
    Records candidates value original beta final.seen final.state final.scores := by
  induction moves generalizing before with
  | nil => exact prior
  | cons move rest ih =>
    let alpha := before.state.alpha
    let score := moveScore true before.seen.isEmpty observe move alpha beta
    let after := before.record beta ⟨square move, alpha, score⟩
    have chosen := available move (by simp)
    have sound := PvsMoveLoop.moveScore_boundSelected (value := fun m => value (square m))
      (pv := true) (first := before.seen.isEmpty) (move := move)
      window (by simp) calls.1 (children move (by simp))
    have recorded : Records candidates value original beta after.seen after.state after.scores :=
      .push ⟨square move, alpha, score⟩ prior chosen.1 chosen.2 (Int.le_refl _) sound.2.2
    have nextWindow : Window after.state.alpha beta :=
      (Threading.SplitWindow.stateContract_proven true before.state.alpha beta before.state [score]
        window (Int.le_max_left _ _)).2.1
    have remaining : ∀ other ∈ rest, square other ∈ candidates ∧ square other ∉ after.seen := by
      intro other member
      have candidate := available other (List.mem_cons_of_mem _ member)
      have unique : square move ∉ rest.map square := (List.nodup_cons.mp distinct).1
      refine ⟨candidate.1, ?_⟩
      rintro repeated
      rcases List.mem_cons.mp repeated with same | old
      · change square other = square move at same
        exact unique (same ▸ List.mem_map_of_mem member)
      · exact candidate.2 old
    change Records candidates value original beta
      (serial square observe beta (move :: rest) before).seen
      (serial square observe beta (move :: rest) before).state
      (serial square observe beta (move :: rest) before).scores
    rw [serial]
    split
    · exact recorded
    · rename_i noCut
      have restCalls := calls.2
      change (match PvsMoveLoop.nextScanState true beta alpha before.state.best score with
        | none => True
        | some (a, s) => ScanCalls completed observe true beta rest false a s) at restCalls
      rw [Prefix.nextScanState_eq, ite_eq_right noCut] at restCalls
      exact ih after nextWindow recorded (List.nodup_cons.mp distinct).2 remaining restCalls
        (fun other mem => children other (List.mem_cons_of_mem _ mem))

theorem selected (final : Snapshot) (covers : Root.Covers mc b roots) (inside : idx < roots.length)
    (window : Window original beta)
    (records : Records (roots.drop idx) (Root.squareValue mc fc b) original beta final.seen final.state final.scores)
    (result : SoundResult (Root.value mc fc b roots idx) original beta final.state.best)
    (high : original < final.state.best) (low : final.state.best < beta) :
    ∃ head tail, (sortFrom roots idx final.scores).drop idx = head :: tail ∧
      SelectedLine mc fc b roots idx final.scores head := by
  have perm := (RootOrder.sortContract_proven roots idx final.scores (Nat.le_of_lt inside)).1
  have remaining : (sortFrom roots idx final.scores).drop idx ≠ [] := by
    rw [ne_eq, List.drop_eq_nil_iff, perm.length_eq]
    omega
  obtain ⟨head, tail, sorted⟩ := List.exists_cons_of_ne_nil remaining
  exact ⟨head, tail, sorted, RootOrder.selectedLine_of_exact covers inside window
    (result.exact high low) records high low sorted⟩

def SerialContract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ (mc : X86Config) (fc : Root.FlipConfig) (b : Position) (roots : List Square) (idx : Nat)
      (moves : List (GeneratedSquare mc b.val)) (observe : ChildSearch (GeneratedSquare mc b.val))
      (original beta : Int),
      Root.Covers mc b roots → idx < roots.length → Window original beta →
      moves.Perm (Root.choices mc b roots idx) →
      ScanCalls (fun m => completed (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b m).val))
        observe true beta moves true original (-scoreInf) →
      let final := serial Subtype.val observe beta moves (start original)
      final.state.best = scan true observe beta moves true original (-scoreInf) ∧
      SoundAttempt mc fc b roots idx original beta final.state.best final.scores

theorem serial_of_children (mc : X86Config) (fc : Root.FlipConfig) (b : Position)
    (roots : List Square) (idx : Nat) (moves : List (GeneratedSquare mc b.val))
    (observe : ChildSearch (GeneratedSquare mc b.val)) (completed : ChildRelation (GeneratedSquare mc b.val))
    (original beta : Int) (covers : Root.Covers mc b roots) (inside : idx < roots.length)
    (window : Window original beta) (order : moves.Perm (Root.choices mc b roots idx))
    (calls : ScanCalls completed observe true beta moves true original (-scoreInf))
    (children : ∀ m ∈ moves, ∀ cp a z s, Window a z → (cp = false → z = a + 1) →
      completed m cp a z s → SoundResult (-(Root.squareValue mc fc b m.val)) a z s) :
    let final := serial Subtype.val observe beta moves (start original)
    final.state.best = scan true observe beta moves true original (-scoreInf) ∧
      SoundAttempt mc fc b roots idx original beta final.state.best final.scores := by
  intro final
  have membership (m : GeneratedSquare mc b.val) (mem : m ∈ moves) : m.val ∈ roots.drop idx :=
    (Root.choices_squares_mem covers m.val).mp (List.mem_map_of_mem (order.mem_iff.mp mem))
  have distinct := (order.map Subtype.val).nodup_iff.mpr (Root.choices_squares_nodup mc b roots idx)
  have records := serial_records Subtype.val observe (Root.squareValue mc fc b) moves (start original)
    completed (original := original) (candidates := roots.drop idx) window .start distinct
    (fun m mem => ⟨membership m mem, by simp [start]⟩) calls children
  have best : final.state.best = scan true observe beta moves true original (-scoreInf) :=
    serial_best Subtype.val observe beta moves (start original)
  have nonempty : moves ≠ [] := fun empty =>
    ((Root.maskContract_proven mc b roots idx covers).2.2.mpr inside) ((empty ▸ order).nil_eq.symm)
  have cap (m : GeneratedSquare mc b.val) : Root.childValue mc fc b m ≤ scoreMax := by
    have range := (PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b m).val)).1
    change -scoreMax ≤ PvsBoardTree.minimax mc fc (PvsBoardTree.play mc fc b m).val at range
    unfold Root.childValue
    omega
  have result := PvsMoveLoop.selectedContract_proven _ true observe (Root.childValue mc fc b)
    completed moves original beta nonempty window (by simp) (fun m _ => cap m) calls
    (fun m mem cp a z s w n recorded => by
      rw [← Root.square_value_eq_child m]
      exact children m mem cp a z s w n recorded)
  have value : bestValue (Root.childValue mc fc b) moves (-scoreInf) = Root.value mc fc b roots idx :=
    PvsMoveLoop.bestValue_eq_of_same_moves _ _ (fun _ => order.mem_iff)
  rw [value, ← best] at result
  exact ⟨best, result, selected final covers inside window records result⟩

theorem serialContract_proven : SerialContract := by
  intro completed follows mc fc b roots idx moves observe original beta covers inside window order calls
  apply serial_of_children mc fc b roots idx moves observe _ original beta covers inside window order calls
  intro m _ cp a z s valid narrow recorded
  rw [Root.square_value_eq_child m, Root.childValue, Int.neg_neg]
  exact sound completed follows _ cp a z s valid narrow recorded

def WipeoutContract : Prop :=
  ∀ (mc : X86Config) (fc : Root.FlipConfig) (board : Position) (roots : List Square) (idx : Nat)
    (alpha beta : Int) (move : GeneratedSquare mc board.val),
    Root.Covers mc board roots → idx < roots.length → Window alpha beta →
    move ∈ Root.choices mc board roots idx → EndgameCutoffs.wipesOut fc board.val move.val = true →
    let final := (start alpha).record beta ⟨move.val, alpha, scoreMax⟩
    SoundAttempt mc fc board roots idx alpha beta scoreMax final.scores

theorem wipeoutContract_proven : WipeoutContract := by
  intro mc fc board roots idx alpha beta move covers inside window chosen wipes final
  have remaining : move.val ∈ roots.drop idx :=
    (Root.choices_squares_mem covers move.val).mp (List.mem_map_of_mem chosen)
  have exactMove : Root.squareValue mc fc board move.val = scoreMax := by
    rw [Root.square_value_eq_child, Root.childValue, EndgameCutoffs.wipeout_child_value mc fc board move wipes]
    decide
  have records : Records (roots.drop idx) (Root.squareValue mc fc board) alpha beta
      final.seen final.state final.scores :=
    .push ⟨move.val, alpha, scoreMax⟩ .start remaining (by simp [start]) (Int.le_refl _)
      ⟨fun _ => Int.le_of_eq exactMove, fun _ => Int.le_of_eq exactMove.symm⟩
  have best : final.state.best = scoreMax := by
    change max (-scoreInf) scoreMax = scoreMax
    decide
  have sound : SoundResult (Root.value mc fc board roots idx) alpha beta scoreMax := by
    rw [Root.wipeout_value chosen wipes]
    exact ⟨by decide, by decide, fun _ => Int.le_refl _, fun _ => Int.le_refl _, fun _ _ => rfl⟩
  exact ⟨sound, by rw [← best] at sound ⊢; exact selected final covers inside window records sound⟩

def twoOpen : Position := ⟨⟨0xde5271007814a8a2#64, 0x21ad0eff87eb175d#64⟩, by rfl⟩

example :
    let mc : X86Config := ⟨false, false⟩
    let fc : Root.FlipConfig := ⟨false, false, false⟩
    let roots : List Square := [14, 47]
    let moves := Root.choices mc twoOpen roots 0
    let child := fun m => PvsBoardTree.build mc fc (PvsBoardTree.play mc fc twoOpen m).val
    let observe := fun m => PvsMoveLoop.run (child m)
    let final := serial Subtype.val observe scoreInf moves (start (-scoreInf))
    final.seen = [47, 14] ∧ final.state.best = -1024 ∧ final.scores 14 = -4096 ∧
      final.scores 47 = -1024 ∧ SelectedLine mc fc twoOpen roots 0 final.scores 47 := by
  intro mc fc roots moves child observe final
  have legal : (PvsBoardTree.legalSquares mc twoOpen.val).map Subtype.val = roots := by decide
  have covers : Root.Covers mc twoOpen roots := List.Perm.of_eq legal.symm
  have matched := serialContract_proven _ sequential_follows mc fc twoOpen roots 0 moves observe
    (-scoreInf) scoreInf covers (by decide) ⟨by decide, by decide, by decide⟩
    (List.Perm.refl _) (ScanCalls.ofAll (by intros; rfl))
  have seen : final.seen = [47, 14] := by decide +kernel
  have best : final.state.best = -1024 := by decide +kernel
  have first : final.scores 14 = -4096 := by decide +kernel
  have second : final.scores 47 = -1024 := by decide +kernel
  obtain ⟨head, tail, sorted, chosen⟩ := matched.2.2 (by rw [best]; decide) (by rw [best]; decide)
  have order : (sortFrom roots 0 final.scores).drop 0 = [47, 14] := by
    change ([14, 47] : List Square).mergeSort _ = _
    rw [List.mergeSort_pair]
    simp [first, second]
  rw [order] at sorted
  cases (List.cons.inj sorted).1
  exact ⟨seen, best, first, second, chosen⟩

end NeuralReversi.PvsParallel.RootReplay
