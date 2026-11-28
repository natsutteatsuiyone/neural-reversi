module
public import NeuralReversi.PvsParallel.RootReplay

@[expose] public section

namespace NeuralReversi.PvsParallel.RootReplay.Split

open Aspiration (SoundResult Window)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (GeneratedSquare Position)
open PvsMoveLoop (ChildSearch ScanCalls Tree bestValue minimax scoreInf scoreMax)
open RootOrder (Records SoundAttempt)
open Threading
open Threading.SplitResult (Report slots)
open Threading.SplitReturn (ReportSteps stateAfter)
open Threading.SplitWindow (reSearch)

def replay (square : Nat → Square) (beta : Int) (reports : List Report) (before : Snapshot) : Snapshot :=
  reports.foldl (fun snapshot report =>
    snapshot.record beta ⟨square report.index, report.searchedAlpha, report.score⟩) before

theorem replay_state (square : Nat → Square) (beta : Int) (reports : List Report) (before : Snapshot) :
    (replay square beta reports before).state = stateAfter true beta before.state reports := by
  induction reports generalizing before with
  | nil => rfl
  | cons report rest ih => exact ih _

theorem mapped_distinct (square : Nat → Square) (reports : List Report)
    (distinct : (reports.map Report.index).Nodup)
    (injective : ∀ a ∈ reports, ∀ b ∈ reports, square a.index = square b.index → a.index = b.index) :
    (reports.map (fun report => square report.index)).Nodup := by
  induction reports with
  | nil => exact List.nodup_nil
  | cons report rest ih =>
    obtain ⟨fresh, tail⟩ := List.nodup_cons.mp distinct
    refine List.nodup_cons.mpr ⟨?_, ih tail (fun a ha b hb => injective a
      (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb))⟩
    intro repeated
    obtain ⟨other, member, same⟩ := List.mem_map.mp repeated
    have equal := injective other (List.mem_cons_of_mem _ member) report (by simp) same
    exact fresh (List.mem_map.mpr ⟨other, member, equal⟩)

theorem records_after (square : Nat → Square) (value : Square → Int) (initial : Snapshot)
    (reports : List Report) (steps : ReportSteps true beta initial.state (value ∘ square) reports)
    (earlier rest : List Report) (partition : reports = earlier ++ rest)
    (distinct : (rest.map fun report => square report.index).Nodup)
    (available : ∀ report ∈ rest, square report.index ∈ candidates ∧
      square report.index ∉ (replay square beta earlier initial).seen)
    (prior : Records candidates value original beta (replay square beta earlier initial).seen
      (replay square beta earlier initial).state (replay square beta earlier initial).scores) :
    let final := replay square beta reports initial
    Records candidates value original beta final.seen final.state final.scores := by
  induction rest generalizing earlier with
  | nil => simpa [partition] using prior
  | cons report tail ih =>
    let before := replay square beta earlier initial
    let after := before.record beta ⟨square report.index, report.searchedAlpha, report.score⟩
    have next : replay square beta (earlier ++ [report]) initial = after := by
      rw [replay, List.foldl_append]
      rfl
    have chosen := available report (by simp)
    have sound := steps earlier report tail partition
    rw [← replay_state square beta earlier initial] at sound
    have recorded : Records candidates value original beta after.seen after.state after.scores :=
      .push ⟨square report.index, report.searchedAlpha, report.score⟩ prior chosen.1 chosen.2 sound.1 sound.2
    apply ih (earlier ++ [report]) (by simpa [List.append_assoc] using partition)
      (List.nodup_cons.mp distinct).2
    · rw [next]
      intro other member
      obtain ⟨included, fresh⟩ := available other (List.mem_cons_of_mem _ member)
      refine ⟨included, ?_⟩
      intro reused
      rcases List.mem_cons.mp reused with same | old
      · change square other.index = square report.index at same
        exact (List.nodup_cons.mp distinct).1 (List.mem_map.mpr ⟨other, member, same⟩)
      · exact fresh old
    · rw [next]
      exact recorded

private theorem seen_nonempty (square : Nat → Square) (beta : Int) (reports : List Report)
    (before : Snapshot) (seen : before.seen ≠ []) :
    (replay square beta reports before).seen ≠ [] := by
  induction reports generalizing before with
  | nil => exact seen
  | cons report rest ih =>
    exact ih (before.record beta ⟨square report.index, report.searchedAlpha, report.score⟩) (by simp [Snapshot.record])

def Contract : Prop :=
  ∀ (len : Nat) (original beta score : Int) (split : SplitExecution len true original beta score)
    (square : Nat → Square) (value : Square → Int) (candidates : List Square) (before : Snapshot),
    split.entry = before.state → before.seen ≠ [] →
    Records candidates value original beta before.seen before.state before.scores →
    (∀ i ∈ slots split.start.toNat len, square i ∈ candidates ∧ square i ∉ before.seen) →
    (∀ i ∈ slots split.start.toNat len, ∀ j ∈ slots split.start.toNat len, square i = square j → i = j) →
    ReportSteps true beta split.entry (value ∘ square) split.reports →
    let final := replay square beta split.reports before
    final.state.best = score ∧
      Records candidates value original beta final.seen final.state final.scores ∧
      ∀ earlier report later, split.reports = earlier ++ report :: later →
        let locked := replay square beta earlier before
        (locked.record beta ⟨square report.index, report.searchedAlpha, report.score⟩).scores
          (square report.index) = if locked.state.alpha < report.score then report.score else -PvsMoveLoop.scoreInf

theorem contract_proven : Contract := by
  intro len original beta score split square value candidates before entry started prior available injective steps final
  have observed := SplitReturn.ownerBestContract_proven split.history split.writes
    split.result split.lastScore split.wait true beta split.entry split.reports split.read split.bestStores
  have result := split.accepted
  rw [observed, entry] at result
  have accepted : split.check.stopped = false := by
    cases stopped : split.check.stopped <;> simp [SplitReturn.checked, stopped] at result ⊢
  have best : final.state.best = score := by
    change (replay square beta split.reports before).state.best = score
    rw [replay_state]
    exact Option.some.inj (by simpa only [SplitReturn.checked, accepted, Bool.false_eq_true, ite_false] using result)
  have tickets := SplitReturn.uniqueReportsContract_proven split.history split.mask split.activation split.wait
    split.width len split.start split.last split.issued split.reports split.check _ split.capacity split.claims split.dispatches
  have distinct := mapped_distinct square split.reports tickets
    (fun a ha b hb => injective a.index (split.valid a ha) b.index (split.valid b hb))
  rw [entry] at steps
  refine ⟨best, records_after square value before split.reports steps [] split.reports rfl
    distinct (fun r member => available r.index (split.valid r member)) prior, ?_⟩
  intro earlier report later _ locked
  have previous := seen_nonempty square beta earlier before started
  have nonempty : locked.seen.isEmpty = false := List.isEmpty_eq_false_iff.mpr previous
  change RootOrder.writeScore locked.scores (square report.index)
    (RootOrder.visibleScore locked.seen.isEmpty locked.state.alpha report.score) (square report.index) = _
  simp [RootOrder.writeScore, RootOrder.visibleScore, nonempty]

def childTree (mc : X86Config) (fc : Root.FlipConfig) (b : Position)
    (move : GeneratedSquare mc b.val) : Tree := PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b move).val

def moveAt (first : Move) (rest : List Move) (i : Nat) : Move :=
  ((first :: rest)[i]?).getD first

private theorem movesAt_map (first : Move) (rest : List Move) (stop : Nat)
    (bounded : stop ≤ (first :: rest).length) (f : Move → Square) :
    (List.range stop).map (fun i => f (moveAt first rest i)) = ((first :: rest).map f).take stop := by
  apply List.ext_getElem
  · simp only [List.length_map, List.length_range, List.length_take, Nat.min_eq_left bounded]
  · intro i h _
    have whereAt : i < stop := by simpa using h
    have inside := Nat.lt_of_lt_of_le whereAt bounded
    simp only [List.getElem_map, List.getElem_range, moveAt, List.getElem?_eq_getElem inside, Option.getD_some, List.getElem_take]

def RootContract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ (mc : X86Config) (fc : Root.FlipConfig) (b : Position) (roots : List Square) (idx : Nat)
      (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
      (observe : ChildSearch Nat) (original beta score : Int)
      (split : SplitExecution ((first :: rest).map (childTree mc fc b)).length true original beta score),
      Root.Covers mc b roots → idx < roots.length → Window original beta →
      (first :: rest).Perm (Root.choices mc b roots idx) →
      0 < split.start.toNat →
      Prefix.run true observe beta (List.range split.start.toNat) true ⟨original, -scoreInf⟩ =
        some split.entry →
      ScanCalls (fun i => completed (childAt ((first :: rest).map (childTree mc fc b)) i))
        observe true beta (List.range split.start.toNat) true original (-scoreInf) →
      (∀ earlier report later (same : split.reports = earlier ++ report :: later),
        let reads := split.reportReads earlier report later same
        let child := childAt ((first :: rest).map (childTree mc fc b)) report.index
        completed child false (-(reads.alpha + 1)) (-reads.alpha) reads.scout ∧
          (reSearch true reads.alpha reads.scout = true →
            completed child true (-beta) (-reads.fresh) reads.wide)) →
      let square := fun i => (moveAt first rest i).val
      let before := serial square observe beta (List.range split.start.toNat) (start original)
      let final := replay square beta split.reports before
      final.state.best = score ∧ SoundAttempt mc fc b roots idx original beta score final.scores

theorem rootContract_proven : RootContract := by
  intro completed follows mc fc b roots idx first rest observe original beta score split covers inside window
    order started continued prefixCalls reportCalls square before final
  let moves := first :: rest
  let trees := moves.map (childTree mc fc b)
  let serialIndices := List.range split.start.toNat
  let value := fun i => -minimax (childAt trees i)
  have size : trees.length = moves.length := List.length_map _
  have bounded : split.start.toNat ≤ moves.length := Nat.le_trans split.startValid (Nat.le_of_eq size)
  have inSuffix (i : Nat) (hi : i ∈ slots split.start.toNat trees.length) :
      split.start.toNat ≤ i ∧ i < moves.length := by
    obtain ⟨lo, high⟩ := List.mem_range'_1.mp hi
    exact ⟨lo, by omega⟩
  have picked (i : Nat) (inList : i < moves.length) : moveAt first rest i = moves[i] := by
    change moves[i]?.getD first = _
    rw [List.getElem?_eq_getElem inList]
    rfl
  have legal (i : Nat) (inList : i < moves.length) : square i ∈ roots.drop idx := by
    have found : moveAt first rest i ∈ moves := picked i inList ▸ List.getElem_mem inList
    exact (Root.choices_squares_mem covers _).mp (List.mem_map_of_mem (order.mem_iff.mp found))
  have sameValue (i : Nat) (inList : i < moves.length) : value i = Root.squareValue mc fc b (square i) := by
    have atTree : i < trees.length := size.symm ▸ inList
    change -minimax (childAt trees i) = Root.squareValue mc fc b (moveAt first rest i).val
    rw [Root.square_value_eq_child, picked i inList, childAt, List.getElem?_eq_getElem atTree]
    change -minimax ((moves.map (childTree mc fc b))[i]) = Root.childValue mc fc b moves[i]
    rw [List.getElem_map]
    rfl
  have cap (i : Nat) : value i ≤ scoreMax := by
    have := (PvsMoveLoop.minimax_range (childAt trees i)).1
    dsimp [value]
    omega
  have prefixSound (i : Nat) (pv : Bool) (a z s : Int) (w : Window a z) (n : pv = false → z = a + 1)
      (recorded : completed (childAt trees i) pv a z s) : SoundResult (-(value i)) a z s := by
    simpa only [value, Int.neg_neg] using sound completed follows (childAt trees i) pv a z s w n recorded
  obtain ⟨openWindow, limit, narrow, seed⟩ := Prefix.selectedContract_proven Nat true observe value serialIndices
    (fun i => completed (childAt trees i)) original beta split.entry window (by simp) continued
    (fun i _ => cap i) prefixCalls (fun i _ => prefixSound i)
  have ⟨scoreBound, steps⟩ := split_bounds split openWindow limit narrow seed (by
    intro earlier report later same
    obtain ⟨scout, wide⟩ := reportCalls earlier report later same
    exact ⟨fun w => sound completed follows _ false _ _ _ w (by intro _; omega) scout,
      fun retry w => sound completed follows _ true _ _ _ w (by simp) (wide retry)⟩)
  have squareSteps : ReportSteps true beta split.entry (Root.squareValue mc fc b ∘ square) split.reports := by
    intro earlier report later same
    have step := steps earlier report later same
    have member : report ∈ split.reports := by simp [same]
    have inList := (inSuffix report.index (split.valid report member)).2
    simpa only [Function.comp_apply, ← sameValue report.index inList] using step
  have unique : (moves.map Subtype.val).Nodup :=
    (order.map Subtype.val).nodup_iff.mpr (Root.choices_squares_nodup mc b roots idx)
  have prefixSquares : serialIndices.map square = (moves.map Subtype.val).take split.start.toNat :=
    movesAt_map first rest split.start.toNat bounded Subtype.val
  have previous := serial_seen square observe beta serialIndices (start original)
  have entry : split.entry = before.state :=
    (serial_prefix square observe beta serialIndices (start original) continued).symm
  have prevRecords := serial_records square observe (Root.squareValue mc fc b) serialIndices (start original)
    (fun i => completed (childAt trees i)) (original := original) (candidates := roots.drop idx) window .start
    (prefixSquares ▸ (List.take_sublist split.start.toNat _).nodup unique)
    (fun i hi => ⟨legal i (Nat.lt_of_lt_of_le (List.mem_range.mp hi) bounded), by simp [start]⟩)
    prefixCalls (fun i hi pv a z s w n recorded => by
      rw [← sameValue i (Nat.lt_of_lt_of_le (List.mem_range.mp hi) bounded)]
      exact prefixSound i pv a z s w n recorded)
  have squareEq (i j : Nat) (hi : i < moves.length) (hj : j < moves.length)
      (equal : square i = square j) : i = j := by
    have mapped : (moves.map Subtype.val)[i]'(by simpa using hi) =
        (moves.map Subtype.val)[j]'(by simpa using hj) := by
      simpa only [List.getElem_map, square, picked i hi, picked j hj] using equal
    exact unique.eq_of_getElem_eq _ _ mapped
  have available (i : Nat) (hi : i ∈ slots split.start.toNat trees.length) :
      square i ∈ roots.drop idx ∧ square i ∉ before.seen := by
    obtain ⟨lo, high⟩ := inSuffix i hi
    refine ⟨legal i high, fun already => ?_⟩
    rcases previous.1 _ already with impossible | used
    · exact List.not_mem_nil impossible
    · obtain ⟨j, hj, eq⟩ := List.mem_map.mp used
      have early : j < split.start.toNat := List.mem_range.mp hj
      have same := squareEq j i (Nat.lt_of_lt_of_le early bounded) high eq
      omega
  have nonempty : before.seen ≠ [] := previous.2 (.inl (by simp [serialIndices, Nat.ne_of_gt started]))
  obtain ⟨best, records, _⟩ := contract_proven trees.length original beta score split square
    (Root.squareValue mc fc b) (roots.drop idx) before entry nonempty prevRecords available
    (fun i hi j hj => squareEq i j (inSuffix i hi).2 (inSuffix j hj).2) squareSteps
  have equalValue : bestValue (fun child => -minimax child) trees (-scoreInf) =
      Root.value mc fc b roots idx := by
    change bestValue (fun child => -minimax child) (moves.map (childTree mc fc b)) (-scoreInf) = _
    rw [PvsMoveLoop.bestValue_map]
    exact PvsMoveLoop.bestValue_eq_of_same_moves (Root.childValue mc fc b) _ (fun _ => order.mem_iff)
  let node := Tree.turn (childTree mc fc b first) (rest.map (childTree mc fc b))
  have nodeValue : minimax node = Root.value mc fc b roots idx := by
    dsimp only [node]
    rw [PvsMoveLoop.minimax_turn]
    exact equalValue
  rw [equalValue, ← nodeValue] at scoreBound
  have result := PvsParallel.bound_sound node window scoreBound
  rw [nodeValue] at result
  refine ⟨best, result, fun high low => ?_⟩
  exact selected final covers inside window records (best.symm ▸ result) (best.symm ▸ high) (best.symm ▸ low)

end NeuralReversi.PvsParallel.RootReplay.Split
