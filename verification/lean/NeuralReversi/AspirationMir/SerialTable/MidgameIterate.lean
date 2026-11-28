module
public import NeuralReversi.AspirationMir.SerialTable.MidgameRows
public import NeuralReversi.AspirationMir.SerialTable.Repeated
public import NeuralReversi.AspirationMir.SerialTable.PvInterior
public import NeuralReversi.MidgameAspiration.Termination

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MidgameIterate

open RootMoveStorage (Row sortRows)
open MidgameAspiration (Attempt FiniteScore)
open PvsBoardTree (Position)
open PvsParallel
open PvsMoveLoop (scoreMax)
open SmallRoot (Answer)

structure State (M : Type) where
  rows : List Row
  memory : M

def Valid (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop) (s : State M) : Prop :=
  Root.Covers mc b (s.rows.map Row.square) ∧ 0 < s.rows.length ∧ MidgameRows.Valid s.rows ∧ safe s.memory

abbrev Trial (M : Type) := Nat → State M → Attempt → Option (Answer × M)

def Works (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop) (trial : Trial M) : Prop :=
  ∀ d s w, Valid mc b safe s → w.window →
    ∃ ans m, trial d s w = some (ans,m) ∧ Valid mc b safe ⟨ans.rows,m⟩ ∧ FiniteScore ans.value ∧
      PvInterior.InsideRange w.alpha w.beta ans.value ∧
      (w.alpha < ans.value → (∃ r ∈ ans.rows, r.score = ans.value) ∧ ∀ r ∈ ans.rows, r.score ≤ ans.value)

/-- Rust does not sort between failed attempts. The next call receives the last attempt's rows and memory. -/
def retry (trial : Trial M) (d : Nat) : Nat → State M → Attempt → Option (Answer × M)
  | 0, _, _ => none
  | fuel + 1, s, w => do
    let (ans,m) ← trial d s w
    match w.retry ans.value with
    | none => return (ans,m)
    | some next => retry trial d fuel ⟨ans.rows,m⟩ next

private theorem retry_returns {trial : Trial M} (works : Works mc b safe trial) (s : State M) (valid : Valid mc b safe s)
    (w : Attempt) (win : w.window) (pos : 0 < w.delta) (budget : MidgameAspiration.HasFuel fuel w) (d : Nat) :
    ∃ ans m, retry trial d fuel s w = some (ans,m) ∧ Valid mc b safe ⟨ans.rows,m⟩ ∧
      (-scoreMax ≤ ans.value ∧ ans.value ≤ scoreMax) ∧
      (∃ r ∈ ans.rows, r.score = ans.value) ∧ ∀ r ∈ ans.rows, r.score ≤ ans.value := by
  induction fuel generalizing s w with
  | zero => contradiction
  | succ fuel ih =>
    obtain ⟨ans,m,frame,after,finite,range,maximum⟩ := works d s w valid win
    cases widened : w.retry ans.value with
    | none =>
      obtain ⟨lo,hi⟩ := MidgameAspiration.retry_none widened
      exact ⟨ans,m,by simp [retry,frame,widened],after,range lo hi,maximum lo⟩
    | some next =>
      obtain ⟨window,width⟩ := MidgameAspiration.retry_window win finite pos widened
      obtain ⟨res,post,ran,ok,bounds,highest⟩ := ih _ after _ window
        (by rw [width]; unfold MidgameAspiration.grow; omega) (budget _ finite _ widened)
      exact ⟨res,post,by simpa [retry,frame,widened] using ran,ok,bounds,highest⟩

structure Completed (M : Type) where
  state : State M
  saved : PublicationMir.Iteration
  base : Row

def iteration (t : Trial M) (sort : PublicationMir.SliceSort) (level depth : Nat) (s : State M) :
    Option (Completed M) := do
  let w := MidgameAspiration.initial depth (s.rows.head?.map MidgameRows.stats) 0
  let (ans,m) ← retry t depth 16 s w
  let rows := sortRows ans.rows 0
  let saved := PublicationMir.capture sort rows depth level false
  let best ← PublicationMir.first saved
  return ⟨⟨sortRows rows 0,m⟩,saved,best⟩

structure Complete (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop)
    (level depth : Nat) (out : Completed M) : Prop where
  valid : Valid mc b safe out.state
  captured : PublicationMir.first out.saved = some out.base
  atDepth : out.saved.depth = depth
  atLevel : out.saved.selectivity = level
  midgame : out.saved.endgame = false
  bounds : -scoreMax ≤ out.base.score ∧ out.base.score ≤ scoreMax

private theorem sorted_valid (s : State M) (prior : Valid mc b safe s) :
    Valid mc b safe ⟨sortRows s.rows 0,s.memory⟩ := by
  have order := (RootMoveStorage.sortRowsContract_proven s.rows 0).1
  exact ⟨(order.map Row.square).trans prior.1,order.length_eq ▸ prior.2.1,
    MidgameRows.perm order prior.2.2.1,prior.2.2.2⟩

theorem iteration_returns {t : Trial M} (works : Works mc b safe t) (s : State M) (ok : Valid mc b safe s)
    (sort : PublicationMir.SliceSort) (level depth : Nat) :
    ∃ out, iteration t sort level depth s = some out ∧ Complete mc b safe level depth out := by
  let hist := s.rows.head?.map MidgameRows.stats
  let w := MidgameAspiration.initial depth hist 0
  have ready := MidgameAspiration.initialContract_proven depth 0 hist (by
    intro h member
    obtain ⟨r,eq,rfl⟩ := Option.map_eq_some_iff.mp member
    exact ok.2.2.1 r (List.mem_of_mem_head? eq))
  obtain ⟨ans,m,ran,valid,range,⟨witness,inRows,attained⟩,upper⟩ :=
    retry_returns works s ok w ready.1 (Int.lt_of_lt_of_le (by decide : (0 : Int) < 512) ready.2.1) (MidgameAspiration.enough_retries ready.1 ready.2.1) depth
  let sorted := sortRows ans.rows 0
  let saved := PublicationMir.capture sort sorted depth level false
  have order := (sort.permutation PublicationMir.key sorted).trans (RootMoveStorage.sortRowsContract_proven ans.rows 0).1
  change saved.rows.Perm ans.rows at order
  have has : saved.rows ≠ [] := by
    intro eq
    have h := order.length_eq
    rw [eq] at h
    exact (Nat.ne_of_gt valid.2.1) h.symm
  obtain ⟨first,tail,eq⟩ := List.exists_cons_of_ne_nil has
  have head : PublicationMir.first saved = some first := by simp [PublicationMir.first,eq]
  have sortedScores := sort.ordered PublicationMir.key sorted
  change saved.rows.Pairwise (fun a b => b.score ≤ a.score) at sortedScores
  rw [eq] at sortedScores
  have chosen : first.score = ans.value := by
    have largest := order.mem_iff.mpr inRows
    rw [eq] at largest
    apply Int.le_antisymm (upper first (order.mem_iff.mp (eq ▸ List.mem_cons_self)))
    rw [← attained]
    rcases List.mem_cons.mp largest with rfl | later
    · exact Int.le_refl _
    · exact (List.pairwise_cons.mp sortedScores).1 _ later
  refine ⟨⟨⟨sortRows sorted 0,m⟩,saved,first⟩,?_,sorted_valid _ (sorted_valid _ valid),head,rfl,rfl,rfl,chosen ▸ range⟩
  change (retry t depth 16 s w).bind _ = _
  rw [ran]
  change (PublicationMir.first saved).bind _ = _
  rw [head]
  rfl

/-- Each predicate can depend on everything reported by the completed iteration.
    The time manager chooses whether the midgame hands off at its final depth. -/
structure Clock (M : Type) where
  deadline : Bool
  enter : Completed M → Bool
  stop : Completed M → Bool
  endgameStop : Nat → Bool

inductive Outcome (M : Type) where
  | done (state : State M)
  | endgame (completed : Completed M)

def startDepth (target : Nat) := if target % 2 = 0 then 2 else 1
def nextDepth (depth : Nat) := if depth ≤ 10 then depth + 2 else depth + 1

abbrev DepthStep (M : Type) := Nat → State M → Option (Completed M)

def loopWith (step : DepthStep M) (target : Nat) (clock : Option (Clock M)) :
    Nat → Nat → State M → Option (Outcome M)
  | 0, _, _ => none
  | fuel + 1, depth, s => do
    if target < depth then return .done s
    let out ← step depth s
    if let some tm := clock then
      if out.state.rows.length == 1 && tm.deadline then return .done out.state
      if target ≤ depth then
        return if tm.enter out then .endgame out else .done out.state
      if tm.stop out then return .done out.state
    loopWith step target clock fuel (nextDepth depth) out.state

def loop (t : Trial M) (sort : PublicationMir.SliceSort) (level target : Nat) (clock : Option (Clock M)) :=
  loopWith (iteration t sort level) target clock

def Ensures (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop) (level target : Nat) : Outcome M → Prop
  | .done s => Valid mc b safe s
  | .endgame out => Complete mc b safe level target out

def DepthWorks (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop) (level : Nat) (step : DepthStep M) : Prop :=
  ∀ depth s, Valid mc b safe s → ∃ out, step depth s = some out ∧ Complete mc b safe level depth out

theorem loop_returns (step : DepthStep M) (works : DepthWorks mc b safe level step) (s : State M) (ok : Valid mc b safe s)
    (target : Nat) (clock : Option (Clock M))
    (fuel depth : Nat) (budget : target + 1 - depth < fuel) :
    ∃ out, loopWith step target clock fuel depth s = some out ∧ Ensures mc b safe level target out := by
  induction fuel generalizing s depth with
  | zero => omega
  | succ fuel ih =>
    by_cases past : target < depth
    · exact ⟨.done s,by simp [loopWith,past],ok⟩
    obtain ⟨out,ran,complete⟩ := works depth s ok
    have later : target + 1 - nextDepth depth < fuel := by unfold nextDepth; split <;> omega
    obtain ⟨last,rest,invariant⟩ := ih out.state complete.valid _ later
    cases clock with
    | none => exact ⟨last,by simpa [loopWith,past,ran] using rest,invariant⟩
    | some tm =>
      by_cases forced : out.state.rows.length = 1 ∧ tm.deadline = true
      · exact ⟨.done out.state,by simp [loopWith,past,ran,forced],complete.valid⟩
      by_cases endDepth : target ≤ depth
      · have eq : depth = target := by omega
        subst target
        cases enter : tm.enter out with
        | false => exact ⟨.done out.state,by simp [loopWith,past,ran,forced,enter],complete.valid⟩
        | true => exact ⟨.endgame out,by simp [loopWith,past,ran,forced,enter],complete⟩
      cases stop : tm.stop out with
      | false => exact ⟨last,by simpa [loopWith,past,ran,forced,endDepth,stop] using rest,invariant⟩
      | true => exact ⟨.done out.state,by simp [loopWith,past,ran,forced,endDepth,stop],complete.valid⟩

def run (t : Trial M) (sort : PublicationMir.SliceSort) (level target : Nat) (clock : Option (Clock M)) (s : State M) :=
  loop t sort level target clock (target + 2) (startDepth target) s

theorem returns {t : Trial M} (works : Works mc b safe t) (s : State M) (ok : Valid mc b safe s)
    (sort : PublicationMir.SliceSort) (level target : Nat) (clock : Option (Clock M)) :
    ∃ out, run t sort level target clock s = some out ∧ Ensures mc b safe level target out :=
  loop_returns (iteration t sort level) (fun d s ok => iteration_returns works s ok sort level d) s ok target clock _ _ (by omega)

/-- With time to continue, the parity-dependent depth schedule reaches the target exactly. -/
def continueToEnd : Clock M := ⟨false,fun _ => true,fun _ => false,fun _ => false⟩

theorem loop_to_end (step : DepthStep M) (works : DepthWorks mc b safe level step) (s : State M) (ok : Valid mc b safe s)
    (target fuel depth : Nat) (within : depth ≤ target)
    (parity : depth ≤ 10 → depth % 2 = target % 2) (budget : target + 1 - depth < fuel) :
    ∃ out, loopWith step target (some continueToEnd) fuel depth s = some (.endgame out) ∧
      Complete mc b safe level target out := by
  induction fuel generalizing depth s with
  | zero => omega
  | succ fuel ih =>
    obtain ⟨it,ran,complete⟩ := works depth s ok
    have active : ¬ target < depth := by omega
    by_cases last : target ≤ depth
    · have eq : depth = target := by omega
      exact ⟨it,by simp [loopWith,active,ran,continueToEnd,last],eq ▸ complete⟩
    have reaches : nextDepth depth ≤ target := by
      unfold nextDepth
      split
      next h => have := parity h; omega
      next => omega
    have aligned (h : nextDepth depth ≤ 10) : nextDepth depth % 2 = target % 2 := by
      by_cases small : depth ≤ 10
      · have := parity small
        simp only [nextDepth,small,ite_true] at h ⊢
        omega
      · simp only [nextDepth,small,ite_false] at h
        omega
    have later : target + 1 - nextDepth depth < fuel := by unfold nextDepth; split <;> omega
    obtain ⟨out,steps,finished⟩ := ih _ complete.valid _ reaches aligned later
    exact ⟨out,by simpa [loopWith,active,ran,last,continueToEnd] using steps,finished⟩

theorem to_end {t : Trial M} (works : Works mc b safe t) (s : State M) (ok : Valid mc b safe s)
    (sort : PublicationMir.SliceSort) (level target : Nat) (positive : 0 < target) :
    ∃ out, run t sort level target (some continueToEnd) s = some (.endgame out) ∧ Complete mc b safe level target out := by
  apply loop_to_end (iteration t sort level) (fun d s ok => iteration_returns works s ok sort level d) s ok
  · unfold startDepth; split <;> omega
  · unfold startDepth; split <;> omega
  · omega

end NeuralReversi.AspirationMir.SerialTable.MidgameIterate
