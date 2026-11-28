module
public import NeuralReversi.PvsParallel.SerialScout

@[expose] public section

namespace NeuralReversi.EndgameSearch.SerialLeafRestore

open Bitboard.Moves (X86Config)
open Flip.Dispatch (TargetConfig)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open EndgameTreeCache (Store Address entryFor)
open Cancellation (EmptyState State Outcome)
open ShallowMir (Stage)

structure Reader where
  shallow : (h : List Store) → (b : Position) → (a : Int) → Option Int
  deep : (fc : FlipConfig) → (paired : Bool) → (b : Position) → (s : State) → (a : Int) →
    (first : Bitboard.Square) → (rest : List Bitboard.Square) → Option Int

def historyReader (addr : Address) : Reader where
  shallow h b a := EndgameCache.probe (entryFor addr h .shallow b.val) b.val a
  deep fc paired b s a first rest := DeepRecursionMir.early fc paired b s addr a first rest

abbrev Post := Result × EmptyState

/-- The next move receives the previous child's returned list, including the
    last child before a beta cutoff. -/
def stage (b : Position) (beta : Int) (child : Move → EmptyState → List Store → Post) :
    List Move → Int → EmptyState → List Store → Stage × EmptyState
  | [], best, s, h => (.next best h,s)
  | m :: ms, best, s, h =>
    let (r,t) := child m s h
    if beta ≤ r.score then (.cutoff (ShallowMir.write b (beta-1) r),t)
    else stage b beta child ms (max best r.score) t r.history

def groups (b : Position) (a : Int) (child : Move → EmptyState → List Store → Post) :
    List (List Move) → Int → EmptyState → List Store → Post
  | [], best, s, h => (ShallowMir.write b a ⟨best,h⟩,s)
  | moves :: ms, best, s, h =>
    match stage b (a+1) child moves best s h with
    | (.cutoff r,t) => (r,t)
    | (.next value after,t) => groups b a child ms value t after

private theorem stage_eq (b : Position) (beta : Int) (child : Move → EmptyState → List Store → Post)
    (project : Move → List Store → Result) (s : EmptyState) (same : ∀ m h, child m s h = (project m h,s))
    (moves : List Move) (best : Int) (h : List Store) :
    stage b beta child moves best s h = (ShallowMir.stage b beta project moves best h,s) := by
  induction moves generalizing best h with
  | nil => rfl
  | cons m rest ih =>
    rw [stage,ShallowMir.stage,same]
    dsimp only
    split
    · rfl
    · exact ih _ _

private theorem groups_eq (b : Position) (a : Int) (child : Move → EmptyState → List Store → Post)
    (project : Move → List Store → Result) (s : EmptyState) (same : ∀ m h, child m s h = (project m h,s))
    (stages : List (List Move)) (best : Int) (h : List Store) :
    groups b a child stages best s h = (ShallowMir.groups b a project stages best h,s) := by
  induction stages generalizing best h with
  | nil => rfl
  | cons ms rest ih =>
    rw [groups,ShallowMir.groups,stage_eq _ _ _ project _ same]
    cases ShallowMir.stage b (a+1) project ms best h with
    | cutoff => rfl
    | next => exact ih _ _

def shallowChild (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (reader : Reader) (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val) (beta : Int) (h : List Store)
    (rec : EmptyState → Int → List Store → Post) : Post :=
  let placed := s.remove m.val
  let (result,after) := if placed.count.toNat = 4 then
    let next := (play mc fc b m).val
    (ShallowMir.child next (-beta) h placed.count.toNat (reader.shallow h next (-beta))
      ((EndgameStability.tryCutoff next.val 4 ⟨-beta⟩).map (·.value))
      (fun _ => ShallowMir.four target bmi2 next placed
        (Empties.oddQuadrants (Board.emptySquares b.val) ^^^ Empties.quadrant m.val) (-beta))
      (fun _ => (rec placed (-beta) h).1),placed)
  else
    let (r,t) := rec placed (-beta) h
    (⟨-r.score,r.history⟩,t)
  (result,after.restore m.val)

private theorem shallowChild_eq (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (addr : Address) (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val) (beta : Int) (h : List Store)
    (rec : EmptyState → Int → List Store → Post) (spec : EmptyState → Int → List Store → Result)
    (aligned : s.matches b) (same : rec (s.remove m.val) (-beta) h = (spec (s.remove m.val) (-beta) h,s.remove m.val)) :
    shallowChild target bmi2 mc fc (historyReader addr) b s m beta h rec =
      (ShallowMir.executeChild target bmi2 mc fc addr b s m beta h spec,s) := by
  have inv := ShallowMir.restore_after mc fc b s m aligned
  by_cases four : (s.remove m.val).count.toNat = 4 <;>
    simp [shallowChild,ShallowMir.executeChild,ShallowMir.child,four,same,inv,historyReader]

def mobile (b : Position) (first : Move) (rest : List Move) (a : Int) (s : EmptyState) (h : List Store)
    (probe stable : Option Int) (stages : List (List Move)) (child : Move → EmptyState → List Store → Post) : Post :=
  match probe, stable with
  | some v, _ | none, some v => (⟨v,h⟩,s)
  | none, none =>
    if rest.isEmpty then
      let (r,t) := child first s h
      (ShallowMir.write b a r,t)
    else groups b a child stages (-127) s h

private theorem mobile_eq (b : Position) (first : Move) (rest : List Move) (a : Int) (s : EmptyState) (h : List Store)
    (probe stable : Option Int) (stages : List (List Move)) (child : Move → EmptyState → List Store → Post)
    (spec : Move → List Store → Result) (same : ∀ m h, child m s h = (spec m h,s)) :
    mobile b first rest a s h probe stable stages child =
      (ShallowMir.node b (first::rest) a h false 0 (fun _ => ⟨0,h⟩) probe stable stages spec,s) := by
  cases probe <;> cases stable <;> try rfl
  simp only [mobile,ShallowMir.node]
  split
  · rw [same]
  · exact groups_eq _ _ _ _ _ same _ _ _

def shallowWith (reader : Reader) (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (s : EmptyState) (a : Int) (h : List Store) : Post :=
  match moves : legalSquares mc b.val with
  | [] =>
    if _other : Board.hasLegalMoves mc (pass b).val = true then
      let (r,t) := shallowWith reader target bmi2 mc fc (pass b) s (-(a+1)) h
      (⟨-r.score,r.history⟩,t)
    else (⟨Board.AwardedScore b.val,h⟩,s)
  | first :: rest =>
    mobile b first rest a s h
      (reader.shallow h b a)
      ((EndgameStability.tryCutoff b.val s.count.toNat ⟨a⟩).map (·.value))
      (ShallowMir.stages mc b) (fun m t h' =>
        shallowChild target bmi2 mc fc reader b t m (a+1) h' (shallowWith reader target bmi2 mc fc (play mc fc b m).val))
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,_other⟩) b.property
  · exact (play mc fc b m).property

abbrev shallow (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address) :=
  shallowWith (historyReader addr) target bmi2 mc fc

theorem shallow_refines (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (s : EmptyState) (a : Int) (h : List Store) (aligned : s.matches b) :
    shallow target bmi2 mc fc addr b s a h = (ShallowMir.run target bmi2 mc fc addr b s a h,s) := by
  rw [shallow,shallowWith,ShallowMir.run]
  cases moves : legalSquares mc b.val with
  | nil =>
    simp only
    split
    next other =>
      have same := shallow_refines target bmi2 mc fc addr (pass b) s (-(a+1)) h (emptyState_pass b _ _ aligned)
      unfold shallow at same
      rw [same]
    next => rfl
  | cons first rest =>
    apply mobile_eq
    intro m h'
    apply shallowChild_eq _ _ _ _ _ _ _ _ _ _ _ _ aligned
    exact shallow_refines target bmi2 mc fc addr (play mc fc b m).val _ _ _
      (emptyStatePlayContract_proven mc fc b m _ _ aligned).2.1
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,by assumption⟩) b.property
  · exact (play mc fc b m).property

/-- Both shallow and deep children return the state that their caller undoes.
    The poll preceding the placement may return without changing the list. -/
def deepChild (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (reader : Reader)
    (b : Position) (m : GeneratedSquare mc b.val) (beta : Int) (input : State)
    (recurse : Int → State → Outcome) : Outcome :=
  let polled := {input with polls := input.polls.after}
  if input.polls.occurred then ⟨none,polled⟩ else
    let placed := {polled with empties := input.empties.remove m.val}
    let out := if placed.empties.count.toNat ≤ 6 then
      let (r,t) := shallowWith reader target bmi2 mc fc (play mc fc b m).val placed.empties (-beta) placed.history
      ⟨some (-r.score), {placed with empties := t, history := r.history}⟩
    else DeepRecursionMir.negate (recurse (-beta) placed)
    {out with state := {out.state with empties := out.state.empties.restore m.val}}

private theorem deepChild_eq (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (m : GeneratedSquare mc b.val) (beta : Int) (input : State)
    (rec project : Int → State → Outcome) (aligned : input.empties.matches b)
    (same : ∀ t : State, t.empties.matches (play mc fc b m).val →
      rec (-beta) t = project (-beta) t ∧ (rec (-beta) t).state.empties = t.empties) :
    let out := deepChild target bmi2 mc fc (historyReader addr) b m beta input rec
    out = DeepRecursionMir.executeChild m.val beta input (ShallowMir.run target bmi2 mc fc addr (play mc fc b m).val)
      project ∧ out.state.empties = input.empties := by
  let placed := {input with polls := input.polls.after, empties := input.empties.remove m.val}
  have fits : placed.empties.matches (play mc fc b m).val :=
    (emptyStatePlayContract_proven mc fc b m _ _ aligned).2.1
  have shallowEq := shallow_refines target bmi2 mc fc addr (play mc fc b m).val placed.empties (-beta) placed.history fits
  have ⟨deepEq,deepState⟩ := same placed fits
  have inv := ShallowMir.restore_after mc fc b input.empties m aligned
  dsimp only
  by_cases poll : input.polls.occurred
  · simp [deepChild,DeepRecursionMir.executeChild,poll]
  by_cases small : placed.empties.count.toNat ≤ 6
  · dsimp only [placed,shallow] at shallowEq small
    simp [deepChild,DeepRecursionMir.executeChild,poll,small,shallowEq,inv]
  · dsimp only [placed] at small deepState deepEq ⊢
    simp [deepChild,DeepRecursionMir.executeChild,poll,small,DeepRecursionMir.negate,← deepEq,deepState,inv]

private theorem ordered_eq (sub spec : Move → State → Outcome) (s : EmptyState)
    (same : ∀ m t, t.empties = s → sub m t = spec m t ∧ (sub m t).state.empties = s)
    (beta : Int) (ms : List Move) (best : Int) (t : State) (live : t.empties = s) :
    let out := DeepRecursionMir.ordered sub beta ms best t
    out = DeepRecursionMir.ordered spec beta ms best t ∧ out.2.empties = s := by
  induction ms generalizing best t with
  | nil => exact ⟨rfl,live⟩
  | cons m rest ih =>
    obtain ⟨score,kept⟩ := same m t live
    simp only [DeepRecursionMir.ordered,← score]
    split
    · exact ⟨rfl,kept⟩
    · exact ih _ _ kept

private theorem moves_eq (b : Position) (a : Int) (t : State) (first : Move) (rest order : List Move)
    (sub spec : Move → State → Outcome)
    (same : ∀ m u, u.empties = t.empties → sub m u = spec m u ∧ (sub m u).state.empties = t.empties) :
    let out := DeepRecursionMir.moves b a t first rest order sub
    out = DeepRecursionMir.moves b a t first rest order spec ∧ out.state.empties = t.empties := by
  have done (best : Int) (u : State) (kept : u.empties = t.empties) :
      (DeepRecursionMir.finish b a best u).state.empties = t.empties := by
    unfold DeepRecursionMir.finish
    split <;> exact kept
  unfold DeepRecursionMir.moves
  by_cases one : rest.isEmpty
  · obtain ⟨eq,kept⟩ := same first t rfl
    simp only [one,ite_true,← eq]
    exact ⟨trivial,done _ _ kept⟩
  · obtain ⟨eq,kept⟩ := ordered_eq sub spec t.empties same (a+1) order (-127) t rfl
    simp only [ite_eq_right one,← eq]
    exact ⟨trivial,done _ _ kept⟩

def deepWith (reader : Reader) (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (a : Int) (s : State) : Outcome :=
  match moves : legalSquares mc b.val with
  | [] =>
    if _other : Board.hasLegalMoves mc (pass b).val = true then
      DeepRecursionMir.negate (deepWith reader target bmi2 paired mc fc (pass b) (-(a+1)) s)
    else ⟨some (Board.AwardedScore b.val),s⟩
  | first :: rest =>
    match reader.deep fc paired b s a first.val (rest.map Subtype.val) with
    | some score => ⟨some score,s⟩
    | none =>
      DeepRecursionMir.moves b a s first rest (deepOrder mc fc b) fun m t =>
        deepChild target bmi2 mc fc reader b m (a+1) t (deepWith reader target bmi2 paired mc fc (play mc fc b m).val)
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,_other⟩) b.property
  · exact (play mc fc b m).property

abbrev deep (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address) :=
  deepWith (historyReader addr) target bmi2 paired mc fc

theorem deep_refines (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (a : Int) (s : State) (aligned : s.empties.matches b) :
    let out := deep target bmi2 paired mc fc addr b a s
    out = DeepRecursionMir.run target bmi2 paired mc fc addr b a s ∧ out.state.empties = s.empties := by
  rw [deep,deepWith,DeepRecursionMir.run]
  dsimp only [historyReader]
  cases moves : legalSquares mc b.val with
  | nil =>
    dsimp only
    split
    next other =>
      obtain ⟨eq,kept⟩ := deep_refines target bmi2 paired mc fc addr (pass b) (-(a+1)) s (emptyState_pass b _ _ aligned)
      exact ⟨congrArg DeepRecursionMir.negate eq,kept⟩
    next => exact ⟨rfl,rfl⟩
  | cons first rest =>
    dsimp only
    cases DeepRecursionMir.early fc paired b s addr a first.val (rest.map Subtype.val) with
    | some => exact ⟨rfl,rfl⟩
    | none =>
      apply moves_eq
      intro m t live
      obtain ⟨eq,kept⟩ := deepChild_eq target bmi2 mc fc addr b m (a+1) t _ _ (live ▸ aligned)
        (fun input fit => deep_refines target bmi2 paired mc fc addr (play mc fc b m).val (-(a+1)) input fit)
      exact ⟨eq,kept.trans live⟩
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,by assumption⟩) b.property
  · exact (play mc fc b m).property

open PvsParallel.SerialScout (Config)
open ScoreRefresh (Score toDiscDiff fromDiscDiff)
open PvsParallel.NonPvMir (sourceRun)

def serialState (s : EmptyState) (past : List Store) : State :=
  ⟨s,past,⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩⟩

def scaled (checked : Bool) (result : Outcome) : Option Int := do
  let score ← (fromDiscDiff checked ⟨BitVec.ofInt 32 (result.score.getD 0)⟩).toOption
  let word ← Cancellation.publish result false score
  return word.raw.toInt

/-- A single specialized NonPV activation. The fixed solvers read the empty
    squares, while recursive arms return the storage they physically restored. -/
def leafWith (reader : Reader) (c : Config) (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState)
    (a _z : Score) (past : List Store) : Option ((Int × List Store) × EmptyState) := do
  let n := s.count.toNat
  let disc := (toDiscDiff a).raw.toInt
  if n ≤ 4 then
    let raw ← sourceRun c.target c.bmi2 mc fc c.address past b s.mask s.count s.count a c.checked
    return ((raw.raw.toInt,past),s)
  else if n ≤ 6 then
    let (r,t) := shallowWith reader c.target c.bmi2 mc fc b s disc past
    let raw ← (fromDiscDiff c.checked ⟨BitVec.ofInt 32 r.score⟩).toOption
    return ((raw.raw.toInt,r.history),t)
  else if n ≤ 12 then
    let out := deepWith reader c.target c.bmi2 c.paired mc fc b disc (serialState s past)
    let value ← scaled c.checked out
    return ((value,out.state.history),out.state.empties)
  else none

abbrev leaf (c : Config) := leafWith (historyReader c.address) c

private theorem middle_score (c : Config) (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState)
    (a : Score) (past : List Store) (lo : 4 < s.count.toNat) (hi : s.count.toNat ≤ 6) :
    sourceRun c.target c.bmi2 mc fc c.address past b s.mask s.count s.count a c.checked =
      (fromDiscDiff c.checked ⟨BitVec.ofInt 32
        (ShallowMir.run c.target c.bmi2 mc fc c.address b s (toDiscDiff a).raw.toInt past).score⟩).toOption := by
  have size : s.count.toNat = 5 ∨ s.count.toNat = 6 := by omega
  unfold sourceRun
  unfold PvsParallel.NonPvMir.withShallow
  rcases size with n|n <;> simp [n,PvsParallel.NonPvMir.dispatch]

private theorem deep_score (c : Config) (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState)
    (a z : Score) (past : List Store) (lo : 6 < s.count.toNat) (hi : s.count.toNat ≤ 12) :
    PvsParallel.DeepLeafMir.sourceRun c.target c.bmi2 c.paired mc fc c.address b a z s.count (serialState s past)
      [] false false c.checked false =
    scaled c.checked (DeepRecursionMir.run c.target c.bmi2 c.paired mc fc c.address b (toDiscDiff a).raw.toInt
      (serialState s past)) := by
  unfold PvsParallel.DeepLeafMir.sourceRun
  unfold PvsParallel.DeepLeafMir.withNws
  unfold PvsParallel.DeepLeafMir.lookup
  simp [PvsParallel.DeepLeafMir.dispatch,serialState,lo,hi]
  rfl

theorem leaf_refines (c : Config) (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState)
    (a z : Score) (past : List Store) (aligned : s.matches b) (size : s.count.toNat ≤ 12) :
    leaf c mc fc b s a z past =
      (PvsParallel.SerialScout.run c mc fc b s a z past).map (fun r => (r,s)) := by
  by_cases fixed : s.count.toNat ≤ 4
  · have small : s.count.toNat ≤ 6 := by omega
    simp only [leaf,leafWith,PvsParallel.SerialScout.run,small,fixed,ite_true]
    cases sourceRun c.target c.bmi2 mc fc c.address past b s.mask s.count s.count a c.checked <;> rfl
  by_cases small : s.count.toNat ≤ 6
  · have eq := shallow_refines c.target c.bmi2 mc fc c.address b s (toDiscDiff a).raw.toInt past aligned
    change shallowWith _ _ _ _ _ _ _ _ _ = _ at eq
    have raw := middle_score c mc fc b s a past (by omega) small
    simp only [leaf,leafWith,PvsParallel.SerialScout.run,small,ite_true,ite_eq_right fixed,eq,raw]
    cases (fromDiscDiff c.checked ⟨BitVec.ofInt 32
      (ShallowMir.run c.target c.bmi2 mc fc c.address b s (toDiscDiff a).raw.toInt past).score⟩).toOption <;> rfl
  · obtain ⟨eq,post⟩ := deep_refines c.target c.bmi2 c.paired mc fc c.address b (toDiscDiff a).raw.toInt (serialState s past) aligned
    change deepWith _ _ _ _ _ _ _ _ _ = _ at eq
    change (deepWith _ _ _ _ _ _ _ _ _).state.empties = _ at post
    have raw := deep_score c mc fc b s a z past (by omega) size
    simp only [leaf,leafWith,PvsParallel.SerialScout.run,ite_eq_right small,ite_eq_right fixed,size,ite_true]
    change (do let value ← scaled c.checked _
               return ((value,_),_)) =
      ((do let score ← PvsParallel.DeepLeafMir.sourceRun c.target c.bmi2 c.paired mc fc c.address b a z s.count
              (serialState s past) [] false false c.checked false
           pure (score,(DeepRecursionMir.run c.target c.bmi2 c.paired mc fc c.address b (toDiscDiff a).raw.toInt
             (serialState s past)).state.history))).map _
    rw [post,eq,raw]
    cases scaled c.checked (DeepRecursionMir.run c.target c.bmi2 c.paired mc fc c.address b (toDiscDiff a).raw.toInt
      (serialState s past)) <;> rfl

def Contract : Prop :=
  ∀ (c : Config) (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState)
    (a z : Score) (past : List Store), s.matches b → s.count.toNat ≤ 12 →
    EndgameTreeCache.soundHistory mc fc past → z.raw.toInt = a.raw.toInt + 1 →
    ∃ raw after, leaf c mc fc b s a z past = some ((raw,after),s) ∧
      PvsParallel.boundedReplay 1 (PvsBoardTree.build mc fc b) false a.raw.toInt z.raw.toInt raw ∧
      EndgameTreeCache.soundHistory mc fc after ∧ past <+: after

theorem contract_proven : Contract := by
  intro c mc fc b s a z past aligned size cache null
  obtain ⟨v,h,ret,valid⟩ := PvsParallel.SerialScout.returns c mc fc b s a z past aligned size cache null
  exact ⟨v,h,by rw [leaf_refines c mc fc b s a z past aligned size,ret]; rfl,valid⟩

end NeuralReversi.EndgameSearch.SerialLeafRestore
