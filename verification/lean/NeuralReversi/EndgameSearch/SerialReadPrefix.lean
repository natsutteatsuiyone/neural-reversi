module
public import NeuralReversi.EndgameSearch.SerialLeafRestore

@[expose] public section

namespace NeuralReversi.EndgameSearch.SerialLeafRestore

open EndgameTreeCache (Store)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open Cancellation (EmptyState State Outcome)
open Flip.Dispatch (TargetConfig)
open Bitboard.Moves (X86Config)
open ShallowMir (Stage)

abbrev History := List Store

structure SameReads (base : History) (left right : Reader) : Prop where
  shallow : ∀ h, base <+: h → ∀ b a, left.shallow h b a = right.shallow h b a
  deep : ∀ fc paired b (s : State), base <+: s.history → ∀ a first rest,
    left.deep fc paired b s a first rest = right.deep fc paired b s a first rest

private def saved : Stage → History
  | .next _ h => h
  | .cutoff r => r.history

def Related (history : α → History) (before : History) (x y : α) : Prop :=
  x = y ∧ before <+: history y

private abbrev Agrees (base : History) (l r : Move → EmptyState → History → Post) :=
  ∀ m t h, base <+: h → Related (fun z : Post => z.1.history) h (l m t h) (r m t h)

private theorem stage_prefix (b : Position) (beta : Int) (l r : Move → EmptyState → History → Post)
    (same : Agrees base l r) (ms : List Move) (best : Int) (t : EmptyState) (h : History) (prefix_ : base <+: h) :
    Related (fun z : Stage × EmptyState => saved z.1) h (stage b beta l ms best t h) (stage b beta r ms best t h) := by
  induction ms generalizing best t h with
  | nil => exact ⟨rfl,List.prefix_refl _⟩
  | cons m ms ih =>
    obtain ⟨eq,grow⟩ := same m t h prefix_
    simp only [stage,eq]
    split
    · exact ⟨rfl,grow.trans (List.prefix_append _ _)⟩
    · obtain ⟨next,more⟩ := ih _ _ _ (prefix_.trans grow)
      exact ⟨next,grow.trans more⟩

private theorem groups_prefix (b : Position) (a : Int) (l r : Move → EmptyState → History → Post)
    (same : Agrees base l r) (ms : List (List Move)) (best : Int) (t : EmptyState) (h : History) (prefix_ : base <+: h) :
    Related (fun z : Post => z.1.history) h (groups b a l ms best t h) (groups b a r ms best t h) := by
  induction ms generalizing best t h with
  | nil => exact ⟨rfl,List.prefix_append _ _⟩
  | cons group tail ih =>
    obtain ⟨eq,grow⟩ := stage_prefix b (a+1) l r same group best t h prefix_
    simp only [groups,eq]
    cases run : stage b (a+1) r group best t h with
    | mk st next =>
      rw [run] at grow
      cases st with
      | cutoff => exact ⟨rfl,grow⟩
      | next best' later =>
        obtain ⟨done_,after⟩ := ih best' next later (prefix_.trans grow)
        exact ⟨done_,grow.trans after⟩

private theorem mobile_prefix (b : Position) (first : Move) (rest : List Move) (a : Int) (t : EmptyState)
    (h : History) (probe stable : Option Int) (ms : List (List Move)) (l r : Move → EmptyState → History → Post)
    (same : Agrees base l r) (prefix_ : base <+: h) :
    Related (fun z : Post => z.1.history) h (mobile b first rest a t h probe stable ms l)
      (mobile b first rest a t h probe stable ms r) := by
  cases probe <;> cases stable <;> try exact ⟨rfl,List.prefix_refl _⟩
  simp only [mobile]
  split
  · obtain ⟨eq,grow⟩ := same first t h prefix_
    simp only [eq]
    exact ⟨rfl,grow.trans (List.prefix_append _ _)⟩
  · exact groups_prefix _ _ _ _ same _ _ _ _ prefix_

theorem shallow_prefix (same : SameReads base l r) (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (b : Position) (s : EmptyState) (a : Int) (h : History) (prefix_ : base <+: h) :
    Related (fun z : Post => z.1.history) h (shallowWith l target bmi2 mc fc b s a h)
      (shallowWith r target bmi2 mc fc b s a h) := by
  rw [shallowWith.eq_def l target bmi2 mc fc b s a h,shallowWith.eq_def r target bmi2 mc fc b s a h]
  cases moves : legalSquares mc b.val with
  | nil =>
    dsimp only
    split
    next other =>
      obtain ⟨eq,grow⟩ := shallow_prefix same target bmi2 mc fc (pass b) s (-(a+1)) h prefix_
      simp only [eq]
      exact ⟨rfl,grow⟩
    next => exact ⟨rfl,List.prefix_refl _⟩
  | cons first rest =>
    dsimp only
    rw [same.shallow h prefix_]
    apply mobile_prefix _ _ _ _ _ _ _ _ _ _ _ _ prefix_
    intro m t past extend
    have ⟨eq,grow⟩ := shallow_prefix same target bmi2 mc fc (play mc fc b m).val (t.remove m.val) (-(a+1)) past extend
    by_cases four : (t.remove m.val).count.toNat = 4
    · simp only [shallowChild,four,ite_true,ShallowMir.child,same.shallow past extend]
      cases r.shallow past (play mc fc b m).val (-(a+1)) with
      | some => exact ⟨rfl,List.prefix_refl _⟩
      | none =>
        cases EndgameStability.tryCutoff (play mc fc b m).val.val 4 ⟨-(a+1)⟩ <;>
          first | exact ⟨rfl,List.prefix_refl _⟩ | exact ⟨rfl,List.prefix_append _ _⟩
    · simp only [shallowChild,ite_eq_right four,eq]; exact ⟨rfl,grow⟩
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,by assumption⟩) b.property
  · exact (play mc fc b m).property

private abbrev Sub (base : History) (l r : Move → State → Outcome) :=
  ∀ m (s : State), base <+: s.history → Related (fun z : Outcome => z.state.history) s.history (l m s) (r m s)

private theorem ordered_prefix (l r : Move → State → Outcome) (same : Sub base l r) (beta : Int) (ms : List Move)
    (best : Int) (s : State) (enter : base <+: s.history) :
    Related (fun z : Int × State => z.2.history) s.history (DeepRecursionMir.ordered l beta ms best s)
      (DeepRecursionMir.ordered r beta ms best s) := by
  induction ms generalizing best s with
  | nil => exact ⟨rfl,List.prefix_refl _⟩
  | cons m ms ih =>
    obtain ⟨eq,grow⟩ := same m s enter
    simp only [DeepRecursionMir.ordered,eq]
    split
    · exact ⟨rfl,grow⟩
    · obtain ⟨done_,more⟩ := ih _ _ (enter.trans grow)
      exact ⟨done_,grow.trans more⟩

private theorem moves_prefix (b : Position) (a : Int) (s : State) (first : Move) (rest order : List Move)
    (l r : Move → State → Outcome) (same : Sub base l r) (enter : base <+: s.history) :
    Related (fun z : Outcome => z.state.history) s.history (DeepRecursionMir.moves b a s first rest order l)
      (DeepRecursionMir.moves b a s first rest order r) := by
  have finish (v : Int) (t : State) (kept : s.history <+: t.history) :
      s.history <+: (DeepRecursionMir.finish b a v t).state.history := by
    unfold DeepRecursionMir.finish
    split
    · exact kept
    · exact kept.trans (List.prefix_append _ _)
  unfold DeepRecursionMir.moves
  by_cases one : rest.isEmpty
  · obtain ⟨eq,grow⟩ := same first s enter
    simp only [one,ite_true,eq]
    exact ⟨rfl,finish _ _ grow⟩
  · obtain ⟨eq,grow⟩ := ordered_prefix l r same (a+1) order (-127) s enter
    simp only [ite_eq_right one,eq]
    exact ⟨rfl,finish _ _ grow⟩

theorem deep_prefix (same : SameReads base l r) (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config)
    (fc : FlipConfig) (b : Position) (a : Int) (s : State) (enter : base <+: s.history) :
    Related (fun z : Outcome => z.state.history) s.history (deepWith l target bmi2 paired mc fc b a s)
      (deepWith r target bmi2 paired mc fc b a s) := by
  rw [deepWith.eq_def l target bmi2 paired mc fc b a s,deepWith.eq_def r target bmi2 paired mc fc b a s]
  cases moves : legalSquares mc b.val with
  | nil =>
    dsimp only
    split
    next other =>
      obtain ⟨eq,grow⟩ := deep_prefix same target bmi2 paired mc fc (pass b) (-(a+1)) s enter
      simp only [eq]
      exact ⟨rfl,grow⟩
    next => exact ⟨rfl,List.prefix_refl _⟩
  | cons first rest =>
    dsimp only
    rw [same.deep fc paired b s enter]
    cases r.deep fc paired b s a first.val (rest.map Subtype.val) with
    | some => exact ⟨rfl,List.prefix_refl _⟩
    | none =>
      apply moves_prefix _ _ _ _ _ _ _ _ _ enter
      intro m t within
      let placed : State := {t with polls := t.polls.after,empties := t.empties.remove m.val}
      have ⟨sh,shGrow⟩ := shallow_prefix same target bmi2 mc fc (play mc fc b m).val placed.empties (-(a+1)) t.history within
      have ⟨dh,dhGrow⟩ := deep_prefix same target bmi2 paired mc fc (play mc fc b m).val (-(a+1)) placed within
      by_cases stop : t.polls.occurred
      · simp only [deepChild,stop,ite_true]; exact ⟨rfl,List.prefix_refl _⟩
      by_cases small : placed.empties.count.toNat ≤ 6
      · dsimp only [placed] at small sh
        simp only [deepChild,ite_eq_right stop,small,ite_true,sh]
        exact ⟨rfl,shGrow⟩
      · dsimp only [placed] at small dh
        simp only [deepChild,ite_eq_right stop,ite_eq_right small,dh]
        exact ⟨rfl,dhGrow⟩
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves,by assumption⟩) b.property
  · exact (play mc fc b m).property

theorem leaf_prefix (same : SameReads base l r) (c : PvsParallel.SerialScout.Config) (mc : X86Config)
    (fc : FlipConfig) (b : Position) (s : EmptyState) (a z : ScoreRefresh.Score) (h : History) (enter : base <+: h) :
    leafWith l c mc fc b s a z h = leafWith r c mc fc b s a z h := by
  by_cases fixed : s.count.toNat ≤ 4
  · simp only [leafWith,fixed,ite_true]
  by_cases middle : s.count.toNat ≤ 6
  · simp only [leafWith,ite_eq_right fixed,middle,ite_true,(shallow_prefix same c.target c.bmi2 mc fc b s _ h enter).1]
  by_cases small : s.count.toNat ≤ 12
  · simp only [leafWith,ite_eq_right fixed,ite_eq_right middle,small,ite_true]
    rw [(deep_prefix same c.target c.bmi2 c.paired mc fc b _ _ enter).1]
  · simp only [leafWith,ite_eq_right fixed,ite_eq_right middle,ite_eq_right small]

end NeuralReversi.EndgameSearch.SerialLeafRestore
