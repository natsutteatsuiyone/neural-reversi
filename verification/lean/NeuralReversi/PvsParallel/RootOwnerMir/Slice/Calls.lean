import NeuralReversi.PvsParallel.RootOwnerMir.Slice

namespace NeuralReversi.PvsParallel.RootOwnerMir.Slice

open RootReplay (Snapshot)
open Bitboard (Square)

/-- The two call sites in a single iteration. Unissued calls have no return word. -/
structure Visit (Move : Type) where
  move : Move
  alpha : Int
  scout : Option Int
  wide : Option Int

def Visit.word (v : Visit Move) (pv : Bool) := if pv then v.wide else v.scout

def Visit.lower (v : Visit Move) (pv : Bool) (beta : Int) :=
  if pv then -beta else -(v.alpha+1)

def Visit.record (m : Move) (a scout wide : Int) (ns nw : Bool) : Visit Move :=
  ⟨m,a,if ns then some scout else none,if nw then some wide else none⟩

private def issuedAt (schedule : Nat → Bool × Bool) (depth limit i : Nat)
    (prior pending : List Move) (fresh : Nat) (s : Snapshot) (scout wide : Int) :=
  headerAction prior.length (prior.length + pending.length) fresh depth limit
    (schedule i).1 (schedule i).2 s.state.alpha scout wide

private abbrev decision (idx : Nat) (s : Snapshot) (beta score : Int) :=
  RootUpdateMir.decision (decide (idx = 1)) s.state.alpha beta s.state.best score

private abbrev scoutAt (source : Nat → PvsMoveLoop.ChildSearch Move) (i : Nat) (m : Move) (s : Snapshot) :=
  source i m false (-(s.state.alpha+1)) (-s.state.alpha)

private abbrev wideAt (source : Nat → PvsMoveLoop.ChildSearch Move) (i : Nat) (m : Move) (s : Snapshot) (beta : Int) :=
  source i m true (-beta) (-s.state.alpha)

/-- A loop execution labelled by precisely the child returns Rust used. The consumed
    prefix and the unsearched suffix concatenate to the physical move array. -/
inductive Path (square : Move → Square) (source : Nat → PvsMoveLoop.ChildSearch Move)
    (schedule : Nat → Bool × Bool) (depth limit : Nat) (beta : Int) :
    Nat → List Move → List Move → Nat → Snapshot → List (Visit Move) → Exit Move → Prop where
  | done : Path square source schedule depth limit beta i prior [] fresh s [] ⟨prior,prior.length,s,false⟩
  | split (act : issuedAt schedule depth limit i prior (m::rest) fresh s 0 0 = .split) :
      Path square source schedule depth limit beta i prior (m::rest) fresh s [] ⟨prior ++ m::rest,prior.length,s,true⟩
  | postpone (act : issuedAt schedule depth limit i prior (m::rest) fresh s 0 0 = .postpone prior.length nextFresh)
      (more : Path square source schedule depth limit beta (i+1) prior (rest ++ [m]) nextFresh s visits out) :
      Path square source schedule depth limit beta i prior (m::rest) fresh s visits out
  | cutoff (act : issuedAt schedule depth limit i prior (m::rest) fresh s (scoutAt source i m s) (wideAt source i m s beta) =
        .child (prior.length+1) ns nw score)
      (cuts : (decision (prior.length+1) s beta score).cuts = true) :
      Path square source schedule depth limit beta i prior (m::rest) fresh s [Visit.record m s.state.alpha (scoutAt source i m s) (wideAt source i m s beta) ns nw]
        ⟨prior ++ m::rest,prior.length+1,RootUpdateMir.after s (square m) score (decision (prior.length+1) s beta score),false⟩
  | next (act : issuedAt schedule depth limit i prior (m::rest) fresh s (scoutAt source i m s) (wideAt source i m s beta) =
        .child (prior.length+1) ns nw score)
      (keep : (decision (prior.length+1) s beta score).cuts = false)
      (more : Path square source schedule depth limit beta (i+1) (prior ++ [m]) rest fresh
        (RootUpdateMir.after s (square m) score (decision (prior.length+1) s beta score)) visits out) :
      Path square source schedule depth limit beta i prior (m::rest) fresh s
        (Visit.record m s.state.alpha (scoutAt source i m s) (wideAt source i m s beta) ns nw :: visits) out

/-- The chronology is generated from arbitrary return words at each iteration.
    No function mapping a board and window to a single score is assumed. -/
theorem path_exists (square : Move → Square) (source : Nat → PvsMoveLoop.ChildSearch Move)
    (schedule : Nat → Bool × Bool) (depth limit : Nat) (beta : Int) (i : Nat)
    (prior pending : List Move) (fresh : Nat) (s : Snapshot) :
    ∃ visits out, Path square source schedule depth limit beta i prior pending fresh s visits out := by
  cases pending with
  | nil => exact ⟨_,_,.done⟩
  | cons m rest =>
    let scout := source i m false (-(s.state.alpha+1)) (-s.state.alpha)
    let wide := source i m true (-beta) (-s.state.alpha)
    let act := issuedAt schedule depth limit i prior (m::rest) fresh s
    have inside : prior.length < prior.length + (m::rest).length := by simp
    by_cases hand : canHandoff prior.length (prior.length+(m::rest).length) depth limit (schedule i).1
    · exact ⟨_,_,.split (by simp only [issuedAt,headerAction_eq,inside,hand,ite_true])⟩
    · by_cases busy : PvsPostpone.shouldPostpone (schedule i).2 prior.length fresh
      · obtain ⟨visits,out,tail⟩ := path_exists square source schedule depth limit beta (i+1) prior
          (rest ++ [m]) (fresh-1) s
        exact ⟨visits,out,.postpone (by simp only [issuedAt,headerAction_eq,inside,hand,busy,ite_true,ite_false]) tail⟩
      · obtain ⟨ns,nw,score,child⟩ : ∃ ns nw score, act scout wide = .child (prior.length+1) ns nw score := by
          simp only [act,issuedAt,headerAction_eq,inside,hand,Bool.eq_false_iff.mpr busy,Bool.false_eq_true,ite_false,ite_true]
          repeat' split
          all_goals exact ⟨_,_,_,rfl⟩
        cases cuts : (decision (prior.length+1) s beta score).cuts with
        | true =>
          exact ⟨_,_,.cutoff child cuts⟩
        | false =>
          obtain ⟨visits,out,tail⟩ := path_exists square source schedule depth limit beta (i+1) (prior ++ [m])
            rest fresh (RootUpdateMir.after s (square m) score (decision (prior.length+1) s beta score))
          exact ⟨_,_,.next child cuts tail⟩
termination_by pending.length + (fresh - prior.length)
decreasing_by
  · simp only [List.length_append,List.length_cons,List.length_nil]
    simp only [PvsPostpone.shouldPostpone,Bool.and_eq_true,decide_eq_true_eq] at busy
    omega
  · simp only [List.length_append,List.length_cons,List.length_nil]
    omega

/-- A real owner may rotate unsearched moves, but it visits a move at most once.
    A scout and its PV re-search are the two fields of that one visit. -/
theorem Path.distinct (path : Path sq src schedule d limit beta i prior pending fresh s visits out)
    (unique : pending.Nodup) :
    (visits.map Visit.move).Nodup ∧ ∀ v ∈ visits, v.move ∈ pending := by
  induction path with
  | done | split => simp
  | postpone act more ih =>
    obtain ⟨unique,members⟩ := ih ((List.perm_append_singleton _ _).nodup_iff.mpr unique)
    exact ⟨unique,fun v mem => (List.perm_append_singleton _ _).mem_iff.mp (members v mem)⟩
  | cutoff => simp [Visit.record]
  | next act keep more ih =>
    obtain ⟨absent,later⟩ := List.nodup_cons.mp unique
    obtain ⟨distinct,members⟩ := ih later
    refine ⟨List.nodup_cons.mpr ⟨?_,distinct⟩,?_⟩
    · intro appears
      obtain ⟨v,mem,eq⟩ := List.mem_map.mp appears
      dsimp only [Visit.record] at eq
      exact absent (eq ▸ members v mem)
    · intro v mem
      rcases List.mem_cons.mp mem with rfl | tail
      · simp [Visit.record]
      · exact List.mem_cons_of_mem _ (members v tail)

private theorem next_window (s : Snapshot) (beta score : Int) (first square)
    (w : Aspiration.Window s.state.alpha beta) :
    Aspiration.Window (RootUpdateMir.after s square score (RootUpdateMir.decision first s.state.alpha beta s.state.best score)).state.alpha beta := by
  obtain ⟨lo,ordered,hi⟩ := w
  by_cases raises : s.state.alpha < score <;> by_cases better : s.state.best < score <;> by_cases inside : score < beta <;>
    simp [RootUpdateMir.after,RootUpdateMir.decision,RootUpdateMir.changes,raises,better,inside] <;>
    constructor <;> omega

private theorem issued_word (pred : Int → Prop) (sc : pred x) (pv : pred y) (role : Bool)
    (got : (Visit.record m a x y ns nw).word role = some v) : pred v := by
  cases role <;> cases ns <;> cases nw <;> simp [Visit.word,Visit.record] at got <;> subst v <;> assumption

/-- Windows remain in the engine's score range as alpha changes. Every return
    word in a visit comes from an issued child, so it retains its native width. -/
theorem Path.operands (path : Path sq src schedule d limit beta i prior pending fresh s visits out)
    (w : Aspiration.Window s.state.alpha beta) (pred : Int → Prop)
    (raw : ∀ j m pv a z, pred (src j m pv a z)) :
    ∀ v ∈ visits, Aspiration.Window v.alpha beta ∧ ∀ pv x, v.word pv = some x → pred x := by
  induction path with
  | done | split => simp
  | postpone _ _ ih => exact ih w
  | cutoff =>
    intro v mem
    obtain rfl := List.mem_singleton.mp mem
    exact ⟨w,fun role _ got => issued_word pred (raw _ _ _ _ _) (raw _ _ _ _ _) role got⟩
  | next _ _ _ ih =>
    intro v mem
    rcases List.mem_cons.mp mem with rfl | later
    · exact ⟨w,fun role _ got => issued_word pred (raw _ _ _ _ _) (raw _ _ _ _ _) role got⟩
    · exact ih (next_window _ _ _ _ _ w) _ later

def Supplies (r : Replies Move) (beta : Int) (visits : List (Visit Move)) : Prop :=
  ∀ v ∈ visits, ∀ pv word, v.word pv = some word → r v.move pv (v.lower pv beta) (-v.alpha) = some word

private def needed (a : HeaderAction) (pv : Bool) := match a with
  | .child _ scout wide _ => if pv then wide else scout
  | _ => false

private theorem replaceWords (eq : headerAction idx count fresh d lim free busy a scout wide = act)
    (sc : needed act false = true → scout' = scout) (pv : needed act true = true → wide' = wide) :
    headerAction idx count fresh d lim free busy a scout' wide' = act := by
  simp only [headerAction_eq] at eq ⊢
  repeat' split at eq
  all_goals cases eq <;> simp_all [needed]
  intro inside; omega

private theorem current (prior : List Move) (m : Move) (rest : List Move) :
    (prior ++ m :: rest)[prior.length]? = some m := by simp

private theorem sameAction (r : Replies Move)
    (act : issuedAt schedule depth limit i prior (m::rest) fresh s scout wide = action)
    (sc : needed action false = true → r m false (-(s.state.alpha+1)) (-s.state.alpha) = some scout)
    (pv : needed action true = true → r m true (-beta) (-s.state.alpha) = some wide) :
    header r (prior ++ m :: rest) prior.length fresh depth limit (schedule i).1 (schedule i).2
      s.state.alpha beta = action := by
  simp only [header,current,Option.bind_some,List.length_append]
  exact replaceWords act (fun h => by simp [sc h]) (fun h => by simp [pv h])

private theorem arguments (ok : Supplies r beta (Visit.record m a scout wide ns nw :: rest)) :
    (ns = true → r m false (-(a+1)) (-a) = some scout) ∧
    (nw = true → r m true (-beta) (-a) = some wide) := by
  constructor
  · intro h; apply ok _ List.mem_cons_self false scout; simp [Visit.word,Visit.record,h]
  · intro h; apply ok _ List.mem_cons_self true wide; simp [Visit.word,Visit.record,h]

private theorem ready {ns nw : Bool} {sc pv : Option Int}
    (first : ns = true → sc = some x) (second : nw = true → pv = some y) :
    ((!ns || sc.isSome) && (!nw || pv.isSome)) = true := by
  cases ns <;> cases nw <;> simp_all

theorem Path.replays {square : Move → Square} (path : Path square source schedule depth limit beta i prior pending fresh s visits out)
    (r : Replies Move) (provided : Supplies r beta visits) :
    run square r schedule depth limit beta i prior.length fresh (prior ++ pending) s = some out := by
  induction path with
  | @done i prior fresh s =>
    have done_ : header r prior prior.length fresh depth limit (schedule i).1 (schedule i).2 s.state.alpha beta = .finish := by
      simp [header,headerAction_eq]
    simp only [List.append_nil]
    rw [run,done_]
  | split act =>
    rw [run,sameAction r act (by simp [needed]) (by simp [needed])]
  | postpone act more ih =>
    rw [run,sameAction r act (by simp [needed]) (by simp [needed])]
    simpa [MoveList.Ordering.postponeMove,List.append_assoc] using ih provided
  | cutoff act cuts =>
    obtain ⟨sc,pv⟩ := arguments provided
    rw [run,sameAction r act sc pv,current]
    simp only [bind,Option.bind]
    rw [ite_eq_left (ready sc pv)]
    simp only [decision] at cuts
    rw [ite_eq_left cuts]
  | next act keep more ih =>
    obtain ⟨sc,pv⟩ := arguments provided
    rw [run,sameAction r act sc pv,current]
    simp only [bind,Option.bind]
    rw [ite_eq_left (ready sc pv)]
    simp only [decision] at keep
    simp only [keep,Bool.false_eq_true,ite_false]
    simpa [List.append_assoc,decision] using ih (fun v mem => provided v (List.mem_cons_of_mem _ mem))

end NeuralReversi.PvsParallel.RootOwnerMir.Slice
