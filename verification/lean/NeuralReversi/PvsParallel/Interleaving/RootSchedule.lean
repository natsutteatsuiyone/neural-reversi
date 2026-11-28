module
public import NeuralReversi.PvsParallel.Interleaving.Root
public import NeuralReversi.PvsParallel.Interleaving.Join

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.RootSchedule

open PvsBoardTree (Position GeneratedSquare)
open PvsMoveLoop (ChildSearch ChildRelation Bound scoreInf scoreMax)
open Aspiration (Window SoundResult)
open Bitboard (Square)
open RootMoveStorage (Row)
open RootOwnerMir (Result Replies fromMoves)
open RootTrial (square labels prefixValue ownerRows)
open Threading

def task (r : Result (GeneratedSquare mc b)) (depth : Nat) (beta : Int) (cut : Bool) : Schedule.Task :=
  ⟨true,true,cut,depth,beta,r.snapshot.state⟩

def indexed (done_ : ChildRelation (GeneratedSquare mc b)) (r : Result (GeneratedSquare mc b)) : ChildRelation Nat :=
  fun index pv a z score => ∃ m, (r.visited ++ r.remaining.pending)[index]? = some m ∧ done_ m pv a z score

/-- Both branches retain the rows written by the search. At a split the iterator starts
    on the actual owner's cursor and its reports come from the scheduler's ticket stream. -/
inductive Trial (mc : Bitboard.Moves.X86Config) (b : Position) (initial : List Row) (idx : Nat)
    (alpha beta : Int) (done_ : ChildRelation (GeneratedSquare mc b.val)) : Int → List Row → Prop where
  | owner (r : Result (GeneratedSquare mc b.val)) (replies : Replies (GeneratedSquare mc b.val))
      (moves : List (GeneratedSquare mc b.val)) (pv : Square → List Square)
      (order : moves.Perm (Root.choices mc b (initial.map Row.square) idx))
      (children : ∀ m cp a z v, replies m cp a z = some v → done_ m cp a z v)
      (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
      (done : r.handoff = false) :
      Trial mc b initial idx alpha beta done_ r.snapshot.state.best (ownerRows alpha beta initial idx r replies pv)
  | split (r : Result (GeneratedSquare mc b.val)) (replies : Replies (GeneratedSquare mc b.val))
      (moves : List (GeneratedSquare mc b.val)) (pv : Square → List Square)
      (order : moves.Perm (Root.choices mc b (initial.map Row.square) idx))
      (children : ∀ m cp a z v, replies m cp a z = some v → done_ m cp a z v)
      (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
      (handed : r.handoff = true) (positive : 0 < depth) (width : 7 ≤ w)
      (j : Join (task r depth beta cut) (indexed done_ r) (labels r).length (BitVec.ofNat w r.remaining.idx))
      (returns : j.answer = some score) :
      Trial mc b initial idx alpha beta done_ score
        (FrameReturns.reportRows (square r) pv beta (ownerRows alpha beta initial idx r replies pv) r.snapshot
          (Schedule.reports j.state))

theorem iterator_layout {b : Position} {r : Result (GeneratedSquare mc b.val)}
    {moves : List (GeneratedSquare mc b.val)} {replies : Replies (GeneratedSquare mc b.val)}
    (order : moves.Perm (Root.choices mc b roots idx))
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r) (width : 7 ≤ n) :
    (labels r).length < 2^n ∧ (BitVec.ofNat n r.remaining.idx).toNat = r.remaining.idx := by
  obtain ⟨layout,cursor⟩ := RootTrial.layout_of_header ran
  have chosen : (Root.choices mc b roots idx).length ≤ 64 :=
    Nat.le_trans (List.length_filter_le _ _) (EndgameOrderMir.generated_layout mc b).2
  have size : (labels r).length = moves.length := by simpa [labels] using layout.length_eq
  have powBound : 64 < 2^n := Nat.lt_of_lt_of_le (by decide : 64 < 2^7) (Nat.pow_le_pow_right (by decide) width)
  have fit : (labels r).length < 2^n := by rw [size,order.length_eq]; omega
  have cursorLe : r.remaining.idx ≤ (labels r).length := by simp [labels,cursor]
  exact ⟨fit,by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]⟩

theorem Trial.certified {b : Position} {done_ : ChildRelation (GeneratedSquare mc b.val)}
    (trial : Trial mc b initial idx alpha beta done_ score rows) (covers : Root.Covers mc b (initial.map Row.square))
    (inside : idx < initial.length) (w : Window alpha beta)
    (children : ∀ m cp a z s, Window a z → (cp = false → z = a+1) → done_ m cp a z s →
      SoundResult (-(Root.squareValue mc fc b m.val)) a z s) :
    ∃ snap : RootReplay.Snapshot, snap.state.best = score ∧ RootSuffix.Writes initial idx alpha beta rows snap ∧
      RootOrder.SoundAttempt mc fc b (initial.map Row.square) idx alpha beta score snap.scores := by
  cases trial with
  | owner r replies moves pv order source ran done =>
    exact ⟨r.snapshot,rfl,RootTrial.owner_writes covers order ran pv,
      RootTrial.returned order ran done w covers (by simpa using inside)
        (fun m cp a z v eq win narrow => children m cp a z v win narrow (source m cp a z v eq))⟩
  | @split schedule depth limit n cut score r replies moves pv order source ran handed positive width j returns =>
    have child : RootTrial.ChildSound mc fc b replies :=
      fun m cp a z v eq win narrow => children m cp a z v win narrow (source m cp a z v eq)
    obtain ⟨_,cursor,entry,cap,seed⟩ := RootTrial.ready ran handed w child
    obtain ⟨fit,startEq⟩ := iterator_layout order ran width
    have sliceCap (m : GeneratedSquare mc b.val) : Root.childValue mc fc b m ≤ scoreMax := RootTrial.child_cap m
    have prefixCap : prefixValue mc fc b r ≤ scoreMax :=
      PvsMoveLoop.bestValue_le _ _ _ _ (by decide) (fun m _ => sliceCap m)
    have squareCap (i : Nat) : Root.squareValue mc fc b (square r i) ≤ scoreMax := by
      unfold Root.squareValue
      split
      · apply sliceCap
      · decide
    have worker (i : Nat) (cp : Bool) (a z v : Int) (win : Window a z) (narrow : cp = false → z = a+1)
        (reply : indexed done_ r i cp a z v) :
        SoundResult (-((Root.squareValue mc fc b ∘ square r) i)) a z v := by
      obtain ⟨m,atMove,completed⟩ := reply
      have label : square r i = m.val := by simp only [square,labels,List.getElem?_map,atMove,Option.map_some,Option.getD_some]
      exact (congrArg (fun sq => SoundResult (-(Root.squareValue mc fc b sq)) a z v) label).mpr
        (children m cp a z v win narrow completed)
    obtain ⟨res,final⟩ := j.accepted (value := Root.squareValue mc fc b ∘ square r) fit returns
      entry cap (by simp [task]) seed prefixCap (fun i _ => squareCap i) (fun i _ _ => worker i)
    rw [startEq] at res
    have tail : RootTrial.Accepted mc fc b r alpha beta (Schedule.reports j.state) score :=
      ⟨res.valid,res.distinct,res.steps,final ▸ res.bound,final⟩
    obtain ⟨best,attempt⟩ := RootTrial.handoff order ran handed w covers (by simpa using inside) child tail
    exact ⟨_,best,RootTrial.handoff_writes covers order ran handed _ res.valid pv,attempt⟩

end NeuralReversi.PvsParallel.Interleaving.RootSchedule
