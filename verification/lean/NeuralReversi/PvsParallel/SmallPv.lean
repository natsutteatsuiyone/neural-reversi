import NeuralReversi.PvsParallel.PvEntryMir
import NeuralReversi.PvsParallel.PvSerialMir
import NeuralReversi.PvsParallel.NonPvMir
import NeuralReversi.PvsParallel.Replay

namespace NeuralReversi.PvsParallel.SmallPv

open Aspiration (Window SoundResult)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass build minimax)
open EndgameSearch.Cancellation (EmptyState)
open PvsMoveLoop (ChildSearch ScanCalls MoveCalls moveScore nextScanState)

structure Config (mc : X86Config) where
  target : Flip.Dispatch.TargetConfig
  bmi2 : Bool
  fc : Cutoffs.FlipConfig
  paired : Bool
  checked : Bool
  memory : MoveList.Initialization.Memory Move.Move
  order : (b : Position) → Int → Int → List (GeneratedSquare mc b.val)

private def raw (a : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 a⟩

def scout (c : Config mc) (b : Position) (s : EmptyState) (a : Int) : Option Int :=
  (NonPvMir.run c.target c.bmi2 mc c.fc (fun _ _ => 0) [] b s.mask s.count s.count
    (raw a) c.checked).map (fun score => score.raw.toInt)

theorem scout_returns (c : Config mc) (b : Position) (s : EmptyState) (a z : Int)
    (aligned : s.matches b) (small : s.count.toNat ≤ 4) (w : Window a z) (null : z = a + 1) :
    ∃ v, scout c b s a = some v ∧ boundedReplay 1 (build mc c.fc b) false a z v := by
  have av : (raw a).raw.toInt = a := BitVec.toInt_ofInt_eq_self (by decide)
    (by have := w.lower; omega) (by have := w.ordered; have := w.upper; omega)
  have zv : (raw z).raw.toInt = z := BitVec.toInt_ofInt_eq_self (by decide)
    (by have := w.ordered; have := w.lower; omega) (by have := w.upper; omega)
  obtain ⟨score, value, trace, _⟩ := NonPvMir.scoreContract_proven Cutoffs.Execution
    (fun _ => boundedReplay 0) c.target c.bmi2 mc c.fc (fun _ _ => 0) [] b s.mask s.count s.count
    (raw a) (raw z) c.checked aligned.1 aligned (by omega) (by simp [EndgameTreeCache.soundHistory])
    (by rwa [av, zv])
  refine ⟨score.raw.toInt, by simp [scout, value], ?_⟩
  change Execution (boundedReplay 0) _ _ _ _ _
  rw [av, zv] at trace
  exact trace

private def observe (c : Config mc) (b : Position) (s : EmptyState)
    (wide : Position → EmptyState → Int → Int → Option Int) : ChildSearch (GeneratedSquare mc b.val) :=
  fun m pv a z =>
    let next := (play mc c.fc b m).val
    let child := s.remove m.val
    (if pv then wide next child a z else scout c next child a).getD 0

private def runFuel (c : Config mc) : Nat → Position → EmptyState → Int → Int → Option Int
  | 0, _, _, _, _ => none
  | fuel + 1, b, s, a, z =>
    let wide := runFuel c fuel
    let leaf := PvEntryMir.run mc c.fc b s.count c.paired c.checked c.memory a z (fun next _ => wide next s)
    match leaf with
    | some v => some v
    | none => (PvSerialMir.run Subtype.val (observe c b s wide) s.count.toNat rootMinimumSplitDepth z
        (c.order b a z) 0 a (-PvsMoveLoop.scoreInf)).map (·.best)

def run (c : Config mc) (b : Position) (s : EmptyState) (alpha beta : Int) : Option Int :=
  runFuel c (Board.playRank mc b.val + 1) b s alpha beta

private theorem windowCalls {Move : Type} {completed : PvsMoveLoop.ChildRelation Move}
    (search : ChildSearch Move) (moves : List Move)
    (first : Bool) (a z best : Int) (w : Window a z)
    (all : ∀ m ∈ moves, ∀ pv lo hi, Window lo hi → (pv = false → hi = lo + 1) →
      completed m pv lo hi (search m pv lo hi)) :
    ScanCalls completed search true z moves first a best := by
  induction moves generalizing first a best with
  | nil => trivial
  | cons m rest ih =>
    refine ⟨?_, ?_⟩
    · unfold MoveCalls
      split
      · exact all m (by simp) true _ _ (PvsMoveLoop.reflectedWindow w) (by simp)
      · exact ⟨all m (by simp) false _ _ (PvsMoveLoop.scoutWindow w) (by omega),
          fun _ => all m (by simp) true _ _ (PvsMoveLoop.reflectedWindow w) (by simp)⟩
    · cases next : nextScanState true z a best (moveScore true first search m a z) with
      | none => trivial
      | some state =>
        rcases state with ⟨nextA, nextB⟩
        apply ih false nextA nextB _ (fun k hk => all k (List.mem_cons_of_mem _ hk))
        unfold nextScanState at next
        split at next
        · split at next
          · split at next
            · rename_i inside
              cases Option.some.inj next
              exact ⟨by have := w.lower; omega, inside.2.1, w.upper⟩
            · cases next
          · cases Option.some.inj next; exact w
        · cases Option.some.inj next; exact w

private theorem fuel_returns (c : Config mc)
    (perm : ∀ b a z, (c.order b a z).Perm (legalSquares mc b.val))
    (fuel : Nat) (b : Position) (s : EmptyState) (a z : Int) (aligned : s.matches b)
    (small : s.count.toNat ≤ 4) (rank : Board.playRank mc b.val < fuel) (w : Window a z) :
    ∃ score, runFuel c fuel b s a z = some score ∧ boundedReplay fuel (build mc c.fc b) true a z score := by
  induction fuel generalizing b s a z with
  | zero => omega
  | succ fuel ih =>
    let wide := runFuel c fuel
    let leaf := PvEntryMir.run mc c.fc b s.count c.paired c.checked c.memory a z (fun next _ => wide next s)
    have passed (stuck : legalSquares mc b.val = []) (mobile : Board.hasLegalMoves mc (pass b).val = true) :
        ∃ score, wide (pass b) s (-z) (-a) = some score ∧
          boundedReplay fuel (build mc c.fc (pass b)) true (-z) (-a) score := by
      have smaller := Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil stuck, mobile⟩) b.property
      change Board.playRank mc (pass b).val < Board.playRank mc b.val at smaller
      exact ih (pass b) s _ _ (EndgameSearch.emptyState_pass b _ _ aligned) small (by omega)
        (PvsMoveLoop.reflectedWindow w)
    cases exit : leaf with
    | some score =>
      refine ⟨score, ?_, ?_⟩
      · change (match leaf with | some v => some v | none => _) = _
        rw [exit]
      · apply PvEntryMir.exitExecution mc c.fc b s.count c.paired c.checked c.memory a z _
          aligned.1 _ exit
        intro stuck mobile answer child
        obtain ⟨got, value, trace⟩ := passed stuck mobile
        exact Option.some.inj (value.symm.trans child) ▸ trace
    | none =>
      cases moves : legalSquares mc b.val with
      | nil =>
        obtain ⟨v, someLeaf⟩ := PvEntryMir.stuck_returns mc c.fc b s.count c.paired c.checked c.memory
          a z (fun next _ => wide next s) aligned.1 moves (fun h => by
            obtain ⟨v, value, _⟩ := passed moves h; exact ⟨v, value⟩)
        change leaf = some v at someLeaf
        rw [exit] at someLeaf
        cases someLeaf
      | cons first rest =>
        let search := observe c b s wide
        have all (m : GeneratedSquare mc b.val) (pv : Bool) (lo hi : Int) (window : Window lo hi)
            (null : pv = false → hi = lo + 1) :
            boundedReplay fuel (build mc c.fc (play mc c.fc b m).val) pv lo hi (search m pv lo hi) := by
          let next := play mc c.fc b m
          let state := s.remove m.val
          obtain ⟨positive, child, _, _, _⟩ :=
            EndgameSearch.emptyStatePlayContract_proven mc c.fc b m _ _ aligned
          have fitted : state.matches next.val := child
          have size : state.count.toNat ≤ 4 := by
            change (s.count - 1).toNat ≤ 4
            rw [BitVec.toNat_sub_of_le (by change 1 ≤ s.count.toNat; omega)]
            omega
          have enough : Board.playRank mc next.val.val < fuel := by have := next.property; omega
          cases pv with
          | true =>
            obtain ⟨v, outcome, trace⟩ := ih next.val state lo hi fitted size enough window
            change boundedReplay fuel (build mc c.fc next.val) true lo hi ((runFuel c fuel next.val state lo hi).getD 0)
            simpa [outcome] using trace
          | false =>
            obtain ⟨v, outcome, trace⟩ := scout_returns c next.val state lo hi fitted size window (null rfl)
            change boundedReplay fuel (build mc c.fc next.val) false lo hi ((scout c next.val state lo).getD 0)
            rw [outcome]
            exact boundedReplay_mono (by omega : 1 ≤ fuel) _ _ _ _ _ trace
        have order := perm b a z
        rw [moves] at order
        obtain ⟨res, numeric, evidence⟩ := PvSerialMir.execution (exit := Cutoffs.Execution)
          (children := fun _ => boundedReplay fuel) mc c.fc b first rest (c.order b a z)
          s.count.toNat rootMinimumSplitDepth a z search moves order
          (windowCalls search _ true a z _ w (fun m _ => all m))
        exact ⟨res.best, by simpa [runFuel, leaf, exit, wide, search] using congrArg (Option.map (·.best)) numeric,
          evidence⟩

theorem normalReturn (c : Config mc) (perm : ∀ b a z, (c.order b a z).Perm (legalSquares mc b.val))
    (b : Position) (s : EmptyState) (alpha beta : Int) (aligned : s.matches b) (small : s.count.toNat ≤ 4)
    (w : Window alpha beta) :
    ∃ score, run c b s alpha beta = some score ∧ Replayed (build mc c.fc b) true alpha beta score ∧
      SoundResult (minimax mc c.fc b) alpha beta score := by
  obtain ⟨score, value, replay⟩ := fuel_returns c perm _ b s alpha beta aligned small (by omega :
    Board.playRank mc b.val < Board.playRank mc b.val + 1) w
  exact ⟨score, value, ⟨_, replay⟩, replayContract_proven _ _ _ _ _ _ w (by simp) replay⟩

def childResult (c : Config mc) (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val)
    (pv : Bool) (lo hi : Int) : Option Int :=
  let next := (play mc c.fc b m).val
  let state := s.remove m.val
  if pv then run c next state lo hi else scout c next state lo

def child (c : Config mc) (b : Position) (s : EmptyState) : ChildSearch (GeneratedSquare mc b.val) :=
  fun m pv lo hi => (childResult c b s m pv lo hi).getD 0

theorem child_sound (c : Config mc) (perm : ∀ b a z, (c.order b a z).Perm (legalSquares mc b.val))
    (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val) (pv : Bool) (lo hi : Int)
    (aligned : s.matches b) (size : s.count.toNat ≤ 5) (w : Window lo hi) (null : pv = false → hi = lo + 1) :
    let score := child c b s m pv lo hi
    childResult c b s m pv lo hi = some score ∧ Replayed (build mc c.fc (play mc c.fc b m).val) pv lo hi score ∧
      SoundResult (minimax mc c.fc (play mc c.fc b m).val) lo hi score := by
  let next := (play mc c.fc b m).val
  let state := s.remove m.val
  have ⟨positive, fit, _, _, _⟩ := EndgameSearch.emptyStatePlayContract_proven mc c.fc b m _ _ aligned
  have aligned' : state.matches next := fit
  have small : state.count.toNat ≤ 4 := by
    change (s.count - 1).toNat ≤ 4
    rw [BitVec.toNat_sub_of_le (by change 1 ≤ s.count.toNat; omega)]
    change s.count.toNat - 1 ≤ 4
    omega
  cases pv with
  | true =>
    obtain ⟨score, got, replay, sound⟩ := normalReturn c perm next state lo hi aligned' small w
    have actual : childResult c b s m true lo hi = some score := got
    simp only [child, actual, Option.getD_some, true_and]
    exact ⟨replay, sound⟩
  | false =>
    obtain ⟨score, got, trace⟩ := scout_returns c next state lo hi aligned' small w (null rfl)
    have actual : childResult c b s m false lo hi = some score := got
    have replay : Replayed (build mc c.fc next) false lo hi score := ⟨1, trace⟩
    have sound := replayContract_proven _ _ _ _ _ 1 w null trace
    simp only [child, actual, Option.getD_some, true_and]
    exact ⟨replay, sound⟩

end NeuralReversi.PvsParallel.SmallPv
