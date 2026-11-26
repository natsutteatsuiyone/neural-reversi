import NeuralReversi.AspirationMir.Serial
import NeuralReversi.PvsParallel.SmallPv
import NeuralReversi.PvsParallel.RootFrameMir
import NeuralReversi.OuterMir.Program

namespace NeuralReversi.AspirationMir.SmallRoot

open Bitboard (Square)
open RootMoveStorage (Row)
open PvsBoardTree (Position GeneratedSquare legalSquares)
open PvsParallel
open EndgameSearch.Cancellation (EmptyState)
open MoveList.Generation (records)

def candidates (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (roots : List Square) (idx : Nat) (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder) :=
  (RootFrameMir.orderedRecords key sort (records fc b.val (Root.remainingMask mc b.val roots idx))).filterMap
    (RootFrameMir.asChoice mc b)

def withOrder (c : SmallPv.Config mc) (key : Position → Int → Int → Square → BitVec 32)
    (sort : RootFrameMir.LibraryOrder) : SmallPv.Config mc :=
  { c with order := fun b a z => candidates mc c.fc b ((legalSquares mc b.val).map Subtype.val) 0 (key b a z) sort }

theorem generatedOrder (c : SmallPv.Config mc) (key : Position → Int → Int → Square → BitVec 32)
    (sort : RootFrameMir.LibraryOrder) (b : Position) (a z : Int) :
    ((withOrder c key sort).order b a z).Perm (legalSquares mc b.val) := by
  let roots := (legalSquares mc b.val).map Subtype.val
  have all : Root.choices mc b roots 0 = legalSquares mc b.val := by
    apply List.filter_eq_self.mpr
    intro m mem
    exact (Root.mask_bit _ _).mpr (by simpa [roots] using (List.mem_map_of_mem (f := Subtype.val) mem))
  exact all ▸ (RootFrameMir.ordered_candidates mc c.fc b roots 0 (key b a z) sort).1

structure Answer where
  value : Int
  rows : List Row

def trial (c : SmallPv.Config mc) (b : Position) (s : EmptyState) (initial : List Row) (idx : Nat)
    (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (a z : Word) : Option Answer := do
  let roots := initial.map Row.square
  let generated := MoveList.Generation.fill false c.paired c.fc b.val (Root.remainingMask mc b.val roots idx) c.memory
  let count ← generated.len
  let cached ← generated.wipeout
  if h : 0 < count then
    match rootEntry count cached.isSome h, cached with
    | .wipeout, some sq =>
      let (writes, value) := rootWipeoutExit sq
      return ⟨value, writes.foldl (applyRootWrite pv) (RootMoveStorage.resetFrom initial idx)⟩
    | .header _ _ _ _, none =>
      let moves := candidates mc c.fc b roots idx key sort
      let child := SmallPv.child c b s
      let start := RootReplay.start a.toInt
      let final ← RootSerialMir.run Subtype.val child s.count.toNat rootMinimumSplitDepth z.toInt moves start
      return ⟨(rootEpilogue idx final.state.best).2, RootSerialMir.rowsAfter Subtype.val (fun m => pv m.val)
        child z.toInt moves start (RootMoveStorage.resetFrom initial idx)⟩
    | _, _ => none
  else none

theorem trial_sound (c : SmallPv.Config mc) (perm : ∀ b a z, (c.order b a z).Perm (legalSquares mc b.val))
    (b : Position) (s : EmptyState) (initial : List Row) (idx : Nat) (key : Square → BitVec 32)
    (sort : RootFrameMir.LibraryOrder) (pv : Square → List Square) (a z : Word)
    (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (aligned : s.matches b) (small : s.count.toNat ≤ 5) (window : Aspiration.Window a.toInt z.toInt) :
    ∃ answer snap, trial c b s initial idx key sort pv a z = some answer ∧ answer.value = snap.state.best ∧
      RootSuffix.Writes initial idx a.toInt z.toInt answer.rows snap ∧
      RootOrder.SoundAttempt mc c.fc b (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores := by
  let roots := initial.map Row.square
  have positive : idx < roots.length := by simpa [roots] using within
  let selected := Root.remainingMask mc b.val roots idx
  have subset : MoveList.Generation.IsSubset selected (Board.getMoves mc b.val) := by
    intro i bit
    simp only [selected, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true] at bit
    exact bit.1
  obtain ⟨_, _, count, nonempty, length⟩ := ConcurrentReplay.generatedRoot_positive mc c.fc b roots idx
    s.count c.paired c.memory covers positive aligned.1
  obtain ⟨_, _, _, _, cache⟩ := MoveList.Generation.generationContract_proven mc c.fc b.val selected false
    c.paired c.memory b.property subset (by simp)
  generalize value : ((MoveList.Generation.candidatesForCache (records c.fc b.val selected)).reverse.find?
    (MoveList.Wipeout.isMatch b.val.opponent)).map MoveList.Wipeout.Move.square = hit at cache
  let start := RootReplay.start a.toInt
  have cleared := RootSuffix.Writes.reset (initial := initial) (idx := idx) (alpha := a.toInt) (beta := z.toInt)
  cases hit with
  | none =>
    let moves := candidates mc c.fc b roots idx key sort
    let child := SmallPv.child c b s
    have order := (RootFrameMir.ordered_candidates mc c.fc b roots idx key sort).1
    have sound := (RootReplay.serial_of_children mc c.fc b roots idx moves child
      (fun m cp lo hi v => v = child m cp lo hi) a.toInt z.toInt covers positive window order
      (PvsMoveLoop.ScanCalls.ofAll (by intros; rfl)) (by
        intro m _ cp lo hi v w n same
        rw [Root.square_value_eq_child m, Root.childValue, Int.neg_neg, same]
        exact (SmallPv.child_sound c perm b s m cp lo hi aligned small w n).2.2)).2
    have writes := RootSerialMir.rowsAfter_writes Subtype.val (fun m => pv m.val) child moves start cleared
      (Root.roots_nodup covers) (by
        intro m mem
        rw [List.map_drop]
        exact (Root.choices_squares_mem covers m.val).mp (List.mem_map_of_mem (order.mem_iff.mp mem)))
    let final := RootReplay.serial Subtype.val child z.toInt moves start
    refine ⟨⟨final.state.best, _⟩, final, ?_, rfl, writes, sound⟩
    dsimp only [roots, selected] at length cache
    simp [trial, length, cache, nonempty, rootEntry, rootEpilogue, RootSerialMir.run_eq, roots,
      moves, child, start, final]
  | some sq =>
    obtain ⟨move, member, square, wipe⟩ := ConcurrentReplay.generatedWipeout mc c.fc b roots idx c.paired
      c.memory sq covers cache
    have legal := (Root.choices_squares_mem covers move.val).mp (List.mem_map_of_mem member)
    have shape := (cleared.scores (Root.roots_nodup covers)).1
    have inRows : sq ∈ ((RootMoveStorage.resetFrom initial idx).drop idx).map Row.square := by
      rw [List.map_drop, shape]
      simpa [roots, square] using legal
    have written := RootSuffix.Writes.record cleared (⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩ : RootOrder.Report)
      true (pv sq) inRows rfl
    have sound := RootReplay.wipeoutContract_proven mc c.fc b roots idx a.toInt z.toInt move covers positive
      window member wipe
    rw [square] at sound
    let final := start.record z.toInt ⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩
    refine ⟨⟨PvsMoveLoop.scoreMax, _⟩, final, ?_, rfl, written, sound⟩
    dsimp only [roots, selected] at length cache
    simp [trial, nonempty, rootEntry, length, cache, rootWipeoutExit, applyRootWrite, PvsMoveLoop.scoreMax]

def queryFor (run : Nat → Word → Word → Option Answer) : Query :=
  fun i a z => BitVec.ofInt 32 (((run i a z).map Answer.value).getD 0)

/-- An uninterrupted line accepts a trial whose score is inside its window.
    Each valid call must produce its own answer and the corresponding root row writes. -/
theorem completeTrials (mc fc) (board : Position) (initial : List Row) (idx : Nat) (m : Machine)
    (run : Nat → Word → Word → Option Answer)
    (actual : ∀ i (a z : Word), Aspiration.Window a.toInt z.toInt →
      ∃ (answer : Answer) (snap : RootReplay.Snapshot), run i a z = some answer ∧ answer.value = snap.state.best ∧
        RootSuffix.Writes initial idx a.toInt z.toInt answer.rows snap ∧
        RootOrder.SoundAttempt mc fc board (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores)
    (entry : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    let query := queryFor run
    ∃ raw last answer snap, Returns query (fun _ => false) m raw last ∧
      run last.index last.alpha last.beta = some answer ∧ answer.value = raw.toInt ∧
      answer.value = snap.state.best ∧
      RootSuffix.Writes initial idx last.alpha.toInt last.beta.toInt answer.rows snap ∧
      (last.alpha.toInt < snap.state.best ∧ snap.state.best < last.beta.toInt) ∧
      RootOrder.SoundAttempt mc fc board (initial.map Row.square) idx last.alpha.toInt last.beta.toInt
        snap.state.best snap.scores := by
  intro query
  have repr (v : Int) (lo : -32767 < v) (hi : v < 32767) : (BitVec.ofInt 32 v).toInt = v :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have finite : FiniteQueries query := by
    intro i a z w
    obtain ⟨answer, _, returned, _, _, h⟩ := actual i a z w
    change run i a z = some answer at returned
    change -32767 < (BitVec.ofInt 32 (((run i a z).map Answer.value).getD 0)).toInt ∧ _
    simp only [query, queryFor, returned, Option.map_some, Option.getD_some, repr _ h.1.lower h.1.upper]
    exact ⟨h.1.lower, h.1.upper⟩
  obtain ⟨raw,last,ran,_,_,w⟩ := completes_in_twelve finite m entry window
  obtain ⟨⟨observed,low,high⟩,rfl⟩ := returned_inside ran entry
  obtain ⟨answer, snap, returned, best, writes, att⟩ := actual last.index last.alpha last.beta w
  change run last.index last.alpha last.beta = some answer at returned
  have score := congrArg BitVec.toInt observed
  simp only [query, queryFor, returned, Option.map_some, Option.getD_some, repr _ att.1.lower att.1.upper] at score
  have inside : last.alpha.toInt < answer.value ∧ answer.value < last.beta.toInt := score ▸ ⟨low, high⟩
  have sound := best ▸ att
  exact ⟨_,last,answer,snap,ran,returned,score,best,writes,best ▸ inside,sound⟩

theorem completeLine (cfg : SmallPv.Config mc) (perm : ∀ b a z, (cfg.order b a z).Perm (legalSquares mc b.val))
    (board : Position) (state : EmptyState) (initial : List Row) (idx : Nat) (m : Machine)
    (rootKey : Nat → Word → Word → Square → BitVec 32) (order : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (covers : Root.Covers mc board (initial.map Row.square)) (within : idx < initial.length)
    (aligned : state.matches board) (small : state.count.toNat ≤ 5) (entry : m.pc = 0)
    (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    let run := fun i a z => trial cfg board state initial idx (rootKey i a z) order pv a z
    let query := queryFor run
    ∃ raw last answer snap, Returns query (fun _ => false) m raw last ∧
      run last.index last.alpha last.beta = some answer ∧ answer.value = raw.toInt ∧
      answer.value = snap.state.best ∧
      RootSuffix.Writes initial idx last.alpha.toInt last.beta.toInt answer.rows snap ∧
      (last.alpha.toInt < snap.state.best ∧ snap.state.best < last.beta.toInt) ∧
      RootOrder.SoundAttempt mc cfg.fc board (initial.map Row.square) idx last.alpha.toInt last.beta.toInt
        snap.state.best snap.scores := by
  intro run query
  exact completeTrials mc cfg.fc board initial idx m run
    (fun i a z w => trial_sound cfg perm board state initial idx (rootKey i a z) order pv a z
      covers within aligned small w) entry window

def PublicScoreContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (c : SmallPv.Config mc) (board : Position)
    (state : EmptyState) (initial : List Row) (m : Machine)
    (rootKey : Nat → Word → Word → Square → BitVec 32) (childKey : Position → Int → Int → Square → BitVec 32)
    (order : RootFrameMir.LibraryOrder) (sort : PublicationMir.SliceSort) (pv : Square → List Square)
    (nodes minimum : Nat) (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult),
    Root.Covers mc board (initial.map Row.square) → 0 < initial.length → state.matches board →
    state.count.toNat ≤ 5 → m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt → minimum ≤ 5 →
    let run := fun i a z => trial (withOrder c childKey order) board state initial 0 (rootKey i a z) order pv a z
    let query : Query := fun i a z => BitVec.ofInt 32 (((run i a z).map Answer.value).getD 0)
    ∃ raw last answer result, Returns query (fun _ => false) m raw last ∧
      run last.index last.alpha last.beta = some answer ∧ answer.value = raw.toInt ∧
      OuterMir.finish minimum 5 sort answer.rows state.count.toNat nodes prior fallback = some result ∧
      PublicationMir.score result = some raw.toInt ∧ raw.toInt = PvsBoardTree.minimax mc c.fc board

theorem publicScore : PublicScoreContract := by
  intro mc c board state initial m rootKey childKey order sort pv nodes minimum prior fallback
    covers positive aligned small entry window reportable
  intro run query
  let cfg := withOrder c childKey order
  obtain ⟨raw,last,answer,_,ran,returned,score,best,writes,inside,sound⟩ := completeLine cfg
    (generatedOrder c childKey order) board state initial 0 m rootKey order pv covers positive aligned small entry window
  obtain ⟨result, published, value⟩ := publishSingle covers positive writes inside sound sort state.count.toNat nodes fallback
  have exactScore := (sound.1.exact inside.1 inside.2).trans (Root.value_zero covers (by simpa using positive))
  have truth := score.symm.trans (best.trans exactScore)
  exact ⟨raw, last, answer, result, ran, returned, score, by simpa [OuterMir.finish, reportable] using published,
    value.trans (congrArg some truth.symm), truth⟩

end NeuralReversi.AspirationMir.SmallRoot
