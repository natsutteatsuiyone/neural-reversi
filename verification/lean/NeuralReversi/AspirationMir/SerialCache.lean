module
public import NeuralReversi.AspirationMir.SerialCache.Driver
public import NeuralReversi.PvsParallel.SerialCacheStack

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialCache

open PvsParallel
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open EndgameTreeCache (soundHistory)
open SerialExecution (Inputs)

abbrev Roots := Nat → Word → Word → SerialRecursion.RootInput

def NextLevel (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (v : Receipt) : Prop :=
  let rows := RootMoveStorage.sortRows v.answer.rows 0
  Root.Covers mc b (rows.map Row.square) ∧ 0 < rows.length ∧ v.call.score.toInt = PvsBoardTree.minimax mc fc b

def search (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots) : Trial :=
  fun s => SerialCacheStack.trial c b (inputs s.index) (roots s.index s.regs.alpha s.regs.beta) s

def Recorded (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (r : Receipt) : Prop :=
  let s := r.state
  let op := inputs s.index
  let a := s.regs.alpha
  let z := s.regs.beta
  Root.Covers mc b (s.rows.map Row.square) ∧ 0 < s.rows.length ∧ Aspiration.Window a.toInt z.toInt ∧
    soundHistory mc fc r.before ∧ r.before <+: r.after ∧ soundHistory mc fc r.after ∧
    SerialCacheStack.SourceProgram c op s.opening (SerialRootExecution.context b a z)
      (SerialRootExecution.attempt c op s.opening b (roots s.index a z) s.rows 0 a z) r.closing r.answer

def Published (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (rows : List Row) (m : Machine) (past : History) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (fallback : Unit → PublicationMir.SearchResult) : Prop :=
  let op := search c b inputs roots
  ∃ prior last result,
    run op 12 (SerialStack.first m rows) past = some (prior,last) ∧
    Path op (SerialStack.first m rows) past prior last ∧ prior.length < 12 ∧
    (prior ++ [last]).map Receipt.before = past :: prior.map Receipt.after ∧
    past <+: last.after ∧ soundHistory mc fc last.after ∧
    (∀ r ∈ prior ++ [last], Recorded c b inputs roots r) ∧
    Returns (SerialStack.queryFor ((prior ++ [last]).map Receipt.toVisit)) (fun _ => false) m last.call.score last.call ∧
    PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
      (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
    PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val) ∧
    NextLevel mc fc b last

def Contract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (rows : List Row) (m : Machine) (past : History) sort nodes fallback,
    (∀ i, SerialCacheStack.SharedFacts (inputs i) ((Board.emptyCount b.val).toNat - 1)) →
    soundHistory mc fc past → Root.Covers mc b (rows.map Row.square) → 0 < rows.length →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    Published c b inputs roots rows m past sort nodes fallback

theorem publication : Contract := by
  intro mc fc c b inputs roots rows m past sort nodes fallback shared sound covers positive pc window
  obtain ⟨prior,last,ran,path,bound,cover,pos,cert,inside,prefix',good,seen⟩ :=
    complete (trial := search c b inputs roots) (evidence := Recorded c b inputs roots)
      (fun s h valid cv ok w => by
        obtain ⟨k,ans,after,returned,closed,admissible,growth,projection⟩ := SerialCacheStack.trial_returns c b
          (inputs s.index) (roots s.index s.regs.alpha s.regs.beta) s h (shared _) valid cv ok w
        exact ⟨k,ans,after,returned,closed,admissible,growth,cv,ok,w,valid,growth,admissible,projection⟩)
      12 (SerialStack.first m rows) past sound (first_ready _ window) rfl covers positive
  obtain ⟨snap,best,writes,att⟩ := cert
  obtain ⟨result,published,score⟩ := publishSingle cover pos writes (best ▸ inside) (best ▸ att)
    (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount b.val).toNat nodes fallback
  refine ⟨prior,last,result,ran,path,bound,path.carried,prefix',good,seen,path.returns m pc,published,?_,?_⟩
  · rw [PublicationMir.ExactScore.completed_tagged published,EndgameTreeCache.scaled_discValue]
    exact score
  · have names := (writes.scores (Root.roots_nodup cover)).1
    have perm := (RootMoveStorage.sortRowsContract_proven last.answer.rows 0).1
    have len : last.answer.rows.length = last.state.rows.length := by simpa using congrArg List.length names
    have word : last.call.score.toInt = last.answer.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by have := att.1.lower; omega) (by have := att.1.upper; omega)
    exact ⟨((perm.map Row.square).trans (List.Perm.of_eq names)).trans cover,
      by rwa [perm.length_eq,len],
      word.trans ((att.1.exact inside.1 inside.2).trans (Root.value_zero cover (by simpa using pos)))⟩

/-- A mobile thirteen-empty root places one stone before the first child
    frame. NonPV descendants at twelve or fewer empties use the local solver.
    The probe used to order PV moves need not provide a valid score or hint. -/
theorem shared_thirteen (inputs : Inputs mc fc) (empty : Nat) (small : empty ≤ 13) :
    SerialCacheStack.SharedFacts inputs (empty-1) := by
  constructor <;> intros <;> omega

def ThirteenFromBoard : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (memory : MoveList.Initialization.Memory Move.Move) (m : Machine) sort nodes fallback,
    (Board.emptyCount b.val).toNat ≤ 13 → Board.hasLegalMoves mc b.val = true →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ rows, NativeRootBirth.started mc fc b c.paired memory = some rows ∧
      Published c b inputs roots rows m [] sort nodes fallback

theorem thirteen : ThirteenFromBoard := by
  intro mc fc c b inputs roots memory m sort nodes fallback small mobile pc window
  have count := legalMoveCount_le_empties mc b
  have nonempty : PvsBoardTree.legalSquares mc b.val ≠ [] :=
    fun none => Bool.noConfusion ((PvsBoardTree.stuck_of_nil none).symm.trans mobile)
  have start := (NativeRootBirth.started_iff mc fc b c.paired memory).2
    ⟨by unfold NativeRootBirth.capacity; omega, nonempty, rfl⟩
  obtain ⟨covers,positive⟩ := NativeRootBirth.covers start
  exact ⟨_,start,publication mc fc c b inputs roots _ m [] sort nodes fallback
    (fun i => shared_thirteen (inputs i) _ small) (by simp [soundHistory])
    covers (List.length_pos_iff.mpr positive) pc window⟩

end NeuralReversi.AspirationMir.SerialCache
