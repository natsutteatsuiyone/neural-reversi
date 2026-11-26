import NeuralReversi.AspirationMir.SerialContext

namespace NeuralReversi.AspirationMir.SerialFive

open PvsParallel
open PvsBoardTree (Position)
open SerialCalls (Program Executes Tree Request)
open SerialExecution (Inputs Frame)

/-- Complete the operand schedule on unreachable frames. Every PV probe and every
    word read on an actual path from this root is retained. -/
private def extend (op : Inputs mc fc) : Inputs mc fc := fun i f =>
  let o := op i f
  if f.request.pv || decide (f.request.depth.toNat ≤ 4) then o
  else { o with cache := [], table := [], probe := [], etc := [] }

private theorem reads (op : Inputs mc fc) :
    SerialExecution.ReadFacts (extend op) ∧ SerialTermination.Hints (extend op) := by
  constructor
  · constructor
    · intro i f role lo _
      simp [extend,role,show ¬ f.request.depth.toNat ≤ 4 by omega,EndgameTreeCache.soundHistory]
    · intro i f role lo
      simp [extend,role,show ¬ f.request.depth.toNat ≤ 4 by omega,TableReads.ProbeReads,TableReads.ProbeReadsFrom]
    · intro i f role lo
      simpa only [extend,role,decide_eq_false (show ¬ f.request.depth.toNat ≤ 4 by omega),Bool.false_or,
        Bool.false_eq_true,ite_false] using (ExactTableHistory.Publications.initial (mc := mc) (fc := fc))
  · intro i f role lo
    simp [extend,role,show ¬ f.request.depth.toNat ≤ 4 by omega,TableReads.LegalHints]

private def Below (f : Frame) (mc : Bitboard.Moves.X86Config) : Prop :=
  ExactDepth.Full f.request.board f.request.depth ∧
    (f.request.depth.toNat ≤ 4 ∨ f.request.depth.toNat ≤ 5 ∧ Board.hasLegalMoves mc f.request.board.val = true)

private theorem child_small {f q child} (start : Below f mc) (entered : SerialExecution.push mc fc f q = some child) :
    Below child mc ∧ child.request.depth.toNat ≤ 4 := by
  obtain ⟨_,hit,rfl⟩ := Option.map_eq_some_iff.mp entered
  have edge := SerialChildState.edge_of_nextState hit
  have small : q.depth.toNat ≤ 4 := by
    have ⟨full,limit⟩ := start
    generalize hb : f.request.board = b, hd : f.request.depth = d, hq : q.board = qb, hcd : q.depth = cd at edge
    rw [hb,hd] at full limit
    cases edge with
    | ply m d =>
      have positive := (ExactDepth.move_depth mc fc _ _ m full).1
      have sub : (d-1).toNat = d.toNat-1 := BitVec.toNat_sub_of_le (by rw [BitVec.le_def]; exact positive)
      rw [sub]
      omega
    | forcedPass _ forced =>
      have stuck := forced.1
      rcases limit with h | ⟨_,mobile⟩
      · exact h
      · simp_all
  exact ⟨⟨edge.full start.1,.inl small⟩,small⟩

private theorem transfer (c : SerialScout.Config) (op : Inputs mc fc)
    (ran : Executes (SerialExecution.body c (extend op)) (SerialExecution.push mc fc) i f code last word forest)
    (start : Below f mc) :
    Executes (SerialExecution.body c op) (SerialExecution.push mc fc) i f code last word forest := by
  induction ran with
  | pure => exact .pure
  | @call f r child i middle w inside resume last v rest enter _ _ down more =>
    obtain ⟨childStart,small⟩ := child_small start enter
    have fields : extend op (i+1) child = op (i+1) child := by simp [extend,small]
    have same : SerialExecution.body c (extend op) (i+1) child = SerialExecution.body c op (i+1) child := by
      unfold SerialExecution.body
      rw [fields]
    exact .call enter (same ▸ down childStart) (more start)

private theorem trial (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (v : SerialStack.Visit)
    (small : (Board.emptyCount b.val).toNat ≤ 5) (mobile : Board.hasLegalMoves mc b.val = true)
    (record : SerialStack.Projected c b (fun i => extend (inputs i)) roots v) :
    SerialStack.trial c b inputs roots v.state = some (v.closing,v.answer) := by
  obtain ⟨forest,seen⟩ := record
  let s := v.state
  let a := s.regs.alpha
  let z := s.regs.beta
  let ctx := SerialRootExecution.context b a z
  have full : ExactDepth.Full b ctx.request.depth := ContextPathMir.At.initial.full (mc := mc) (fc := fc)
  have start : Below ctx mc := ⟨full,.inr ⟨full.symm ▸ small,mobile⟩⟩
  have same : SerialRootExecution.attempt c (extend (inputs s.index)) s.opening b (roots s.index a z) s.rows 0 a z =
      SerialRootExecution.attempt c (inputs s.index) s.opening b (roots s.index a z) s.rows 0 a z := by
    simp only [SerialRootExecution.attempt,extend,SerialRootExecution.context,Bool.true_or,ite_true]
  have ran := transfer c (inputs s.index) seen.1 start
  rw [same] at ran
  obtain ⟨ans,actual,_,issued⟩ := ran.outer_returns
  have recorded := seen.2.2.1
  rw [same] at recorded
  exact SerialStack.same_result issued recorded ▸ actual

private theorem drive (search : SerialStack.Trial) (chain : SerialStack.Trials other s prior last)
    (matchesRun : ∀ v ∈ prior ++ [last], search v.state = some (v.closing,v.answer))
    (fuel : Nat) (fits : prior.length < fuel) : SerialStack.run search fuel s = some (prior,last) := by
  induction chain generalizing fuel with
  | accept _ stopped =>
    cases fuel with
    | zero => contradiction
    | succ _ => simp [SerialStack.run,matchesRun _ List.mem_cons_self,SerialStack.Visit.call,stopped]
  | retry found widened tail ih =>
    cases fuel with
    | zero => contradiction
    | succ n =>
      have inner := ih (fun v m => matchesRun v (.tail _ m)) n (by simpa using fits)
      simp [SerialStack.run,matchesRun _ List.mem_cons_self,SerialStack.Visit.call,widened,inner]

def Contract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (rows : List RootMoveStorage.Row) (m : Machine)
    (inputs : Nat → Inputs mc fc) (roots : Nat → Word → Word → SerialRecursion.RootInput)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    (Board.emptyCount b.val).toNat ≤ 5 → Board.hasLegalMoves mc b.val = true →
    Root.Covers mc b (rows.map RootMoveStorage.Row.square) → 0 < rows.length →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ prior last result, SerialStack.run (SerialStack.trial c b inputs roots) 12 (SerialStack.first m rows) =
      some (prior,last) ∧ prior.length < 12 ∧
      PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
        (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem publication : Contract := by
  intro mc fc c b rows m inputs roots sort nodes fallback small mobile covers within pc window
  obtain ⟨prior,last,res,_,chain,limit,seen,_,_,_,published,word⟩ := SerialStack.publicScore mc fc c b rows m
    (fun i => extend (inputs i)) roots sort nodes fallback covers within (fun i => (reads _).1)
    (fun i => (reads _).2) pc window
  exact ⟨prior,last,res,drive _ chain (fun v hv => trial c b inputs roots v small mobile (seen v hv)) 12 limit,
    limit,published,word⟩

def FromBoard : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (m : Machine) (memory : MoveList.Initialization.Memory Move.Move)
    (inputs : Nat → Inputs mc fc) (roots : Nat → Word → Word → SerialRecursion.RootInput)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    (Board.emptyCount b.val).toNat ≤ 5 → Board.hasLegalMoves mc b.val = true →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ rows prior last result,
      NativeRootBirth.started mc fc b c.paired memory = some rows ∧
      SerialStack.run (SerialStack.trial c b inputs roots) 12 (SerialStack.first m rows) = some (prior,last) ∧
      prior.length < 12 ∧ PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
        (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem from_board : FromBoard := by
  intro mc fc c b m memory inputs roots sort nodes fallback small mobile pc window
  have count := legalMoveCount_le_empties mc b
  have nonempty : PvsBoardTree.legalSquares mc b.val ≠ [] :=
    fun none => Bool.noConfusion ((PvsBoardTree.stuck_of_nil none).symm.trans mobile)
  have start := (NativeRootBirth.started_iff mc fc b c.paired memory).2
    ⟨by unfold NativeRootBirth.capacity; omega, nonempty, rfl⟩
  have ⟨covers, positive⟩ := NativeRootBirth.covers start
  obtain ⟨prior,last,result,ran,fit,published,correct⟩ :=
    publication mc fc c b _ m inputs roots sort nodes fallback small mobile covers
      (List.length_pos_iff.mpr positive) pc window
  exact ⟨_,prior,last,result,start,ran,fit,published,correct⟩

end NeuralReversi.AspirationMir.SerialFive
