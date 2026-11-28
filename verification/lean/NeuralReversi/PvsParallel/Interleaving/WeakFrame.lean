module
public import NeuralReversi.PvsParallel.Interleaving.BoardFrame
public import NeuralReversi.PvsParallel.Interleaving.RootFinite
public import NeuralReversi.PvsParallel.SerialTable.SelectiveFrame

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving

open Threading
open SerialCalls.ScoreWords (Finite)
open PvsMoveLoop (scoreInf)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open Aspiration (Window)
open RootSchedule (scan_finite locked_finite reported_finite)

/-- If the owner hands off before completing its first move, either a helper
    publishes a cutoff or retirement accounts for every remaining ticket. -/
theorem Turn.finite (turn : Turn done_ moves pv original beta score) (nonempty : moves ≠ [])
    (children : ∀ m cp a z v, done_ m cp a z v → Finite v) : Finite score := by
  cases turn with
  | returned reply exec =>
    obtain ⟨perm,_,same,calls⟩ := exec.reproduced
    simp only [PvsPostpone.Queue.pending,List.append_nil] at perm
    exact same ▸ scan_finite calls children (by change (-32767:Int) ≤ -32767; decide) (by change (-32767:Int) < 32767; decide) (.inl (fun empty => nonempty (empty ▸ perm).nil_eq.symm))
  | @joined visited remaining entry _ reply fallback root cut depth width exec fit j return_ =>
    obtain ⟨perm,idx,continued,calls⟩ := exec.reproduced
    simp only [PvsPostpone.Queue.pending,List.append_nil] at perm
    obtain ⟨retired,normal,result⟩ := j.finished return_
    have fs := reported_finite j.path (fun i => children _)
    have scored := Prefix.continued_scan pv reply beta visited _ _ entry continued
    by_cases any : visited ≠ []
    · have f := scored ▸ scan_finite calls children (by change (-32767:Int) ≤ -32767; decide) (by change (-32767:Int) < 32767; decide) (.inl any)
      exact result ▸ locked_finite pv beta entry _ f fs
    · have nil := Classical.not_not.mp any
      simp only [nil,Prefix.run,Option.some.injEq] at continued
      have lenPos : 0 < (visited ++ remaining.pending).length := List.length_pos_iff.mpr
        (fun eq => nonempty (eq ▸ perm).nil_eq.symm)
      have zero : remaining.idx = 0 := by simpa [nil] using idx
      have posted : j.state.reports ≠ [] := by
        intro empty
        rcases j.path.tickets.outcome fit retired with stop | ⟨p,mem,_⟩ | covers
        · simp_all
        · simp_all
        · have n := covers.length_eq
          simp [Tickets.State.indices,empty,zero,BitVec.toNat_ofNat] at n
          rw [List.length_append] at lenPos
          omega
      obtain ⟨p,ps,posts⟩ := List.exists_cons_of_ne_nil posted
      have finite := fs p (by simp [posts])
      have entryBest : entry.best = -scoreInf := congrArg SplitWindow.State.best continued.symm
      have best : Finite (SplitWindow.publish pv beta entry p.report.score).best := by
        change Finite (max entry.best p.report.score)
        rw [entryBest,Int.max_eq_right (by have := finite.1; change -32767 ≤ _; omega)]
        exact finite
      rw [←result]
      change Finite (SplitReturn.stateAfter pv beta entry (j.state.reports.map Tickets.Post.report)).best
      rw [posts]
      exact locked_finite pv beta _ ps best (fun q mem => fs q (posts ▸ List.mem_cons_of_mem p mem))

namespace WeakFrame

open CausalTable (Values Lookups)
open SerialTable.SelectiveFrame (Estimates probCut probCut_finite)

/-- Selective node returns. A local terminal, stability or shallow solver uses the
    same primitive as the exact node. An ordinary move scan can itself split. -/
inductive Frame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (prior : Values) (reads : Lookups)
    (b : Position) : Bool → Int → Int → Int → Prop where
  | localReturn (win : Window a z) (narrow : pv = false → z = a+1)
      (leaf : BoardFrame.Frame mc fc (fun _ _ _ _ _ => False) (fun _ _ => False) b pv a z v) :
      Frame mc fc prior reads b pv a z v
  | leaf (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker opening : Nat)
      (i : EndgameSearch.CacheHistory.Input) (board : i.board = b) (count : i.depth = i.empties.count)
      (size : i.depth.toNat ≤ 12) (window : Window i.alpha.raw.toInt i.beta.raw.toInt)
      (null : i.beta.raw.toInt = i.alpha.raw.toInt+1)
      (seen : ∀ slot e, TranspositionTable.ProbeRetry.probe (i.attempts.map (TranspositionTable.ProbeRetry.observe (Cutoffs.key b))) =
        .hit slot e → reads b e)
      (run : (i.run mc fc p.address (timeline.readAt mc fc worker opening).history).score = some v) :
      Frame mc fc prior reads b false i.alpha.raw.toInt i.beta.raw.toInt v
  | pass (stuck : legalSquares mc b.val = []) (mobile : Board.hasLegalMoves mc (pass b).val = true)
      (child : prior (pass b) pv (-z) (-a) v) : Frame mc fc prior reads b pv a z (-v)
  | table (beta : TranspositionTable.Score) (seen : reads b e)
      (cut : TranspositionTable.canCut checked e beta d level true = .ok true) :
      Frame mc fc prior reads b false a beta.raw.toInt e.score.toInt
  | enhanced (q : PvsEnhancedCutoff.Query) (probes : List PvsEnhancedCutoff.Probe) (hit : PvsEnhancedCutoff.Hit)
      (seen : ∀ p ∈ probes, ∀ e, p.2 = some e → ∃ m : GeneratedSquare mc b.val,
        p.1 = m.val ∧ reads (play mc fc b m).val e)
      (ran : PvsEnhancedCutoff.scan q probes = some hit) :
      Frame mc fc prior reads b false q.alpha.raw.toInt (q.alpha.raw.toInt+1) hit.2
  | probability (beta : ScoreRefresh.Score) (scores : Estimates)
      (win : Window a beta.raw.toInt) (ran : probCut d level cut beta scores = some v) :
      Frame mc fc prior reads b false a beta.raw.toInt v
  | moves (first : GeneratedSquare mc b.val) (rest order : List (GeneratedSquare mc b.val))
      (generated : legalSquares mc b.val = first :: rest) (perm : order.Perm (first :: rest))
      (ran : Turn (BoardFrame.Children mc fc b prior) order pv a z v) : Frame mc fc prior reads b pv a z v

theorem preserves (prior : Values) (reads : Lookups) (children : ∀ b p a z v, prior b p a z v → Finite v)
    (words : ∀ b e, reads b e → Finite e.score.toInt) (b : Position) (p : Bool) (a z v : Int)
    (ran : Frame mc fc prior reads b p a z v) : Finite v := by
  cases ran with
  | localReturn win narrow leaf =>
    have sound := BoardFrame.preserves _ _ (fun _ _ _ _ _ _ _ h => h.elim) (fun _ _ h => h.elim)
      _ _ _ _ _ win narrow leaf
    exact ⟨sound.lower,sound.upper⟩
  | leaf p timeline worker opening i board count _ window null seen run =>
    subst b
    let hist := (timeline.readAt mc fc worker opening).history
    have cache := (timeline.reads mc fc worker opening).1
    by_cases small : i.empties.count.toNat ≤ 6
    · have aligned := i.aligned
      obtain ⟨s,ran,cert⟩ := NonPvMir.run_returns i.target i.bmi2 mc fc p.address hist i.board _ _ _
        i.alpha i.beta i.checked (count.trans aligned.1) aligned small (fun _ => cache) null
      rw [EndgameSearch.CacheHistory.shallow_score i count small,NonPvMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ aligned (fun _ => cache),ran] at run
      cases run
      have answer := bound_sound _ window (cert.bound i.beta null)
      exact ⟨answer.lower,answer.upper⟩
    · have dispatch : NonPvMir.dispatch i.empties.count.toNat = none := by
        obtain ⟨n,eq⟩ : ∃ n, i.empties.count.toNat = n+7 := ⟨i.empties.count.toNat-7,by omega⟩
        rw [eq]; rfl
      unfold EndgameSearch.CacheHistory.Input.run at run
      cases hit : i.lookup with
      | none => simp [EndgameSearch.CacheHistory.Input.arm,hit,EndgameSearch.CacheHistory.Input.eval] at run
      | some entry =>
        cases entry with
        | some e =>
          simp [EndgameSearch.CacheHistory.Input.arm,hit,EndgameSearch.CacheHistory.Input.eval] at run
          unfold EndgameSearch.CacheHistory.Input.lookup at hit
          split at hit
          · cases read : TranspositionTable.ProbeRetry.probe (i.attempts.map (TranspositionTable.ProbeRetry.observe (Cutoffs.key i.board))) with
            | miss slot => simp [read] at hit
            | hit slot data =>
              cases accepted : TranspositionTable.canCut i.checked data ⟨i.beta.raw⟩ i.depth i.selectivity true with
              | error err => simp [read,accepted,Except.toOption] at hit
              | ok yes =>
                cases yes <;> simp [read,accepted,Except.toOption] at hit
                exact run ▸ hit ▸ words _ _ (seen slot data read)
          · cases hit
        | none =>
          rw [show i.arm = .deep i.active by simp [EndgameSearch.CacheHistory.Input.arm,hit,dispatch]] at run
          have valid := bound_sound _ window (DeepLeafMir.local_bound i.target i.bmi2 i.paired mc fc p.address
            i.board i.alpha i.beta (i.state hist) i.active i.checked i.aborted i.aligned cache null run)
          exact ⟨valid.lower,valid.upper⟩
  | pass _ _ used =>
    obtain ⟨lo,hi⟩ := children _ _ _ _ _ used
    exact ⟨by omega,by omega⟩
  | table _ used _ => exact words _ _ used
  | enhanced q probes hit seen ran =>
    obtain ⟨before,row,after,layout,⟨e,found,_,_,_,eq⟩,_⟩ := ((PvsEnhancedCutoff.contract_proven _ _).2.1 hit).mp ran
    obtain ⟨m,_,used⟩ := seen row (by simp [layout]) e found
    obtain ⟨lo,hi⟩ := words _ _ used
    rw [←eq]
    exact ⟨by dsimp; omega,by dsimp; omega⟩
  | probability _ _ win hit => exact (probCut_finite win hit).2
  | moves first rest order _ perm path =>
    exact path.finite (fun e => by simp [e] at perm) (fun m => children _)

end WeakFrame

end NeuralReversi.PvsParallel.Interleaving
