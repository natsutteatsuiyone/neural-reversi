import NeuralReversi.AspirationMir.SerialTable.MultiBatch
import NeuralReversi.AspirationMir.SerialTable.MultiClock
import NeuralReversi.AspirationMir.SerialTable.MultiMidgame

namespace NeuralReversi.AspirationMir.SerialTable.MultiOrdinary

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open PvsParallel
open PvsParallel.SerialTable (Memory)
open Repeated (ReadyMemory)
open SelectiveLadder (Start)
open RootMoveStorage (Row sortRows)
open PvsMoveLoop (scoreMax)

/-- Any stored score at least -SCORE_INF gives a nonempty window on the following PV. -/
theorem seed_window (idx : Nat) (base : Word) (rows : List Row)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax)
    (prev : ∀ v ∈ MultiLine.previous idx rows, -32767 ≤ v) :
    Aspiration.Window (MultiLine.seed idx base rows).regs.alpha.toInt (MultiLine.seed idx base rows).regs.beta.toInt := by
  apply SerialEntry.windowMachine_valid
  by_cases first : idx = 0
  · subst idx
    exact RootAspiration.seedWindowsContract_proven.1 0 base.toInt none range.1 range.2 (by simp)
  · unfold RootAspiration.initialWindow
    rw [ite_eq_right first]
    cases h : MultiLine.previous idx rows with
    | none => exact ⟨by decide,by decide,by decide⟩
    | some v =>
      have floor := prev v h
      exact ⟨Int.le_refl _,by change -32767 < min (v+1) 32767; omega,Int.min_le_right ..⟩

private theorem interpreted {code : SerialCalls.Program SerialCalls.Request α}
    (ran : PvsParallel.SerialTable.Stack.interpret (mc := mc) (fc := fc) descend i ctx m code = some (k,v,after)) :
    ∃ cs, SerialCalls.Program.Runs code cs v := by
  induction code generalizing i m with
  | pure => cases ran; exact ⟨[],.pure⟩
  | fail => cases ran
  | call q cont ih =>
    obtain ⟨child,_,rest⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨⟨j,w,next⟩,_,tail⟩ := Option.bind_eq_some_iff.mp rest
    obtain ⟨cs,returned⟩ := ih w tail
    exact ⟨(q,w)::cs,.call returned⟩

private theorem trial_peak (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position)
    (ops : FullLadder.Inputs mc) (fc : Cutoffs.FlipConfig) (level : Nat)
    (s : SerialStack.State) (m : Memory l)
    (ran : MultiLevel.trial idx p c b ops fc level s m = some (k,ans,next))
    (cover : Root.Covers mc b (s.rows.map Row.square))
    (win : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (improved : s.regs.alpha.toInt < ans.value) : MidgameRoot.RowMaximum (ans.rows.drop idx) ans.value := by
  unfold MultiLevel.trial at ran
  split at ran
  all_goals
    obtain ⟨⟨_,out,_⟩,executed,eq⟩ := Option.bind_eq_some_iff.mp ran
    cases eq
    obtain ⟨calls,issued⟩ := interpreted executed
    exact (MidgameRoot.root_suffix_rows issued cover win).2 improved

private theorem last_trial {t : SerialCache.Trial M} (fuel : Nat)
    (ran : SerialCache.run t fuel s m = some (prior,last)) :
    t last.state last.before = some (last.closing,last.answer,last.after) := by
  induction fuel generalizing s m prior with
  | zero => cases ran
  | succ fuel ih =>
    obtain ⟨⟨k,ans,next⟩,executed,widened⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨⟨regs,again⟩,_,done_⟩ := Option.bind_eq_some_iff.mp widened
    cases again with
    | false => cases done_; exact executed
    | true =>
      obtain ⟨⟨_,out⟩,tail,eq⟩ := Option.bind_eq_some_iff.mp done_
      cases eq
      exact ih tail


/-- One level couples its accepted score to the actual suffix row read by the next PV.
    This bound also applies after selective cutoffs. -/
theorem stage (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Nat) (s : Start (Memory l)) (addr : c.address = p.address)
    (ok : MultiLevel.Valid mc b idx (ReadyMemory p mc fc) s) :
    ∃ out, MultiLevel.advance idx (MultiLevel.trial idx p c b ops fc) s level = some out ∧
      MultiLevel.Valid mc b idx (ReadyMemory p mc fc) out ∧ out.rows.length = s.rows.length ∧
      ∀ v ∈ MultiLine.previous (idx+1) out.rows, -32767 ≤ v := by
  obtain ⟨prior,last,ran,acc⟩ := MultiLevel.retries idx p c b ops fc level 12
    (SerialStack.first s.machine s.rows) s.memory addr ok.safe ok.cover ok.index (first_ready _ ok.window) rfl
  obtain ⟨last',step,_,valid,keys,_⟩ := MultiLevel.stage idx p c b ops fc level s addr ok
  have actual : MultiLevel.advance idx (MultiLevel.trial idx p c b ops fc) s level =
      some (MultiLevel.successor idx last) := by simp [MultiLevel.advance,ran]
  rw [actual] at step
  have pos : idx < last.answer.rows.length := by
    have len : last.answer.rows.length = s.rows.length := by simpa [SerialStack.first] using congrArg List.length acc.afterKeys
    exact len ▸ ok.index
  have pk := trial_peak idx p c b ops fc level last.state last.before (last_trial _ ran)
    (acc.keys ▸ ok.cover) acc.window acc.inside.1
  have row := MultiMidgame.selected_score last.answer.rows idx pos pk
    (by rw [(RootMoveStorage.sortRowsContract_proven ..).1.length_eq]; exact pos)
  rw [← Option.some.inj step] at valid keys
  refine ⟨_,actual,valid,by simpa using keys.length_eq,?_⟩
  intro v hv
  change (Option.map Row.score (sortRows last.answer.rows idx)[idx]?) = some v at hv
  rw [List.getElem?_eq_getElem (by rw [(RootMoveStorage.sortRowsContract_proven ..).1.length_eq]; exact pos)] at hv
  have eq := (Option.some.inj hv).symm.trans row
  exact eq ▸ Int.le_trans acc.window.lower (Int.le_of_lt acc.inside.1)

theorem stages (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (levels : List Nat) (s : Start (Memory l)) (addr : c.address = p.address)
    (ok : MultiLevel.Valid mc b idx (ReadyMemory p mc fc) s) (nonempty : levels ≠ []) :
    ∃ out, levels.foldlM (MultiLevel.advance idx (MultiLevel.trial idx p c b ops fc)) s = some out ∧
      MultiLevel.Valid mc b idx (ReadyMemory p mc fc) out ∧ out.rows.length = s.rows.length ∧
      ∀ v ∈ MultiLine.previous (idx+1) out.rows, -32767 ≤ v := by
  induction levels generalizing s with
  | nil => contradiction
  | cons level tail ih =>
    obtain ⟨now,ran,valid,len,bound⟩ := stage idx p c b ops fc level s addr ok
    cases tail with
    | nil => exact ⟨now,by simpa [List.foldlM_cons] using ran,valid,len,bound⟩
    | cons x xs =>
      obtain ⟨out,more,good,size,lower⟩ := ih now valid (by simp)
      exact ⟨out,by simpa [List.foldlM_cons,ran] using more,good,size.trans len,lower⟩

theorem lines_returns (p : Pool) (c : SerialScout.Config) (b : Position) (ops : MultiBatch.Inputs mc)
    (fc : Cutoffs.FlipConfig) (levels : Nat → List Nat) (base : Word) (count idx : Nat)
    (rows : List Row) (m : Memory l) (addr : c.address = p.address) (ready : ReadyMemory p mc fc m)
    (cover : Root.Covers mc b (rows.map Row.square)) (fit : idx+count ≤ rows.length)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax)
    (prev : ∀ v ∈ MultiLine.previous idx rows, -32767 ≤ v)
    (nonempty : ∀ i, idx ≤ i → i < idx+count → levels i ≠ []) :
    ∃ final next, MultiBatch.lines (MultiBatch.trials p c b ops fc) levels base count idx rows m = some (final,next) ∧
      ReadyMemory p mc fc next ∧ Root.Covers mc b (final.map Row.square) ∧ final.length = rows.length ∧
      ∀ v ∈ MultiLine.previous (idx+count) final, -32767 ≤ v := by
  induction count generalizing rows m idx with
  | zero => exact ⟨rows,m,rfl,ready,cover,rfl,prev⟩
  | succ n ih =>
    obtain ⟨out,stageRun,good,len,bounded⟩ := stages idx p c b (ops idx) fc (levels idx)
      ⟨rows,MultiLine.seed idx base rows,m⟩ addr ⟨cover,by simpa using (show idx < rows.length by omega),ready,rfl,
        seed_window idx base rows range prev⟩ (nonempty idx (by omega) (by omega))
    change out.rows.length = rows.length at len
    obtain ⟨final,next,tail,safe,covers,size,floor⟩ := ih (idx+1) out.rows out.memory good.safe good.cover
      (by omega) bounded (fun i lo hi => nonempty i (by omega) (by omega))
    refine ⟨final,next,?_,safe,covers,size.trans len,by rwa [show idx+(n+1) = (idx+1)+n by omega]⟩
    change (MultiLine.run idx base _ _ _ _).bind _ = _
    have line : MultiLine.run idx base (MultiBatch.trials p c b ops fc idx) (levels idx) rows m = some out := stageRun
    rw [line]
    exact tail

theorem clock_nonempty (enabled : 1 ≤ cap) :
    MultiClock.levels cap minimum prediction nanos idx ≠ [] := by
  simp [MultiClock.levels,OuterMir.ladder,OuterMir.lineLevels,OuterMir.step,show ¬cap < 1 by omega]

end NeuralReversi.AspirationMir.SerialTable.MultiOrdinary
