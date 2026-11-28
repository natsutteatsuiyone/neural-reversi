module
public import NeuralReversi.AspirationMir.SerialStack

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialCache

open PvsParallel
open PvsBoardTree (Position)
open SerialStack (State Visit Certified)
open EndgameTreeCache (Store soundHistory)

abbrev History := List Store
abbrev Trial (M : Type := History) := State → M → Option (Nat × SmallRoot.Answer × M)

structure Receipt (M : Type := History) extends Visit where
  before : M
  after : M

/-- The aspiration retry uses the worker cache left by its preceding attempt,
    including a failed attempt which stopped at its first cutoff. -/
def run (trial : Trial M) : Nat → State → M → Option (List (Receipt M) × Receipt M)
  | 0, _, _ => none
  | fuel+1, s, h => do
    let (k,ans,h') ← trial s h
    let v : Receipt M := ⟨⟨s,k,ans⟩,h,h'⟩
    let (regs,again) ← applyWiden s.regs v.call.score
    if again then
      let (earlier,last) ← run trial fuel (SerialStack.next v.toVisit regs) h'
      return (v :: earlier,last)
    else return ([],v)

inductive Path (trial : Trial M) : State → M → List (Receipt M) → Receipt M → Prop
  | accept (returned : trial s h = some (k,ans,h'))
      (stop : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (s.regs,false)) :
      Path trial s h [] ⟨⟨s,k,ans⟩,h,h'⟩
  | retry (returned : trial s h = some (k,ans,h'))
      (wide : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (regs,true))
      (more : Path trial (SerialStack.next ⟨s,k,ans⟩ regs) h' prior last) :
      Path trial s h (⟨⟨s,k,ans⟩,h,h'⟩ :: prior) last

theorem Path.carried {trial : Trial M} (path : Path trial s h prior last) :
    (prior ++ [last]).map Receipt.before = h :: prior.map Receipt.after := by
  induction path with
  | accept => rfl
  | retry _ _ _ ih => exact congrArg (_ :: ·) ih

theorem Path.indices {trial : Trial M} (path : Path trial s h prior last) :
    ((prior ++ [last]).map fun v => v.state.index) = List.range' s.index (prior.length+1) := by
  induction path with
  | accept => simp [List.range'_succ]
  | retry _ _ _ ih =>
    simp only [List.cons_append,List.map_cons,List.length_cons,List.range'_succ]
    exact congrArg (_ :: ·) ih

private def pastFor (fallback : M) (all : List (Receipt M)) (i : Nat) : M :=
  ((all.map fun r => (r.state.index,r.before)).lookup i).getD fallback

private theorem lookup_past {trial : Trial M} (path : Path trial s h prior last) (v : Receipt M) (mem : v ∈ prior ++ [last]) :
    pastFor h (prior ++ [last]) v.state.index = v.before := by
  let all := prior ++ [last]
  have distinct : (all.map fun r => r.state.index).Nodup := path.indices ▸ List.nodup_range'
  have selected := SerialCalls.Program.lookup_mem (calls := all.map fun r => (r.state.index,r.before))
    (by simpa only [List.map_map,Function.comp_def] using distinct)
    (List.mem_map_of_mem (f := fun r : Receipt M => (r.state.index,r.before)) mem)
  change (all.map (fun r => (r.state.index,r.before)) |>.lookup v.state.index).getD h = v.before
  rw [selected]
  rfl

private def replayTrial (t : Trial M) (history : Nat → M) : SerialStack.Trial := fun s =>
  (t s (history s.index)).map fun (k,ans,_) => (k,ans)

private theorem erased {trial : Trial M} (path : Path trial s h prior last) (history : Nat → M)
    (observed : ∀ r ∈ prior ++ [last], history r.state.index = r.before) :
    SerialStack.Trials (replayTrial trial history) s (prior.map Receipt.toVisit) last.toVisit := by
  induction path with
  | @accept s h k ans after found stop =>
    have exactHistory := observed _ List.mem_cons_self
    exact .accept (by simp only [replayTrial,exactHistory,found,Option.map_some]) stop
  | @retry s h k ans after regs prior last found wide tail ih =>
    have exactHistory := observed _ List.mem_cons_self
    exact .retry (by simp only [replayTrial,exactHistory,found,Option.map_some]) wide
      (ih (fun r mem => observed r (.tail _ mem)))

theorem Path.trials {trial : Trial M} (path : Path trial s h prior last) :
    ∃ search, SerialStack.Trials search s (prior.map Receipt.toVisit) last.toVisit :=
  ⟨_,erased path _ (lookup_past path)⟩

theorem Path.returns {trial : Trial M} (m : Machine) (pc : m.pc = 0) (path : Path trial (SerialStack.first m rows) h prior last) :
    Returns (SerialStack.queryFor ((prior ++ [last]).map Receipt.toVisit)) (fun _ => false)
      m last.call.score last.call := by
  simpa only [List.map_append,List.map_cons,List.map_nil] using
    (erased path _ (lookup_past path)).returns m pc

theorem completeState {b : Position} {trial : Trial M} {evidence : Receipt M → Prop}
    {valid : M → Prop} {extendsState : M → M → Prop}
    (compose : ∀ {x y z}, extendsState x y → extendsState y z → extendsState x z)
    (works : ∀ s h, valid h → Root.Covers mc b (s.rows.map RootMoveStorage.Row.square) →
      0 < s.rows.length → Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt →
      ∃ k ans after, trial s h = some (k,ans,after) ∧ Certified mc fc b s ans ∧
        valid after ∧ extendsState h after ∧ evidence ⟨⟨s,k,ans⟩,h,after⟩)
    (fuel : Nat) (s : State) (h : M) (good : valid h) (ready : Ready n s.regs)
    (budget : n+fuel = 12) (cover : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square))
    (positive : 0 < s.rows.length) :
    ∃ prior last, run trial fuel s h = some (prior,last) ∧ Path trial s h prior last ∧ prior.length < fuel ∧
      Root.Covers mc b (last.state.rows.map RootMoveStorage.Row.square) ∧ 0 < last.state.rows.length ∧
      Certified mc fc b last.state last.answer ∧
      (last.state.regs.alpha.toInt < last.answer.value ∧ last.answer.value < last.state.regs.beta.toInt) ∧
      extendsState h last.after ∧ valid last.after ∧ ∀ r ∈ prior ++ [last], evidence r := by
  induction fuel generalizing n s h with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    obtain ⟨k,ans,after,ret,cert,safe,extended,ev⟩ := works s h good cover positive ready.window
    obtain ⟨snap,best,writes,att⟩ := cert
    let v : Receipt M := ⟨⟨s,k,ans⟩,h,after⟩
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by have := att.1.lower; omega) (by have := att.1.upper; omega)
    have finite := show -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ _ from
      signed ▸ And.intro att.1.lower att.1.upper
    rcases ready_widen ready (BitVec.ofInt 32 ans.value) finite with stop | ⟨regs,wide,fits⟩
    · exact ⟨[],v,by simp [run,ret,SerialStack.Visit.call,stop,v],.accept ret stop,Nat.zero_lt_succ _,
        cover,positive,⟨snap,best,writes,att⟩,by simpa only [v,signed] using widen_false stop,
        extended,safe,by simpa [v]⟩
    · have same := (writes.scores (Root.roots_nodup cover)).1
      have length : ans.rows.length = s.rows.length := by simpa using congrArg List.length same
      obtain ⟨prior,last,rest,path,bound,covers,pos,proof,inside,prefix',sound,more⟩ :=
        ih (SerialStack.next v.toVisit regs) after safe fits (by omega) (same ▸ cover) (length ▸ positive)
      refine ⟨v::prior,last,?_,.retry ret wide path,by simpa using bound,covers,pos,proof,inside,
        compose extended prefix',sound,?_⟩
      · dsimp only [v] at rest ⊢
        simp [run,ret,SerialStack.Visit.call,wide,rest]
      · intro r mem
        rcases List.mem_cons.mp mem with rfl | tail
        · exact ev
        · exact more r tail

theorem complete {b : Position} {trial : Trial} {evidence : Receipt → Prop}
    (works : ∀ s h, soundHistory mc fc h → Root.Covers mc b (s.rows.map RootMoveStorage.Row.square) →
      0 < s.rows.length → Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt →
      ∃ k ans after, trial s h = some (k,ans,after) ∧ Certified mc fc b s ans ∧
        soundHistory mc fc after ∧ h <+: after ∧ evidence ⟨⟨s,k,ans⟩,h,after⟩)
    (fuel : Nat) (s : State) (h : History) (good : soundHistory mc fc h) (ready : Ready n s.regs)
    (budget : n+fuel = 12) (cover : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square))
    (positive : 0 < s.rows.length) :
    ∃ prior last, run trial fuel s h = some (prior,last) ∧ Path trial s h prior last ∧ prior.length < fuel ∧
      Root.Covers mc b (last.state.rows.map RootMoveStorage.Row.square) ∧ 0 < last.state.rows.length ∧
      Certified mc fc b last.state last.answer ∧
      (last.state.regs.alpha.toInt < last.answer.value ∧ last.answer.value < last.state.regs.beta.toInt) ∧
      h <+: last.after ∧ soundHistory mc fc last.after ∧ ∀ r ∈ prior ++ [last], evidence r :=
  completeState (fun h₁ h₂ => h₁.trans h₂) works fuel s h good ready budget cover positive

end NeuralReversi.AspirationMir.SerialCache
