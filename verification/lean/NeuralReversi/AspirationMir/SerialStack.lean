import NeuralReversi.PvsParallel.SerialNativeRows
import NeuralReversi.PvsParallel.SerialCalls.Driver
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.AspirationMir.SerialStack

open PvsParallel
open RootMoveStorage (Row)
open PvsBoardTree (Position)
open SmallRoot (Answer)

structure State where
  index : Nat
  opening : Nat
  regs : WindowRegs
  rows : List Row

structure Visit where
  state : State
  closing : Nat
  answer : Answer

def Visit.call (v : Visit) : Call :=
  ⟨v.state.index,v.state.regs.alpha,v.state.regs.beta,BitVec.ofInt 32 v.answer.value⟩

def first (m : Machine) (rows : List Row) : State :=
  ⟨m.issued,0,{m.regs with delta := BitVec.ofInt 32 initialDelta},rows⟩

def next (v : Visit) (after : WindowRegs) : State :=
  ⟨v.state.index+1,v.closing+1,doubled after,v.answer.rows⟩

abbrev Trial := State → Option (Nat × Answer)

def trial (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (s : State) : Option (Nat × Answer) :=
  let a := s.regs.alpha
  let z := s.regs.beta
  let op := inputs s.index
  let frame := SerialRootExecution.attempt c op s.opening b (roots s.index a z) s.rows 0 a z
  SerialCalls.outer (SerialExecution.body c op) (SerialExecution.push mc fc) s.opening
    (SerialRootExecution.context b a z) frame

theorem same_result {p : SerialCalls.Program Req α} (one : p.Runs children a)
    (two : p.Runs children z) : a = z := by
  induction one generalizing z with
  | pure => cases two; rfl
  | call _ ih => cases two with | call rest => exact ih rest

def Certified (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (s : State) (ans : Answer) : Prop :=
  ∃ snap : RootReplay.Snapshot, ans.value = snap.state.best ∧
    RootSuffix.Writes s.rows 0 s.regs.alpha.toInt s.regs.beta.toInt ans.rows snap ∧
    RootOrder.SoundAttempt mc fc b (s.rows.map Row.square) 0 s.regs.alpha.toInt s.regs.beta.toInt ans.value snap.scores

def OnForest (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (v : Visit)
    (forest : List (SerialCalls.Tree SerialCalls.Request)) : Prop :=
  let s := v.state
  let op := inputs s.index
  let a := s.regs.alpha
  let z := s.regs.beta
  let frame := SerialRootExecution.attempt c op s.opening b (roots s.index a z) s.rows 0 a z
  SerialCalls.Executes (SerialExecution.body c op) (SerialExecution.push mc fc) s.opening
      (SerialRootExecution.context b a z) (frame.map Answer.value) v.closing v.answer.value forest ∧
    SerialCalls.openingNumbers forest = List.range' (s.opening+1) (v.closing-s.opening) ∧
    frame.Runs (forest.map SerialCalls.Tree.direct) v.answer ∧
    ∃ scored decoded rootRow, let clock := 3*s.opening
      SerialNativeRows.Rows 1 0 (NativeFrames.frameKey 1 0 clock) (clock+1) forest scored decoded ∧
      SerialNativeRows.Recorded (SerialRootExecution.context b a z).request v.answer.value (some 0)
        1 0 clock 0 scored (scored+1) rootRow ∧ scored+2 = 3*(v.closing+1)

def Projected (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (v : Visit) : Prop :=
  ∃ forest, OnForest c b inputs roots v forest

theorem trial_returns (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (s : State)
    (valid : SerialExecution.ReadFacts (inputs s.index)) (hints : SerialTermination.Hints (inputs s.index))
    (covers : Root.Covers mc b (s.rows.map Row.square)) (within : 0 < s.rows.length)
    (w : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans, trial c b inputs roots s = some (k,ans) ∧ Certified mc fc b s ans ∧
      Projected c b inputs roots ⟨s,k,ans⟩ := by
  let a := s.regs.alpha
  let z := s.regs.beta
  obtain ⟨count,k,value,stepped⟩ := SerialTermination.root_terminates c (inputs s.index) valid hints s.opening
    b (roots s.index a z) s.rows 0 a z covers within w
  obtain ⟨forest,ans,snap,exec,issued,val,best,writes,_,sound⟩ :=
    SerialRootExecution.returns c (inputs s.index) valid s.opening b (roots s.index a z) s.rows 0 a z covers within w stepped
  obtain ⟨other,actual,_,again⟩ := exec.outer_returns
  cases same_result issued again
  let clock := 3*s.opening
  obtain ⟨scored,decoded,rootRow,children,recorded⟩ := SerialNativeRows.root_rows c (inputs s.index) valid hints b
    (roots s.index a z) s.rows 0 a z covers within w exec 1 0 0 clock
  have countRows := children.count
  rw [exec.openings.1, List.length_range'] at countRows
  have elapsed := children.elapsed
  have earlier := exec.openings.2
  refine ⟨k,ans,actual,⟨snap,val.trans best,writes,val ▸ sound⟩,forest,val ▸ exec,exec.openings.1,issued,
    scored,decoded,rootRow,children,val ▸ recorded,?_⟩
  dsimp only [clock] at elapsed
  change scored+2 = 3*(k+1)
  omega

/-- Each retry keeps the root rows written by the previous attempt. The first
    opening of the next trial follows the last nested activation of its predecessor. -/
def run (search : Trial) : Nat → State → Option (List Visit × Visit)
  | 0, _ => none
  | fuel+1, s => do
    let (k,ans) ← search s
    let v : Visit := ⟨s,k,ans⟩
    let (after,again) ← applyWiden s.regs v.call.score
    if again then
      let (more,last) ← run search fuel (next v after)
      return (v :: more,last)
    else return ([],v)

inductive Trials (search : Trial) : State → List Visit → Visit → Prop
  | accept (returned : search s = some (k,ans))
      (stop : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (s.regs,false)) :
      Trials search s [] ⟨s,k,ans⟩
  | retry (returned : search s = some (k,ans))
      (step : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (after,true))
      (more : Trials search (next ⟨s,k,ans⟩ after) prior last) :
      Trials search s (⟨s,k,ans⟩ :: prior) last

private theorem complete {b : Position} {search : Trial} {evidence : Visit → Prop}
    (works : ∀ s, Root.Covers mc b (s.rows.map Row.square) → 0 < s.rows.length →
      Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt →
      ∃ k ans, search s = some (k,ans) ∧ Certified mc fc b s ans ∧ evidence ⟨s,k,ans⟩)
    (fuel : Nat) (s : State) (ready : Ready n s.regs) (left : n+fuel=12)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length) :
    ∃ prior last, run search fuel s = some (prior,last) ∧ Trials search s prior last ∧
      prior.length < fuel ∧ Root.Covers mc b (last.state.rows.map Row.square) ∧ 0 < last.state.rows.length ∧
      Certified mc fc b last.state last.answer ∧
      (last.state.regs.alpha.toInt < last.answer.value ∧ last.answer.value < last.state.regs.beta.toInt) ∧
      ∀ v ∈ prior ++ [last], evidence v := by
  induction fuel generalizing s n with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    obtain ⟨k,ans,returned,proof,issued⟩ := works s cover positive ready.window
    obtain ⟨snap,best,writes,sound⟩ := proof
    let v : Visit := ⟨s,k,ans⟩
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by have := sound.1.lower; omega) (by have := sound.1.upper; omega)
    have bounded := show -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ _ from
      signed ▸ And.intro sound.1.lower sound.1.upper
    rcases ready_widen ready (BitVec.ofInt 32 ans.value) bounded with stop | ⟨after,step,valid⟩
    · exact ⟨[],v,by simp [run,returned,Visit.call,stop,v], .accept returned stop, Nat.zero_lt_succ _,
        cover,positive,⟨snap,best,writes,sound⟩, by simpa [v,signed] using widen_false stop, by simpa [v]⟩
    · have squares := (writes.scores (Root.roots_nodup cover)).1
      have size : ans.rows.length = s.rows.length := by simpa using congrArg List.length squares
      obtain ⟨prior,last,rest,chain,bound,lastCover,lastPos,certified,inside,more⟩ := ih (next v after) valid (by omega)
        (squares ▸ cover) (size ▸ positive)
      refine ⟨v::prior,last,?_,.retry returned step chain,by simpa using bound,lastCover,lastPos,certified,inside,?_⟩
      · dsimp only [v] at rest ⊢
        simp [run,returned,Visit.call,step,rest]
      · intro t member
        rcases List.mem_cons.mp member with rfl | later
        · exact issued
        · exact more t later

theorem Trials.indices (chain : Trials search s prior last) :
    ((prior ++ [last]).map (fun v => v.state.index)) = List.range' s.index (prior.length+1) := by
  induction chain with
  | accept => simp [List.range'_succ]
  | retry _ _ _ ih =>
    simp only [List.cons_append, List.map_cons, List.length_cons, List.range'_succ]
    exact congrArg (_ :: ·) ih

/-- Every stored row field and the counter for the next trial are carried from
    the preceding completion, including retries which cut off part of the root list. -/
theorem Trials.carried (chain : Trials search s prior last) :
    (prior ++ [last]).map (fun v => v.state.rows) = s.rows :: prior.map (fun v => v.answer.rows) ∧
    (prior ++ [last]).map (fun v => v.state.opening) = s.opening :: prior.map (fun v => v.closing+1) := by
  induction chain with
  | accept => exact ⟨rfl,rfl⟩
  | retry _ _ _ ih => exact ⟨congrArg (_ :: ·) ih.1,congrArg (_ :: ·) ih.2⟩

def Visit.openings (v : Visit) := List.range' v.state.opening (v.closing+1 - v.state.opening)

theorem Trials.openings (chain : Trials search s prior last)
    (ordered : ∀ v ∈ prior ++ [last], v.state.opening ≤ v.closing) :
    (prior ++ [last]).flatMap Visit.openings = List.range' s.opening (last.closing+1 - s.opening) ∧
      s.opening ≤ last.closing := by
  induction chain with
  | accept => exact ⟨List.flatMap_singleton .., ordered _ List.mem_cons_self⟩
  | @retry s k ans after prior last found step more ih =>
    have reached : s.opening ≤ k := ordered _ List.mem_cons_self
    obtain ⟨tail, later⟩ := ih (fun t mem => ordered t (.tail _ mem))
    change k+1 ≤ last.closing at later
    refine ⟨?_,by omega⟩
    simp only [List.cons_append,List.flatMap_cons,tail,Visit.openings,next]
    have offset : k+1 = s.opening + 1*(k+1-s.opening) := by omega
    have joined := List.range'_append (s := s.opening) (m := k+1-s.opening) (n := last.closing+1-(k+1)) (step := 1)
    rw [← offset, show (k+1-s.opening)+(last.closing+1-(k+1)) = last.closing+1-s.opening by omega] at joined
    exact joined

private def key (v : Visit) := (v.state.index, v.state.regs.alpha, v.state.regs.beta)

/-- The MIR query reads a score from the actual activation at the requested
    trial index and window. Its default is unreachable on the proved driver run. -/
def queryFor (visits : List Visit) : Query := fun i a z =>
  BitVec.ofInt 32 (((visits.map (fun v => (key v,v.answer.value))).lookup (i,a,z)).getD 0)

private theorem observed (chain : Trials search s prior last) (v : Visit) (mem : v ∈ prior ++ [last]) :
    v.call.Observed (queryFor (prior ++ [last])) := by
  let visits := prior ++ [last]
  have unique : (visits.map fun v => v.state.index).Nodup := chain.indices ▸ List.nodup_range'
  have keyed : ((visits.map key).map Prod.fst).Nodup := by
    simpa only [List.map_map, Function.comp_def, key] using unique
  have un : (visits.map key).Nodup := keyed.of_map Prod.fst (fun _ _ ne eq => ne (congrArg Prod.fst eq))
  have actual := SerialCalls.Program.lookup_mem (calls := visits.map (fun v => (key v,v.answer.value)))
    (by simpa only [List.map_map, Function.comp_def] using un)
    (List.mem_map_of_mem (f := fun v => (key v,v.answer.value)) mem)
  change BitVec.ofInt 32 (((visits.map _).lookup (key v)).getD 0) = _
  rw [actual]
  rfl

private theorem steps (chain : Trials search s prior last) (query : Query)
    (seen : ∀ v ∈ prior ++ [last], v.call.Observed query) (m : Machine)
    (pc : m.pc = 2) (idx : m.issued = s.index) (regs : m.regs = s.regs) :
    Returns query (fun _ => false) m last.call.score last.call := by
  induction chain generalizing m with
  | @accept s k ans found stop =>
    let v : Visit := ⟨s,k,ans⟩
    have value : query s.index s.regs.alpha s.regs.beta = v.call.score := seen v (by simp [v])
    have current : currentCall query m = v.call := by simp [currentCall, idx, regs, value, v, Visit.call]
    rw [← current]
    exact AspirationMir.accept pc (by rw [current,regs]; exact stop)
  | @retry s k ans after prior last found step rest ih =>
    let v : Visit := ⟨s,k,ans⟩
    have value : query s.index s.regs.alpha s.regs.beta = v.call.score := seen v List.mem_cons_self
    have current : currentCall query m = v.call := by simp [currentCall, idx, regs, value, v, Visit.call]
    apply AspirationMir.retry pc (show applyWiden m.regs (currentCall query m).score = some (after,true) by
      rw [current,regs]; exact step)
    exact ih (fun t h => seen t (.tail _ h)) (nextTrial query m after) rfl (congrArg (·+1) idx) rfl

theorem Trials.returns (m : Machine) (pc : m.pc = 0) (chain : Trials search (first m rows) prior last) :
    Returns (queryFor (prior ++ [last])) (fun _ => false) m last.call.score last.call := by
  cases m with
  | mk pc' regs idx old flag out =>
    change pc' = 0 at pc
    subst pc'
    exact .next rfl (.next rfl (steps chain _ (observed chain)
      ⟨2,{regs with delta := BitVec.ofInt 32 initialDelta},idx,old,flag,out⟩ rfl rfl rfl))

def PublicScoreContract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (rows : List Row) (m : Machine)
    (inputs : Nat → SerialExecution.Inputs mc fc) (roots : Nat → Word → Word → SerialRecursion.RootInput)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    Root.Covers mc b (rows.map Row.square) → 0 < rows.length →
    (∀ i, SerialExecution.ReadFacts (inputs i)) → (∀ i, SerialTermination.Hints (inputs i)) →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    let search := trial c b inputs roots
    ∃ prior last result,
      run search 12 (first m rows) = some (prior,last) ∧ Trials search (first m rows) prior last ∧
      prior.length < 12 ∧ (∀ v ∈ prior ++ [last], Projected c b inputs roots v) ∧
      (prior ++ [last]).flatMap Visit.openings = List.range (last.closing+1) ∧
      Returns (queryFor (prior ++ [last])) (fun _ => false) m last.call.score last.call ∧
      last.answer.value = last.call.score.toInt ∧
      PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
        (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem publicScore : PublicScoreContract := by
  intro mc fc c b rows m inputs roots sort nodes fallback covers within valid hints entry window search
  obtain ⟨prior,last,ran,chain,limit,cover,positive,certified,inside,projected⟩ := complete
    (fun s cv pos w => trial_returns c b inputs roots s (valid _) (hints _) cv pos w)
    12 (first m rows) (first_ready _ window) rfl covers within
  obtain ⟨snap,best,writes,att⟩ := certified
  obtain ⟨result,published,exactScore⟩ := publishSingle cover positive writes (best ▸ inside) (best ▸ att)
    (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount b.val).toNat nodes fallback
  have signed : (BitVec.ofInt 32 last.answer.value).toInt = last.answer.value :=
    BitVec.toInt_ofInt_eq_self (by decide) (by have := att.1.lower; omega) (by have := att.1.upper; omega)
  have clock := chain.openings (fun v mem => by
    obtain ⟨_,exec,_,_⟩ := projected v mem
    exact exec.openings.2)
  refine ⟨prior,last,result,ran,chain,limit,projected,?_,chain.returns _ entry,signed.symm,published,?_⟩
  · simpa only [first, Nat.sub_zero, List.range_eq_range'] using clock.1
  · rw [PublicationMir.ExactScore.completed_tagged published, EndgameTreeCache.scaled_discValue]
    exact exactScore

end NeuralReversi.AspirationMir.SerialStack
