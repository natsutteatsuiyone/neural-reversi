import NeuralReversi.AspirationMir.SerialCache
import NeuralReversi.PvsParallel.SerialCacheStack.Cells

namespace NeuralReversi.AspirationMir.SerialCache.Physical

open PvsParallel
open PvsBoardTree (Position)
open SerialExecution (Inputs)
open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.SerialLeafCache (Reachable snapshot)
open RootMoveStorage (Row)

/-- Both cache arrays survive a failed window, along with the root rows. -/
def run (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots) :
    Nat → SerialStack.State → Local → Option (List Receipt × Receipt × Local)
  | 0, _, _ => none
  | fuel+1, s, prev => do
    let (k,ans,next) ← SerialCacheStack.Cells.trial p c b (inputs s.index)
      (roots s.index s.regs.alpha s.regs.beta) s prev
    let v : Receipt := ⟨⟨s,k,ans⟩,prev.history,next.history⟩
    let (regs,retry) ← applyWiden s.regs v.call.score
    if retry then
      let (earlier,last,post) ← run p c b inputs roots fuel (SerialStack.next v.toVisit regs) next
      return (v::earlier,last,post)
    else return ([],v,next)

theorem carried (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (facts : ∀ i, SerialCacheStack.SharedFacts (inputs i) ((Board.emptyCount b.val).toNat-1))
    (path : Path (search c b inputs roots) s h prior last)
    (observed : ∀ r ∈ prior ++ [last], Recorded c b inputs roots r)
    (fuel : Nat) (fits : prior.length < fuel) (prev : Local) (eq : prev.history = h)
    (boot : Reachable p mc fc prev) :
    run p c b inputs roots fuel s prev = some (prior,last,⟨last.after,snapshot p last.after⟩) ∧
      Reachable p mc fc ⟨last.after,snapshot p last.after⟩ := by
  induction path generalizing fuel prev with
  | @accept s h k ans after returned stop =>
    cases fuel with
    | zero => omega
    | succ n =>
      obtain ⟨covers,pos,window,_⟩ := observed _ List.mem_cons_self
      obtain ⟨t,ok⟩ := SerialCacheStack.Cells.trial_eq p c address b (inputs s.index) _ s prev
        (facts _) boot covers pos window (eq ▸ returned)
      exact ⟨by simp [run,t,eq,SerialStack.Visit.call,stop],ok⟩
  | @retry s h k ans after regs prior last returned wide tail ih =>
    cases fuel with
    | zero => omega
    | succ n =>
      obtain ⟨covers,pos,window,_⟩ := observed _ List.mem_cons_self
      obtain ⟨t,ok⟩ := SerialCacheStack.Cells.trial_eq p c address b (inputs s.index) _ s prev
        (facts _) boot covers pos window (eq ▸ returned)
      obtain ⟨rest,post⟩ := ih (fun r mem => observed r (.tail _ mem)) n
        (by simpa using fits) _ rfl ok
      exact ⟨by simp [run,t,eq,SerialStack.Visit.call,wide,rest],post⟩

def Published (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc)
    (roots : Roots) (rows : List Row) (m : Machine) (prev : Local) (sort : PublicationMir.SliceSort)
    (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) : Prop :=
  ∃ prior last post result,
    run p c b inputs roots 12 (SerialStack.first m rows) prev = some (prior,last,post) ∧
    Reachable p mc fc post ∧ post.history = last.after ∧ prev.history <+: post.history ∧
    (prior ++ [last]).map Receipt.before = prev.history :: prior.map Receipt.after ∧
    Returns (SerialStack.queryFor ((prior ++ [last]).map Receipt.toVisit)) (fun _ => false) m last.call.score last.call ∧
    PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
      (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
    PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val) ∧
    NextLevel mc fc b last

def Contract : Prop :=
  ∀ mc fc (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc) (roots : Roots)
    (rows : List Row) (m : Machine) (prev : Local) sort nodes fallback,
    c.address = p.address → (∀ i, SerialCacheStack.SharedFacts (inputs i) ((Board.emptyCount b.val).toNat-1)) →
    Reachable p mc fc prev → Root.Covers mc b (rows.map Row.square) → 0 < rows.length → m.pc = 0 →
    Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    Published p c b inputs roots rows m prev sort nodes fallback

theorem publication : Contract := by
  intro mc fc p c b inputs roots rows m prev sort nodes fallback address facts boot covers pos pc window
  obtain ⟨prior,last,result,_,path,bound,hist,grows,_,obs,machine,printed,score,level⟩ :=
    SerialCache.publication mc fc c b inputs roots rows m prev.history sort nodes fallback facts boot.1 covers pos pc window
  obtain ⟨run_,ok⟩ := carried p c address b inputs roots facts path obs 12 bound prev rfl boot
  exact ⟨prior,last,_,result,run_,ok,rfl,grows,hist,machine,printed,score,level⟩

/-- With the true value strictly inside the incoming window, neither bound can
    request widening. This concerns the returned word even on a warm worker. -/
theorem first_accepts (p : Pool) (c : SerialScout.Config) (b : Position)
    (inputs : Nat → Inputs mc fc) (roots : Roots) (s : SerialStack.State) (prev : Local)
    (address : c.address = p.address) (facts : SerialCacheStack.SharedFacts (inputs s.index)
      ((Board.emptyCount b.val).toNat-1)) (boot : Reachable p mc fc prev)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (pos : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (lo : s.regs.alpha.toInt < PvsBoardTree.minimax mc fc b)
    (hi : PvsBoardTree.minimax mc fc b < s.regs.beta.toInt)
    (ran : run p c b inputs roots (fuel+1) s prev = some (prior,last,post)) : prior = [] := by
  obtain ⟨k,ans,after,res,cert,_⟩ := SerialCacheStack.trial_returns c b (inputs s.index)
    (roots s.index s.regs.alpha s.regs.beta) s prev.history facts boot.1 cover pos window
  obtain ⟨concrete,_⟩ := SerialCacheStack.Cells.trial_eq p c address b (inputs s.index) _ s prev
    facts boot cover pos window res
  obtain ⟨snap,_,_,sound⟩ := cert
  have valid := sound.1
  rw [Root.value_zero cover (by simpa using pos)] at valid
  have word : (BitVec.ofInt 32 ans.value).toInt = ans.value := BitVec.toInt_ofInt_eq_self (by decide)
    (by have := valid.lower; omega) (by have := valid.upper; omega)
  have inside := widen_inside s.regs (BitVec.ofInt 32 ans.value)
    (by rw [word]; have := valid.failLow; omega) (by rw [word]; have := valid.failHigh; omega)
  simp [run,concrete,SerialStack.Visit.call,inside] at ran
  exact ran.1

end NeuralReversi.AspirationMir.SerialCache.Physical
