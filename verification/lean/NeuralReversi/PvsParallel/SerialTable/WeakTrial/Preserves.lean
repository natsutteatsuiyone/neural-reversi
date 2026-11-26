import NeuralReversi.PvsParallel.SerialTable.WeakTrial.Bounds

namespace NeuralReversi.PvsParallel.SerialTable.WeakTrial

open SerialExecution (Frame Inputs push)
open SerialCalls (Request Program)
open EndgameSearch.CacheHistory (Pool)
open TranspositionTable (Selectivity)
open SerialCalls.ScoreWords (Finite encode)
open PvsMoveLoop (scoreMax)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open AspirationMir.SerialStack (State)
open AspirationMir.SerialTable.PvInterior (InsideRange)

private theorem encoded (v : Int) (lo : -32767 ≤ v) (hi : v ≤ 32767) : (encode v).raw.toInt = v :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

private theorem exact_range {w : Writes.Proposal} {q : Request} (win : Aspiration.Window q.alpha q.beta)
    (fin : Finite score) (range : Range q score) (value : w.score = score)
    (tag : w.bound = TranspositionTable.classify q.pv (Writes.word score) (Writes.word q.alpha) (Writes.word q.beta))
    (exactTag : w.bound = .exact) : -scoreMax ≤ w.score ∧ w.score ≤ scoreMax := by
  have h := ((TranspositionTable.classify_contract ..).2.1).mp (tag.symm.trans exactTag)
  obtain ⟨lo,ordered,hi⟩ := win
  have ⟨sl,sh⟩ := fin
  change q.pv = true ∧ (encode q.alpha).raw.toInt < (encode score).raw.toInt ∧ (encode score).raw.toInt < (encode q.beta).raw.toInt at h
  rw [encoded score (by omega) (by omega),encoded q.alpha lo (by omega),encoded q.beta (by omega) hi] at h
  exact value ▸ range h.1 h.2.1 h.2.2

private theorem frame_range (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc)
    (level : Selectivity) (screen : WeakStack.Screen) (i : Nat) (ctx : Frame) (entry : SerialTermination.Entry ctx)
    (ran : Program.Runs (WeakStack.body m c base level screen i ctx) cs out) (finite : Finite out.score)
    (child : ∀ t ∈ cs, Range t.1 t.2) :
    Range ctx.request out.score ∧ ∀ w, out.store = some w → w.bound = .exact → -scoreMax ≤ w.score ∧ w.score ≤ scoreMax := by
  dsimp only [WeakStack.body] at ran
  cases role : ctx.request.pv <;> simp only [role,Bool.false_eq_true,ite_false,ite_true] at ran
  · have null : (encode ctx.request.beta).raw.toInt = (encode ctx.request.alpha).raw.toInt+1 := by
      obtain ⟨lo,mid,hi⟩ := entry.2.2.1
      rw [encoded _ (by omega) hi,encoded _ lo (by omega)]
      exact entry.2.2.2 role
    refine ⟨fun t => Bool.noConfusion (role.symm.trans t),?_⟩
    intro w saved exactTag
    have prod := SelectiveFrame.frame_produced ran saved null finite
    have impossible := ((TranspositionTable.classify_contract ..).2.1).mp (prod.bound.symm.trans exactTag)
    exact Bool.noConfusion impossible.1
  · have range := pv_range entry.2.1 (EndgameOrderMir.ordinary_perm _ _ _ _ _) entry.2.2.1 ran child
    refine ⟨fun _ => range,?_⟩
    intro w saved tag
    have prod := Writes.pv_produced ran saved
    exact exact_range entry.2.2.1 finite (fun _ => range) prod.value (by rw [role]; exact prod.bound) tag

private theorem children_range {mc fc ctx} {cs : List (Request × Int)} (found : ∀ t ∈ cs, ∃ child, push mc fc ctx t.1 = some child ∧ Range child.request t.2) :
    ∀ t ∈ cs, Range t.1 t.2 := by
  intro t mem
  obtain ⟨child,pushed,range⟩ := found t mem
  exact SerialCacheStack.request_of_push pushed ▸ range

theorem weak_fuel (p : Pool) (c : SerialScout.Config) (addr : c.address = p.address) (base : Inputs mc fc)
    (level : Selectivity) (weak : level.val < 5) (screen : WeakStack.Screen) (fuel i : Nat) (ctx : Frame) (m : Memory l)
    (rank : Board.playRank mc ctx.request.board.val < fuel) (entry : SerialTermination.Entry ctx)
    (valid : m.Valid p mc fc) (before : ExactScores m.tt.history) :
    ∃ k w next, WeakStack.runFuel p c base level screen fuel i ctx m = some (k,w,next) ∧ next.Valid p mc fc ∧
      ExactScores next.tt.history ∧ Range ctx.request w := by
  induction fuel generalizing i ctx m with
  | zero => omega
  | succ n ih =>
    obtain ⟨k,w,next,full,holds,finite⟩ := WeakStack.fuel_returns p c addr base level weak screen (n+1) i ctx m rank entry valid
    refine ⟨k,w,next,full,holds,?_⟩
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · simp only [WeakStack.runFuel,leaf,Stack.run,Stack.runFuel] at full
      obtain ⟨⟨_,_,_⟩,_,eq⟩ := Option.map_eq_some_iff.mp full
      cases eq
      exact ⟨before,fun pv => Bool.noConfusion (leaf.1.symm.trans pv)⟩
    · let code := WeakStack.body m c base level screen i ctx
      have ready := WeakStack.body_ready m c base level screen i ctx valid entry leaf
      obtain ⟨j,out,after,returned,sound,cs,issued,words⟩ := Stack.interpret_valid
        (valid := fun m : Memory l => m.Valid p mc fc ∧ ExactScores m.tt.history) (word := fun f v => Range f.request v)
        (code := code) ready (WeakStack.runFuel p c base level screen n) (fun q ok j prior good => by
          obtain ⟨child,pushed,childEntry,decrease⟩ := SerialTermination.child_entry entry q ok
          obtain ⟨last,v,next,ran,safe,range,score⟩ := ih j child prior (by omega) childEntry good.1 good.2
          exact ⟨child,last,v,next,pushed,ran,⟨safe,range⟩,score⟩) i m ⟨valid,before⟩
      dsimp only [code] at returned
      simp only [WeakStack.runFuel,leaf,ite_false,returned,bind,Option.bind,pure] at full
      obtain ⟨rfl,rfl,rfl⟩ := Prod.mk.inj (Option.some.inj full)
      obtain ⟨score,stored⟩ := frame_range m c base level screen i ctx entry issued finite (children_range words)
      exact ⟨after.apply_exactScores sound.2 _ _ _ stored,score⟩

/-- Every completed selective trial preserves the bounded exact TT entries, including across
    the root probe's retained store and all earlier rejected child windows. -/
theorem preserves (p : Pool) (c : SerialScout.Config) (addr : c.address = p.address) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (weak : level.val < 5) (screen : WeakStack.Screen)
    (s : State) (m : Memory l) (valid : m.Valid p mc fc) (before : ExactScores m.tt.history)
    (covers : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (obtained : trial p c b base root level screen s m = some (k,ans,next)) :
    ExactScores next.tt.history ∧ InsideRange s.regs.alpha.toInt s.regs.beta.toInt ans.value := by
  obtain ⟨j,v,m',done_,_,finite,_⟩ := trial_returns p c addr b base root level weak screen s m valid covers within window
  rw [obtained] at done_
  cases done_
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let order := Trial.ordering m base b root s
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows 0 order s.regs.alpha s.regs.beta
  obtain ⟨entry,ready⟩ := SerialCacheStack.root_ready c b base order s covers within window
  change Program.Ready _ (SerialRootCalls.frame ..) at ready
  rw [← Root.erase] at ready
  obtain ⟨j,out,after,ran,good,cs,issued,words⟩ := Stack.interpret_valid (ctx := ctx) (code := code)
    (valid := fun m : Memory l => m.Valid p mc fc ∧ ExactScores m.tt.history) (word := fun f v => Range f.request v)
    (Stack.ready_of_map ready) (WeakStack.run p c base level screen) (fun q ok j prior g => by
      obtain ⟨child,pushed,entered,_⟩ := SerialTermination.child_entry entry q ok.1
      obtain ⟨last,v,out,ran,valid,exactTT,word⟩ := weak_fuel p c addr base level weak screen _ j child prior
        (Nat.lt_succ_self _) entered g.1 g.2
      exact ⟨child,last,v,out,pushed,ran,⟨valid,exactTT⟩,word⟩) s.opening m ⟨valid,before⟩
  have range := root_range issued window (children_range words)
  dsimp only [trial] at obtained
  rw [ran] at obtained
  cases obtained
  refine ⟨after.apply_exactScores good.2 _ _ _ (fun w saved tag => ?_),range⟩
  obtain ⟨_,_,_,score,_,classified,_⟩ := Root.writes issued saved
  apply exact_range (q := ctx.request) window finite (fun _ => range) score _ tag
  simpa [ctx,SerialRootExecution.context,Writes.word] using classified

end NeuralReversi.PvsParallel.SerialTable.WeakTrial
