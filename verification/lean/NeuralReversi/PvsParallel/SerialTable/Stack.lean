module
public import NeuralReversi.PvsParallel.SerialTable.State
public import NeuralReversi.PvsParallel.SerialTable.Execution
public import NeuralReversi.PvsParallel.SerialCacheStack.Cells

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.Stack

open EndgameSearch.CacheHistory (Pool)
open SerialExecution (Frame Inputs Invariant push)
open SerialCalls (Program Request)
open SerialTermination (Entry Allowed)

abbrev Output (l : Layout) (α : Type) := Nat × α × Memory l
abbrev Child (l : Layout) := Nat → Frame → Memory l → Option (Output l Int)

def interpret {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (descend : Child l)
    (i : Nat) (ctx : Frame) (prior : Memory l) : Program Request α → Option (Output l α)
  | .pure v => some (i,v,prior)
  | .fail => none
  | .call q resume => do
    let child ← push mc fc ctx q
    let (k,v,next) ← descend (i+1) child prior
    interpret (mc := mc) (fc := fc) descend k ctx next (resume v)

theorem ready_of_map {code : Program Req α} {f : α → β} (ready : Program.Ready allowed (code.map f)) :
    Program.Ready allowed code := by
  induction code with
  | fail => cases ready
  | pure => exact .pure
  | call q resume ih =>
    cases ready with
    | call admissible more => exact .call admissible (fun w => ih w (more w))

theorem runs_map {code : Program Req α} (ran : Program.Runs code calls value) (f : α → β) :
    Program.Runs (code.map f) calls (f value) := by
  induction ran with
  | pure => exact .pure
  | call _ ih => exact .call ih

theorem interpret_valid {valid : Memory l → Prop} {word : Frame → Int → Prop}
    {code : Program Request α} (ready : Program.Ready allowed code)
    (descend : Child l) (enters : ∀ q, allowed q → ∀ j (m : Memory l), valid m →
      ∃ child k w next, push mc fc ctx q = some child ∧ descend j child m = some (k,w,next) ∧
        valid next ∧ word child w) (i : Nat) (prior : Memory l) (good : valid prior) :
    ∃ k v after, interpret (mc := mc) (fc := fc) descend i ctx prior code = some (k,v,after) ∧
      valid after ∧ ∃ calls, Program.Runs code calls v ∧
        ∀ t ∈ calls, ∃ child, push mc fc ctx t.1 = some child ∧ word child t.2 := by
  induction ready generalizing i prior with
  | pure => exact ⟨_,_,_,rfl,good,[],.pure,by simp⟩
  | @call q resume ok more ih =>
    obtain ⟨child,k,w,next,pushed,ran,sound,result⟩ := enters q ok (i+1) prior good
    obtain ⟨j,v,post,cont,g,cs,rest,children⟩ := ih w k next sound
    refine ⟨j,v,post,by simpa [interpret,pushed,ran] using cont,g,(_,w)::cs,.call rest,?_⟩
    intro t h
    rcases List.mem_cons.mp h with rfl | later
    · exact ⟨child,pushed,result⟩
    · exact children _ later

theorem interpret_returns {code : Program Request α} (ready : Program.Ready allowed code)
    (descend : Child l) (enters : ∀ q, allowed q → ∀ j (m : Memory l), m.Good p mc fc →
      ∃ child k w next, push mc fc ctx q = some child ∧ descend j child m = some (k,w,next) ∧
        next.Good p mc fc ∧ Invariant mc fc child w) (i : Nat) (prior : Memory l) (good : prior.Good p mc fc) :
    ∃ k v after, interpret (mc := mc) (fc := fc) descend i ctx prior code = some (k,v,after) ∧
      after.Good p mc fc ∧ ∃ calls, Program.Runs code calls v ∧
        ∀ t ∈ calls, ∃ child, push mc fc ctx t.1 = some child ∧ Invariant mc fc child t.2 :=
  interpret_valid ready descend enters i prior good

/-- The entry probe supplies the destination. Children may replace its contents before save executes. -/
def runFuel (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) : Nat → Child l
  | 0, _, _, _ => none
  | n+1, i, ctx, m =>
    if ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12 then
      (SerialCacheStack.Cells.run p c (m.tt.image.inputs m.supported c base) i ctx m.cache).map
        fun (k,w,cells) => (k,w,{m with cache := cells})
    else do
      let slot := m.tt.image.slot ctx.request.board
      let (k,out,after) ← interpret (mc := mc) (fc := fc) (runFuel p c base n) i ctx m
        (Execution.body c (m.inputs c base) i ctx)
      return (k,out.score,after.apply slot out.store)

def run (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (i : Nat) (ctx : Frame) (m : Memory l) :=
  runFuel p c base (Board.playRank mc ctx.request.board.val + 1) i ctx m

theorem fuel_returns (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (base : Inputs mc fc) (fuel i : Nat) (ctx : Frame) (m : Memory l)
    (rank : Board.playRank mc ctx.request.board.val < fuel) (entry : Entry ctx) (good : m.Good p mc fc) :
    ∃ k w next, runFuel p c base fuel i ctx m = some (k,w,next) ∧ next.Good p mc fc ∧ Invariant mc fc ctx w := by
  induction fuel generalizing i ctx m with
  | zero => omega
  | succ n ih =>
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · let f := m.tt.image.inputs m.supported c base
      have shared := m.tt.image.inputs_shared (limit := ctx.request.depth.toNat) m.supported c base
        good.1.publications good.1.hints
      let size := Board.playRank mc ctx.request.board.val + 1
      obtain ⟨k,w,next,actual,hist,reachable⟩ := SerialCacheStack.Cells.fuel_together p c address f shared size i
        ctx m.cache (by dsimp [size]; omega) (Nat.le_refl _) entry good.2
      obtain ⟨_,_,_,returned,_,_,result,_⟩ := SerialCacheStack.fuel_returns c f shared size i ctx m.cache.history
        (by dsimp [size]; omega) (Nat.le_refl _) entry good.2.1
      rw [hist] at returned
      obtain ⟨rfl,rfl,_⟩ := Prod.mk.inj (Option.some.inj returned)
      refine ⟨k,w,{m with cache := next},?_,⟨good.1,reachable⟩,result⟩
      simp only [runFuel,leaf,SerialCacheStack.Cells.run]
      rw [actual]; rfl
    · let inputs := m.inputs c base
      obtain ⟨valid,hints⟩ := m.facts c base good
      have ready := SerialTermination.frame_ready c inputs valid hints i ctx entry
      rw [← Execution.body_score c inputs i ctx] at ready
      obtain ⟨k,out,after,ran,post,cs,issued,children⟩ := interpret_returns (ready_of_map ready)
        (runFuel p c base n) (fun q accepted j prev ok => by
          obtain ⟨child,pushed,childEntry,smaller⟩ := SerialTermination.child_entry entry q accepted
          obtain ⟨last,v,next,done_,g,inv⟩ := ih j child prev (by omega) childEntry ok
          exact ⟨child,last,v,next,pushed,done_,g,inv⟩) i m good
      have returned := runs_map issued Writes.Result.score
      rw [Execution.body_score] at returned
      have inv := SerialExecution.localRule c inputs valid i ctx cs out.score returned children
      have replay := inv entry.1 entry.2.1 entry.2.2.1 entry.2.2.2
      have finite := SerialCalls.ScoreWords.finite_of_replay entry.2.2.1 entry.2.2.2 replay
      have sound := replayContract_proven _ _ _ _ _ _ entry.2.2.1 entry.2.2.2 replay
      let slot := m.tt.image.slot ctx.request.board
      refine ⟨k,out.score,after.apply slot out.store,?_,?_,inv⟩
      · simp [runFuel,leaf,inputs,slot] at ran ⊢
        rw [ran]; rfl
      · exact after.apply_good post slot (fun w saved =>
          ⟨_,Execution.produced c inputs i ctx entry issued saved finite⟩) entry.2.2.1 sound

theorem run_returns (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (base : Inputs mc fc) (i : Nat) (ctx : Frame) (m : Memory l) (entry : Entry ctx) (good : m.Good p mc fc) :
    ∃ k w next, run p c base i ctx m = some (k,w,next) ∧ next.Good p mc fc ∧ Invariant mc fc ctx w :=
  fuel_returns p c address base _ i ctx m (by omega) entry good

end NeuralReversi.PvsParallel.SerialTable.Stack
