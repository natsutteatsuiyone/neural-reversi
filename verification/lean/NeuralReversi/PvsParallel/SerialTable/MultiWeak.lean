module
public import NeuralReversi.AspirationMir.SerialTable.Repeated

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MultiWeak

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position GeneratedSquare)
open SerialExecution (Inputs)
open SerialCalls (Program MoveRequest)
open SerialCalls.ScoreWords (Finite)
open TranspositionTable (Selectivity)
open AspirationMir.SerialStack (State)
open AspirationMir.SmallRoot (Answer candidates)
open AspirationMir.SerialTable.Repeated (ReadyMemory)
open RootMoveStorage (Row)

private theorem wipeout {n wipe sq} (found : SerialRootCalls.entry n wipe = some (some sq)) :
    wipe = some (some sq) := by
  cases n with
  | none => contradiction
  | some n =>
    cases wipe with
    | none => contradiction
    | some v =>
      cases v <;> by_cases positive : 0 < n <;> simp [SerialRootCalls.entry,rootEntry,positive] at found ⊢
      exact found

/-- A root trial resets only its suffix and writes only generated moves from that suffix.
    This holds for selective child replies, including rejected aspiration attempts. -/
theorem frame_rows {mc fc b d paired mem rows idx root a z}
    (issued : Program.Runs (Root.frame mc fc b d paired mem rows idx root a z) cs out)
    (cover : PvsParallel.Root.Covers mc b (rows.map Row.square)) :
    out.answer.rows.map Row.square = rows.map Row.square ∧ out.answer.rows.take idx = rows.take idx := by
  let roots := rows.map Row.square
  let start := RootReplay.start a.toInt
  have clear := RootSuffix.Writes.reset (initial := rows) (idx := idx) (alpha := a.toInt) (beta := z.toInt)
  have unique := PvsParallel.Root.roots_nodup cover
  dsimp only [Root.frame] at issued
  split at issued
  · cases issued
  next sq hit =>
    cases issued
    obtain ⟨m,member,name,_⟩ := ConcurrentReplay.generatedWipeout mc fc b roots idx paired mem sq cover (wipeout hit)
    have inRows : sq ∈ ((RootMoveStorage.resetFrom rows idx).drop idx).map Row.square := by
      rw [List.map_drop,(clear.scores unique).1]
      simpa only [roots,name] using (PvsParallel.Root.choices_squares_mem cover m.val).mp (List.mem_map_of_mem member)
    have written := (RootSuffix.Writes.record clear (⟨sq,a.toInt,PvsMoveLoop.scoreMax⟩ : RootOrder.Report) true
      (root.pvLine sq) inRows rfl).scores unique
    exact ⟨written.1,written.2.1⟩
  next =>
    obtain ⟨owner,requests,rfl⟩ := Program.runs_map issued
    obtain ⟨words,ran,rfl⟩ := Program.runs_mapRequests requests
    let moves := candidates mc fc b roots idx root.key root.sort
    have order : moves.Perm (PvsParallel.Root.choices mc b roots idx) := (RootFrameMir.ordered_candidates ..).1
    have distinct := order.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc b).1.filter _)
    let key := fun q : MoveRequest (GeneratedSquare mc b.val) => (q.move,q.pv)
    let search : PvsMoveLoop.ChildSearch (GeneratedSquare mc b.val) := fun m p _ _ => Program.returnAt key words (m,p)
    have supplied t (h : t ∈ words) : search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 :=
      Program.returnAt_mem key words (SerialOwnerCalls.runs_distinct ran distinct) t h
    have stored := SerialOwnerCalls.runs_rows ran (fun m => root.pvLine m.val) (RootMoveStorage.resetFrom rows idx) search supplied
    have written := RootSerialMir.rowsAfter_writes Subtype.val (fun m => root.pvLine m.val) search moves start clear unique (by
      intro m h
      rw [List.map_drop]
      exact (PvsParallel.Root.choices_squares_mem cover m.val).mp (List.mem_map_of_mem (order.mem_iff.mp h)))
    have facts := written.scores unique
    change (owner.1.foldl (SerialOwnerCalls.writeVisit Subtype.val (fun m => root.pvLine m.val)) (true,RootMoveStorage.resetFrom rows idx)).2 = _ at stored
    dsimp only [SerialRootCalls.ownerAnswer]
    rw [stored]
    exact ⟨facts.1,facts.2.1⟩

def trial (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (lvl : Selectivity) (screen : WeakStack.Screen)
    (s : State) (prior : Memory l) : Option (Nat × Answer × Memory l) := do
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows idx
    (Trial.ordering prior base b root s) s.regs.alpha s.regs.beta
  let (k,out,after) ← Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base lvl screen) s.opening ctx prior code
  return (k,out.answer,after.apply (prior.tt.image.slot b) out.proposal lvl)

theorem trial_returns (idx : Nat) (p : Pool) (c : SerialScout.Config) (addr : c.address = p.address)
    (b : Position) (base : Inputs mc fc) (root : SerialRecursion.RootInput) (lvl : Selectivity) (weak : lvl.val < 5)
    (screen : WeakStack.Screen) (s : State) (prior : Memory l) (good : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (within : idx < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans next, trial idx p c b base root lvl screen s prior = some (k,ans,next) ∧ ReadyMemory p mc fc next ∧
      Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square ∧ ans.rows.take idx = s.rows.take idx := by
  by_cases first : idx = 0
  · subst idx
    obtain ⟨k,ans,next,ran,valid,finite,keys⟩ := WeakTrial.trial_returns p c addr b base root lvl weak screen s prior good.1 cover within window
    have exactTT := (WeakTrial.preserves p c addr b base root lvl weak screen s prior good.1 good.2 cover within window ran).1
    exact ⟨k,ans,next,ran,⟨valid,exactTT⟩,finite,keys,rfl⟩
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let order := Trial.ordering prior base b root s
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows idx order s.regs.alpha s.regs.beta
  have entry : SerialTermination.Entry ctx := by
    simp only [ctx,SerialTermination.Entry,SerialRootExecution.context,ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b,rfl,window,by simp⟩
  have live : Program.Ready (SerialTermination.Allowed mc fc ctx) (code.map Root.Exit.answer) := by
    rw [show code.map Root.Exit.answer = _ from Root.erase ..]
    exact SerialRootCalls.frame_ready_at_depth cover within window
  obtain ⟨k,out,after,ran,post,cs,issued,words⟩ := Stack.interpret_valid (ctx := ctx) (code := code)
    (valid := ReadyMemory p mc fc) (word := fun _ w => Finite w)
    (Stack.ready_of_map live) (WeakStack.run p c base lvl screen) (fun q allowed j before ready => by
      obtain ⟨child,pushed,childEntry,_⟩ := SerialTermination.child_entry entry q allowed
      obtain ⟨k,w,next,done_,valid,finite⟩ := WeakStack.run_returns p c addr base lvl weak screen j child before childEntry ready.1
      obtain ⟨k',w',next',done',_,exactTT,_⟩ := WeakTrial.weak_fuel p c addr base lvl weak screen _ j child before (Nat.lt_succ_self _) childEntry ready.1 ready.2
      rw [show WeakStack.runFuel p c base lvl screen _ j child before = _ from done_] at done'
      cases done'
      exact ⟨child,k,w,next,pushed,done_,⟨valid,exactTT⟩,finite⟩) s.opening prior good
  have finite := (WeakRoot.result issued cover within (fun t h => by
    obtain ⟨_,_,word⟩ := words t h
    exact word)).1
  have absent : out.proposal = none := by
    cases found : out.proposal with
    | none => rfl
    | some w => exact (first (Root.writes issued found).1).elim
  exact ⟨k,out.answer,after,by simp [trial,code,ctx,order] at ran ⊢; rw [ran]; simp [absent,Memory.apply],
    post,finite,frame_rows issued cover⟩

end NeuralReversi.PvsParallel.SerialTable.MultiWeak
