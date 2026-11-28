module
public import NeuralReversi.AspirationMir.SerialTable.MidgameRoot
public import NeuralReversi.PvsParallel.SerialTable.MidgameChild

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MidgamePolledRoot

open PvsParallel
open EndgameSearch.CacheHistory (Pool)
open PvsParallel.SerialTable (Memory)
open PvsBoardTree (Position)
open SerialCalls (Request Program)
open SerialCalls.ScoreWords (Finite)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)

def runAt (idx : Nat) (p : Pool) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (level : TranspositionTable.Selectivity) (plans : (r : Request) → SerialTable.MidgameChild.Plan mc fc r)
    (stop : Request → Bool) (rows : List Row) (a z : Word) (m : Memory l) : Option (Memory l) := do
  let code := PvsParallel.SerialTable.Root.frame mc fc b depth paired scratch rows idx order a z
  let (out,after) ← PvsParallel.SerialTable.MidgamePoll.run (fun r => PvsParallel.SerialTable.MidgameChild.eval p (plans r))
    (fun r v => PvsParallel.SerialTable.MidgamePoll.afterMove r.pv r.beta v && stop r) m code
  match out with
  | none => return after
  | some finished => return MidgameRoot.save after (m.tt.image.slot b) level finished.proposal

theorem returnsAt (idx : Nat) (p : Pool) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (level : TranspositionTable.Selectivity) (plans : (r : Request) → PvsParallel.SerialTable.MidgameChild.Plan mc fc r)
    (stop : Request → Bool) (rows : List Row) (a z : Word) (m : Memory l) (good : ReadyMemory p mc fc m)
    (cover : Root.Covers mc b (rows.map Row.square)) (positive : idx < rows.length) (window : Aspiration.Window a.toInt z.toInt) :
    ∃ after, runAt idx p b depth paired scratch order level plans stop rows a z m = some after ∧ ReadyMemory p mc fc after := by
  let code := PvsParallel.SerialTable.Root.frame mc fc b depth paired scratch rows idx order a z
  have live := SerialRootCalls.frame_ready_at_depth (fc := fc) (key := order.key) (sort := order.sort) (pv := order.pvLine)
    (depth := depth) (paired := paired) (mem := scratch) cover positive window
  rw [← PvsParallel.SerialTable.Root.erase mc fc b depth paired scratch rows idx order] at live
  let reply := fun r => PvsParallel.SerialTable.MidgameChild.eval p (plans r) (l := l)
  let poll := fun (r : Request) v => PvsParallel.SerialTable.MidgamePoll.afterMove r.pv r.beta v && stop r
  obtain ⟨out,n,ran,post,done_⟩ := PvsParallel.SerialTable.MidgamePoll.returns (code := code)
    (word := fun (r : Request) v => Finite v ∧ PvsParallel.SerialTable.WeakTrial.Range r v)
    (safe := ReadyMemory p mc fc) (PvsParallel.SerialTable.Stack.ready_of_map live) reply poll
    (fun r ok prev valid => PvsParallel.SerialTable.MidgameChild.eval_ready p (plans r) prev ok.2.1 valid) m good
  cases out with
  | none =>
    refine ⟨n,?_,post⟩
    change (PvsParallel.SerialTable.MidgamePoll.run reply poll m code).bind _ = _
    rw [ran]
    rfl
  | some result =>
    obtain ⟨calls,issued,words⟩ := done_ result rfl
    have finite := (PvsParallel.SerialTable.WeakRoot.result issued cover positive (fun t h => (words t h).1)).1
    have range := PvsParallel.SerialTable.WeakTrial.root_range issued window (fun t h => (words t h).2)
    refine ⟨_,?_,MidgameRoot.save_ready n post (m.tt.image.slot b) level issued finite range⟩
    change (PvsParallel.SerialTable.MidgamePoll.run reply poll m code).bind _ = _
    rw [ran]
    rfl

def run (p : Pool) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (level : TranspositionTable.Selectivity) (plans : (r : Request) → SerialTable.MidgameChild.Plan mc fc r)
    (stop : Request → Bool) (rows : List Row) (a z : Word) (m : Memory l) : Option (Memory l) :=
  runAt 0 p b depth paired scratch order level plans stop rows a z m

theorem returns (p : Pool) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (level : TranspositionTable.Selectivity) (plans : (r : Request) → PvsParallel.SerialTable.MidgameChild.Plan mc fc r)
    (stop : Request → Bool) (rows : List Row) (a z : Word) (m : Memory l) (good : ReadyMemory p mc fc m)
    (cover : Root.Covers mc b (rows.map Row.square)) (positive : 0 < rows.length) (window : Aspiration.Window a.toInt z.toInt) :
    ∃ after, run p b depth paired scratch order level plans stop rows a z m = some after ∧ ReadyMemory p mc fc after := by
  exact returnsAt 0 p b depth paired scratch order level plans stop rows a z m good cover positive window

/-- With no observed stop, this interpreter performs the identical TT write to the normal
    trial and leaves the identical physical memory, including all promoted-leaf cache writes. -/
theorem uninterrupted (p : Pool) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (level : TranspositionTable.Selectivity) (plans : (r : Request) → PvsParallel.SerialTable.MidgameChild.Plan mc fc r)
    (rows : List Row) (a z : Word) (m : Memory l) :
    run p b depth paired scratch order level plans (fun _ => false) rows a z m =
      (MidgameRoot.run mc fc b depth paired scratch order level (PvsParallel.SerialTable.MidgameChild.run p plans) rows a z m).map Prod.snd := by
  let reply := fun r => PvsParallel.SerialTable.MidgameChild.eval p (plans r) (l := l)
  let child := PvsParallel.SerialTable.MidgameChild.run p plans (l := l)
  have same {A : Type} (code : Program Request A) (mem : Memory l) :
      PvsParallel.SerialTable.MidgamePoll.run reply (fun _ _ => false) mem code =
        (MidgameRoot.interpret child code mem).map (fun out => (some out.1,out.2)) := by
    induction code generalizing mem with
    | fail | pure => rfl
    | call q resume ih =>
      change PvsParallel.SerialTable.MidgamePoll.run reply _ (reply q mem).2 (resume (reply q mem).1) =
        (MidgameRoot.interpret child (resume (reply q mem).1) (reply q mem).2).map _
      exact ih _ _
  unfold run runAt MidgameRoot.run MidgameRoot.runAt
  simp only [Bool.and_false]
  change (PvsParallel.SerialTable.MidgamePoll.run reply (fun _ _ => false) m _).bind _ = _
  rw [same]
  cases MidgameRoot.interpret child _ m <;> rfl

end NeuralReversi.AspirationMir.SerialTable.MidgamePolledRoot
