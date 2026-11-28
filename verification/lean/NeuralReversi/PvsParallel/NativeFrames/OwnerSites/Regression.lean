module
public import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Body
public import NeuralReversi.PvsParallel.NativeFrames.SplitReturn
meta import NeuralReversi.PvsParallel.NativeFrames
meta import NeuralReversi.PvsParallel.NativeFrames.Invocations
meta import NeuralReversi.PvsParallel.NativeFrames.OwnerSites
meta import NeuralReversi.PvsParallel.NativeFrames.Returns
meta import NeuralReversi.PvsParallel.NativeFrames.SplitReturn

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Regression

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheOwner (Node Resource)
open Invocations (Calls)

private def pool : Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def mainW : Worker pool := ⟨0,.main,by decide⟩
private def otherW : Worker pool := ⟨1,.worker,by decide⟩
private def tick (n : Nat) := 3*n+2
private def args (pv : Bool) : Args :=
  ⟨⟨0,7⟩,61,⟨0⟩,⟨1⟩,true,Cutoffs.exactSelectivity,pv,!pv,none⟩

private def children := [Returns.argsFields (args false) (-3),Returns.argsFields (args true) 7]
private def nested (exit : Invocations.Exit) : Calls :=
  .call ⟨Role.scout.site,mainW.root,none⟩ exit .nil .nil .nil
private def sites (k : Nat) : Site :=
  if k = 0 then ⟨nested (.discarded ⟨0⟩),nested (.normal ⟨9⟩),[]⟩
  else ⟨.nil,.nil,[⟨⟨6,mainW.root,none⟩,.leave .split none none,by intro; simp,.nil,.nil⟩]⟩
private def calls := issue (moveSite false) mainW.root sites [] 0 children

private def before (n : Nat) (node : Node) (receiver : Resource) : Event :=
  .entry (tick n) 0 node receiver none
private def owner := before 4 .pvs mainW.root
private def scoreAt := tick (5+calls.count)
private def mainEvents : List Event :=
  [before 0 .root mainW.root,before 1 .endRoot mainW.root,before 2 .solve mainW.root,
    before 3 .endWindow mainW.root,owner] ++ calls.events tick 5 ++ [.scored scoreAt ⟨7⟩]

/-- An unrelated child with the same opening time and board on a second worker. -/
private def otherEvents : List Event :=
  [before 0 .active otherW.root,before 1 .dispatch otherW.root,before 2 .splitSearch otherW.root,
    before 3 .pvs otherW.root,.entry (tick 5) Role.scout.site .pvs otherW.root none,.scored (tick 6) ⟨9⟩,
    .leave (tick 7) (tick 5) .pvs none (some ⟨9⟩)]
private def stream := (mainEvents.take 6).map (Record.localCall 0) ++ otherEvents.map (Record.localCall 1) ++
  (mainEvents.drop 6).map (Record.localCall 0)
private def background (w t : Nat) : Option Args :=
  if w = 0 then
    if t ∈ [tick 4,tick 6,tick 9] then some (args false) else none
  else if t ∈ [tick 3,tick 5] then some (args false) else none
private def inputs := callInputs 0 sites tick 5 children background

#guard ((entries sites tick 0 5 children).map Prod.fst) == [tick 5,tick 15]
#guard (calls.direct tick (inputs 0) 5) == children
#guard (decode pool inputs stream).isSome
#guard resumed (tick 14) scoreAt 7 (Returns.localEvents 0 stream)
#guard !(resumed (tick 14) scoreAt 9 (Returns.localEvents 0 stream))
#guard !(resumed (tick 14) scoreAt 7 (Returns.localEvents 1 stream))
#guard !(resumed (tick 14) (tick 6) 9 (Returns.localEvents 0 stream))
#guard ((Returns.poolReturns pool inputs stream).filter (fun ret =>
  ret.worker == 0 && ret.parent == frameKey 2 0 (tick 4))).map Returns.Returned.call == children
#guard (Returns.poolReturns pool inputs stream).any (fun ret => ret.worker == 1 && ret.opening == tick 5)
#guard !(decode pool (fun w t => if w = 0 && t = tick 6 then none else inputs w t) stream).isSome

#guard !(decode pool (fun w t => if w = 0 && t = tick 5 then some {args false with depth := 60} else inputs w t) stream).isSome
#guard !(decode pool (fun w t => if w = 0 && t = tick 5 then some {args false with beta := ⟨2⟩} else inputs w t) stream).isSome

/-- A real pass preserves depth while reversing both bounds. Generate its
    register write and parse all three instructions at arbitrarily spaced times. -/
example {p : Pool} (w : Worker p) (r : Row) (word : ScoreRefresh.Score) (clock : Nat → Nat)
    (ordered : ∀ {a b}, a < b → clock a < clock b) (owner i : Nat) (receiver : Resource) :
    let c := FrameReturns.OwnerCalls.passCall r.frame r.depth r.frame.pv word.raw.toInt
    let site : Nat → Site := fun _ => ⟨.nil,.nil,[]⟩
    let reg := install site clock i [c] (fun _ => none)
    Body w reg receiver owner clock i (issue (fun _ _ => .passChild) receiver site [] 0 [c]) := by
  dsimp only
  apply issue_body _ _ _ _ _ ordered
  · intro c mem
    exact List.mem_singleton.mp mem ▸ pass_admissible r word.raw.toInt
  · intro c mem
    exact List.mem_singleton.mp mem ▸ pass_fits r r.frame.pv word
  · exact installed _ clock ordered _ _ _
  · exact ⟨Body.nil,Body.nil,Body.nil,Body.nil⟩

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Regression
