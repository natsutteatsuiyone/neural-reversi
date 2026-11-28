module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheOwner (Node Thread)
open EndgameSearch.CacheHistory.Payloads (Request Site Kind site? target)
open EndgameTreeCache (Table)
open Invocations (Calls)

def Access.tableKind : Access → Table × Kind
  | .slot t _ => (t,.slot)
  | .probe t .. => (t,.probe)
  | .store t .. => (t,.store)

theorem Access.primitive (op : Access) : target op.node = some op.tableKind := by
  cases op with
  | slot t | probe t | store t => cases t <;> rfl

structure Issued where
  index : Nat
  op : Access

def Issued.request (clock : Nat → Nat) (x : Issued) : Request :=
  ⟨clock x.index,clock (x.index+1),x.op.tableKind.1,x.op.tableKind.2⟩

def Program.schedule (th : Thread) (i : Nat) : Program node h last → List Issued
  | .done => []
  | .call _ _ _ _ _ body next =>
    body.schedule th (i+1) ++ next.schedule th (i+1+(body.forest.calls th).count+1)
  | .access _ op _ _ next => ⟨i,op⟩ :: next.schedule th (i+2)

theorem Program.schedule_ops (code : Program node h last) (th : Thread) (i : Nat) :
    (code.schedule th i).map Issued.op = code.accesses := by
  induction code generalizing i with
  | done => rfl
  | call _ _ _ _ _ _ _ first rest => simp [schedule,accesses,first,rest]
  | access _ _ _ _ _ ih => simp [schedule,accesses,ih]

def sitesOf (worker : Nat) (ev : List Event) :=
  (NativeFrameCache.project (ev.map (Record.localCall worker))).filterMap site?

private theorem sites_append (w : Nat) (xs ys : List Event) :
    sitesOf w (xs ++ ys) = sitesOf w xs ++ sitesOf w ys := by
  simp [sitesOf,NativeFrameCache.project,List.filterMap_append]

private theorem sites_call (w site i : Nat) (th : Thread) (child : Node) (body later : Calls) (clock : Nat → Nat) :
    sitesOf w ((Calls.call ⟨site,child.kind.resource th,none⟩ (.leave child none none) body .nil later).events clock i) =
      sitesOf w (body.events clock (i+1)) ++
      ((target child).map fun k => Site.mk w ⟨clock i,clock (i+1+body.count),k.1,k.2⟩).toList ++
      sitesOf w (later.events clock (i+1+body.count+1)) := by
  simp only [Calls.events,Invocations.Exit.node,Invocations.Exit.score,Option.map_none,Option.toList_none,
    List.length_nil,Calls.count,Nat.add_zero,Invocations.Exit.event,List.append_nil,sites_append]
  change [] ++ _ ++ (site? (EndgameSearch.CacheHistory.NativeTrace.Event.worker (Shared:=ConcurrentReplay.Event) w
    (.leave (clock (i+1+body.count)) (clock i) child none))).toList ++ _ = _
  cases h : target child <;> simp [site?,h]

theorem Program.schedule_sites (code : Program node h last) (w : Nat) (th : Thread) (clock : Nat → Nat) (i : Nat) :
    sitesOf w ((code.forest.calls th).events clock i) = (code.schedule th i).map (fun x => ⟨w,x.request clock⟩) := by
  induction code generalizing i with
  | done => rfl
  | call _ _ _ _ routine _ _ first rest =>
    simp [forest,Forest.calls,sites_call,schedule,routine,first,rest]
  | access _ op _ _ _ tail =>
    simp only [forest,Forest.calls,sites_call,schedule,op.primitive,Calls.count,Nat.add_zero,tail]
    rfl

/-- The clock positions are distinct, increasing and internal to this invocation,
    even if other workers run between any two of its instructions. -/
theorem Program.schedule_order (code : Program node h last) (th : Thread) (i : Nat) :
    (code.schedule th i).Pairwise (fun a b => a.index < b.index) ∧
    ∀ x ∈ code.schedule th i, i ≤ x.index ∧ x.index+1 < i+(code.forest.calls th).count := by
  induction code generalizing i with
  | done => simp [schedule]
  | call site c flow edge routine body next ihBody ihNext =>
    obtain ⟨ordered₁,bound₁⟩ := ihBody (i+1)
    obtain ⟨ordered₂,bound₂⟩ := ihNext (i+1+(body.forest.calls th).count+1)
    simp only [schedule,List.pairwise_append,ordered₁,ordered₂,true_and,forest,Forest.calls,Calls.count,
      Invocations.Exit.score,Option.toList_none,List.length_nil,Nat.add_zero]
    refine ⟨fun a ha b hb => ?_,fun x hx => ?_⟩
    · have := (bound₁ a ha).2
      have := (bound₂ b hb).1
      omega
    · rcases List.mem_append.mp hx with l | r
      · obtain ⟨_,_⟩ := bound₁ x l; omega
      · obtain ⟨_,_⟩ := bound₂ x r; omega
  | access site op flow edge next ih =>
    obtain ⟨ordered,bound⟩ := ih (i+2)
    simp only [schedule,List.pairwise_cons,ordered,and_true,forest,Forest.calls,Calls.count,
      Invocations.Exit.score,Option.toList_none,List.length_nil,Nat.add_zero]
    refine ⟨fun a ha => ?_,fun x hx => ?_⟩
    · have := (bound a ha).1; omega
    · rcases List.mem_cons.mp hx with rfl | tail
      · dsimp only; constructor <;> omega
      · obtain ⟨_,_⟩ := bound x tail; omega

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls
