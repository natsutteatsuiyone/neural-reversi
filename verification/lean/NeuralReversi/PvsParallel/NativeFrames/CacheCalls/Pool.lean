module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Registers

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Visit)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor Activation group_workers)
open EndgameSearch.CacheHistory.Payloads (Values completeVisit pendingWorker)
open NativeFrameCache (project)

abbrev Item (p : Pool) := (w : Worker p) × Cursor w

/-- A complete invocation has an observed return time. An outstanding
    invocation has an observed opening, but its unexecuted tail may finish later. -/
structure Plans (p : Pool) where
  finished : Visit p → Block p
  pending : (w : Worker p) → (s : Cursor w) → Activation w → Block p
  finished_at : ∀ v, (finished v).window = ⟨v.worker,v.first,v.last⟩
  pending_at : ∀ w s f, (pending w s f).window.worker = w.ordinal ∧ (pending w s f).window.first = f.first

def Plans.openBlock (plan : Plans p) (s : Item p) : Option (Block p) :=
  (s.2.stack.find? (fun f => f.node == .boundary)).map (plan.pending s.1 s.2)

def Plans.workerBlocks (plan : Plans p) (s : Item p) :=
  s.2.completed.map plan.finished ++ (plan.openBlock s).toList

def Plans.blocks (plan : Plans p) (c : EndgameSearch.CacheHistory.NativeTrace.Capture p S) :=
  c.workers.flatMap plan.workerBlocks

private theorem Plans.local_order (plan : Plans p) (w : Worker p) (s : Cursor w) (stream : List (EndgameSearch.CacheHistory.NativeTrace.Event S))
    (parsed : EndgameSearch.CacheHistory.NativeTrace.run w stream = some s) :
    (plan.workerBlocks ⟨w,s⟩).Pairwise Block.earlier ∧ ∀ b ∈ plan.workerBlocks ⟨w,s⟩, b.window.worker = w.ordinal := by
  obtain ⟨ordered,owner⟩ := EndgameSearch.CacheHistory.NativeTrace.run_ordered w stream parsed
  have prior := Block.visit_order ⟨s.completed,ordered⟩ plan.finished plan.finished_at
  unfold workerBlocks openBlock
  cases found : s.stack.find? (fun f => f.node == .boundary) with
  | none =>
    simp only [Option.map_none,Option.toList_none,List.append_nil,List.mem_map]
    exact ⟨prior,fun b ⟨v,hv,eq⟩ => eq ▸ plan.finished_at v ▸ owner v hv⟩
  | some f =>
    have below := List.mem_of_find?_eq_some found
    have tag : f.node = .boundary := by simpa using List.find?_some found
    simp only [Option.map_some,Option.toList_some,List.pairwise_append,List.pairwise_singleton,true_and,
      List.mem_map,List.mem_append,List.mem_singleton]
    refine ⟨⟨prior,?_⟩,?_⟩
    · intro a ⟨v,hv,eq⟩ b same _
      subst a b
      rw [plan.finished_at,(plan.pending_at w s f).2]
      exact EndgameSearch.CacheHistory.NativeTrace.pending_after w stream parsed f below tag v hv
    · intro b present
      rcases present with ⟨v,hv,rfl⟩ | rfl
      · rw [plan.finished_at]; exact owner v hv
      · exact (plan.pending_at w s f).1

/-- Local replays can be proved independently for any unrelated register
    values. The decoder receives one register file for the actual worker pool,
    including all open invocations at the end of this stream. -/
theorem Plans.capture (plan : Plans p) {frames : Capture p} (native : NativeFrames.decode p inputs stream = some frames)
    (visits : ∀ v ∈ frames.timeline.visits, ∀ other, ∃ a,
      completeVisit p frames.timeline mc fc ((plan.finished v).install other) (project stream) v = some a)
    (partialRuns : ∀ (w : Worker p) s, EndgameSearch.CacheHistory.NativeTrace.run w (project stream) = some s →
      ∀ f, s.stack.find? (fun a => a.node == .boundary) = some f → ∀ other, ∃ a,
        pendingWorker p frames.timeline mc fc ((plan.pending w s f).install other) (project stream) ⟨w,s⟩ = some a) :
    ∃ values out, NativeFrameCache.decode p mc fc inputs values stream = some out ∧ out.physical = frames := by
  obtain ⟨c,parsed,timeline,_⟩ := NativeFrameCache.projected_capture native
  obtain ⟨_,complete,ran⟩ := EndgameSearch.CacheHistory.NativeTrace.decode_keeps _ _ parsed
  have ordered := (group_workers p (project stream) plan.workerBlocks (fun b => b.window.worker) Block.earlier
    (fun _ _ ne eq => (ne eq).elim) (fun w s => plan.local_order w s _) _ List.nodup_range ran).1
  let all_ := plan.blocks c
  let base : Values := fun _ _ => none
  let values := Block.registers all_ base
  have states := EndgameSearch.CacheHistory.Payloads.capture_workers _ _ _ parsed
  obtain ⟨cache,audit,_⟩ := EndgameSearch.CacheHistory.Payloads.decode_of_visits p _ c values parsed (by
    intro v mem
    obtain ⟨s,worker_,completed⟩ := List.mem_flatMap.mp (complete ▸ mem)
    have member : plan.finished v ∈ all_ := List.mem_flatMap.mpr ⟨s,worker_,
      List.mem_append_left _ (List.mem_map.mpr ⟨_,completed,rfl⟩)⟩
    obtain ⟨a,ok⟩ := visits v (timeline ▸ mem) values
    exact ⟨a,timeline ▸ Block.install_registers all_ ordered base _ member ▸ ok⟩) (by
    intro s member
    cases found : s.2.stack.find? (fun f => f.node == .boundary) with
    | none => exact ⟨none,by simp [pendingWorker,found]⟩
    | some f =>
      have block_ : plan.pending s.1 s.2 f ∈ all_ := List.mem_flatMap.mpr ⟨s,member,List.mem_append_right _ (by simp [openBlock,found])⟩
      obtain ⟨a,ok⟩ := partialRuns s.1 s.2 (states s member) f found values
      rw [show (plan.pending s.1 s.2 f).install values = values from Block.install_registers all_ ordered base _ block_] at ok
      exact ⟨a,timeline ▸ ok⟩)
  have collected : NativeFrameCache.decode p mc fc inputs values stream = some ⟨frames,cache⟩ := by
    simp [NativeFrameCache.decode,native,audit]
  exact ⟨_,_,collected,Option.some.inj ((NativeFrameCache.decode_physical collected).symm.trans native)⟩

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls
