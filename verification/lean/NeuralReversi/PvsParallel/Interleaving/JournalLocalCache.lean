module
public import NeuralReversi.PvsParallel.Interleaving.JournalBodies

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalLocalCache

open EndgameSearch.CacheHistory

/-- Calls that finished on this worker after the session began and before this
    particular boundary. Both limits are worker-local instruction numbers. -/
def during (visits : List (Visit p)) (worker opening stop : Nat) :=
  visits.filter (fun v => v.worker == worker && decide (opening ≤ v.last ∧ v.last < stop))

private theorem partition (visits : List (Visit p)) (ord : visits.Pairwise Earlier) (worker opening stop : Nat)
    (time : opening ≤ stop) :
    beforeStep visits worker stop = beforeStep visits worker opening ++ during visits worker opening stop := by
  induction visits with
  | nil => rfl
  | cons v vs ih =>
    obtain ⟨next,tail⟩ := List.pairwise_cons.mp ord
    have rest := ih tail
    by_cases mine : v.worker = worker
    · by_cases previous : v.last < opening
      · have already : v.last < stop := Nat.lt_of_lt_of_le previous time
        simp [beforeStep,during,mine,previous,already,Nat.not_le_of_gt previous] at rest ⊢
        exact rest
      · have none : beforeStep vs worker opening = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro g mem
          by_cases same : g.worker = worker
          · have lt := next g mem (mine.trans same.symm)
            have ended := g.inOrder
            simp [same,show ¬g.last < opening by omega]
          · simp [same]
        have after : opening ≤ v.last := Nat.le_of_not_gt previous
        unfold beforeStep at none
        by_cases finished : v.last < stop <;>
          simp [beforeStep,during,mine,previous,after,finished,none] at rest ⊢ <;> exact rest
    · simpa [beforeStep,during,mine] using rest

/-- Replaying the full pool lifetime and resuming the next search with the
    preceding worker's cache image give identical histories and cache arrays. -/
theorem resume (t : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (worker opening stop : Nat) (time : opening ≤ stop) :
    t.readAt mc fc worker stop = (during t.visits worker opening stop).foldl
      (fun state v => v.input.after mc fc p.address state) (t.readAt mc fc worker opening) := by
  unfold Timeline.readAt replay
  rw [partition t.visits t.ordered worker opening stop time,List.foldl_append]

/-- Use the same pool decoder as the depth-twelve and shallow frame checks.
    Its initial image for a late session is computed from earlier decoded calls. -/
theorem captured (es : Joint.Tape Op) (pool : Pool) (cap : NativeFrames.Capture pool)
    (_captured : JournalBodies.capture pool es = some cap) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (worker opening stop : Nat) (time : opening ≤ stop) :
    let warm := cap.timeline.readAt mc fc worker opening
    Good mc fc pool.width pool.sizes pool.key warm ∧ cap.timeline.readAt mc fc worker stop =
      (during cap.timeline.visits worker opening stop).foldl (fun state v => v.input.after mc fc pool.address state) warm :=
  ⟨replay_good ..,resume cap.timeline mc fc worker opening stop time⟩

/-- Normalize only the worker-local time stamps used by the cache replay.
    The board, cache inputs and worker identity stay tied to the original visit. -/
def shift (offset : Nat) (v : Visit p) : Visit p := {
  v with first := offset + v.first, last := offset + v.last,
         inOrder := Nat.add_le_add_left v.inOrder offset,
         scoredAt := v.scoredAt.map (offset + ·),
         scoreAfter := by
           intro t found
           obtain ⟨n,src,rfl⟩ := Option.map_eq_some_iff.mp found
           exact Nat.add_le_add_left (v.scoreAfter _ src) offset }

def offset (previous : Timeline p) :=
  ((previous.visits.map (fun v => max v.last (v.scoredAt.getD 0))).max?.getD 0) + 1

private theorem ended (previous : Timeline p) {v : Visit p} (mem : v ∈ previous.visits) : v.last < offset previous := by
  have member := List.mem_map_of_mem (f := fun v : Visit p => max v.last (v.scoredAt.getD 0)) mem
  have upper := List.le_max?_getD_of_mem (k := 0) member
  exact Nat.lt_succ_of_le (Nat.le_trans (Nat.le_max_left ..) upper)

/-- Cache-call histories of two consecutive searches on the same pool. Each
    source can number its local steps starting at zero. -/
def sessions (previous current : Timeline p) : Timeline p where
  visits := previous.visits ++ current.visits.map (shift (offset previous))
  ordered := by
    refine List.pairwise_append.mpr ⟨previous.ordered,?_,?_⟩
    · apply List.pairwise_map.mpr
      exact current.ordered.imp (fun h same => Nat.add_lt_add_left (h same) _)
    · intro a inPast b inCurrent _
      obtain ⟨v,_,rfl⟩ := List.mem_map.mp inCurrent
      exact Nat.lt_of_lt_of_le (ended previous inPast) (Nat.le_add_right ..)

def readSessions (previous current : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (worker step : Nat) :=
  (beforeStep current.visits worker step).foldl (fun s v => v.input.after mc fc p.address s)
    (replay mc fc p (byOwner previous.visits worker))

/-- All previous cache cells are reconstructed from this worker's past calls.
    This also holds if another worker uses the same numerical slot index. -/
theorem sessions_read (previous current : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (worker step : Nat) : (sessions previous current).readAt mc fc worker (offset previous + step) =
      readSessions previous current mc fc worker step := by
  have older : beforeStep previous.visits worker (offset previous + step) = byOwner previous.visits worker := by
    apply List.filter_congr
    intro v mem
    have lt := Nat.lt_of_lt_of_le (ended previous mem) (Nat.le_add_right (offset previous) step)
    simp [lt]
  have early : beforeStep ((sessions previous current).visits) worker (offset previous + step) =
      byOwner previous.visits worker ++ (beforeStep current.visits worker step).map (shift (offset previous)) := by
    unfold sessions beforeStep at *
    rw [List.filter_append,List.filter_map,older]
    simp [shift,Function.comp_def]
  simp only [Timeline.readAt,early,replay,readSessions,List.foldl_append,List.foldl_map,shift]

theorem sessions_good (previous current : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (worker step : Nat) : Good mc fc p.width p.sizes p.key (readSessions previous current mc fc worker step) := by
  rw [← sessions_read]
  exact replay_good ..

/-- Starting a new shared-table journal with the previous pool's call samples
    feeds exactly the same local-cache decoder as a single pool-lifetime tape. -/
theorem prefix_capture (pool : Pool) (past current : Joint.Tape Op)
    (ordinary : JournalBodies.noPoolHeader (past ++ current) = true) :
    JournalBodies.capture pool (.poolHistory pool.base pool.size (past.filterMap Joint.Event.frameSample) :: current) =
      NativeFrames.EntryStream.capture pool ((past ++ current).filterMap Joint.Event.frameSample) := by
  have both : JournalBodies.noPoolHeader past = true ∧ JournalBodies.noPoolHeader current = true := by
    simpa [JournalBodies.noPoolHeader] using ordinary
  simp [JournalBodies.capture,JournalBodies.cacheSamples,JournalBodies.poolHeader,both.2,List.filterMap_append,guard]

theorem resumed_capture (pool : Pool) (earlier : Joint.Tape Old) (current : Joint.Tape New)
    {joined : Joint.Tape New} (made : JournalBodies.resumePool pool earlier current = some joined) :
    JournalBodies.capture pool joined = (JournalBodies.cacheSamples pool earlier).bind fun before =>
      NativeFrames.EntryStream.capture pool (before ++ current.filterMap Joint.Event.frameSample) := by
  unfold JournalBodies.resumePool at made
  obtain ⟨past,seen,more⟩ := Option.bind_eq_some_iff.mp made
  by_cases clean : JournalBodies.noPoolHeader current = true
  · simp [guard,clean] at more
    subst joined
    rw [seen]
    simp [JournalBodies.capture,JournalBodies.cacheSamples,JournalBodies.poolHeader,clean,guard]
  · simp [guard,clean] at more
    contradiction

def Contract : Prop :=
  (∀ (Old New : Type) (p : Pool) (earlier : Joint.Tape Old) (current joined : Joint.Tape New),
    JournalBodies.resumePool p earlier current = some joined →
    JournalBodies.capture p joined = (JournalBodies.cacheSamples p earlier).bind fun before =>
      NativeFrames.EntryStream.capture p (before ++ current.filterMap Joint.Event.frameSample)) ∧
  (∀ (p : Pool) (previous current : Timeline p) mc fc worker step,
    (sessions previous current).readAt mc fc worker (offset previous + step) =
      readSessions previous current mc fc worker step ∧
    Good mc fc p.width p.sizes p.key (readSessions previous current mc fc worker step))

theorem carries : Contract :=
  ⟨fun _ _ p earlier current _ h => resumed_capture p earlier current h,
   fun _ previous current mc fc worker step =>
    ⟨sessions_read previous current mc fc worker step,sessions_good previous current mc fc worker step⟩⟩

end NeuralReversi.PvsParallel.Interleaving.JournalLocalCache
