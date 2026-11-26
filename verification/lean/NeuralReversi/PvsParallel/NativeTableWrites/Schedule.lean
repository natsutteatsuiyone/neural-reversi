import NeuralReversi.PvsParallel.NativeTableWrites.Memory

namespace NeuralReversi.PvsParallel.NativeTableWrites.Journal

open TranspositionTable

private def pcOf : WriterMir.Atom → Pc
  | .cas .. => .cas
  | .fence .. => .fence
  | .store (.payload f) .. => .field f
  | .store .sequence .. => .finish

def Attempt.fetch (w : Attempt) (endian : Bool) (n : Nat) : Option Pc :=
  (w.program endian).1[n]?.map pcOf

private theorem Attempt.predecessor (w : Attempt) (endian : Bool) (atPc : w.fetch endian n = some pc)
    (lt : earlier.index < pc.index) : earlier.index < n ∧ w.fetch endian earlier.index = some earlier := by
  have bound : n < (w.program endian).1.length := by
    obtain ⟨_,found,_⟩ := Option.map_eq_some_iff.mp atPc
    exact (List.getElem?_eq_some_iff.mp found).1
  cases success : w.success <;> simp [Attempt.program,WriterMir.writeStable,success] at bound
  · obtain rfl : n = 0 := by omega
    simp [Attempt.fetch,Attempt.program,WriterMir.writeStable,success,pcOf] at atPc
    subst pc
    contradiction
  · have indices : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 := by omega
    rcases indices with rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Attempt.fetch,Attempt.program,WriterMir.writeStable,success,pcOf] at atPc <;> subst pc
    all_goals
      cases earlier with
      | field f => cases f <;> simp_all [Attempt.fetch,Attempt.program,WriterMir.writeStable,Pc.index,pcOf]
      | _ => simp_all [Attempt.fetch,Attempt.program,WriterMir.writeStable,Pc.index,pcOf]

inductive Turn where
  | localCall (worker : Nat) (event : NativeFrames.Event)
  | writer (invocation : Nat)

structure ScheduleState where
  next : Nat → Nat
  records : List Record

def issue (c : Calls) (endian : Bool) (s : ScheduleState) : Turn → ScheduleState
  | .localCall w e => {s with records := s.records ++ [.localCall w e]}
  | .writer k =>
    let out := ((c k).fetch endian (s.next k)).map (fun pc => Record.writer ⟨k,pc⟩)
    ⟨fun i => if i = k then s.next i + 1 else s.next i,s.records ++ out.toList⟩

def stateAfter (c : Calls) (endian : Bool) (turns : List Turn) :=
  turns.foldl (issue c endian) ⟨fun _ => 0,[]⟩

def schedule (c : Calls) (endian : Bool) (turns : List Turn) := (stateAfter c endian turns).records

private structure Reached (c : Calls) (endian : Bool) (s : ScheduleState) : Prop where
  seen : ∀ k n pc, n < s.next k → (c k).fetch endian n = some pc → Record.writer ⟨k,pc⟩ ∈ s.records
  prior : ∀ site, Record.writer site ∈ s.records → ∀ pc : Pc, pc.index < site.pc.index →
    Record.writer ⟨site.invocation,pc⟩ ∈ s.records

private theorem Reached.issue (old : Reached c endian s) (t : Turn) : Reached c endian (issue c endian s t) := by
  cases t with
  | localCall w e =>
    constructor
    · intro k n pc lt hit
      exact List.mem_append_left _ (old.seen k n pc lt hit)
    · intro site present pc lt
      have member : Record.writer site ∈ s.records := by simpa [Journal.issue] using present
      exact List.mem_append_left _ (old.prior site member pc lt)
  | writer k =>
    constructor
    · intro i n pc lt hit
      change n < (if i = k then s.next i + 1 else s.next i) at lt
      by_cases h : n < s.next i
      · exact List.mem_append_left _ (old.seen i n pc h hit)
      · split at lt <;> try contradiction
        rename_i eq
        subst i
        obtain rfl : n = s.next k := by omega
        simp [Journal.issue,hit]
    · intro site present pc lt
      rcases List.mem_append.mp present with before | now
      · exact List.mem_append_left _ (old.prior site before pc lt)
      · cases pos : (c k).fetch endian (s.next k) with
        | none => simp [pos] at now
        | some next =>
          simp [pos] at now
          subst site
          obtain ⟨bound,hit⟩ := (c k).predecessor endian pos lt
          exact List.mem_append_left _ (old.seen k _ pc bound hit)

private theorem reachedAfter (c : Calls) (endian : Bool) (turns : List Turn) :
    Reached c endian (stateAfter c endian turns) := by
  have reached (s : ScheduleState) (ok : Reached c endian s) : Reached c endian (turns.foldl (issue c endian) s) := by
    induction turns generalizing s with
    | nil => exact ok
    | cons t _ ih => exact ih _ (ok.issue t)
  exact reached ⟨fun _ => 0,[]⟩ ⟨by simp,by simp⟩

theorem schedule_earlier (c : Calls) (endian : Bool) (turns : List Turn) (site : Site)
    (member : Record.writer site ∈ schedule c endian turns) (pc : Pc) (earlier : pc.index < site.pc.index) :
    Record.writer ⟨site.invocation,pc⟩ ∈ schedule c endian turns :=
  (reachedAfter c endian turns).prior site member pc earlier

/-- At a copy boundary all writers which started on the source slot have run
    out of MIR instructions. Other slots may still have writers in flight. -/
def ScheduleState.idleAt (s : ScheduleState) (c : Calls) (endian : Bool) (a : FrameTableReads.Address) :=
  s.records.all fun
    | .writer ⟨k,.cas⟩ => (c k).address != a || ((c k).fetch endian (s.next k)).isNone
    | _ => true

theorem releaseBeforeCopy (c : Calls) (endian : Bool) (turns : List Turn) (k : Nat)
    (idle : (stateAfter c endian turns).idleAt c endian (c k).address = true)
    (cas : Record.writer ⟨k,.cas⟩ ∈ schedule c endian turns) (success : (c k).success = true) :
    Record.writer ⟨k,.finish⟩ ∈ schedule c endian turns := by
  let state := stateAfter c endian turns
  have closed := List.all_eq_true.mp idle _ cas
  have exhausted : ((c k).fetch endian (state.next k)).isNone = true := by simpa using closed
  have stop : ((c k).program endian).1.length ≤ state.next k := by
    simpa [Attempt.fetch] using exhausted
  have enough : 5 < state.next k := by
    simp [Attempt.program,WriterMir.writeStable,success] at stop
    omega
  apply (reachedAfter c endian turns).seen k 5 .finish enough
  simp [Attempt.fetch,Attempt.program,WriterMir.writeStable,success,pcOf]

/-- Scheduling an invocation selects its next instruction from the extracted
    MIR. Only the language's sequenced-before relation for issued instructions
    of that invocation is supplied by the memory execution. -/
def Atomics.ofSchedule {c : Calls} {endian : Bool} {turns : List Turn}
    (x : Atomics c endian (schedule c endian turns))
    (localOrder : ∀ k (a b : Pc), Record.writer ⟨k,a⟩ ∈ schedule c endian turns →
      Record.writer ⟨k,b⟩ ∈ schedule c endian turns → a.index < b.index →
      (x.memory (c k).address).sb ((c k).event a.index) ((c k).event b.index)) :
    Execution c endian (schedule c endian turns) where
  toAtomics := x
  earlier site member pc lt := by
    have earlier := schedule_earlier c endian turns site member pc lt
    exact ⟨earlier,localOrder _ _ _ earlier member lt⟩

end NeuralReversi.PvsParallel.NativeTableWrites.Journal
