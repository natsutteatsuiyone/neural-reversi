import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Segments

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

open FrameReturns.OwnerCalls (ChildReturn)
open Invocations (Calls Exit)
open EndgameSearch.CacheOwner (Resource)
open EndgameSearch.CacheHistory.NativeTrace (Worker)

variable {p : EndgameSearch.CacheHistory.Pool} {w : Worker p}

private theorem monotone {clock : Nat → Nat} (time : ∀ {a b : Nat}, a < b → clock a < clock b) {i j : Nat} (le : i ≤ j) :
    clock i ≤ clock j := by
  rcases Nat.eq_or_lt_of_le le with rfl | lt
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (time lt)

private theorem extend (run : Segment w inputs caller receiver owner lo hi events) (l : a ≤ lo) (h : hi ≤ z) :
    Segment w inputs caller receiver owner a z events := by
  intro s callerAt tick old
  obtain ⟨out,added,steps,stack,clock,rows,range⟩ := run s callerAt (Nat.le_trans tick l)
    (fun r mem => Nat.lt_of_lt_of_le (old r mem) l)
  exact ⟨out,added,steps,stack,Nat.le_trans clock h,rows,fun r mem =>
    ⟨Nat.le_trans l (range r mem).1,Nat.lt_of_lt_of_le (range r mem).2 h⟩⟩

/-- A sequence of instructions with the clock and event offsets of the issuing PVS. -/
def Body (w : Worker p) (inputs : Nat → Option Args) (receiver : Resource) (owner : Nat) (clock : Nat → Nat) (i : Nat) (calls : Calls) :=
  Segment w inputs .pvs receiver owner (clock i) (clock (i+calls.count)) (calls.events clock i)

theorem Body.nil : Body w inputs receiver owner clock i .nil := by
  exact Segment.nil (Nat.le_refl _)

private theorem wrap_count (ss : List SideCall) (tail : Calls) :
    (wrap ss tail).count = (ss.map SideCall.count).sum + tail.count := by
  induction ss with
  | nil => simp [wrap]
  | cons a ss ih =>
    simp only [wrap,Calls.count,ih,List.map_cons,List.sum_cons,SideCall.count]
    omega

private theorem wrap_events (ss : List SideCall) (tail : Calls) (clock : Nat → Nat) (i : Nat) :
    (wrap ss tail).events clock i = (wrap ss .nil).events clock i ++
      tail.events clock (i+(ss.map SideCall.count).sum) := by
  induction ss generalizing i with
  | nil => simp [wrap,Calls.events]
  | cons a ss ih =>
    let j := i+1+a.beforeScore.count+a.exit.score.toList.length+a.afterScore.count+1
    have eq : j+(ss.map SideCall.count).sum = i+((a::ss).map SideCall.count).sum := by
      simp only [j,List.map_cons,List.sum_cons,SideCall.count]; omega
    simp only [wrap,Calls.events]
    rw [ih j,eq]
    simp only [List.append_assoc,j]

private theorem wrap_body (time : ∀ {a b : Nat}, a < b → clock a < clock b) (ss : List SideCall)
    (first : Body w inputs receiver owner clock i (wrap ss .nil))
    (rest : Body w inputs receiver owner clock (i+(ss.map SideCall.count).sum) tail) :
    Body w inputs receiver owner clock i (wrap ss tail) := by
  simp only [Body,wrap_count,Calls.count,Nat.add_zero] at first
  unfold Body at rest ⊢
  rw [wrap_events,wrap_count,← Nat.add_assoc]
  exact first.append rest (monotone time (by omega)) (monotone time (by omega))

/-- The enclosing MIR loop needs only the side calls and the interior of each
    child. Its own entry, score, return and continuation are constructed here. -/
def Interiors (w : Worker p) (inputs : Nat → Option Args) (receiver : Resource) (owner : Nat) (clock : Nat → Nat)
    (sites : Nat → Site) (after : List SideCall) (k i : Nat) : List ChildReturn → Prop
  | [] => Body w inputs receiver owner clock i (wrap after .nil)
  | _::cs =>
    let s := sites k
    let start := i+(s.preceding.map SideCall.count).sum
    let mark := start+1+s.beforeScore.count
    let finish := mark+1+s.afterScore.count
    Body w inputs receiver owner clock i (wrap s.preceding .nil) ∧
      Body w inputs receiver (clock start) clock (start+1) s.beforeScore ∧
      Body w inputs receiver (clock start) clock (mark+1) s.afterScore ∧
      Interiors w inputs receiver owner clock sites after (k+1) (finish+1) cs

theorem issue_body (op : Nat → ChildReturn → Role) (receiver : Resource) (sites : Nat → Site) (after : List SideCall)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (k i : Nat) (rows : List ChildReturn)
    (valid : ∀ c ∈ rows, Admissible c) (fits : ∀ c ∈ rows, Fits c)
    (registers : ∀ x ∈ entries sites clock k i rows, inputs x.1 = some x.2)
    (nested : Interiors w inputs receiver owner clock sites after k i rows) :
    Body w inputs receiver owner clock i (issue op receiver sites after k rows) := by
  induction rows generalizing i k with
  | nil => exact nested
  | cons c cs ih =>
    let s := sites k
    let start := i+(s.preceding.map SideCall.count).sum
    let mark := start+1+s.beforeScore.count
    let finish := mark+1+s.afterScore.count
    let tail := issue op receiver sites after (k+1) cs
    obtain ⟨side,pre,post,more⟩ := nested
    have rest : Body w inputs receiver owner clock (finish+1) tail :=
      ih (k+1) (finish+1) (fun a mem => valid _ (List.mem_cons_of_mem c mem))
        (fun a mem => fits _ (List.mem_cons_of_mem c mem)) (fun x mem => registers x (List.mem_cons_of_mem _ mem)) more
    have reg : inputs (clock start) = some (params c) := registers _ List.mem_cons_self
    have admissible : Admissible (Returns.argsFields (params c) (value c).raw.toInt) :=
      (congrArg Admissible (fits c List.mem_cons_self)).mpr (valid c List.mem_cons_self)
    have preRun : Segment w inputs .pvs receiver (clock start) (clock start+1) (clock mark)
        (s.beforeScore.events clock (start+1)) :=
      extend pre (time (Nat.lt_succ_self start)) (Nat.le_refl _)
    have postRun : Segment w inputs .pvs receiver (clock start) (clock mark+1) (clock finish)
        (s.afterScore.events clock (mark+1)) :=
      extend post (time (Nat.lt_succ_self mark)) (Nat.le_refl _)
    have first := Segment.normal (owner:=owner) (forwards (op k c)) (params c) (value c) admissible rfl reg preRun postRun
      (time (by dsimp [mark]; omega)) (time (by dsimp [finish]; omega))
    have both := (extend first (Nat.le_refl _) (time (Nat.lt_succ_self finish))).append rest
      (monotone time (by dsimp [finish,mark]; omega)) (monotone time (by omega))
    apply wrap_body time _ side
    unfold Body
    simp only [Calls.events,Calls.count,Exit.node,Exit.score,Exit.event,Option.toList_some,Option.map_some,List.length_cons,List.length_nil]
    change Segment w inputs .pvs receiver owner _ (clock (start+(1+s.beforeScore.count+1+s.afterScore.count+1+tail.count))) _
    have offset : start+(1+s.beforeScore.count+1+s.afterScore.count+1+tail.count) = finish+1+tail.count := by
      dsimp [finish,mark]; omega
    rw [offset]
    exact both

theorem Execution.body {r : Row} {w : Worker p} (worker : w.ordinal = r.frame.worker)
    (ex : Execution p inputs stream r.frame op rows) (valid : ∀ c ∈ rows, Admissible c)
    (nested : Interiors w (inputs w.ordinal) ex.scope.entry.received (ex.scope.clock ex.scope.opening) ex.scope.clock ex.sites ex.afterCalls
      0 (ex.scope.opening+1) rows) :
    Body w (inputs w.ordinal) ex.scope.entry.received (ex.scope.clock ex.scope.opening) ex.scope.clock (ex.scope.opening+1) ex.scope.body := by
  rw [ex.instructions]
  exact issue_body _ _ _ _ _ ex.scope.time _ _ _ valid ex.widths (worker ▸ ex.args) nested

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites
