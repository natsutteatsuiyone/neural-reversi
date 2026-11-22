import NeuralReversi.Board.TransformModel

namespace NeuralReversi.Board.Transform
open NeuralReversi
variable {Frame Fn Payload Stop Context Local World Event : Type}
variable (oracle : Oracle Frame Fn Payload Stop Context Local World Event)

def rank : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local) → Nat
  | .readyP _ _ _ _ => 4
  | .external (.player _ _ _ _) _ _ => 3
  | .readyO _ _ _ _ _ => 2
  | .external (.opponent _ _ _ _ _) _ _ => 1
  | .terminal _ => 0

def charge : Label Frame Fn Payload Stop Event → Nat
  | .control _ | .completion _ _ => 1
  | _ => 0

theorem resume_rank (inv : Invocation Frame Fn) (reply : Reply Payload Stop)
    (ctx : Context) (loc : Local) :
    rank (Context:=Context) (Local:=Local) (resume inv reply) + 1 ≤ rank (Payload:=Payload) (Stop:=Stop) (.external inv ctx loc) := by
  cases inv <;> cases reply <;> simp [resume,rank]

theorem administrative_rank (initial : Invocation Frame Fn → World → Context × Local)
    (state : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local)) (w : World) (action next)
    (h : administrativeNext initial state w = some (action,next)) :
    rank next + 1 ≤ rank state := by
  cases state <;> simp only [administrativeNext] at h
  all_goals try {cases h}
  all_goals simp only [Option.some.injEq,Prod.mk.injEq] at h
  all_goals rcases h with ⟨rfl,rfl⟩
  all_goals simp [rank]

theorem step_rank {a b label} (h : Step oracle a label b) :
    rank b.1 + charge label ≤ rank a.1 := by
  cases h with
  | administrative state w action next h => exact administrative_rank oracle.initial state w action next h
  | complete inv ctx loc w reply event w' h => exact resume_rank inv reply ctx loc
  | internal inv ctx loc w event loc' w' h => cases inv <;> simp [rank,charge]
  | environment state w event w' h live => simp [charge]
  | environmentStop state w event reason w' h live => simp [rank,charge]

theorem trace_rank {a b labels} (h : Trace oracle a labels b) :
    rank b.1 + (labels.map charge).sum ≤ rank a.1 := by
  induction h with
  | refl => simp
  | cons step tail ih =>
    have hs := step_rank oracle step
    simp only [List.map_cons,List.sum_cons]
    omega

theorem administrative_bound (frame : Frame) (fn : Fn) (p o : Bits) (w : World) {b labels}
    (h : Trace oracle (.readyP frame fn p o,w) labels b) : (labels.map charge).sum ≤ 4 := by
  have hh := trace_rank oracle h
  simp only [rank] at hh
  omega

inductive Continuation : Invocation Frame Fn → Reply Payload Stop →
    ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) → Prop
  | playerReturn (frame fn p o r) : Continuation (.player frame fn p o) (.returned r) (.readyO frame fn p o r)
  | opponentReturn (frame fn p o r s) : Continuation (.opponent frame fn p o r) (.returned s) (.terminal (.returned ⟨r,s⟩))
  | unwind (inv payload) : Continuation inv (.unwinding payload) (.terminal (.unwound payload))
  | stop (inv reason) : Continuation inv (.stopped reason) (.terminal (.stopped reason))

theorem resume_exact (inv : Invocation Frame Fn) (reply : Reply Payload Stop)
    (next : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local)) :
    resume inv reply = next ↔ Continuation inv reply next := by
  cases inv <;> cases reply <;> constructor <;> intro h
  all_goals first | (cases h <;> rfl) | (simp [resume] at h; subst next; constructor)

inductive AdministrativeSpec (initial : Invocation Frame Fn → World → Context × Local) :
    ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) → World → Control Frame Fn Payload Stop →
    ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) → Prop
  | player (frame fn p o w) : AdministrativeSpec initial (.readyP frame fn p o) w
      (.enter (.player frame fn p o))
      (.external (.player frame fn p o) (initial (.player frame fn p o) w).1 (initial (.player frame fn p o) w).2)
  | opponent (frame fn p o r w) : AdministrativeSpec initial (.readyO frame fn p o r) w
      (.enter (.opponent frame fn p o r))
      (.external (.opponent frame fn p o r) (initial (.opponent frame fn p o r) w).1 (initial (.opponent frame fn p o r) w).2)

theorem administrative_exact (initial : Invocation Frame Fn → World → Context × Local)
    (state : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local)) (w : World) (action next) :
    administrativeNext initial state w = some (action,next) ↔ AdministrativeSpec initial state w action next := by
  constructor
  · intro h
    cases state <;> simp only [administrativeNext] at h
    all_goals cases h <;> constructor
  · intro h
    cases h <;> simp_all [administrativeNext]

inductive SpecStep : Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local) (World:=World) → Label Frame Fn Payload Stop Event →
    Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World) → Prop
  | administrative (state w action next) (h : AdministrativeSpec oracle.initial state w action next) :
      SpecStep (state,w) (.control action) (next,w)
  | complete (inv ctx loc w reply event w' next)
      (ho : oracle.complete inv ctx loc w reply event w') (hr : Continuation inv reply next) :
      SpecStep (.external inv ctx loc,w) (.completion (.finish inv reply) event) (next,w')
  | internal (inv ctx loc w event loc' w') (h : oracle.internal inv ctx loc w event loc' w') :
      SpecStep (.external inv ctx loc,w) (.external event) (.external inv ctx loc',w')
  | environment (state w event w') (h : oracle.environment state w event w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      SpecStep (state,w) (.environment event) (state,w')
  | environmentStop (state w event reason w') (h : oracle.stopping state w event reason w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      SpecStep (state,w) (.environment event) (.terminal (.stopped reason),w')

theorem strong_step_equivalence {a label b} : Step oracle a label b ↔ SpecStep oracle a label b := by
  constructor
  · intro h
    cases h with
    | administrative state w action next h => exact .administrative state w action next ((administrative_exact _ _ _ _ _).mp h)
    | complete inv ctx loc w reply event w' h => exact .complete inv ctx loc w reply event w' _ h ((resume_exact _ _ _).mp rfl)
    | internal inv ctx loc w event loc' w' h => exact .internal inv ctx loc w event loc' w' h
    | environment state w event w' h live => exact .environment state w event w' h live
    | environmentStop state w event reason w' h live => exact .environmentStop state w event reason w' h live
  · intro h
    cases h with
    | administrative state w action next h => exact .administrative state w action next ((administrative_exact _ _ _ _ _).mpr h)
    | complete inv ctx loc w reply event w' next ho hr =>
      have eq := (resume_exact inv reply next).mpr hr
      subst next
      exact .complete inv ctx loc w reply event w' ho
    | internal inv ctx loc w event loc' w' h => exact .internal inv ctx loc w event loc' w' h
    | environment state w event w' h live => exact .environment state w event w' h live
    | environmentStop state w event reason w' h live => exact .environmentStop state w event reason w' h live

def spent (labels : Nat → Label Frame Fn Payload Stop Event) : Nat → Nat
  | 0 => 0
  | n+1 => spent labels n + charge (labels n)

theorem infinite_prefix_bound
    (states : Nat → Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label Frame Fn Payload Stop Event)
    (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∀ n, rank (states n).1 + spent labels n ≤ rank (states 0).1 := by
  intro n
  induction n with
  | zero => simp [spent]
  | succ n ih =>
    have hs := step_rank oracle (run n)
    simp only [spent]
    omega

def sourceReady : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local) → Bool
  | .readyP _ _ _ _ | .readyO _ _ _ _ _ => true
  | _ => false

theorem source_progress (state : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local)) (w : World) (ready : sourceReady state = true) :
    ∃ action next, administrativeNext oracle.initial state w = some (action,next) ∧
      Step oracle (state,w) (.control action) (next,w) ∧ rank next < rank state := by
  have existsNext : ∃ action next, administrativeNext oracle.initial state w = some (action,next) := by
    cases state <;> simp_all [sourceReady,administrativeNext]
  obtain ⟨action,next,hn⟩ := existsNext
  refine ⟨action,next,hn,.administrative state w action next hn,?_⟩
  have h := administrative_rank oracle.initial state w action next hn
  omega

theorem completion_progress (inv : Invocation Frame Fn) (ctx : Context) (loc : Local) (w : World)
    (reply : Reply Payload Stop) (event : Event) (w' : World)
    (available : oracle.complete inv ctx loc w reply event w') :
    Step oracle (.external inv ctx loc,w) (.completion (.finish inv reply) event) (resume inv reply,w') :=
  .complete inv ctx loc w reply event w' available

theorem terminal_no_step (outcome : Outcome Payload Stop) (w : World) (label next) :
    ¬ Step oracle (.terminal outcome,w) label next := by
  intro h
  cases h with
  | administrative state w action next h => simp [administrativeNext] at h
  | environment state w event w' h live => exact live outcome rfl
  | environmentStop state w event reason w' h live => exact live outcome rfl

private theorem bounded_eventually_constant (f : Nat → Nat)
    (monotone : ∀ n m, n ≤ m → f n ≤ f m) :
    ∀ bound, (∀ n, f n ≤ bound) → ∃ first, ∀ n, first ≤ n → f n = f first := by
  intro bound
  induction bound with
  | zero =>
    intro bounded
    refine ⟨0, ?_⟩
    intro n _
    have hn := bounded n
    have hz := bounded 0
    omega
  | succ bound ih =>
    intro bounded
    classical
    by_cases attained : ∃ n, f n = bound + 1
    · obtain ⟨first, exactBound⟩ := attained
      refine ⟨first, ?_⟩
      intro n after
      have lower := monotone first n after
      have upper := bounded n
      omega
    · apply ih
      intro n
      have upper := bounded n
      have unequal : f n ≠ bound + 1 := fun h => attained ⟨n, h⟩
      omega

theorem eventually_external
    (states : Nat → Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label Frame Fn Payload Stop Event)
    (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∃ first, ∀ n, first ≤ n → charge (labels n) = 0 := by
  have monotone : ∀ n m, n ≤ m → spent labels n ≤ spent labels m := by
    intro n m after
    induction after with
    | refl => exact Nat.le_refl _
    | step _ ih => simp only [spent]; omega
  have bounded : ∀ n, spent labels n ≤ rank (states 0).1 := by
    intro n
    have bound := infinite_prefix_bound oracle states labels run n
    omega
  obtain ⟨first, stable⟩ := bounded_eventually_constant (spent labels) monotone _ bounded
  refine ⟨first, ?_⟩
  intro n after
  have here := stable n after
  have next := stable (n + 1) (by omega)
  simp only [spent] at next
  omega

def projectOne : Label Frame Fn Payload Stop Event → List (Control Frame Fn Payload Stop)
  | .control action | .completion action _ => [action]
  | _ => []

def projectControl (labels : List (Label Frame Fn Payload Stop Event)) : List (Control Frame Fn Payload Stop) :=
  labels.flatMap projectOne

inductive LivePrefix (frame : Frame) (fn : Fn) (p o : Bits) :
    List (Control Frame Fn Payload Stop) → ModelState (Frame:=Frame) (Fn:=Fn)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | playerReady : LivePrefix frame fn p o [] (.readyP frame fn p o)
  | playerRunning (ctx loc) : LivePrefix frame fn p o
      [.enter (.player frame fn p o)] (.external (.player frame fn p o) ctx loc)
  | opponentReady (r) : LivePrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r)]
      (.readyO frame fn p o r)
  | opponentRunning (r ctx loc) : LivePrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r),
       .enter (.opponent frame fn p o r)] (.external (.opponent frame fn p o r) ctx loc)

inductive ExactPrefix (frame : Frame) (fn : Fn) (p o : Bits) :
    List (Control Frame Fn Payload Stop) → ModelState (Frame:=Frame) (Fn:=Fn)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | live {history state} (h : LivePrefix (Context:=Context) (Local:=Local) frame fn p o history state) :
      ExactPrefix frame fn p o history state
  | returned (r s) : ExactPrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r),
       .enter (.opponent frame fn p o r), .finish (.opponent frame fn p o r) (.returned s)]
      (.terminal (.returned ⟨r,s⟩))
  | playerUnwind (payload) : ExactPrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.unwinding payload)]
      (.terminal (.unwound payload))
  | opponentUnwind (r payload) : ExactPrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r),
       .enter (.opponent frame fn p o r), .finish (.opponent frame fn p o r) (.unwinding payload)]
      (.terminal (.unwound payload))
  | playerStop (reason) : ExactPrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.stopped reason)]
      (.terminal (.stopped reason))
  | opponentStop (r reason) : ExactPrefix frame fn p o
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r),
       .enter (.opponent frame fn p o r), .finish (.opponent frame fn p o r) (.stopped reason)]
      (.terminal (.stopped reason))
  | ambientStop {history state} (h : LivePrefix (Context:=Context) (Local:=Local) frame fn p o history state) (reason) :
      ExactPrefix frame fn p o history (.terminal (.stopped reason))

theorem exactPrefix_step {frame fn p o history a label b}
    (prefixOK : ExactPrefix frame fn p o history a.1)
    (step : Step oracle a label b) :
    ExactPrefix frame fn p o (history ++ projectOne label) b.1 := by
  rcases a with ⟨state,w⟩
  rcases b with ⟨next,w'⟩

  cases prefixOK with
  | live livePrefix =>
    cases livePrefix with
    | playerReady =>
      cases step with
      | administrative _ _ action next h =>
        simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
        rcases h with ⟨rfl,rfl⟩
        exact .live (.playerRunning _ _)
      | environment _ _ _ _ _ _ => exact .live .playerReady
      | environmentStop _ _ _ reason _ _ _ => exact .ambientStop .playerReady reason
    | playerRunning ctx loc =>
      cases step with
      | administrative _ _ _ _ h => simp [administrativeNext] at h
      | complete inv ctx loc w reply event w' h =>
        cases reply with
        | returned r => exact .live (.opponentReady r)
        | unwinding payload => exact .playerUnwind payload
        | stopped reason => exact .playerStop reason
      | internal _ _ _ _ _ _ _ _ => exact .live (.playerRunning _ _)
      | environment _ _ _ _ _ _ => exact .live (.playerRunning _ _)
      | environmentStop _ _ _ reason _ _ _ => exact .ambientStop (.playerRunning ctx loc) reason
    | opponentReady r =>
      cases step with
      | administrative _ _ action next h =>
        simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
        rcases h with ⟨rfl,rfl⟩
        exact .live (.opponentRunning r _ _)
      | environment _ _ _ _ _ _ => exact .live (.opponentReady r)
      | environmentStop _ _ _ reason _ _ _ => exact .ambientStop (.opponentReady r) reason
    | opponentRunning r ctx loc =>
      cases step with
      | administrative _ _ _ _ h => simp [administrativeNext] at h
      | complete inv ctx loc w reply event w' h =>
        cases reply with
        | returned s => exact .returned r s
        | unwinding payload => exact .opponentUnwind r payload
        | stopped reason => exact .opponentStop r reason
      | internal _ _ _ _ _ _ _ _ => exact .live (.opponentRunning r _ _)
      | environment _ _ _ _ _ _ => exact .live (.opponentRunning r _ _)
      | environmentStop _ _ _ reason _ _ _ => exact .ambientStop (.opponentRunning r ctx loc) reason
  | returned r s => exact False.elim (terminal_no_step oracle _ _ _ _ step)
  | playerUnwind payload => exact False.elim (terminal_no_step oracle _ _ _ _ step)
  | opponentUnwind r payload => exact False.elim (terminal_no_step oracle _ _ _ _ step)
  | playerStop reason => exact False.elim (terminal_no_step oracle _ _ _ _ step)
  | opponentStop r reason => exact False.elim (terminal_no_step oracle _ _ _ _ step)
  | ambientStop livePrefix reason => exact False.elim (terminal_no_step oracle _ _ _ _ step)

theorem exactPrefix_trace {frame fn p o history a labels b}
    (prefixOK : ExactPrefix frame fn p o history a.1) (trace : Trace oracle a labels b) :
    ExactPrefix frame fn p o (history ++ projectControl labels) b.1 := by
  induction trace generalizing history with
  | refl a => simpa [projectControl] using prefixOK
  | cons step tail ih =>
    have nextPrefix := exactPrefix_step oracle prefixOK step
    have result := ih nextPrefix
    simpa [projectControl,List.append_assoc] using result

theorem initial_trace_exact (frame : Frame) (fn : Fn) (p o : Bits) (w : World) {labels b}
    (trace : Trace oracle (.readyP frame fn p o,w) labels b) :
    ExactPrefix frame fn p o (projectControl labels) b.1 := by
  have result := exactPrefix_trace oracle (ExactPrefix.live LivePrefix.playerReady) trace
  simpa using result

theorem returned_trace_exact (frame : Frame) (fn : Fn) (p o : Bits) (w : World)
    (board : NeuralReversi.Board.Board) (w' : World) (labels)
    (trace : Trace oracle (.readyP frame fn p o,w) labels (.terminal (.returned board),w')) :
    projectControl labels =
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned board.player),
       .enter (.opponent frame fn p o board.player),
       .finish (.opponent frame fn p o board.player) (.returned board.opponent)] := by
  have grammar := initial_trace_exact oracle frame fn p o w trace
  generalize projectControl labels = history at grammar ⊢
  cases grammar with
  | live h => cases h
  | returned r s => rfl

theorem unwound_trace_exact (frame : Frame) (fn : Fn) (p o : Bits) (w : World)
    (payload : Payload) (w' : World) (labels)
    (trace : Trace oracle (.readyP frame fn p o,w) labels (.terminal (.unwound payload),w')) :
    projectControl labels =
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.unwinding payload)] ∨
    ∃ r, projectControl labels =
      [.enter (.player frame fn p o), .finish (.player frame fn p o) (.returned r),
       .enter (.opponent frame fn p o r), .finish (.opponent frame fn p o r) (.unwinding payload)] := by
  have grammar := initial_trace_exact oracle frame fn p o w trace
  generalize projectControl labels = history at grammar ⊢
  cases grammar with
  | live h => cases h
  | playerUnwind => exact Or.inl rfl
  | opponentUnwind r => exact Or.inr ⟨r,rfl⟩

def ControlContract : Prop :=
  ∀ (Frame Fn Payload Stop Context Local World Event : Type)
    (oracle : Oracle Frame Fn Payload Stop Context Local World Event) a label b,
    Step oracle a label b ↔ SpecStep oracle a label b

theorem controlContract_proven : ControlContract := by
  intro Frame Fn Payload Stop Context Local World Event oracle a label b
  exact strong_step_equivalence oracle

def ProgressContract : Prop :=
  ∀ (Frame Fn Payload Stop Context Local World Event : Type)
    (oracle : Oracle Frame Fn Payload Stop Context Local World Event),
    (∀ frame fn p o w b labels, Trace oracle (.readyP frame fn p o,w) labels b →
      (labels.map charge).sum ≤ 4) ∧
    (∀ state w, sourceReady state = true →
      ∃ action next, administrativeNext oracle.initial state w = some (action,next) ∧
        Step oracle (state,w) (.control action) (next,w) ∧ rank next < rank state) ∧
    (∀ (states : Nat → Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
        (Context:=Context) (Local:=Local) (World:=World))
        (labels : Nat → Label Frame Fn Payload Stop Event),
      (∀ n, Step oracle (states n) (labels n) (states (n+1))) →
      ∃ first, ∀ n, first ≤ n → charge (labels n) = 0) ∧
    (∀ outcome w label next, ¬ Step oracle (.terminal outcome,w) label next)

theorem progressContract_proven : ProgressContract := by
  intro Frame Fn Payload Stop Context Local World Event oracle
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro frame fn p o w b labels trace
    exact administrative_bound oracle frame fn p o w trace
  · intro state w ready
    exact source_progress oracle state w ready
  · intro states labels run
    exact eventually_external oracle states labels run
  · intro outcome w label next
    exact terminal_no_step oracle outcome w label next

def TraceContract : Prop :=
  ∀ (Frame Fn Payload Stop Context Local World Event : Type)
    (oracle : Oracle Frame Fn Payload Stop Context Local World Event)
    frame fn p o w labels b,
    Trace oracle (.readyP frame fn p o,w) labels b →
      ExactPrefix frame fn p o (projectControl labels) b.1

theorem traceContract_proven : TraceContract := by
  intro Frame Fn Payload Stop Context Local World Event oracle frame fn p o w labels b trace
  exact initial_trace_exact oracle frame fn p o w trace

end NeuralReversi.Board.Transform
