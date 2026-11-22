import NeuralReversi.Board.ConstructionModel

namespace NeuralReversi.Board.Construction
open NeuralReversi
variable {P O Payload Stop Context Local World Event : Type}
variable (oracle : Oracle P O Payload Stop Context Local World Event)

def rank : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Nat
  | .readyP _ _ => 7
  | .readyO _ _ => 5
  | .check _ _ => 3
  | .readyCleanup _ _ => 5
  | .readyAssert _ _ => 2
  | .readyFatal _ _ => 2
  | .external (.intoPlayer _ _) _ _ => 6
  | .external (.intoOpponent _ _) _ _ => 4
  | .external (.cleanupOpponent _ _) _ _ => 4
  | .external (.assertion _ _ _ _) _ _ => 1
  | .external (.fatalCleanup _ _) _ _ => 1
  | .terminal _ => 0

def charge : Label P O Payload Stop Event → Nat
  | .control _ => 1
  | .completion _ _ => 1
  | _ => 0

theorem resume_rank (inv : InvocationType (P:=P) (O:=O) (Payload:=Payload)) (reply : ReplyType (Payload:=Payload) (Stop:=Stop))
    (ctx : Context) (loc : Local) (next : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (h : resume inv reply = some next) : rank next + 1 ≤ rank (Stop:=Stop) (.external inv ctx loc) := by
  cases inv <;> cases reply <;> simp [resume] at h <;> subst next <;> simp [rank]

theorem administrative_rank (initial : Invocation P O Payload → World → Context × Local)
    (state : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (w : World) (action next) (h : administrativeNext initial state w = some (action,next)) :
    rank next + 1 ≤ rank state := by
  cases state <;> simp only [administrativeNext] at h
  all_goals try {cases h}
  all_goals try {split at h <;> cases h <;> simp [rank]}
  all_goals simp only [Option.some.injEq,Prod.mk.injEq] at h
  all_goals rcases h with ⟨rfl,rfl⟩
  all_goals simp [rank]

theorem step_rank {a b label} (h : Step oracle a label b) : rank b.1 + charge label ≤ rank a.1 := by
  cases h with
  | administrative state w action next h => exact administrative_rank oracle.initial state w action next h
  | complete inv ctx loc w reply event w' next ho hr => exact resume_rank inv reply ctx loc next hr
  | internal inv ctx loc w event loc' w' h => cases inv <;> simp [rank,charge]
  | environment state w event w' h live => simp [charge]
  | environmentStop state w event reason w' h live => simp [rank, charge]

theorem trace_rank {a b labels} (h : Trace oracle a labels b) :
    rank b.1 + (labels.map charge).sum ≤ rank a.1 := by
  induction h with
  | refl => simp
  | cons step tail ih =>
    have hs := step_rank oracle step
    simp only [List.map_cons,List.sum_cons]
    omega

theorem administrative_bound (p : P) (o : O) (w : World) {b labels}
    (h : Trace oracle (.readyP p o,w) labels b) : (labels.map charge).sum ≤ 7 := by
  have hh := trace_rank oracle h
  simp only [rank] at hh
  omega

inductive Continuation : Invocation P O Payload → Reply Payload Stop →
    ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | playerReturn (p o bits) : Continuation (.intoPlayer p o) (.converted bits) (.readyO bits o)
  | playerUnwind (p o payload) : Continuation (.intoPlayer p o) (.unwinding payload) (.readyCleanup o payload)
  | opponentReturn (p o bits) : Continuation (.intoOpponent p o) (.converted bits) (.check p bits)
  | opponentUnwind (p o payload) : Continuation (.intoOpponent p o) (.unwinding payload) (.terminal (.unwound payload))
  | cleanupReturn (o original) : Continuation (.cleanupOpponent o original) .cleaned (.terminal (.unwound original))
  | cleanupUnwind (o original fresh) : Continuation (.cleanupOpponent o original) (.unwinding fresh) (.readyFatal original fresh)
  | assertionUnwind (p o message sourceOrigin payload) : Continuation (.assertion p o message sourceOrigin) (.unwinding payload) (.terminal (.unwound payload))
  | stop (inv reason) : Continuation inv (.stopped reason) (.terminal (.stopped reason))

theorem resume_exact (inv : Invocation P O Payload) (reply : Reply Payload Stop)
    (next : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) :
    resume inv reply = some next ↔ Continuation inv reply next := by
  cases inv <;> cases reply <;> constructor <;> intro h
  all_goals first | (cases h <;> rfl) | (simp [resume] at h; subst next; constructor)

inductive AdministrativeSpec (initial : Invocation P O Payload → World → Context × Local) :
    ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → World →
    Control P O Payload Stop → ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | player (p o w) : AdministrativeSpec initial (.readyP p o) w (.enter (.intoPlayer p o))
      (.external (.intoPlayer p o) (initial (.intoPlayer p o) w).1 (initial (.intoPlayer p o) w).2)
  | opponent (p o w) : AdministrativeSpec initial (.readyO p o) w (.enter (.intoOpponent p o))
      (.external (.intoOpponent p o) (initial (.intoOpponent p o) w).1 (initial (.intoOpponent p o) w).2)
  | cleanup (o original w) : AdministrativeSpec initial (.readyCleanup o original) w (.enter (.cleanupOpponent o original))
      (.external (.cleanupOpponent o original) (initial (.cleanupOpponent o original) w).1 (initial (.cleanupOpponent o original) w).2)
  | assertion (p o w) : AdministrativeSpec initial (.readyAssert p o) w (.enter (.assertion p o "player and opponent bitboards must not overlap" "f6fd54154c270f6db63ab78953663fe090841e26:crates/reversi-core/src/board.rs:64"))
      (.external (.assertion p o "player and opponent bitboards must not overlap" "f6fd54154c270f6db63ab78953663fe090841e26:crates/reversi-core/src/board.rs:64") (initial (.assertion p o "player and opponent bitboards must not overlap" "f6fd54154c270f6db63ab78953663fe090841e26:crates/reversi-core/src/board.rs:64") w).1 (initial (.assertion p o "player and opponent bitboards must not overlap" "f6fd54154c270f6db63ab78953663fe090841e26:crates/reversi-core/src/board.rs:64") w).2)
  | fatal (old fresh w) : AdministrativeSpec initial (.readyFatal old fresh) w (.enter (.fatalCleanup old fresh))
      (.external (.fatalCleanup old fresh) (initial (.fatalCleanup old fresh) w).1 (initial (.fatalCleanup old fresh) w).2)
  | clear (p o w) (disjoint : p &&& o = 0) : AdministrativeSpec initial (.check p o) w (.disjoint p o)
      (.terminal (.returned ⟨p,o⟩))
  | overlap (p o w) (overlap : p &&& o ≠ 0) : AdministrativeSpec initial (.check p o) w (.overlap p o)
      (.readyAssert p o)

theorem administrative_exact (initial : Invocation P O Payload → World → Context × Local)
    (state : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (w : World) (action next) :
    administrativeNext initial state w = some (action,next) ↔ AdministrativeSpec initial state w action next := by
  constructor
  · intro h
    cases state <;> simp only [administrativeNext] at h
    all_goals first | (cases h <;> constructor) | (split at h <;> cases h <;> constructor <;> assumption)
  · intro h
    cases h <;> simp_all [administrativeNext]

inductive SpecStep : Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    Label P O Payload Stop Event → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
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
    | complete inv ctx loc w reply event w' next ho hr => exact .complete inv ctx loc w reply event w' next ho ((resume_exact _ _ _).mp hr)
    | internal inv ctx loc w event loc' w' h => exact .internal inv ctx loc w event loc' w' h
    | environment state w event w' h live => exact .environment state w event w' h live
    | environmentStop state w event reason w' h live => exact .environmentStop state w event reason w' h live
  · intro h
    cases h with
    | administrative state w action next h => exact .administrative state w action next ((administrative_exact _ _ _ _ _).mpr h)
    | complete inv ctx loc w reply event w' next ho hr => exact .complete inv ctx loc w reply event w' next ho ((resume_exact _ _ _).mpr hr)
    | internal inv ctx loc w event loc' w' h => exact .internal inv ctx loc w event loc' w' h
    | environment state w event w' h live => exact .environment state w event w' h live
    | environmentStop state w event reason w' h live => exact .environmentStop state w event reason w' h live

def spent (labels : Nat → Label P O Payload Stop Event) : Nat → Nat
  | 0 => 0
  | n+1 => spent labels n + charge (labels n)

theorem infinite_prefix_bound
    (states : Nat → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label P O Payload Stop Event) (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∀ n, rank (states n).1 + spent labels n ≤ rank (states 0).1 := by
  intro n
  induction n with
  | zero => simp [spent]
  | succ n ih =>
    have hs := step_rank oracle (run n)
    simp only [spent]
    omega

def sourceReady : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Bool
  | .external _ _ _ | .terminal _ => false
  | _ => true

theorem source_progress (state : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (w : World) (ready : sourceReady state = true) :
    ∃ action next, administrativeNext oracle.initial state w = some (action,next) ∧
      Step oracle (state,w) (.control action) (next,w) ∧ rank next < rank state := by
  have existsNext : ∃ action next, administrativeNext oracle.initial state w = some (action,next) := by
    cases state <;> simp_all [sourceReady,administrativeNext]
    split <;> simp
  obtain ⟨action,next,hn⟩ := existsNext
  refine ⟨action,next,hn,.administrative state w action next hn,?_⟩
  have h := administrative_rank oracle.initial state w action next hn
  omega

theorem completion_progress (inv : Invocation P O Payload) (ctx : Context) (loc : Local) (w : World)
    (reply : Reply Payload Stop) (event : Event) (w' : World)
    (next : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (available : oracle.complete inv ctx loc w reply event w') (allowed : Continuation inv reply next) :
    Step oracle (.external inv ctx loc,w) (.completion (.finish inv reply) event) (next,w') :=
  .complete inv ctx loc w reply event w' next available ((resume_exact inv reply next).mpr allowed)

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
    (states : Nat → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label P O Payload Stop Event)
    (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∃ first, ∀ n, first ≤ n → charge (labels n) = 0 := by
  have monotone : ∀ n m, n ≤ m → spent labels n ≤ spent labels m := by
    intro n m after
    induction after with
    | refl => exact Nat.le_refl _
    | step _ ih =>
      simp only [spent]
      omega
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

def ControlContract : Prop :=
  ∀ (P O Payload Stop Context Local World Event : Type)
    (oracle : Oracle P O Payload Stop Context Local World Event) a label b,
    Step oracle a label b ↔ SpecStep oracle a label b

theorem controlContract_proven : ControlContract := by
  intro P O Payload Stop Context Local World Event oracle a label b
  exact strong_step_equivalence oracle

def ProgressContract : Prop :=
  ∀ (P O Payload Stop Context Local World Event : Type)
    (oracle : Oracle P O Payload Stop Context Local World Event),
    (∀ p o w b labels, Trace oracle (.readyP p o,w) labels b →
      (labels.map charge).sum ≤ 7) ∧
    (∀ state w, sourceReady state = true →
      ∃ action next, administrativeNext oracle.initial state w = some (action,next) ∧
        Step oracle (state,w) (.control action) (next,w) ∧ rank next < rank state) ∧
    (∀ (states : Nat → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop)
        (Context:=Context) (Local:=Local) (World:=World))
        (labels : Nat → Label P O Payload Stop Event),
      (∀ n, Step oracle (states n) (labels n) (states (n+1))) →
      ∃ first, ∀ n, first ≤ n → charge (labels n) = 0) ∧
    (∀ outcome w label next, ¬ Step oracle (.terminal outcome,w) label next)

theorem progressContract_proven : ProgressContract := by
  intro P O Payload Stop Context Local World Event oracle
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro p o w b labels trace
    exact administrative_bound oracle p o w trace
  · intro state w ready
    exact source_progress oracle state w ready
  · intro states labels run
    exact eventually_external oracle states labels run
  · intro outcome w label next
    exact terminal_no_step oracle outcome w label next

end NeuralReversi.Board.Construction
