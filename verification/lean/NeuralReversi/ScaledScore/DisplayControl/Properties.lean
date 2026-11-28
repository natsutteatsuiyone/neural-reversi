module
public import NeuralReversi.ScaledScore.DisplayControl.Spec

@[expose] public section

namespace NeuralReversi.ScaledScore.DisplayControl
open NeuralReversi.ScaledScore (magnitudeChunks displayChunks)
variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}
variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem byteLength_exact (inv : Invocation Frame Receiver Ref)
    (fits : inv.text.utf8ByteSize < 2^64) : inv.byteLength.toNat = inv.text.utf8ByteSize := by
  rw [Invocation.byteLength,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt fits

theorem resume_exact (inv : Invocation Frame Receiver Ref) (rest : List String)
    (reply : Reply Payload Stop) (next : ModelState (Frame:=Frame) (Receiver:=Receiver)
      (Ref:=Ref) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) :
    resume inv rest reply = next ↔ Continuation inv rest reply next := by
  cases reply <;> constructor <;> intro h
  all_goals first | (cases h <;> rfl) | (simp only [resume] at h; subst next; constructor)

theorem administrative_exact
    (initial : Scope Frame Receiver → Nat → String → World → SliceRef Ref × Context × Local)
    (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (world : World) (action next) :
    administrativeNext initial state world = some (action,next) ↔
      AdministrativeSpec initial state world action next := by
  constructor
  · intro h
    cases state with
    | ready scope ordinal remaining =>
      cases remaining with
      | nil =>
        simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
        rcases h with ⟨rfl,rfl⟩
        constructor
      | cons chunk rest =>
        simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
        rcases h with ⟨rfl,rfl⟩
        apply AdministrativeSpec.call
        rfl
    | external => simp [administrativeNext] at h
    | terminal => simp [administrativeNext] at h
  · intro h
    cases h with
    | done => rfl
    | call scope ordinal chunk rest world reference context localState snapshot =>
      simp only [administrativeNext,snapshot]

theorem strong_step_equivalence {a label b} : Step oracle a label b ↔ SpecStep oracle a label b := by
  constructor
  · intro h
    cases h with
    | administrative state world action next h =>
      exact .administrative state world action next ((administrative_exact _ _ _ _ _).mp h)
    | complete inv ctx loc rest world reply event world' h =>
      exact .complete inv ctx loc rest world reply event world' _ h ((resume_exact _ _ _ _).mp rfl)
    | internal inv ctx loc rest world event loc' world' h =>
      exact .internal inv ctx loc rest world event loc' world' h
    | environment state world event world' live h => exact .environment state world event world' live h
    | environmentStop state world event reason world' live h =>
      exact .environmentStop state world event reason world' live h
  · intro h
    cases h with
    | administrative state world action next h =>
      exact .administrative state world action next ((administrative_exact _ _ _ _ _).mpr h)
    | complete inv ctx loc rest world reply event world' next external continuation =>
      have heq := (resume_exact inv rest reply next).mpr continuation
      subst next
      exact .complete inv ctx loc rest world reply event world' external
    | internal inv ctx loc rest world event loc' world' h =>
      exact .internal inv ctx loc rest world event loc' world' h
    | environment state world event world' live h => exact .environment state world event world' live h
    | environmentStop state world event reason world' live h =>
      exact .environmentStop state world event reason world' live h

theorem terminal_no_step (scope : Scope Frame Receiver) (outcome : Outcome Payload Stop)
    (world : World) (label next) : ¬ Step oracle (.terminal scope outcome,world) label next := by
  intro h
  cases h with
  | administrative state world action next h => simp [administrativeNext] at h
  | environment state world event world' live h => exact live scope outcome rfl
  | environmentStop state world event reason world' live h => exact live scope outcome rfl

theorem success_snoc {scope : Scope Frame Receiver} {start chunks controls ordinal remaining}
    (prior : SuccessPrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      scope start chunks controls ordinal remaining) (chunk : String) (rest : List String)
    (same : remaining = chunk::rest)
    (inv : Invocation Frame Receiver Ref) (matching : Matches scope ordinal chunk inv) :
    SuccessPrefix scope start chunks (controls ++ [.enter inv,.finish inv .ok]) (ordinal+1) rest := by
  induction prior with
  | empty ordinal chunks =>
    subst chunks
    exact .next ordinal chunk rest inv [] (ordinal+1) rest matching (.empty _ _)
  | next start first tail earlier controls after remaining hm hp ih =>
    exact .next start first tail earlier _ _ _ hm (ih same matching)

theorem live_scope {scope : Scope Frame Receiver} {chunks controls state}
    (prior : LivePrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) : stateScope state = scope := by
  cases prior with
  | ready => rfl
  | running controls ordinal chunk rest inv context localState hp matching => exact matching.1

theorem live_not_terminal {scope : Scope Frame Receiver} {chunks controls state}
    (prior : LivePrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) : Live state := by
  cases prior <;> intro s o h <;> cases h

theorem step_history {scope : Scope Frame Receiver} {chunks controls a label b}
    (prior : History scope chunks controls a.1) (step : Step oracle a label b) :
    History scope chunks (controls ++ projectOne label) b.1 := by
  rcases a with ⟨state,world⟩
  rcases b with ⟨next,world'⟩
  change History scope chunks controls state at prior
  cases prior with
  | live controls state hp =>
    cases hp with
    | ready controls ordinal remaining hp =>
      cases step with
      | administrative state world action next h =>
        cases remaining with
        | nil =>
          simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
          rcases h with ⟨rfl,rfl⟩
          exact .returned controls ordinal hp
        | cons chunk rest =>
          simp only [administrativeNext,Option.some.injEq,Prod.mk.injEq] at h
          rcases h with ⟨rfl,rfl⟩
          exact .live _ _ (.running controls ordinal chunk rest _ _ _ hp ⟨rfl,rfl,rfl⟩)
      | environment state world event world' live h =>
        simpa only [projectOne,List.append_nil] using History.live _ _ (LivePrefix.ready controls ordinal remaining hp)
      | environmentStop state world event reason world' live h =>
        simpa only [projectOne,List.append_nil,stateScope] using
          History.ambientStop controls (.ready scope ordinal remaining) reason (LivePrefix.ready controls ordinal remaining hp)
    | running controls ordinal chunk rest inv ctx loc hp matching =>
      cases step with
      | administrative state world action next h => simp [administrativeNext] at h
      | complete inv ctx loc rest world reply event world' h =>
        cases reply with
        | ok =>
          have hs := success_snoc hp chunk rest rfl inv matching
          have hm := matching.1
          have hi := matching.2.1
          simpa only [projectOne,resume,List.append_assoc,List.cons_append,List.nil_append,hm,hi] using
            History.live _ _ (LivePrefix.ready _ (ordinal+1) rest hs)
        | error =>
          simpa only [projectOne,resume,List.append_assoc,List.cons_append,List.nil_append,matching.1] using
            History.error controls ordinal chunk rest inv hp matching
        | unwind payload =>
          simpa only [projectOne,resume,List.append_assoc,List.cons_append,List.nil_append,matching.1] using
            History.unwind controls ordinal chunk rest inv payload hp matching
        | stopped reason =>
          simpa only [projectOne,resume,List.append_assoc,List.cons_append,List.nil_append,matching.1] using
            History.callbackStop controls ordinal chunk rest inv reason hp matching
      | internal inv ctx loc rest world event loc' world' h =>
        simpa only [projectOne,List.append_nil] using
          History.live _ _ (LivePrefix.running controls ordinal chunk rest inv ctx loc' hp matching)
      | environment state world event world' live h =>
        simpa only [projectOne,List.append_nil] using
          History.live _ _ (LivePrefix.running controls ordinal chunk rest inv ctx loc hp matching)
      | environmentStop state world event reason world' live h =>
        simpa only [projectOne,List.append_nil,stateScope,matching.1] using
          History.ambientStop (controls ++ [.enter inv]) (.external inv rest ctx loc) reason
            (LivePrefix.running controls ordinal chunk rest inv ctx loc hp matching)
  | returned => exact False.elim ((terminal_no_step oracle _ _ _ _ _) step)
  | error => exact False.elim ((terminal_no_step oracle _ _ _ _ _) step)
  | unwind => exact False.elim ((terminal_no_step oracle _ _ _ _ _) step)
  | callbackStop => exact False.elim ((terminal_no_step oracle _ _ _ _ _) step)
  | ambientStop => exact False.elim ((terminal_no_step oracle _ _ _ _ _) step)

theorem trace_history {scope : Scope Frame Receiver} {chunks a labels b}
    (trace : Trace oracle a labels b) : ∀ controls,
    History scope chunks controls a.1 → History scope chunks (controls ++ projectControl labels) b.1 := by
  induction trace with
  | refl => intro controls h; simpa [projectControl] using h
  | cons step tail ih =>
    intro controls h
    have hn := step_history oracle h step
    have ht := ih _ hn
    simpa only [projectControl,List.flatMap_cons,List.append_assoc] using ht

theorem initial_history (scope : Scope Frame Receiver) (world : World) {labels b}
    (trace : Trace oracle (initialState scope,world) labels b) :
    History scope (displayChunks scope.raw) (projectControl labels) b.1 := by
  have initial : History scope (displayChunks scope.raw) []
      (initialState (Ref:=Ref) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) scope) :=
    .live [] _ (.ready [] 0 _ (.empty _ _))
  simpa only [List.nil_append] using trace_history oracle trace [] initial

end NeuralReversi.ScaledScore.DisplayControl
