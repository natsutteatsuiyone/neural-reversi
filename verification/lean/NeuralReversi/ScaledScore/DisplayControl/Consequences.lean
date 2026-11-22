import NeuralReversi.ScaledScore.DisplayControl.Properties

namespace NeuralReversi.ScaledScore.DisplayControl
open NeuralReversi.ScaledScore (magnitudeChunks displayChunks)
variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}
variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem entries_append (a b : List (Control Frame Receiver Ref Payload Stop)) :
    entries (a ++ b) = entries a ++ entries b := by
  induction a with
  | nil => rfl
  | cons action tail ih => cases action <;> simp only [List.cons_append,entries,ih]

theorem success_texts {scope : Scope Frame Receiver} {start chunks controls after remaining}
    (h : SuccessPrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      scope start chunks controls after remaining) :
    (entries controls).map Invocation.text ++ remaining = chunks := by
  induction h with
  | empty => rfl
  | next ordinal chunk rest inv controls after remaining matching tail ih =>
    simp only [entries,List.map_cons,List.cons_append,matching.2.2]
    exact congrArg (List.cons chunk) ih

theorem success_ordinal {scope : Scope Frame Receiver} {start chunks controls after remaining}
    (h : SuccessPrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      scope start chunks controls after remaining) : start + (entries controls).length = after := by
  induction h with
  | empty => simp [entries]
  | next ordinal chunk rest inv controls after remaining matching tail ih =>
    simp only [entries,List.length_cons]
    omega

theorem live_texts {scope : Scope Frame Receiver} {chunks controls state}
    (h : LivePrefix (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) :
    ∃ remaining, (entries controls).map Invocation.text ++ remaining = chunks := by
  cases h with
  | ready controls ordinal remaining hp => exact ⟨remaining,success_texts hp⟩
  | running controls ordinal chunk rest inv ctx loc hp matching =>
    refine ⟨rest,?_⟩
    simpa only [entries_append,entries,List.map_append,List.map_cons,List.map_nil,
      List.append_assoc,List.cons_append,List.nil_append,matching.2.2] using success_texts hp

theorem history_texts {scope : Scope Frame Receiver} {chunks controls state}
    (h : History (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) :
    ∃ remaining, (entries controls).map Invocation.text ++ remaining = chunks := by
  cases h with
  | live controls state hp => exact live_texts hp
  | returned controls ordinal hp =>
    refine ⟨[],?_⟩
    simpa only [entries_append,entries,List.append_nil] using success_texts hp
  | error controls ordinal chunk rest inv hp matching =>
    refine ⟨rest,?_⟩
    simpa only [entries_append,entries,List.map_append,List.map_cons,List.map_nil,
      List.append_assoc,List.cons_append,List.nil_append,matching.2.2] using success_texts hp
  | unwind controls ordinal chunk rest inv payload hp matching =>
    refine ⟨rest,?_⟩
    simpa only [entries_append,entries,List.map_append,List.map_cons,List.map_nil,
      List.append_assoc,List.cons_append,List.nil_append,matching.2.2] using success_texts hp
  | callbackStop controls ordinal chunk rest inv reason hp matching =>
    refine ⟨rest,?_⟩
    simpa only [entries_append,entries,List.map_append,List.map_cons,List.map_nil,
      List.append_assoc,List.cons_append,List.nil_append,matching.2.2] using success_texts hp
  | ambientStop controls state reason hp => exact live_texts hp

theorem history_entry_bound {scope : Scope Frame Receiver} {chunks controls state}
    (h : History (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) :
    (entries controls).length ≤ chunks.length := by
  obtain ⟨rest,heq⟩ := history_texts h
  have hn := congrArg List.length heq
  simp only [List.length_append,List.length_map] at hn
  omega

theorem history_scope {scope : Scope Frame Receiver} {chunks controls state}
    (h : History (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) scope chunks controls state) : stateScope state = scope := by
  cases h with
  | live controls state hp => exact live_scope hp
  | returned => rfl
  | error => rfl
  | unwind => rfl
  | callbackStop => rfl
  | ambientStop => rfl

theorem magnitude_call_bound (cents : Nat) : (magnitudeChunks cents).length ≤ 3 := by
  unfold magnitudeChunks
  split
  · simp
  · split
    · simp
    · split <;> simp

theorem display_call_bound (raw : NeuralReversi.ScaledScore.Word) : (displayChunks raw).length ≤ 4 := by
  have hm := magnitude_call_bound (NeuralReversi.ScaledScore.displayCents raw)
  unfold displayChunks
  rw [List.length_append]
  split <;> simp only [List.length_cons,List.length_nil] <;> omega

theorem outer_call_bound (scope : Scope Frame Receiver) (world : World) {labels b}
    (trace : Trace oracle (initialState scope,world) labels b) :
    (entries (projectControl labels)).length ≤ 4 := by
  exact Nat.le_trans (history_entry_bound (initial_history oracle scope world trace))
    (display_call_bound scope.raw)

theorem final_scope_preserved (scope : Scope Frame Receiver) (world : World) {labels b}
    (trace : Trace oracle (initialState scope,world) labels b) : stateScope b.1 = scope :=
  history_scope (initial_history oracle scope world trace)

def rank : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Nat
  | .ready _ _ remaining => 2 * remaining.length + 1
  | .external _ remaining _ _ => 2 * remaining.length + 2
  | .terminal _ _ => 0

def charge : Label Frame Receiver Ref Payload Stop Child Event → Nat
  | .control _ | .completion _ _ _ => 1
  | _ => 0

theorem resume_rank (inv : Invocation Frame Receiver Ref) (rest : List String)
    (reply : Reply Payload Stop) (ctx : Context) (loc : Local) :
    rank (Context:=Context) (Local:=Local) (resume inv rest reply) + 1 ≤
      rank (Payload:=Payload) (Stop:=Stop) (.external inv rest ctx loc) := by
  cases reply <;> simp [resume,rank]

theorem administrative_rank
    (initial : Scope Frame Receiver → Nat → String → World → SliceRef Ref × Context × Local)
    (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local))
    (world : World) (action next)
    (h : administrativeNext initial state world = some (action,next)) :
    rank next + 1 ≤ rank state := by
  have spec := (administrative_exact initial state world action next).mp h
  cases spec <;> simp [rank] <;> omega

theorem step_rank {a label b} (h : Step oracle a label b) : rank b.1 + charge label ≤ rank a.1 := by
  cases h with
  | administrative state world action next h => exact administrative_rank _ _ _ _ _ h
  | complete inv ctx loc rest world reply event world' h => exact resume_rank inv rest reply ctx loc
  | internal => simp [rank,charge]
  | environment => simp [charge]
  | environmentStop => simp [rank,charge]

theorem trace_rank {a labels b} (h : Trace oracle a labels b) :
    rank b.1 + (labels.map charge).sum ≤ rank a.1 := by
  induction h with
  | refl => simp
  | cons step tail ih =>
    have hs := step_rank oracle step
    simp only [List.map_cons,List.sum_cons]
    omega

theorem charged_bound (scope : Scope Frame Receiver) (world : World) {labels b}
    (trace : Trace oracle (initialState scope,world) labels b) : (labels.map charge).sum ≤ 9 := by
  have hr := trace_rank oracle trace
  have hc := display_call_bound scope.raw
  simp only [initialState,rank] at hr
  omega

def sourceReady : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Bool
  | .ready _ _ _ => true
  | _ => false

theorem source_progress (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) (world : World)
    (ready : sourceReady state = true) :
    ∃ action next, administrativeNext oracle.initial state world = some (action,next) ∧
      Step oracle (state,world) (.control action) (next,world) ∧ rank next < rank state := by
  have available : ∃ action next, administrativeNext oracle.initial state world = some (action,next) := by
    cases state with
    | ready scope ordinal remaining =>
      cases remaining <;> exact ⟨_,_,rfl⟩
    | external => simp [sourceReady] at ready
    | terminal => simp [sourceReady] at ready
  obtain ⟨action,next,h⟩ := available
  refine ⟨action,next,h,.administrative state world action next h,?_⟩
  have hr := administrative_rank oracle.initial state world action next h
  omega

end NeuralReversi.ScaledScore.DisplayControl
