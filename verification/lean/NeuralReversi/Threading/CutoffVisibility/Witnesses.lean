import NeuralReversi.Threading.CutoffVisibility

namespace NeuralReversi.Threading.CutoffVisibility.Witnesses

private def epochWrites (release rmw : Bool) : List EpochWrite :=
  [⟨⟨3, 0⟩, 0, false, false⟩, ⟨⟨0, 1⟩, 1, release, true⟩, ⟨⟨1, 0⟩, 2, true, rmw⟩]

private def flags (_ : Nat) : List FlagWrite :=
  [⟨⟨3, 1⟩, false⟩, ⟨⟨0, 0⟩, true⟩]

private def epochRead (release acquire rmw : Bool) : EpochRead (epochWrites release rmw) :=
  ⟨⟨2, 0⟩, ⟨2, by change 2 < 3; decide⟩, acquire⟩

private def staleFlag : FlagRead flags :=
  ⟨0, ⟨2, 1⟩, ⟨0, by decide⟩⟩

private def history (release acquire rmw : Bool) : History :=
  ⟨epochWrites release rmw, flags, [epochRead release acquire rmw], [staleFlag], fun _ _ => False⟩

private theorem epoch_cases (head : Fin (epochWrites release rmw).length) :
    head = ⟨0, by change 0 < 3; decide⟩ ∨
      head = ⟨1, by change 1 < 3; decide⟩ ∨ head = ⟨2, by change 2 < 3; decide⟩ := by
  have h := head.isLt
  change head.val < 3 at h
  have vals : head.val = 0 ∨ head.val = 1 ∨ head.val = 2 := by omega
  rcases vals with e | e | e
  · exact Or.inl (Fin.ext e)
  · exact Or.inr (Or.inl (Fin.ext e))
  · exact Or.inr (Or.inr (Fin.ext e))

private theorem flag_cases (stamp : Fin 2) :
    stamp = ⟨0, by decide⟩ ∨ stamp = ⟨1, by decide⟩ := by
  have h := stamp.isLt
  change stamp.val < 2 at h
  have vals : stamp.val = 0 ∨ stamp.val = 1 := by omega
  exact vals.imp Fin.ext Fin.ext

private theorem worker_closed (hist : History) (worker : Nat) (a b : Event)
    (closed : ∀ head (r : EpochRead hist.epochs), r ∈ hist.epochReads → r.acquire = true →
      InReleaseSequence hist head r.source → (hist.epochs.get head).event.worker ≠ worker)
    (noOther : ∀ a b, ¬hist.otherSync a b)
    (order : HappensBefore hist a b) : a.worker = worker → b.worker = worker := by
  induction order with
  | sequenced earlier => exact fun same => earlier.1.symm.trans same
  | synchronized head load present acquire seq =>
    exact fun same => False.elim (closed head load present acquire seq same)
  | other edge => exact fun _ => False.elim (noOther _ _ edge)
  | trans _ _ first last => exact fun same => last (first same)

private theorem reader_closed (historyOrder : HappensBefore (history release acquire rmw) a b) :
    a.worker = 2 → b.worker = 2 := by
  apply worker_closed _ _ _ _ _ (by intro a b; simp [history]) historyOrder
  intro head read present ordered inSequence
  rcases epoch_cases head with eq | eq | eq <;> rw [eq]
  · change 3 ≠ 2
    decide
  · change 0 ≠ 2
    decide
  · change 1 ≠ 2
    decide

private theorem marker_closed (weakened : release = false ∨ acquire = false ∨ rmw = false)
    (order : HappensBefore (history release acquire rmw) a b) : a.worker = 0 → b.worker = 0 := by
  apply worker_closed _ _ _ _ _ (by intro a b; simp [history]) order
  intro head load present acquireRead seq
  have same : load = epochRead release acquire rmw :=
    List.mem_singleton.mp present
  rcases epoch_cases head with eq | eq | eq
  · rw [eq]
    change 3 ≠ 0
    decide
  · rw [eq] at seq ⊢
    rcases weakened with releaseOff | acquireOff | chainBreak
    · have rel := seq.1
      change release = true at rel
      simp [releaseOff] at rel
    · rw [same] at acquireRead
      change acquire = true at acquireRead
      simp [acquireOff] at acquireRead
    · have later := seq.2.2 ⟨2, by change 2 < 3; decide⟩ (by change 1 < 2; decide) (by rw [same]; exact Nat.le_refl _)
      change rmw = true at later
      simp [chainBreak] at later
  · rw [eq]
    change 1 ≠ 0
    decide

private theorem stale_coherent (weakened : release = false ∨ acquire = false ∨ rmw = false) :
    Coherent (history release acquire rmw) := by
  constructor
  · intro a aIn b bIn _
    rw [List.mem_singleton.mp aIn, List.mem_singleton.mp bIn]
    exact Nat.le_refl _
  · intro r seen stamp ordered
    have only : r = staleFlag := List.mem_singleton.mp seen
    subst r
    rcases flag_cases stamp with eq | eq
    · rw [eq]
      exact Nat.zero_le _
    · rw [eq] at ordered
      have inconsistent := marker_closed weakened ordered rfl
      contradiction
  · intro r seen stamp ordered
    have only : r = staleFlag := List.mem_singleton.mp seen
    subst r
    rcases flag_cases stamp with eq | eq <;> rw [eq] at ordered
    all_goals
      have inconsistent := reader_closed ordered rfl
      simp [history, flags, staleFlag] at inconsistent

private theorem counter_correct : Counter (history true true true) := by
  constructor
  · intro i zero
    rw [show i = ⟨0, by decide⟩ from Fin.ext zero]
    rfl
  · intro i positive
    rcases epoch_cases i with eq | eq | eq
    · rw [eq] at positive
      contradiction
    · rw [eq]
      decide
    · rw [eq]
      decide

private def marked : Publication (history true true true) :=
  ⟨0, ⟨1, by decide⟩, ⟨1, by decide⟩, by decide, rfl, ⟨rfl, by decide⟩⟩

private def staleScan : FlagReads (history true true true) (epochRead true true true) :=
  ⟨[0], [staleFlag], by
    intro read member
    rw [List.mem_singleton.mp member]
    exact ⟨List.mem_singleton.mpr rfl, ⟨rfl, by decide⟩⟩⟩

def LitmusContract : Prop :=
  ¬ Coherent (history true true true) ∧
    Coherent (history false true true) ∧
    Coherent (history true false true) ∧
    Coherent (history true true false)

theorem litmusContract_proven : LitmusContract := by
  refine ⟨?_, stale_coherent (Or.inl rfl), stale_coherent (Or.inr (Or.inl rfl)),
    stale_coherent (Or.inr (Or.inr rfl))⟩
  intro coherent
  have cannotReadStale := scanContract_proven _ counter_correct coherent marked _ staleScan
    (List.mem_singleton.mpr rfl) rfl (by simp [staleScan, marked])
    (.descend staleFlag rfl .empty) (by
      intro r _ _ stamp after _
      have limit := stamp.isLt
      change 1 < stamp.val at after
      change stamp.val < 2 at limit
      omega) (by decide)
  change true ∈ [false] at cannotReadStale
  simp at cannotReadStale

end NeuralReversi.Threading.CutoffVisibility.Witnesses
