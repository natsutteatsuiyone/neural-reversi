import NeuralReversi.PvsParallel.SplitSchedulerMir.Correct

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFields

open Threading.AtomicWord (Event Write Read SequencedBefore)

/-- Each guard contributes its load to the owning thread's program order. -/
def phases (guards : Nat) : List Phase := ownerPhases.flatMap fun
  | .init => initPhases
  | .idle => List.replicate guards .idle
  | site => [site]

theorem phases_eq (guards : Nat) : phases guards =
    [.clear, .mark, .seed, .seedMove, .seedAlpha, .assign] ++
      List.replicate guards .idle ++ [.finalize, .best, .picked, .returned] := by
  simp [phases, ownerPhases, initPhases]

/-- The existing mask proof projects out the two additional initializer stores. -/
def maskClock (owner : Clock) : Clock where
  worker := owner.worker
  tick i := owner.tick (if i < 3 then i else i + 2)
  advances i j lt := by apply owner.advances; split <;> split <;> omega

def returnIndex (guards : Nat) (site : Phase) : Nat :=
  initPhases.length + guards + (ownerPhases.drop 2).idxOf site

theorem return_indices (guards : Nat) :
    returnIndex guards .best = guards + 7 ∧ returnIndex guards .picked = guards + 8 := by
  change 6 + guards + 1 = guards + 7 ∧ 6 + guards + 2 = guards + 8
  omega

def moveLoad (owner : Clock) (guards : Nat) (src : Fin ws.length) : Read ws :=
  ⟨owner.atStep (returnIndex guards .picked), src, false⟩

theorem seed_sites (guards : Nat) :
    (phases guards).take 6 = [.clear, .mark, .seed, .seedMove, .seedAlpha, .assign] := by
  rw [phases_eq]; rfl

theorem return_sites (guards : Nat) :
    (phases guards).drop (guards + 7) = [.best, .picked, .returned] := by
  simp [phases_eq, List.drop_append]

def returned (owner : Clock) (guards : Nat) : Event := owner.atStep (returnIndex guards .returned)

theorem move_before_assign (owner : Clock) :
    SequencedBefore (owner.atStep 3) ((maskClock owner).atStep 3) :=
  ⟨rfl, owner.advances 3 5 (by decide)⟩

theorem score_event (owner : Clock) (log : List (Guard mask)) (src : Fin ws.length) :
    (scoreLoad ws src (maskClock owner) log).event = owner.atStep (log.length + 7) := by
  simp [scoreLoad, maskClock, Clock.atStep, show ¬ log.length + 5 < 3 by omega, Nat.add_assoc]

theorem read_order {bs ms : List Write} (owner : Clock) (log : List (Guard mask)) (score : Fin bs.length) (move : Fin ms.length) :
    SequencedBefore (scoreLoad bs score (maskClock owner) log).event (moveLoad owner log.length move).event := by
  rw [score_event]
  exact ⟨rfl, owner.advances _ _ (by rw [(return_indices _).2]; omega)⟩

theorem read_before_return (owner : Clock) (log : List (Guard mask)) (score : Fin bs.length) :
    SequencedBefore (scoreLoad bs score (maskClock owner) log).event (returned owner log.length) := by
  rw [score_event]
  refine ⟨rfl, owner.advances _ _ ?_⟩
  change log.length + 7 < 6 + log.length + 3
  omega

inductive Field | score | move | alpha

def Field.seedIndex : Field → Nat
  | .score => initPhases.idxOf .seed
  | .move => initPhases.idxOf .seedMove
  | .alpha => initPhases.idxOf .seedAlpha

/-- A later initializer on the same thread can address another slot, or a finite execution
    can end partway through it. Each field's following modifications are in address order. -/
structure Initialization where
  index : Nat
  count : Fin (initPhases.length + 1)
  score : BitVec 32
  move : BitVec 8
  alpha : BitVec 32
  scoreStores : List Write
  moveStores : List Write
  alphaStores : List Write

def Initialization.stores (a : Initialization) (owner : Clock) (start : Nat) (field : Field) : List Write :=
  let (value, later) := match field with
    | .score => (a.score.zeroExtend 64, a.scoreStores)
    | .move => (a.move.zeroExtend 64, a.moveStores)
    | .alpha => (a.alpha.zeroExtend 64, a.alphaStores)
  ⟨owner.atStep (start + field.seedIndex), value, false, false⟩ :: later

def laterWrites (owner : Clock) (slot : Nat) (field : Field) (start : Nat) :
    List Initialization → List Write
  | [] => []
  | a :: rest =>
    let tail := laterWrites owner slot field (start + a.count.val) rest
    if a.index = slot ∧ field.seedIndex < a.count.val then a.stores owner start field ++ tail else tail

private theorem first_later (owner : Clock) (slot : Nat) (field : Field) (later : List Initialization)
    (start read : Nat) (before : read < start) (w : Write)
    (first : (laterWrites owner slot field start later)[0]? = some w) :
    SequencedBefore (owner.atStep read) w.event := by
  induction later generalizing start with
  | nil => simp [laterWrites] at first
  | cons a rest ih =>
    simp only [laterWrites] at first
    split at first
    · cases field <;> simp [Initialization.stores] at first <;> subst w <;>
        exact ⟨rfl, owner.advances _ _ (by omega)⟩
    · exact ih _ (by omega) first

theorem before_reuse (owner : Clock) (slot : Nat) (field : Field) (guards : Nat) (later : List Initialization)
    (pre current : List Write) (read : Nat) (within : read < (phases guards).length)
    (i : Fin (pre ++ current ++ laterWrites owner slot field (phases guards).length later).length)
    (pos : i.val = pre.length + current.length) :
    SequencedBefore (owner.atStep read) ((pre ++ current ++ laterWrites owner slot field (phases guards).length later).get i).event := by
  apply first_later owner slot field later _ _ within
  have atPos := List.getElem?_eq_getElem (l := pre ++ current ++ laterWrites owner slot field (phases guards).length later) i.isLt
  rw [List.getElem?_append_right (by simpa using Nat.le_of_eq pos.symm)] at atPos
  rw [show i.val - (pre ++ current).length = 0 by simp [pos]] at atPos
  exact atPos

end NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFields
