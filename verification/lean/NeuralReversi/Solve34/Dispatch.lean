import NeuralReversi.Solve2.Guard

namespace NeuralReversi.Solve34.Dispatch
set_option linter.unusedSimpArgs false
abbrev Word := BitVec 64
abbrev Score := Solve2.Score
abbrev Square := Fin 64
abbrev Flip := Solve2.Dispatch.Flip
abbrev Adjacent := Solve2.Dispatch.Adjacent

structure Call where
  player : Word
  opponent : Word
  alpha : Score
  remaining : List Square
  deriving DecidableEq, Repr

structure ChildResult where
  score : Score
  nodes : Nat
  deriving DecidableEq, Repr
abbrev Oracle := Call → ChildResult

inductive Event where
  | root | pass | child (call : Call)
  deriving DecidableEq, Repr

structure Outcome where
  score : Score
  events : List Event
  nodes : Nat
  deriving DecidableEq, Repr

def Event.cost (oracle : Oracle) : Event → Nat
  | .root | .pass => 1
  | .child c => (oracle c).nodes

def Outcome.calls (out : Outcome) : List Call :=
  out.events.filterMap fun e => match e with | .child c => some c | _ => none

def finish (score : Score) : Outcome := ⟨score,[],0⟩

def emit (oracle : Oracle) (event : Event) (out : Outcome) : Outcome :=
  ⟨out.score,event :: out.events,event.cost oracle + out.nodes⟩

def maxRun (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) : Outcome :=
  match xs with
  | [] => if best = -(127 : Score) then fallback else finish best
  | [none] => if best = -(127 : Score) then fallback else finish best
  | [some c] => emit oracle (.child c) (finish (Solve2.maxWord (-(oracle c).score) best))
  | none :: rest => maxRun oracle alpha false best rest fallback
  | some c :: rest =>
      let score := -(oracle c).score
      if (alpha+1).toInt ≤ score.toInt then emit oracle (.child c) (finish score)
      else emit oracle (.child c)
        (maxRun oracle alpha false (if first then score else Solve2.maxWord score best) rest fallback)

def minRun (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) : Outcome :=
  match xs with
  | [] => if best = (127 : Score) then fallback else finish best
  | [none] => if best = (127 : Score) then fallback else finish best
  | [some c] => emit oracle (.child c) (finish (Solve2.minWord (oracle c).score best))
  | none :: rest => minRun oracle alpha false best rest fallback
  | some c :: rest =>
      let score := (oracle c).score
      if score.toInt ≤ alpha.toInt then emit oracle (.child c) (finish score)
      else emit oracle (.child c)
        (minRun oracle alpha false (if first then score else Solve2.minWord score best) rest fallback)

def run (charge : Bool) (oracle : Oracle) (alpha : Score)
    (playerCalls opponentCalls : List (Option Call)) (terminal : Score) : Outcome :=
  let pass := minRun oracle alpha true 127 opponentCalls (finish terminal)
  let pass := if charge then emit oracle .pass pass else pass
  let out := maxRun oracle alpha true (-127) playerCalls pass
  if charge then emit oracle .root out else out

def candidate (player opponent flipped : Word) (alpha : Score) (played : Square)
    (remaining : List Square) : Option Call :=
  if flipped = 0 then none else
    some ⟨opponent ^^^ flipped, player ^^^ flipped ^^^ ((1 : Word) <<< played.val),alpha,remaining⟩

def guardedCandidate (adjacent : Bool) (player opponent flipped : Word)
    (alpha : Score) (played : Square) (remaining : List Square) : Option Call :=
  if adjacent then candidate player opponent flipped alpha played remaining else none

theorem guardedCandidate_eq (adjacent : Bool) (player opponent flipped : Word)
    (alpha : Score) (played : Square) (remaining : List Square)
    (guard : adjacent = false → flipped = 0) :
    guardedCandidate adjacent player opponent flipped alpha played remaining =
      candidate player opponent flipped alpha played remaining := by
  cases ha : adjacent
  · simp [guardedCandidate, ha, candidate, guard ha]
  · rfl

abbrev Slot := Square × List Square

def slots3 (a b c : Square) : List Slot := [(a,[b,c]),(b,[a,c]),(c,[a,b])]
def slots4 (a b c d : Square) : List Slot := [(a,[b,c,d]),(b,[a,c,d]),(c,[a,b,d]),(d,[a,b,c])]

def eagerCandidates (player opponent : Word) (alpha : Score)
    (slots : List Slot) (flips : List Word) : List (Option Call) :=
  (slots.zip flips).map fun (slot,mask) => candidate player opponent mask alpha slot.1 slot.2

def fallbackCandidates (flip : Flip) (adjacent : Adjacent) (player opponent : Word)
    (alpha : Score) (slots : List Slot) : List (Option Call) :=
  slots.map fun slot => guardedCandidate (adjacent opponent slot.1) player opponent
    (flip slot.1 player opponent) alpha slot.1 slot.2

abbrev BatchFlip := List Square → Word → Word → List Word

def eager (charge : Bool) (oracle : Oracle) (batch : BatchFlip)
    (player opponent : Word) (alpha : Score) (slots : List Slot) (terminal : Score) : Outcome :=
  run charge oracle alpha
    (eagerCandidates player opponent (Solve2.childAlpha alpha) slots (batch (slots.map Prod.fst) player opponent))
    (eagerCandidates opponent player alpha slots (batch (slots.map Prod.fst) opponent player)) terminal

def fallback (charge : Bool) (oracle : Oracle) (flip : Flip) (adjacent : Adjacent)
    (player opponent : Word) (alpha : Score) (slots : List Slot) (terminal : Score) : Outcome :=
  run charge oracle alpha
    (fallbackCandidates flip adjacent player opponent (Solve2.childAlpha alpha) slots)
    (fallbackCandidates flip adjacent opponent player alpha slots) terminal

def eager3 (oracle : Oracle) (batch : BatchFlip) (p o : Word) (alpha : Score)
    (a b c : Square) (terminal : Score) : Outcome := eager true oracle batch p o alpha (slots3 a b c) terminal

def fallback3 (oracle : Oracle) (flip : Flip) (adjacent : Adjacent) (p o : Word) (alpha : Score)
    (a b c : Square) (terminal : Score) : Outcome := fallback true oracle flip adjacent p o alpha (slots3 a b c) terminal

def eager4 (oracle : Oracle) (batch : BatchFlip) (p o : Word) (alpha : Score)
    (a b c d : Square) (terminal : Score) : Outcome := eager false oracle batch p o alpha (slots4 a b c d) terminal

def fallback4 (oracle : Oracle) (flip : Flip) (adjacent : Adjacent) (p o : Word) (alpha : Score)
    (a b c d : Square) (terminal : Score) : Outcome := fallback false oracle flip adjacent p o alpha (slots4 a b c d) terminal

def eagerEnabled (x86 avx512cd avx512vl avx2 aarch64 neon wasm simd128 : Bool) : Bool :=
  (x86 && avx512cd && avx512vl) || (x86 && avx2) || (aarch64 && neon) || (wasm && simd128)

def dispatch (enabled charge : Bool) (oracle : Oracle) (flip : Flip) (batch : BatchFlip)
    (adjacent : Adjacent) (p o : Word) (alpha : Score) (slots : List Slot) (terminal : Score) : Outcome :=
  if enabled then eager charge oracle batch p o alpha slots terminal
  else fallback charge oracle flip adjacent p o alpha slots terminal

structure Compatible (flip : Flip) (batch : BatchFlip) (adjacent : Adjacent) : Prop where
  batch : ∀ squares p o, batch squares p o = squares.map (fun sq => flip sq p o)
  guard : ∀ sq p o, adjacent o sq = false → flip sq p o = 0

theorem candidates_eq (flip : Flip) (adjacent : Adjacent) (p o : Word) (alpha : Score)
    (slots : List Slot) (guard : ∀ sq, adjacent o sq = false → flip sq p o = 0) :
    fallbackCandidates flip adjacent p o alpha slots =
      eagerCandidates p o alpha slots (slots.map fun slot => flip slot.1 p o) := by
  induction slots with
  | nil => rfl
  | cons slot slots ih =>
    simp only [fallbackCandidates, eagerCandidates, List.map_cons, List.zip_cons_cons]
    rw [guardedCandidate_eq _ _ _ _ _ _ _ (guard slot.1)]
    congr 1

theorem eager_eq_fallback (charge : Bool) (oracle : Oracle) (flip : Flip) (batch : BatchFlip)
    (adjacent : Adjacent) (compatible : Compatible flip batch adjacent)
    (p o : Word) (alpha : Score) (slots : List Slot) (terminal : Score) :
    eager charge oracle batch p o alpha slots terminal =
      fallback charge oracle flip adjacent p o alpha slots terminal := by
  simp only [eager, fallback, compatible.batch, List.map_map, Function.comp_def]
  rw [candidates_eq flip adjacent p o _ slots (fun sq => compatible.guard sq p o),
    candidates_eq flip adjacent o p _ slots (fun sq => compatible.guard sq o p)]

def scalarBatch : BatchFlip := fun squares p o => squares.map (fun sq => Solve2.Guard.scalarFlip sq p o)

theorem scalar_compatible : Compatible Solve2.Guard.scalarFlip scalarBatch Solve2.Guard.adjacent := by
  constructor
  · intros; rfl
  · exact Solve2.Guard.scalar_compatible.guard

theorem scalar_eager_eq_fallback (charge : Bool) (oracle : Oracle) (p o : Word)
    (alpha : Score) (slots : List Slot) (terminal : Score) :
    eager charge oracle scalarBatch p o alpha slots terminal =
      fallback charge oracle Solve2.Guard.scalarFlip Solve2.Guard.adjacent p o alpha slots terminal :=
  eager_eq_fallback charge oracle _ _ _ scalar_compatible p o alpha slots terminal


def CostCorrect (oracle : Oracle) (out : Outcome) : Prop :=
  out.nodes = (out.events.map (Event.cost oracle)).sum

theorem finish_cost (oracle : Oracle) (score : Score) : CostCorrect oracle (finish score) := rfl

theorem emit_cost (oracle : Oracle) (event : Event) (out : Outcome)
    (h : CostCorrect oracle out) : CostCorrect oracle (emit oracle event out) := by
  simpa only [CostCorrect, emit, List.map_cons, List.sum_cons, h] using
    congrArg (event.cost oracle + ·) h

theorem maxRun_cost (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) (hf : CostCorrect oracle fallback) :
    CostCorrect oracle (maxRun oracle alpha first best xs fallback) := by
  induction xs generalizing first best with
  | nil => simp only [maxRun]; split; exact hf; exact finish_cost _ _
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [maxRun]; split; exact hf; exact finish_cost _ _
      | some c => exact emit_cost _ _ _ (finish_cost _ _)
    | cons y ys =>
      cases x with
      | none => exact ih false best
      | some c =>
        simp only [maxRun]
        split
        · exact emit_cost _ _ _ (finish_cost _ _)
        · exact emit_cost _ _ _ (ih false _)

theorem minRun_cost (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) (hf : CostCorrect oracle fallback) :
    CostCorrect oracle (minRun oracle alpha first best xs fallback) := by
  induction xs generalizing first best with
  | nil => simp only [minRun]; split; exact hf; exact finish_cost _ _
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [minRun]; split; exact hf; exact finish_cost _ _
      | some c => exact emit_cost _ _ _ (finish_cost _ _)
    | cons y ys =>
      cases x with
      | none => exact ih false best
      | some c =>
        simp only [minRun]
        split
        · exact emit_cost _ _ _ (finish_cost _ _)
        · exact emit_cost _ _ _ (ih false _)

theorem run_cost (charge : Bool) (oracle : Oracle) (alpha : Score)
    (playerCalls opponentCalls : List (Option Call)) (terminal : Score) :
    CostCorrect oracle (run charge oracle alpha playerCalls opponentCalls terminal) := by
  have pass := minRun_cost oracle alpha true 127 opponentCalls (finish terminal) (finish_cost _ _)
  cases charge
  · exact maxRun_cost _ _ _ _ _ _ pass
  · exact emit_cost _ _ _ (maxRun_cost _ _ _ _ _ _ (emit_cost _ _ _ pass))

theorem maxRun_local_count (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) (event : Event)
    (isLocal : event = .root ∨ event = .pass) :
    (maxRun oracle alpha first best xs fallback).events.count event ≤ fallback.events.count event := by
  have child_ne : ∀ c, Event.child c ≠ event := by
    intro c; rcases isLocal with rfl | rfl <;> simp
  induction xs generalizing first best with
  | nil => simp only [maxRun]; split; omega; simp [finish]
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [maxRun]; split; omega; simp [finish]
      | some c => simp [maxRun, emit, finish, child_ne c]
    | cons y ys =>
      cases x with
      | none => exact ih false best
      | some c =>
        simp only [maxRun]
        split
        · simp [emit, finish, child_ne c]
        · simpa [emit, child_ne c] using ih false (if first then -(oracle c).score else Solve2.maxWord (-(oracle c).score) best)

theorem minRun_local_count (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) (event : Event)
    (isLocal : event = .root ∨ event = .pass) :
    (minRun oracle alpha first best xs fallback).events.count event ≤ fallback.events.count event := by
  have child_ne : ∀ c, Event.child c ≠ event := by
    intro c; rcases isLocal with rfl | rfl <;> simp
  induction xs generalizing first best with
  | nil => simp only [minRun]; split; omega; simp [finish]
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [minRun]; split; omega; simp [finish]
      | some c => simp [minRun, emit, finish, child_ne c]
    | cons y ys =>
      cases x with
      | none => exact ih false best
      | some c =>
        simp only [minRun]
        split
        · simp [emit, finish, child_ne c]
        · simpa [emit, child_ne c] using ih false (if first then (oracle c).score else Solve2.minWord (oracle c).score best)

theorem run_local_nodes (charge : Bool) (oracle : Oracle) (alpha : Score)
    (playerCalls opponentCalls : List (Option Call)) (terminal : Score) :
    (run charge oracle alpha playerCalls opponentCalls terminal).events.count .root = charge.toNat ∧
    (run charge oracle alpha playerCalls opponentCalls terminal).events.count .pass ≤ charge.toNat := by
  let pass := minRun oracle alpha true 127 opponentCalls (finish terminal)
  have pr := minRun_local_count oracle alpha true 127 opponentCalls (finish terminal) .root (Or.inl rfl)
  have pp := minRun_local_count oracle alpha true 127 opponentCalls (finish terminal) .pass (Or.inr rfl)
  have pr0 : pass.events.count .root = 0 := by simpa [pass,finish] using pr
  have pp0 : pass.events.count .pass = 0 := by simpa [pass,finish] using pp
  cases charge
  · have hr := maxRun_local_count oracle alpha true (-127) playerCalls pass .root (Or.inl rfl)
    have hp := maxRun_local_count oracle alpha true (-127) playerCalls pass .pass (Or.inr rfl)
    change _ = 0 ∧ _ ≤ 0
    change (maxRun oracle alpha true (-127) playerCalls pass).events.count .root = 0 ∧
      (maxRun oracle alpha true (-127) playerCalls pass).events.count .pass ≤ 0
    omega
  · have hr := maxRun_local_count oracle alpha true (-127) playerCalls (emit oracle .pass pass) .root (Or.inl rfl)
    have hp := maxRun_local_count oracle alpha true (-127) playerCalls (emit oracle .pass pass) .pass (Or.inr rfl)
    have hr0 : (maxRun oracle alpha true (-127) playerCalls (emit oracle .pass pass)).events.count .root = 0 := by
      simpa [emit, pr0] using hr
    change (emit oracle .root (maxRun oracle alpha true (-127) playerCalls (emit oracle .pass pass))).events.count .root = 1 ∧
      (emit oracle .root (maxRun oracle alpha true (-127) playerCalls (emit oracle .pass pass))).events.count .pass ≤ 1
    constructor
    · simpa [emit] using hr0
    · simpa [emit, pp0] using hp


theorem event_costs_split (oracle : Oracle) (events : List Event) :
    (events.map (Event.cost oracle)).sum = events.count .root + events.count .pass +
      ((events.filterMap fun e => match e with | .child c => some c | _ => none).map
        (fun c => (oracle c).nodes)).sum := by
  induction events with
  | nil => rfl
  | cons event events ih =>
    cases event <;> simp [Event.cost, List.count_cons, ih] <;> omega

theorem run_node_formula (charge : Bool) (oracle : Oracle) (alpha : Score)
    (playerCalls opponentCalls : List (Option Call)) (terminal : Score) :
    let out := run charge oracle alpha playerCalls opponentCalls terminal
    out.nodes = charge.toNat + out.events.count .pass +
      (out.calls.map fun c => (oracle c).nodes).sum := by
  have cost := run_cost charge oracle alpha playerCalls opponentCalls terminal
  have localCounts := run_local_nodes charge oracle alpha playerCalls opponentCalls terminal
  unfold CostCorrect at cost
  dsimp only
  rw [cost, event_costs_split, localCounts.1]
  rfl

theorem slots3_order (a b c : Square) :
    (slots3 a b c).map Prod.fst = [a,b,c] ∧
    (slots3 a b c).map Prod.snd = [[b,c],[a,c],[a,b]] := ⟨rfl,rfl⟩

theorem slots4_order (a b c d : Square) :
    (slots4 a b c d).map Prod.fst = [a,b,c,d] ∧
    (slots4 a b c d).map Prod.snd = [[b,c,d],[a,c,d],[a,b,d],[a,b,c]] := ⟨rfl,rfl⟩

theorem physical_last_slot (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (call : Call) (fallback : Outcome) :
    maxRun oracle alpha first best [some call] fallback =
      emit oracle (.child call) (finish (Solve2.maxWord (-(oracle call).score) best)) ∧
    minRun oracle alpha first best [some call] fallback =
      emit oracle (.child call) (finish (Solve2.minWord (oracle call).score best)) := ⟨rfl,rfl⟩

theorem dispatch_eq_fallback (enabled charge : Bool) (oracle : Oracle) (flip : Flip)
    (batch : BatchFlip) (adjacent : Adjacent) (compatible : Compatible flip batch adjacent)
    (p o : Word) (alpha : Score) (slots : List Slot) (terminal : Score) :
    dispatch enabled charge oracle flip batch adjacent p o alpha slots terminal =
      fallback charge oracle flip adjacent p o alpha slots terminal := by
  unfold dispatch
  split
  · exact eager_eq_fallback charge oracle flip batch adjacent compatible p o alpha slots terminal
  · rfl

theorem eager_fallback_effects (charge : Bool) (oracle : Oracle) (flip : Flip)
    (batch : BatchFlip) (adjacent : Adjacent) (compatible : Compatible flip batch adjacent)
    (p o : Word) (alpha : Score) (slots : List Slot) (terminal : Score) :
    let e := eager charge oracle batch p o alpha slots terminal
    let f := fallback charge oracle flip adjacent p o alpha slots terminal
    e.score = f.score ∧ e.calls = f.calls ∧ e.events = f.events ∧ e.nodes = f.nodes := by
  rw [eager_eq_fallback charge oracle flip batch adjacent compatible p o alpha slots terminal]
  exact ⟨rfl,rfl,rfl,rfl⟩

end NeuralReversi.Solve34.Dispatch
