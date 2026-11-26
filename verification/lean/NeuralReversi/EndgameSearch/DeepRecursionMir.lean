import Lean
import NeuralReversi.EndgameSearch.DeepRecursionMir.Program
import NeuralReversi.EndgameSearch.ShallowMir

namespace NeuralReversi.EndgameSearch.DeepRecursionMir

open Cancellation (EmptyState Outcome State Valid)
open EndgameTreeCache (Address Store entryFor discValue soundHistory)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open Flip.Dispatch (TargetConfig)

open Lean in
local macro "cancelDef%" n:ident : term =>
  pure (mkIdent (mkPrivateNameCore `NeuralReversi.EndgameSearch.Cancellation
    (`NeuralReversi.EndgameSearch.Cancellation ++ n.getId)))

open Lean in
local macro "unfoldCancel" n:ident : tactic => do
  let id := mkIdent (mkPrivateNameCore `NeuralReversi.EndgameSearch.Cancellation
    (`NeuralReversi.EndgameSearch.Cancellation ++ n.getId))
  `(tactic| unfold $id:ident)

private theorem early_model (fc : FlipConfig) (paired : Bool) (b : Position) (s : State) (addr : Address)
    (a : Int) (first : Bitboard.Square) (rest : List Bitboard.Square) (aligned : s.empties.matches b) :
    early fc paired b s addr a first rest =
      EndgameCutoffs.beforeMoves fc b.val s.history addr .deep a (first :: rest) := by
  simp only [early, EndgameCutoffs.beforeMoves, List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, aligned.1]
  cases h : EndgameCache.probe (entryFor addr s.history .deep b.val) b.val a with
  | some score => rfl
  | none =>
    cases d : EndgameStability.tryCutoff b.val (Board.emptyCount b.val).toNat ⟨a⟩ with
    | some score => rfl
    | none =>
      dsimp only
      cases rest with
      | nil => simp [List.any_cons]
      | cons y ys =>
        let wipe := EndgameCutoffs.wipesOut fc b.val
        let choices := first :: y :: ys
        have h := MoveList.SmallGenerate.generate_spec paired wipe choices
        cases clear : (MoveList.SmallGenerate.generate paired wipe choices).1 with
        | false =>
          have someWipe : choices.any wipe = true := by
            cases flag : choices.any wipe with
            | true => rfl
            | false =>
              have pass := h.1.mpr (fun x hx => Bool.eq_false_iff.mpr (List.any_eq_false.mp flag x hx))
              simp [clear] at pass
          dsimp [choices, wipe] at someWipe
          simp [someWipe]
        | true =>
          have noWipe : choices.any wipe = false := List.any_eq_false.mpr (fun x hx => Bool.eq_false_iff.mp (h.1.mp clear x hx))
          have ordered := h.2.1 clear
          dsimp [choices, wipe] at noWipe ordered
          simp [noWipe, ordered]

def run (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (a : Int) (s : State) : Outcome :=
  match movesNow : legalSquares mc b.val with
  | [] =>
    if _other : Board.hasLegalMoves mc (pass b).val = true then
      negate (run target bmi2 paired mc fc addr (pass b) (-(a+1)) s)
    else ⟨some (Board.AwardedScore b.val), s⟩
  | m :: rest =>
    match early fc paired b s addr a m.val (rest.map Subtype.val) with
    | some score => ⟨some score, s⟩
    | none =>
      moves b a s m rest (deepOrder mc fc b) fun sq input =>
        let next := (play mc fc b sq).val
        executeChild sq.val (a+1) input (ShallowMir.run target bmi2 mc fc addr next)
          (run target bmi2 paired mc fc addr next)
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil movesNow, _other⟩) b.property
  · exact (play mc fc b sq).property

private theorem ordered_model {Move : Type} (child : Move → State → Outcome) (a : Int)
    (ms : List Move) (best : Int) (s : State) :
    ordered child (a+1) ms best s = (cancelDef% scanRaw) child a ms best s := by
  induction ms generalizing best s with
  | nil => rfl
  | cons m ms ih =>
    rw [ordered]
    unfoldCancel scanRaw
    dsimp only [stopped]
    simp only [ih]

private theorem scan_replace {Move : Type} (mc : X86Config) (fc : FlipConfig) (b : Position)
    (left right : Move → State → Outcome) (a : Int) (ms : List Move) (best : Int) (s : State)
    (aligned : s.empties.matches b) (sound : soundHistory mc fc s.history)
    (calls : ∀ m state, state.empties.matches b → soundHistory mc fc state.history →
      left m state = right m state ∧ (right m state).state.empties = state.empties ∧
      soundHistory mc fc (right m state).state.history) :
    ordered left (a+1) ms best s = ordered right (a+1) ms best s := by
  induction ms generalizing best s with
  | nil => rfl
  | cons m ms ih =>
    obtain ⟨same, undone, valid⟩ := calls m s aligned sound
    simp only [ordered, same]
    split
    · rfl
    · exact ih _ _ (undone.symm ▸ aligned) valid

private def basicChild (opts : Options mc) (fc : FlipConfig) (b : Position) (a : Int)
    (sq : GeneratedSquare mc b.val) (input : State) : Outcome :=
  (cancelDef% moveFrame) sq (fun placed =>
    let next := (play mc fc b sq).val
    if placed.empties.count.toNat ≤ 6 then
      let r := EndgameSearch.run mc fc opts .shallow next (-(a + 1)) placed.history
      ⟨some r.score, { placed with history := r.history }⟩
    else Cancellation.run mc fc opts next (-(a + 1)) placed) input

private theorem basic_valid (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (a : Int) (sq : GeneratedSquare mc b.val) (input : State) (safe : SafeThreshold (-(a+1)))
    (aligned : input.empties.matches b) (sound : soundHistory mc fc input.history) :
    Valid mc fc input a (-discValue mc fc (play mc fc b sq).val.val)
      (basicChild (orderedOptions target bmi2 mc fc addr) fc b a sq input) := by
  let opts := orderedOptions target bmi2 mc fc addr
  have correct := orderedOptions_correct target bmi2 mc fc addr
  apply (cancelDef% moveFrame_valid) mc fc b sq a input aligned sound
  intro placed childAligned ready
  split
  · have ⟨range, bound, sound, grew⟩ := EndgameSearch.contract_proven mc fc opts .shallow
      (play mc fc b sq).val (-(a+1)) placed.history correct ready (fun _ _ => safe)
    exact ⟨rfl, sound, grew, range, bound⟩
  · exact Cancellation.contract_proven mc fc opts _ _ placed correct childAligned ready safe

private theorem frame_agrees (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (a : Int) (sq : GeneratedSquare mc b.val) (input : State) (safe : SafeThreshold (-(a+1)))
    (aligned : input.empties.matches b) (sound : soundHistory mc fc input.history)
    (deep : Int → State → Outcome)
    (same : ∀ state, state.empties.matches (play mc fc b sq).val → soundHistory mc fc state.history →
      deep (-(a+1)) state = Cancellation.run mc fc (orderedOptions target bmi2 mc fc addr) (play mc fc b sq).val (-(a+1)) state) :
    executeChild sq.val (a+1) input (ShallowMir.run target bmi2 mc fc addr (play mc fc b sq).val) deep =
      basicChild (orderedOptions target bmi2 mc fc addr) fc b a sq input := by
  unfold executeChild basicChild
  unfoldCancel moveFrame
  unfoldCancel State.afterPoll
  split
  · rfl
  · let placed : State := { input with polls := input.polls.after, empties := input.empties.remove sq.val }
    have placedAligned : placed.empties.matches (play mc fc b sq).val :=
      (emptyStatePlayContract_proven mc fc b sq _ _ _ aligned).2.2.1
    have shallow := ShallowMir.agrees target bmi2 mc fc addr (play mc fc b sq).val placed.empties (-(a+1)) placed.history placedAligned safe sound
    have eq := same placed placedAligned sound
    cases small : decide (placed.empties.count.toNat ≤ 6) with
    | true =>
      have small := of_decide_eq_true small
      dsimp [placed] at small shallow
      simp only [small, ↓reduceIte, shallow]
      rfl
    | false =>
      have large := of_decide_eq_false small
      dsimp [placed] at large eq
      simp only [large, ↓reduceIte, eq]
      rfl

private theorem singleton {Move : Type} (child : Move → State → Outcome) (a : Int) (m : Move) (s : State)
    (lo : -127 ≤ (child m s).score.getD stopped) :
    ordered child (a+1) [m] (-127) s = ((child m s).score.getD stopped, (child m s).state) := by
  simp only [ordered, Int.max_eq_right lo]
  split <;> rfl

private theorem atRank (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (addr : Address) (k : Nat) : ∀ (b : Position), Board.playRank mc b.val = k → ∀ (a : Int) (s : State),
      s.empties.matches b → SafeThreshold a → soundHistory mc fc s.history →
      run target bmi2 paired mc fc addr b a s =
        Cancellation.run mc fc (orderedOptions target bmi2 mc fc addr) b a s := by
  induction k using Nat.strongRecOn with
  | ind k ih =>
    intro b rank a s aligned safe sound
    subst k
    have nextSafe : SafeThreshold (-(a+1)) := by obtain ⟨_, _⟩ := safe; constructor <;> omega
    rw [run.eq_def, Cancellation.run.eq_def]
    cases ms : legalSquares mc b.val with
    | nil =>
      cases other : Board.hasLegalMoves mc (pass b).val with
      | false => simp
      | true =>
        have smaller := Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil ms, other⟩) b.property
        have eq := ih _ smaller (pass b) rfl (-(a+1)) s (emptyState_pass b _ _ _ aligned) nextSafe sound
        simp [eq]
        rfl
    | cons first rest =>
      have address : (orderedOptions target bmi2 mc fc addr).address = addr := rfl
      simp only [early_model _ _ _ _ _ _ _ _ aligned, List.map_cons, address]
      cases cut : EndgameCutoffs.beforeMoves fc b.val s.history addr .deep a (first.val :: rest.map Subtype.val) with
      | some score => rfl
      | none =>
        let opts := orderedOptions target bmi2 mc fc addr
        let sub := fun (sq : GeneratedSquare mc b.val) (input : State) =>
          executeChild sq.val (a+1) input (ShallowMir.run target bmi2 mc fc addr (play mc fc b sq).val)
            (run target bmi2 paired mc fc addr (play mc fc b sq).val)
        let spec := basicChild opts fc b a
        have same (m : GeneratedSquare mc b.val) (st : State) (al : st.empties.matches b) (good : soundHistory mc fc st.history) :
            sub m st = spec m st := frame_agrees target bmi2 mc fc addr b a m st nextSafe al good _
            (fun s aligned good => ih _ (play mc fc b m).property _ rfl _ s aligned nextSafe good)
        have valid (m : GeneratedSquare mc b.val) (s : State) (al : s.empties.matches b) (good : soundHistory mc fc s.history) :
            Valid mc fc s a (-discValue mc fc (play mc fc b m).val.val) (spec m s) :=
          basic_valid target bmi2 mc fc addr b a m s nextSafe al good
        change moves b a s first rest (deepOrder mc fc b) sub =
          (cancelDef% storeDeep) b a ((cancelDef% scanRaw) spec a ((cancelDef% searchOrder) opts b first rest) (-127) s)
        rw [← ordered_model]
        unfoldCancel searchOrder
        cases rest with
        | nil =>
          have bounded := (valid first s aligned sound).2.2.2
          have lo : -127 ≤ (spec first s).score.getD stopped := by
            cases score : (spec first s).score with
            | none => decide
            | some v =>
              rw [score] at bounded
              exact bounded.1.1
          simp only [moves, List.isEmpty_nil, ↓reduceIte, same first s aligned sound, singleton _ _ _ _ lo]
          rfl
        | cons x xs =>
          have eq := scan_replace mc fc b sub spec a (deepOrder mc fc b) (-127) s aligned sound (fun m st al good =>
            ⟨same m st al good, (valid m st al good).1, (valid m st al good).2.1⟩)
          simp only [moves, List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, eq]
          rfl

theorem agrees (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (a : Int) (s : State) (aligned : s.empties.matches b)
    (safe : SafeThreshold a) (sound : soundHistory mc fc s.history) :
    run target bmi2 paired mc fc addr b a s =
      Cancellation.run mc fc (orderedOptions target bmi2 mc fc addr) b a s :=
  atRank target bmi2 paired mc fc addr _ b rfl a s aligned safe sound

theorem returned_score (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (addr : Address) (b : Position) (a : Int) (s : State) (aligned : s.empties.matches b)
    (safe : SafeThreshold a) (sound : soundHistory mc fc s.history) :
    let out := run target bmi2 paired mc fc addr b a s
    out.state.empties = s.empties ∧ soundHistory mc fc out.state.history ∧ s.history <+: out.state.history ∧
      ∀ score, out.score = some score → EndgameCache.SearchRange score ∧ Solve2.NWS a (discValue mc fc b.val) score := by
  rw [agrees target bmi2 paired mc fc addr b a s aligned safe sound]
  obtain ⟨restored, valid, growing, result⟩ := Cancellation.contract_proven mc fc
    (orderedOptions target bmi2 mc fc addr) b a s (orderedOptions_correct target bmi2 mc fc addr) aligned sound safe
  refine ⟨restored, valid, growing, fun score output => ?_⟩
  rw [output] at result
  exact result

end NeuralReversi.EndgameSearch.DeepRecursionMir
