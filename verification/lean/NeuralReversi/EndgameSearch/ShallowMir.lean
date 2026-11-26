import Lean
import NeuralReversi.EndgameSearch.ShallowMir.Program
import NeuralReversi.EndgameSearch.Direct.Mir
import NeuralReversi.EndgameSearch.Cancellation

namespace NeuralReversi.EndgameSearch.ShallowMir

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open EndgameTreeCache (Store Address entryFor)
open Cancellation (EmptyState)
open Flip.Dispatch (TargetConfig)

private def modelScan {Move : Type} (sub : Move → List Store → Result) (alpha : Int) :
    List Move → Int → List Store → Result
  | [], best, h => ⟨best, h⟩
  | m :: rest, best, h =>
    let r := sub m h
    if alpha + 1 ≤ r.score then r else modelScan sub alpha rest (max best r.score) r.history

open Lean in
local macro "searchDef%" n:ident : term =>
  pure (mkIdent (mkPrivateNameCore `NeuralReversi.EndgameSearch (`NeuralReversi.EndgameSearch ++ n.getId)))

open Lean in
local macro "unfoldSearch" n:ident : tactic => do
  let id := mkIdent (mkPrivateNameCore `NeuralReversi.EndgameSearch (`NeuralReversi.EndgameSearch ++ n.getId))
  `(tactic| unfold $id:ident)

open Lean in
local macro "directDef%" n:ident : term =>
  pure (mkIdent (mkPrivateNameCore `NeuralReversi.EndgameSearch.Direct.Mir
    (`NeuralReversi.EndgameSearch.Direct.Mir ++ n.getId)))

private theorem scan_model (sub : Move → List Store → Result) (alpha : Int)
    (ms : List Move) (best : Int) (h : List Store) :
    modelScan sub alpha ms best h = (searchDef% scan) sub alpha ms best h := by
  induction ms generalizing best h with
  | nil => rfl
  | cons m rest ih =>
    unfold modelScan
    change (if _ then _ else _) = (if alpha + 1 ≤ (sub m h).score then (sub m h) else _)
    split
    · rfl
    · exact ih _ _

private theorem stage_then (b : Position) (sub : Move → List Store → Result) (alpha : Int)
    (ms rest : List Move) (best : Int) (h : List Store) :
    (match stage b (alpha + 1) sub ms best h with
     | .cutoff r => r
     | .next best' h' => write b alpha (modelScan sub alpha rest best' h')) =
      write b alpha (modelScan sub alpha (ms ++ rest) best h) := by
  induction ms generalizing best h with
  | nil => rfl
  | cons m ms ih =>
    rw [List.cons_append, stage, modelScan]
    by_cases cut : alpha + 1 ≤ (sub m h).score
    · simp [cut]
    · simpa only [ite_eq_right cut] using ih (max best (sub m h).score) (sub m h).history

theorem groups_flatten (b : Position) (sub : Move → List Store → Result) (alpha : Int)
    (stages : List (List Move)) (best : Int) (history : List Store) :
    groups b alpha sub stages best history = write b alpha (modelScan sub alpha stages.flatten best history) := by
  induction stages generalizing best history with
  | nil => rfl
  | cons ms rest ih =>
    rw [groups, List.flatten_cons, ← stage_then]
    cases stage b (alpha + 1) sub ms best history <;> simp [ih]

theorem masks_model (moves : LegalMoves.Raw) (bits : Bitboard.Bits) (p : Fin 16) :
    masks moves bits (ShallowMoveOrder.quadrantMask p) = ShallowMoveOrder.groups moves bits p := by rfl

private def stagesFor (mc : X86Config) (b : Position) (odd : Bitboard.Bits) :
    List (List (GeneratedSquare mc b.val)) :=
  (masks ⟨Board.getMoves mc b.val⟩ (Board.emptySquares b.val) odd).map fun group =>
    (legalSquares mc b.val).filter fun m => group.bits.getLsbD m.val.val

private theorem stagesFor_order (mc : X86Config) (b : Position) (p : Fin 16) :
    (stagesFor mc b (ShallowMoveOrder.quadrantMask p)).flatten = shallowOrder mc b p := by rfl

/-- The move groups of `shallow_search`, split by `odd_quadrants(board.empty_squares())`. -/
def stages (mc : X86Config) (b : Position) : List (List (GeneratedSquare mc b.val)) :=
  stagesFor mc b (Empties.oddQuadrants (Board.emptySquares b.val))

theorem stages_order (mc : X86Config) (b : Position) :
    (stages mc b).flatten = shallowOrder mc b (boardParity b.val) := by
  unfold stages
  rw [oddQuadrants_board]
  exact stagesFor_order mc b _

theorem singleton_order (mc : X86Config) (b : Position) (m : GeneratedSquare mc b.val) (p : Fin 16)
    (single : legalSquares mc b.val = [m]) : shallowOrder mc b p = [m] := by
  rw [← stagesFor_order]
  simp only [stagesFor, masks_model, ShallowMoveOrder.groups, single, List.map_cons, List.map_nil, List.flatten_cons,
    List.flatten_nil, List.filter_cons, List.filter_nil]
  generalize mask : ShallowMoveOrder.neighbors (Board.emptySquares b.val) = adjacent
  simp only [LegalMoves.bitand, LegalMoves.andNot, LegalMoves.nonCorners, LegalMoves.corners,
    Bitboard.corners, Bitboard.nonCorners, BitVec.getLsbD_not, BitVec.getLsbD_and, m.val.isLt, decide_true,
    Bool.true_and, m.property]
  repeat' first | split | simp_all

def four (target : TargetConfig) (bmi2 : Bool) (b : Position) (s : EmptyState) (odd : Bitboard.Bits) (alpha : Int) :
    Int :=
  ((Direct.Mir.solveFour target bmi2 b.val s odd (BitVec.ofInt 32 alpha)).map (fun disc => disc.raw.toInt)).getD 0

theorem four_score (target : TargetConfig) (bmi2 : Bool) (b : Position) (s : EmptyState) (a : Int)
    (aligned : s.matches b) (n : (Board.emptyCount b.val).toNat = 4) :
    four target bmi2 b s (Empties.oddQuadrants (Board.emptySquares b.val)) a =
      (Solve34.target4Run target bmi2 (presortedFourOrder b n).val (BitVec.ofInt 32 a)).score.toInt := by
  have slots := presortedFourOrder_reads b n s aligned.2
  let arg := (presortedFourOrder b n).val
  have board := (presortedFourOrder b n).property
  unfold four Direct.Mir.solveFour
  simp only [slots]
  have recover := (directDef% args_recover) arg
  change (directDef% args?) (presortedFourOrder b n).val.asBoard 4 (presortedFourOrder b n).val.squares = _ at recover
  rw [board] at recover
  simp [recover, arg]

def executeChild (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val) (beta : Int) (h : List Store)
    (rec : EmptyState → Int → List Store → Result) : Result :=
  let n := (play mc fc b m).val
  let placed := s.remove m.val
  let odd := Empties.oddQuadrants (Board.emptySquares b.val) ^^^ Empties.quadrant m.val
  let a := -beta
  child n a h placed.count.toNat
    (EndgameCache.probe (entryFor addr h .shallow n.val) n.val a)
    ((EndgameStability.tryCutoff n.val 4 ⟨a⟩).map (·.value))
    (fun _ => four target bmi2 n placed odd a) (fun _ => rec placed a h)

theorem restore_after (mc : X86Config) (fc : FlipConfig) (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val)
    (aligned : s.matches b) : (s.remove m.val).restore m.val = s := by
  obtain ⟨_, _, _, restored, count⟩ :=
    emptyStatePlayContract_proven mc fc b m s.mask s.count aligned
  cases s
  simp only [EmptyState.restore_eq, EmptyState.remove_eq] at restored count ⊢
  rw [restored, count]

private theorem child_model (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (s : EmptyState) (m : GeneratedSquare mc b.val) (a : Int) (h : List Store)
    (aligned : s.matches b) (rec : EmptyState → Int → List Store → Result)
    (same : rec (s.remove m.val) (-(a + 1)) h = EndgameSearch.run mc fc (orderedOptions target bmi2 mc fc addr)
      .shallow (play mc fc b m).val (-(a + 1)) h) :
    executeChild target bmi2 mc fc addr b s m (a + 1) h rec =
    (searchDef% searchChild) (orderedOptions target bmi2 mc fc addr) .shallow (play mc fc b m).val a h
      (fun tier a' h' => EndgameSearch.run mc fc (orderedOptions target bmi2 mc fc addr) tier (play mc fc b m).val a' h') := by
  let n := (play mc fc b m).val
  have placed : (s.remove m.val).matches n :=
    (emptyStatePlayContract_proven mc fc b m s.mask s.count aligned).2.1
  simp only [executeChild, child]
  rw [placed.1]
  by_cases count : (Board.emptyCount n.val).toNat = 4
  · change (if (Board.emptyCount n.val).toNat = 4 then _ else _) = _
    rw [ite_eq_left count]
    unfoldSearch searchChild
    dsimp only [orderedOptions, withSourceFour]
    dsimp [n] at count
    simp only [true_and, count, ↓reduceIte]
    unfoldSearch fourResult
    cases hit : EndgameCache.probe (entryFor addr h .shallow (play mc fc b m).val.val) (play mc fc b m).val.val (-(a + 1)) with
    | some v => rfl
    | none =>
      cases stable : EndgameStability.tryCutoff (play mc fc b m).val.val 4 ⟨-(a + 1)⟩ with
      | some v => rfl
      | none =>
        have fixed := four_score target bmi2 n (s.remove m.val) (-(a + 1)) placed count
        rw [← oddQuadrants_play mc fc b m] at fixed
        dsimp [n] at fixed
        simp only [fixed, Option.map_none, count, ↓reduceDIte]
        rfl
  · change (if (Board.emptyCount n.val).toNat = 4 then _ else _) = _
    rw [ite_eq_right count, same]
    unfoldSearch searchChild
    dsimp [n] at count
    simp only [orderedOptions, withSourceFour, true_and, count, ↓reduceIte]
    unfoldSearch childTable
    rfl

def run (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (s : EmptyState) (a : Int) (h : List Store) : Result :=
  match moves : legalSquares mc b.val with
  | [] =>
    if _other : Board.hasLegalMoves mc (pass b).val = true then
      let r := run target bmi2 mc fc addr (pass b) s (-(a + 1)) h
      ⟨-r.score, r.history⟩
    else ⟨Board.AwardedScore b.val, h⟩
  | first :: rest =>
    node b (first :: rest) a h false 0 (fun _ => ⟨0, h⟩)
      (EndgameCache.probe (entryFor addr h .shallow b.val) b.val a)
      ((EndgameStability.tryCutoff b.val s.count.toNat ⟨a⟩).map (·.value))
      (stages mc b) (fun m h' => executeChild target bmi2 mc fc addr b s m (a + 1) h'
        (fun s' a' stores => run target bmi2 mc fc addr (play mc fc b m).val s' a' stores))
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves, _other⟩) b.property
  · exact (play mc fc b m).property

private theorem scan_congr (mc : X86Config) (fc : FlipConfig) (sub spec : Move → List Store → Result)
    (a : Int) (ms : List Move) (best : Int) (h : List Store) (valid : EndgameTreeCache.soundHistory mc fc h)
    (agree : ∀ m h, EndgameTreeCache.soundHistory mc fc h → sub m h = spec m h ∧
      EndgameTreeCache.soundHistory mc fc (spec m h).history) :
    modelScan sub a ms best h = modelScan spec a ms best h := by
  induction ms generalizing h best with
  | nil => rfl
  | cons m ms ih =>
    obtain ⟨same, next⟩ := agree m h valid
    simp only [modelScan, same]
    split
    · rfl
    · exact ih _ _ next

private theorem scan_one (sub : Move → List Store → Result) (m : Move) (a : Int) (history : List Store)
    (limit : -127 ≤ (sub m history).score) : modelScan sub a [m] (-127) history = sub m history := by
  simp only [modelScan, Int.max_eq_right limit]
  split <;> rfl

private theorem atRank (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (k : Nat) : ∀ (b : Position), Board.playRank mc b.val = k → ∀ (s : EmptyState) (a : Int) (h : List Store),
      s.matches b → SafeThreshold a → EndgameTreeCache.soundHistory mc fc h →
      run target bmi2 mc fc addr b s a h = EndgameSearch.run mc fc (orderedOptions target bmi2 mc fc addr) .shallow b a h := by
  induction k using Nat.strongRecOn with
  | ind k ih =>
    intro b rank s a h aligned safe sound
    subst k
    have nextSafe : SafeThreshold (-(a + 1)) := by obtain ⟨l, u⟩ := safe; constructor <;> omega
    rw [run.eq_def, EndgameSearch.run.eq_def]
    cases moves : legalSquares mc b.val with
    | nil =>
      cases other : Board.hasLegalMoves mc (pass b).val with
      | false => simp
      | true =>
        have smaller := Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves, other⟩) b.property
        have passed := ih _ smaller (pass b) rfl s (-(a + 1)) h (emptyState_pass b _ _ aligned) nextSafe sound
        simp [passed]
        rfl
    | cons first rest =>
      let opts := orderedOptions target bmi2 mc fc addr
      let sub := fun m input => executeChild target bmi2 mc fc addr b s m (a + 1) input
        (fun s' a' hs => run target bmi2 mc fc addr (play mc fc b m).val s' a' hs)
      let spec := fun m input => (searchDef% searchChild) opts EndgameTreeCache.Table.shallow (play mc fc b m).val a input
        (fun tier a' hs => EndgameSearch.run mc fc opts tier (play mc fc b m).val a' hs)
      have childFacts (m : GeneratedSquare mc b.val) (input : List Store)
          (prior : EndgameTreeCache.soundHistory mc fc input) : sub m input = spec m input ∧
          EndgameCache.SearchRange (spec m input).score ∧ EndgameTreeCache.soundHistory mc fc (spec m input).history := by
        let n := (play mc fc b m).val
        have placed : (s.remove m.val).matches n :=
          (emptyStatePlayContract_proven mc fc b m s.mask s.count aligned).2.1
        refine ⟨child_model target bmi2 mc fc addr b s m a input aligned _ (ih _ (play mc fc b m).property n rfl
          (s.remove m.val) (-(a+1)) input placed nextSafe prior), ?_⟩
        have correct := orderedOptions_correct target bmi2 mc fc addr
        have window : ValidThreshold opts (-(a+1)) := fun _ _ => nextSafe
        let delegated := fun tier a' hs => EndgameSearch.run mc fc opts tier n a' hs
        have delegatedSound := contract_proven mc fc opts ((searchDef% childTable) .shallow n) n (-(a+1)) input correct prior window
        have proven := (searchDef% child_sound) mc fc opts correct .shallow n a input prior window delegated delegatedSound
        exact ⟨proven.1, proven.2.2.1⟩
      simp only [node, EndgameCutoffs.beforeMoves, aligned.1, List.map_cons, List.isEmpty_cons,
        Bool.false_eq_true, ↓reduceIte, orderedOptions, withSourceFour]
      cases hit : EndgameCache.probe (entryFor addr h .shallow b.val) b.val a with
      | some hit => rfl
      | none =>
        cases stable : EndgameStability.tryCutoff b.val (Board.emptyCount b.val).toNat ⟨a⟩ with
        | some v => rfl
        | none =>
          simp only [Option.map_none, reduceCtorEq, ↓reduceIte]
          change (if rest.isEmpty then write b a (sub first h) else groups b a sub (stages mc b) (-127) h) =
            write b a ((searchDef% scan) spec a (shallowOrder mc b (boardParity b.val)) (-127) h)
          rw [← scan_model]
          cases rest with
          | nil =>
            have single := singleton_order mc b first (boardParity b.val) moves
            simp only [List.isEmpty_nil, ↓reduceIte, single, scan_one _ _ _ _ (childFacts first h sound).2.1.1,
              (childFacts first h sound).1]
          | cons y ys =>
            simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, groups_flatten, stages_order]
            rw [scan_congr mc fc sub spec a _ (-127) h sound (fun m hs valid =>
              ⟨(childFacts m hs valid).1, (childFacts m hs valid).2.2⟩)]

theorem agrees (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig) (addr : Address)
    (b : Position) (s : EmptyState) (a : Int) (h : List Store) (aligned : s.matches b)
    (safe : SafeThreshold a) (sound : EndgameTreeCache.soundHistory mc fc h) :
    run target bmi2 mc fc addr b s a h = EndgameSearch.run mc fc (orderedOptions target bmi2 mc fc addr) .shallow b a h :=
  atRank target bmi2 mc fc addr _ b rfl s a h aligned safe sound

theorem scoreThreshold (alpha : ScoreRefresh.Score) : SafeThreshold (ScoreRefresh.toDiscDiff alpha).raw.toInt := by
  rw [ScoreRefresh.disc_floor]
  have lower := BitVec.le_toInt alpha.raw
  have upper := BitVec.toInt_lt (x := alpha.raw)
  change -2147483648 < _ ∧ _ < 2147483647
  constructor <;> omega

end NeuralReversi.EndgameSearch.ShallowMir
