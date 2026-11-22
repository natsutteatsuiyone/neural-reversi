import NeuralReversi.EndgameCutoffs
import NeuralReversi.Solve34.Control

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position GeneratedSquare legalSquares play pass minimax)
open Bitboard.Moves (X86Config)
open EndgameCache (SearchRange)
open EndgameTreeCache (Table Store Address discValue soundHistory entryFor)
open Solve2 (NWS)

abbrev FlipConfig := Flip.Dispatch.X86Config

structure Options (mc : X86Config) where
  address : Address
  order : (b : Position) → Table → List (GeneratedSquare mc b.val)
  four : Option (Position → Int → Int)

def SafeThreshold (alpha : Int) : Prop := -2147483648 < alpha ∧ alpha < 2147483647

def ValidThreshold (opts : Options mc) (alpha : Int) : Prop :=
  ∀ f, opts.four = some f → SafeThreshold alpha

private theorem child_threshold (valid : ValidThreshold opts alpha) :
    ValidThreshold opts (-(alpha + 1)) := by
  intro f chosen
  obtain ⟨lo, hi⟩ := valid f chosen
  exact ⟨by omega, by omega⟩

def CorrectOptions (mc : X86Config) (fc : FlipConfig) (opts : Options mc) : Prop :=
  (∀ b table sq, sq ∈ opts.order b table ↔ sq ∈ legalSquares mc b.val) ∧
  ∀ f, opts.four = some f → ∀ (b : Position) (alpha : Int), (Board.emptyCount b.val).toNat = 4 →
    SafeThreshold alpha →
    SearchRange (f b alpha) ∧ NWS alpha (discValue mc fc b.val) (f b alpha)

structure Result where
  score : Int
  history : List Store

private def storeResult (table : Table) (b : Position) (alpha score : Int) (history : List Store) :
    Result :=
  ⟨score, history ++ [⟨table, ⟨b.val, alpha, score⟩⟩]⟩

private def negateResult (result : Result) : Result :=
  ⟨-result.score, result.history⟩

private def scan {Move : Type} (child : Move → List Store → Result) (alpha : Int) :
    List Move → Int → List Store → Result
  | [], best, history => ⟨best, history⟩
  | move :: rest, best, history =>
    let result := child move history
    if alpha + 1 ≤ result.score then result
    else scan child alpha rest (max best result.score) result.history

private def fourResult (address : Address) (solve : Position → Int → Int)
    (b : Position) (alpha : Int) (history : List Store) : Result :=
  match EndgameCache.probe (entryFor address history .shallow b.val) b.val alpha with
  | some score => ⟨score, history⟩
  | none =>
    match EndgameStability.tryCutoff b.val 4 ⟨alpha⟩ with
    | some score => ⟨score.value, history⟩
    | none => storeResult .shallow b alpha (solve b alpha) history

private def childTable (table : Table) (b : Position) : Table :=
  if table = .deep ∧ 6 < (Board.emptyCount b.val).toNat then .deep else .shallow

private def searchChild (opts : Options mc) (table : Table) (next : Position) (alpha : Int)
    (history : List Store) (call : Table → Int → List Store → Result) : Result :=
  let childAlpha := -(alpha + 1)
  match opts.four with
  | some solve =>
    if table = .shallow ∧ (Board.emptyCount next.val).toNat = 4 then
      negateResult (fourResult opts.address solve next childAlpha history)
    else negateResult (call (childTable table next) childAlpha history)
  | none => negateResult (call (childTable table next) childAlpha history)

def run (mc : X86Config) (fc : FlipConfig) (opts : Options mc) (table : Table)
    (b : Position) (alpha : Int) (history : List Store) : Result :=
  match moves : legalSquares mc b.val with
  | [] =>
    if available : Board.hasLegalMoves mc (pass b).val = true then
      negateResult (run mc fc opts table (pass b) (-(alpha + 1)) history)
    else ⟨Board.AwardedScore b.val, history⟩
  | first :: rest =>
    match EndgameCutoffs.beforeMoves fc b.val history opts.address table alpha
      ((first :: rest).map Subtype.val) with
    | some score => ⟨score, history⟩
    | none =>
      let child := fun sq input =>
        searchChild opts table (play mc fc b sq).val alpha input
          (fun tier a stores => run mc fc opts tier (play mc fc b sq).val a stores)
      let result := scan child alpha (opts.order b table) (-127) history
      storeResult table b alpha result.score result.history
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves, available⟩) b.property
  · exact (play mc fc b sq).property

private theorem seed_lower {Move : Type} (v : Move → Int) (moves : List Move) (seed : Int) :
    seed ≤ PvsMoveLoop.bestValue v moves seed := by
  induction moves generalizing seed with
  | nil => exact Int.le_refl _
  | cons sq rest ih =>
    exact Int.le_trans (Int.le_max_left _ _) (ih _)

private def Valid (mc : X86Config) (fc : FlipConfig) (history : List Store)
    (alpha value : Int) (result : Result) : Prop :=
  SearchRange result.score ∧ NWS alpha value result.score ∧
    soundHistory mc fc result.history ∧ history <+: result.history

private theorem unchanged {mc : X86Config} {fc : FlipConfig} {history : List Store}
    {alpha value score : Int} (sound : soundHistory mc fc history)
    (range : SearchRange score) (bounds : NWS alpha value score) :
    Valid mc fc history alpha value ⟨score, history⟩ :=
  ⟨range, bounds, sound, ⟨[], by simp⟩⟩

private theorem stored {mc : X86Config} {fc : FlipConfig} {history : List Store}
    {alpha : Int} {result : Result} (table : Table) (b : Position)
    (valid : Valid mc fc history alpha (discValue mc fc b.val) result) :
    Valid mc fc history alpha (discValue mc fc b.val)
      (storeResult table b alpha result.score result.history) := by
  obtain ⟨range, bounds, sound, grew⟩ := valid
  exact ⟨range, bounds,
    EndgameTreeCache.append_sound mc fc result.history sound _ (EndgameCache.searchRange_store range) bounds,
    grew.trans (List.prefix_append _ _)⟩

private theorem negated {mc : X86Config} {fc : FlipConfig} {history : List Store}
    {alpha value : Int} {result : Result}
    (valid : Valid mc fc history (-(alpha + 1)) value result) :
    Valid mc fc history alpha (-value) (negateResult result) := by
  obtain ⟨⟨lo, hi⟩, bounds, sound, grew⟩ := valid
  exact ⟨⟨by simp [negateResult]; omega, by simp [negateResult]; omega⟩,
    Solve34.negate_nws bounds, sound, grew⟩

private theorem scan_sound {Move : Type} (mc : X86Config) (fc : FlipConfig)
    (v : Move → Int) (child : Move → List Store → Result) (alpha : Int)
    (moves : List Move) (seed best : Int) (history : List Store)
    (sound : soundHistory mc fc history)
    (calls : ∀ sq input, sq ∈ moves → soundHistory mc fc input →
      Valid mc fc input alpha (v sq) (child sq input))
    (seedSound : NWS alpha seed best) (bestRange : SearchRange best) :
    Valid mc fc history alpha (PvsMoveLoop.bestValue v moves seed) (scan child alpha moves best history) := by
  induction moves generalizing seed best history with
  | nil =>
    exact unchanged sound bestRange seedSound
  | cons first rest ih =>
    have response := calls first history (by simp) sound
    let result := child first history
    change Valid mc fc history alpha (v first) result at response
    obtain ⟨range, bounds, after, grew⟩ := response
    rw [scan, PvsMoveLoop.bestValue]
    change Valid mc fc history alpha (PvsMoveLoop.bestValue v rest (max seed (v first)))
      (if alpha + 1 ≤ result.score then result else scan child alpha rest (max best result.score) result.history)
    split
    · rename_i high
      have valueBelow := Int.le_trans (Int.le_max_right seed (v first))
        (seed_lower v rest (max seed (v first)))
      exact ⟨range, ⟨fun h => by omega, fun h => Int.le_trans (bounds.2 h) valueBelow⟩, after, grew⟩
    · have merged := Solve34.max_nws bounds seedSound
      have mergedRange : SearchRange (max best result.score) := by
        unfold SearchRange at bestRange range ⊢
        omega
      have h := ih (max seed (v first)) (max best result.score) result.history after
        (fun sq input mem => calls sq input (List.mem_cons_of_mem _ mem))
        merged mergedRange
      exact ⟨h.1, h.2.1, h.2.2.1, grew.trans h.2.2.2⟩

private theorem four_sound (mc : X86Config) (fc : FlipConfig) (address : Address)
    (solve : Position → Int → Int) (b : Position) (alpha : Int) (history : List Store)
    (sound : soundHistory mc fc history)
    (solver : SearchRange (solve b alpha) ∧ NWS alpha (discValue mc fc b.val) (solve b alpha)) :
    Valid mc fc history alpha (discValue mc fc b.val) (fourResult address solve b alpha history) := by
  unfold fourResult
  split
  next score hit =>
    obtain ⟨range, bounds⟩ := EndgameTreeCache.probe_sound sound address .shallow b.val alpha score hit
    exact unchanged sound (EndgameCache.cacheRange_search range) bounds
  next =>
    split
    next score hit =>
      obtain ⟨lo, hi, bound, threshold⟩ :=
        (EndgameStability.cutoffContract_proven mc fc b 4 ⟨alpha⟩).2 score hit
      have scaled := EndgameTreeCache.scaled_discValue mc fc b
      change score.value ≤ alpha at threshold
      exact unchanged sound ⟨by omega, by omega⟩ ⟨fun _ => by omega, fun _ => by omega⟩
    next => exact stored .shallow b (unchanged sound solver.1 solver.2)

private theorem child_sound (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (correct : CorrectOptions mc fc opts) (table : Table) (next : Position) (alpha : Int)
    (history : List Store) (sound : soundHistory mc fc history)
    (window : ValidThreshold opts (-(alpha + 1)))
    (call : Table → Int → List Store → Result)
    (valid : Valid mc fc history (-(alpha + 1)) (discValue mc fc next.val)
      (call (childTable table next) (-(alpha + 1)) history)) :
    Valid mc fc history alpha (-discValue mc fc next.val) (searchChild opts table next alpha history call) := by
  unfold searchChild
  split
  next solve chosen =>
    split
    next four =>
      exact negated (four_sound mc fc opts.address solve next _ history sound
        (correct.2 solve chosen next _ four.2 (window solve chosen)))
    next => exact negated valid
  next => exact negated valid

theorem run_sound (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (correct : CorrectOptions mc fc opts) (table : Table) (b : Position) (alpha : Int)
    (history : List Store) (sound : soundHistory mc fc history) (window : ValidThreshold opts alpha) :
    Valid mc fc history alpha (discValue mc fc b.val) (run mc fc opts table b alpha history) := by
  rw [run.eq_def]
  split
  next moves =>
    split
    next available =>
      have forced : Board.ForcedPass mc b.val := ⟨PvsBoardTree.stuck_of_nil moves, available⟩
      have a := EndgameTreeCache.scaled_discValue mc fc b
      have c := EndgameTreeCache.scaled_discValue mc fc (pass b)
      rw [PvsBoardTree.minimax_pass mc fc b moves available] at a
      have eq : discValue mc fc b.val = -discValue mc fc (pass b).val := by omega
      rw [eq]
      exact negated (run_sound mc fc opts correct table (pass b) (-(alpha + 1)) history sound
        (child_threshold window))
    next unavailable =>
      have no : Board.hasLegalMoves mc (pass b).val = false := by
        cases available : Board.hasLegalMoves mc (pass b).val <;> simp_all
      have scaled := EndgameTreeCache.scaled_discValue mc fc b
      rw [PvsBoardTree.terminal_value mc fc b moves no] at scaled
      have eq : discValue mc fc b.val = Board.AwardedScore b.val := by omega
      have ⟨lo, hi⟩ := EndgameTreeCache.discValue_range mc fc b.val
      exact unchanged sound ⟨by omega, by omega⟩ (eq ▸ Solve2.nws_exact _ _)
  next first rest moves =>
    split
    next score hit =>
      obtain ⟨range, bounds⟩ := EndgameCutoffs.contract_proven mc fc b history opts.address table
        alpha score (first :: rest) sound (fun sq member => moves.symm ▸ member) hit
      exact unchanged sound (EndgameCache.cacheRange_search range) bounds
    next =>
      apply stored table b
      let v := fun sq => -discValue mc fc (play mc fc b sq).val.val
      have eq : discValue mc fc b.val = PvsMoveLoop.bestValue v (opts.order b table) (-127) := by
        rw [EndgameTreeCache.discValue_moves mc fc b first rest moves]
        apply PvsMoveLoop.bestValue_eq_of_same_moves
        intro sq
        rw [← moves]
        exact (correct.1 b table sq).symm
      rw [eq]
      apply scan_sound mc fc v _ alpha (opts.order b table) (-127) (-127) history sound
      · intro sq input _ valid
        exact child_sound mc fc opts correct table (play mc fc b sq).val alpha input valid
          (child_threshold window) _
          (run_sound mc fc opts correct _ (play mc fc b sq).val _ input valid (child_threshold window))
      · exact Solve2.nws_exact _ _
      · exact ⟨by decide, by decide⟩
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass forced) b.property
  · exact (play mc fc b sq).property

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (opts : Options mc) (table : Table)
    (b : Position) (alpha : Int) (history : List Store),
    CorrectOptions mc fc opts → soundHistory mc fc history → ValidThreshold opts alpha →
    let result := run mc fc opts table b alpha history
    SearchRange result.score ∧ NWS alpha (discValue mc fc b.val) result.score ∧
      soundHistory mc fc result.history ∧ history <+: result.history

theorem contract_proven : Contract := by
  intro mc fc opts table b alpha history correct sound window
  exact run_sound mc fc opts correct table b alpha history sound window

def RawWindowContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (opts : Options mc) (table : Table)
    (b : Position) (alpha : Int) (history : List Store),
    CorrectOptions mc fc opts → soundHistory mc fc history → ValidThreshold opts (alpha / 256) →
    let result := run mc fc opts table b (alpha / 256) history
    Aspiration.SoundResult (minimax mc fc b) alpha (alpha + 1) (256 * result.score) ∧
      soundHistory mc fc result.history ∧ history <+: result.history

theorem rawWindowContract_proven : RawWindowContract := by
  intro mc fc opts table b alpha history correct sound window
  obtain ⟨range, bounds, after, grew⟩ :=
    contract_proven mc fc opts table b (alpha / 256) history correct sound window
  exact ⟨EndgameTreeCache.rawWindow_sound mc fc b alpha _ range bounds, after, grew⟩

theorem cold_sound (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (correct : CorrectOptions mc fc opts) (table : Table) (b : Position) (alpha : Int)
    (window : ValidThreshold opts alpha) :
    let result := run mc fc opts table b alpha []
    NWS alpha (discValue mc fc b.val) result.score ∧ soundHistory mc fc result.history := by
  have ready : soundHistory mc fc [] := by simp [soundHistory]
  have proof := contract_proven mc fc opts table b alpha [] correct ready window
  exact ⟨proof.2.1, proof.2.2.1⟩

theorem cold_generated_order_sound (mc : X86Config) (fc : FlipConfig) (address : Address)
    (table : Table) (b : Position) (alpha : Int) :
    let opts : Options mc := ⟨address, fun p _ => legalSquares mc p.val, none⟩
    let result := run mc fc opts table b alpha []
    NWS alpha (discValue mc fc b.val) result.score ∧ soundHistory mc fc result.history := by
  let opts : Options mc := ⟨address, fun p _ => legalSquares mc p.val, none⟩
  exact cold_sound mc fc opts ⟨fun _ _ _ => Iff.rfl, fun _ h => by cases h⟩ table b alpha
    (fun _ h => by cases h)

end NeuralReversi.EndgameSearch
