module
public import NeuralReversi.TranspositionTable.WriterMir.Store
public import NeuralReversi.TranspositionTable.WriterMir.SourceProof

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterMir

open PvsParallel
open PvsBoardTree (Position build)
open CausalReplay (SearchSave PrefixWriteFrom SearchHint)

theorem makeData_agrees (score move bound depth selectivity generation endgame pv) :
    makeData score move bound depth selectivity generation endgame pv =
      Packing.newData score move bound depth selectivity generation endgame pv := by
  cases endgame <;> cases pv <;> rfl

theorem store_input {σ β : Type} (slot : σ) (board : β)
    (score bound depth move selectivity generation endgame pv) :
    storeInput slot board score bound depth move selectivity generation endgame pv =
      if score.raw.toInt.natAbs < 32767 then
        some (slot,board,Packing.newData score move bound depth selectivity generation endgame pv)
      else none := by
  have mask : (generation &&& 127#8) &&& 127#8 = generation &&& 127#8 := by rw [BitVec.and_assoc, BitVec.and_self]
  simp [storeInput, makeData_agrees, Packing.newData, mask]

/-- A vacant snapshot reads just metadata between the two sequence loads. Only an occupied
    snapshot has key loads from which the writer can inherit a move. -/
def observations (source : Cutoffs.RawSample → Prop) (endian : Bool)
    (obs : Save.Observation) (step : SourceStep) : Prop :=
  .snapshot ∈ step.ops → ∀ s, Snapshot.read endian obs.first obs.last obs.sample = some s →
    (Packing.unpack endian s.metadata).bound ≠ 0 → ∃ raw, source raw ∧ raw.bigEndian = endian ∧
    raw.first = obs.first ∧ raw.last = obs.last ∧ raw.payload = obs.sample

theorem preserves_observed_hint {board : Position} {source endian entry obs}
    (recorded : observations source endian obs
      (saveRound endian board.val.player board.val.opponent entry obs)) :
    TableReads.PreservedHintFrom source board endian entry obs := by
  intro snapshot bound _ sample _ occupied
  apply recorded _ snapshot sample occupied
  simp only [saveRound, sample, Option.getD_some, Option.isSome_some]
  rw [expand_eq (ordinary_stop bound ..).1]
  simp only [saveStep, bound, ↓reduceIte]
  repeat' split
  all_goals simp

def storeRound (event : Nat → Nat) (endian : Bool) (slot : Nat) (board : Position)
    (save : SearchSave) (bound : Bound) : Option (Request × WriterTrace.Call Nat) := do
  let (_,b,entry) ← storeInput slot board save.score bound save.depth save.move Cutoffs.exactSelectivity
    save.generation true save.ttPv
  publishRound event endian save.observed.current
    (saveRound endian b.val.player b.val.opponent entry save.observed)

theorem store_publication {board : Position} (frame : Option Nat)
    {event endian slot save bound req call}
    (hit : storeRound event endian slot board save bound = some (req,call)) :
    Save.attempt endian (Cutoffs.key board) (save.incoming bound) save.observed =
        .written (SeqLock.publish call.expected) req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  unfold storeRound at hit
  rw [store_input] at hit
  split at hit <;> try contradiction
  exact save_publication frame hit

theorem exact_prefix {board : Position} (frame : Option Nat)
    {mc fc computed source event endian slot} (save : SearchSave) (pv : Bool)
    {req call} (window : Aspiration.Window save.alpha.raw.toInt save.beta.raw.toInt)
    (scout : pv = false → save.beta.raw.toInt = save.alpha.raw.toInt + 1)
    (result : computed (build mc fc board) pv save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt)
    (hint : SearchHint mc board pv save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt save.move)
    (reads : observations source endian save.observed (saveRound endian board.val.player board.val.opponent
      (save.incoming (classify pv save.score save.alpha save.beta)) save.observed))
    (hit : storeRound event endian slot board save (classify pv save.score save.alpha save.beta) =
      some (req,call)) :
    PrefixWriteFrom mc fc endian computed source board req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  have accepted := store_publication frame hit
  let s := {save with sequence := SeqLock.publish call.expected}
  exact ⟨.exact s pv window scout result hint (preserves_observed_hint reads) accepted.1,accepted.2⟩

theorem root_prefix {board : Position} (frame : Option Nat)
    {mc fc computed source event endian slot} (save : SearchSave) (roots : List Bitboard.Square) (idx : Nat)
    {req call} (window : Aspiration.Window save.alpha.raw.toInt save.beta.raw.toInt)
    (covers : Root.Covers mc board roots) (index : idx < roots.length) (first : Root.storeAllowed true idx = true)
    (result : Root.Execution computed mc fc board roots idx save.alpha.raw.toInt save.beta.raw.toInt
      save.score.raw.toInt)
    (hint : SearchHint mc board true save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt save.move)
    (reads : observations source endian save.observed (saveRound endian board.val.player board.val.opponent
      (save.incoming (classify true save.score save.alpha save.beta)) save.observed))
    (hit : storeRound event endian slot board save (classify true save.score save.alpha save.beta) =
      some (req,call)) :
    PrefixWriteFrom mc fc endian computed source board req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  have accepted := store_publication frame hit
  let s := {save with sequence := SeqLock.publish call.expected}
  exact ⟨.root s roots idx window covers index first result hint (preserves_observed_hint reads) accepted.1,accepted.2⟩

end NeuralReversi.TranspositionTable.WriterMir

