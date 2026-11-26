import NeuralReversi.PvsParallel.Root

namespace NeuralReversi.PvsParallel.ExactTableHistory

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position minimax)
open TranspositionTable
open Cutoffs (History FlipConfig SoundHistory ExactEndgame exactSelectivity key PublishedProbe)

inductive Admission (mc : X86Config) (fc : FlipConfig) (history : History) :
    Position → EntryData → Prop
  | other (board : Position) (entry : EntryData) :
      ¬ExactEndgame entry → Admission mc fc history board entry
  | penalized (board : Position) (entry : EntryData) :
      (board, entry) ∈ history → 0 < entry.depth.toNat →
      Admission mc fc history board (Packing.penalized entry)
  | searched (board : Position) (pv : Bool)
      (alpha beta score : Score) (move : Option (Fin 64)) (depth : Depth) (generation : BitVec 8)
      (ttPv : Bool) (previousKey : Save.Key) (previous written : EntryData) :
      Window alpha.raw.toInt beta.raw.toInt →
      SoundResult (minimax mc fc board) alpha.raw.toInt beta.raw.toInt score.raw.toInt →
      Save.proposal (key board) previousKey
        (Packing.newData score move (classify pv score alpha beta) depth
          exactSelectivity generation true ttPv) previous = some written →
      Admission mc fc history board written

theorem Admission.sound (write : Admission mc fc history board entry) (prior : SoundHistory mc fc history) :
    SoundHistory mc fc (history ++ [(board, entry)]) := by
  cases write with
  | other _ outside => exact Cutoffs.SoundHistory.append_other prior _ _ outside
  | penalized entry source _ => exact Cutoffs.SoundHistory.append_penalty prior _ entry source
  | searched pv alpha beta score move depth gen ttPv previousKey previous written window result saved =>
    exact (Cutoffs.storeContract_proven mc fc _ _ pv alpha beta score move depth gen ttPv
      previousKey previous _ prior window result saved).2

inductive Publications (mc : X86Config) (fc : FlipConfig) : History → Prop
  | initial : Publications mc fc []
  | append : Publications mc fc history → Admission mc fc history board entry →
      Publications mc fc (history ++ [(board, entry)])

theorem Publications.sound (trace : Publications mc fc history) : SoundHistory mc fc history := by
  induction trace with
  | initial => simp [SoundHistory]
  | append _ write ih => exact write.sound ih

theorem Admission.saved (history : History) (board : Position) (pv : Bool)
    (alpha beta score : Score) (move : Option (Fin 64)) (depth : Depth) (generation : BitVec 8) (ttPv : Bool)
    (window : Window alpha.raw.toInt beta.raw.toInt)
    (result : SoundResult (minimax mc fc board) alpha.raw.toInt beta.raw.toInt score.raw.toInt)
    (written : Save.attempt bigEndian (key board)
      (Packing.newData score move (classify pv score alpha beta) depth exactSelectivity generation true ttPv)
      observed = .written seq entry) : Admission mc fc history board entry := by
  obtain ⟨previousKey, previous, proposed⟩ := Save.attemptProposal written
  exact .searched board pv alpha beta score move depth generation ttPv previousKey previous entry
    window result proposed

theorem Admission.otherSaved (history : History) (board : Position) (incoming : EntryData)
    (outside : ¬ExactEndgame incoming)
    (written : Save.attempt bigEndian (key board) incoming observed = .written seq entry) :
    Admission mc fc history board entry := by
  obtain ⟨previousKey, previous, proposed⟩ := Save.attemptProposal written
  rw [(Save.contract_proven (key board) previousKey incoming previous).1] at proposed
  split at proposed
  · contradiction
  · have data := (Option.some.inj proposed).symm
    refine Admission.other board entry ?_
    intro admitted
    apply outside
    rw [data] at admitted
    unfold Save.payload at admitted
    split at admitted <;> exact admitted

theorem Admission.completed (history : History) (board : Position)
    (pv : Bool) (alpha beta score : Score) (move : Option (Fin 64)) (depth : Depth)
    (generation : BitVec 8) (ttPv : Bool)
    (completed : Search) (follows : Follows completed)
    (window : Window alpha.raw.toInt beta.raw.toInt)
    (narrow : pv = false → beta.raw.toInt = alpha.raw.toInt + 1)
    (trace : completed (PvsBoardTree.build mc fc board) pv alpha.raw.toInt beta.raw.toInt score.raw.toInt)
    (writtenData : Save.attempt bigEndian (key board)
      (Packing.newData score move (classify pv score alpha beta) depth exactSelectivity generation
        true ttPv) observed = .written seq entry) :
    Admission mc fc history board entry :=
  .saved history board pv alpha beta score move depth generation ttPv
    window (boardContract_proven completed follows mc fc board pv _ _ _ window narrow trace) writtenData

theorem Admission.rootCompleted (history : History) (board : Position)
    (roots : List Bitboard.Square) (idx : Nat) (alpha beta score : Score)
    (move : Option (Fin 64)) (depth : Depth) (generation : BitVec 8)
    (ttPv : Bool)
    (completed : Search) (follows : Follows completed)
    (covers : Root.Covers mc board roots) (inBounds : idx < roots.length)
    (guard : Root.storeAllowed true idx = true)
    (window : Window alpha.raw.toInt beta.raw.toInt)
    (trace : Root.Execution completed mc fc board roots idx alpha.raw.toInt beta.raw.toInt
      score.raw.toInt)
    (writtenData : Save.attempt bigEndian (key board)
      (Packing.newData score move (classify true score alpha beta) depth exactSelectivity generation
        true ttPv) observed = .written seq entry) :
    Admission mc fc history board entry := by
  have whole := (Root.storeGateContract_proven mc fc board roots idx covers inBounds).2 guard
  have result := Root.scoreContract_proven completed follows mc fc board roots idx _ _ _ window trace
  rw [whole] at result
  exact .saved history board true alpha beta score move depth generation ttPv window result writtenData

def Contract : Prop :=
  ∀ mc fc board history attempts slot entry (alpha beta : Score) depth checked,
    Publications mc fc history → PublishedProbe history board attempts →
    beta.raw.toInt = alpha.raw.toInt + 1 →
    ProbeRetry.probe (attempts.map (ProbeRetry.observe (key board))) = .hit slot entry →
    canCut checked entry beta depth exactSelectivity true = .ok true →
    PvsMoveLoop.Bound (minimax mc fc board) alpha.raw.toInt beta.raw.toInt entry.score.toInt

theorem contract_proven : Contract := by
  intro mc fc board history attempts slot entry alpha beta depth checked historyTrace
    reads narrow hit cut
  exact Cutoffs.publishedCutContract_proven mc fc board history attempts slot entry alpha beta depth
    checked historyTrace.sound reads narrow hit cut

end NeuralReversi.PvsParallel.ExactTableHistory
