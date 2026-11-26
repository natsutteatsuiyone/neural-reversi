import NeuralReversi.PvsParallel.SerialTable.MidgameValue
import NeuralReversi.Stability.Cutoff

namespace NeuralReversi.PvsParallel.SerialTable.MidgameNativeWords

open AspirationMir.SerialTable.Repeated (ReadyMemory)
open SerialCalls (Request)
open SerialCalls.ScoreWords (Finite)
open TranspositionTable
open PvsBoardTree (Position)
open EndgameSearch.CacheHistory (Pool)

/-- The entry probe supplies both the score read now and the copy held across verification. -/
theorem sampled (m : Memory l) (ok : ReadyMemory p mc fc m) (b : Position)
    (read : m.tt.image.probe b = .hit slot e) : Finite e.score.toInt := by
  have reads := m.tt.image.probe_reads m.supported b
  obtain ⟨attempt,tried,seen,occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ read
  exact ok.1.2 b e (reads.published.matches slot e attempt tried seen occupied)

theorem margin (m : Memory l) (ok : ReadyMemory p mc fc m) (q : Request)
    (read : m.tt.image.probe q.board = .hit slot e) (window : Aspiration.Window q.alpha q.beta)
    (eligible : q.beta + MidgameScoreRange.ttMargin ≤ e.score.toInt) :
    Finite (q.beta + MidgameScoreRange.ttMargin) :=
  MidgameScoreRange.margin_finite window (sampled m ok q.board read) eligible

theorem candidate (e : EntryData) (finite : Finite e.score.toInt) (q : PvsEnhancedCutoff.Query)
    (sq : Bitboard.Square) (hit : PvsEnhancedCutoff.candidate q (sq,some e) = some ans) : Finite ans.2 := by
  simp only [PvsEnhancedCutoff.candidate,bind,Option.bind] at hit
  split at hit
  · cases Option.some.inj hit
    change Finite (-(e.score.signExtend 32)).toInt
    have widened : (e.score.signExtend 32).toInt = e.score.toInt := BitVec.toInt_signExtend_of_le (by decide)
    have neg : (-(e.score.signExtend 32)).toInt = -e.score.toInt := by
      apply (BitVec.toInt_neg_of_ne_intMin ?_).trans (congrArg Neg.neg widened)
      intro bad
      rw [bad] at widened
      change -2147483648 = e.score.toInt at widened
      unfold Finite at finite
      omega
    rw [neg]
    exact MidgameScoreRange.finite_neg finite
  · contradiction

theorem enhanced (m : Memory l) (ok : ReadyMemory p mc fc m) (next : Position)
    (q : PvsEnhancedCutoff.Query) (sq : Bitboard.Square)
    (hit : PvsEnhancedCutoff.candidate q (sq,m.tt.image.lookup next) = some ans) : Finite ans.2 := by
  cases read : m.tt.image.lookup next with
  | none => simp [PvsEnhancedCutoff.candidate,read] at hit
  | some e =>
    obtain ⟨source⟩ := m.tt.image.lookup_source m.supported next read
    exact candidate e (ok.1.2 next e source.source) q sq (read ▸ hit)

theorem stability (q : Request) (alpha : Stability.Cutoff.DiscScore)
    (hit : Stability.Cutoff.cutoffAtEmptyCount q.board.val.player q.board.val.opponent
        (Board.emptyCount q.board.val).toNat alpha = some score) : Finite (256 * score.value) := by
  have value := ((Stability.Cutoff.sourceCutoffContract_proven ..).2 score hit).1
  have count : Stability.Cutoff.popcount (Stability.Cutoff.cutoffStable q.board.val.opponent q.board.val.player alpha) ≤ 64 :=
    calc _ ≤ (List.finRange 64).length := List.countP_le_length
         _ = 64 := List.length_finRange
  unfold Finite
  omega

theorem after_child (p : Pool) (m : Memory l) (q : Request) (child : MidgameChild.Plan mc fc q)
    (win : Aspiration.Window q.alpha q.beta) (ready : ReadyMemory p mc fc m) :
    let returned := (MidgameChild.eval p child m).1
    Finite (-returned) ∧ Finite (max (-32767) (-returned)) ∧
      ∀ best, Finite best → Finite (max best (-returned)) := by
  have inside := MidgameScoreRange.finite_neg (MidgameChild.eval_ready p child m win ready).2.1
  exact ⟨inside, by rwa [Int.max_eq_right (Int.le_of_lt inside.1)],
    fun _ earlier => MidgameScoreRange.finite_max earlier inside⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameNativeWords
