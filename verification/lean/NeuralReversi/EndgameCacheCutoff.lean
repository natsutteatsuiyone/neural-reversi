module
public import NeuralReversi.EndgameCache
public import NeuralReversi.ScoreRefresh.Properties
public import NeuralReversi.TranspositionTable.Save
public import NeuralReversi.TranspositionTable.StoredBound

@[expose] public section

namespace NeuralReversi.EndgameCacheCutoff

open Board (Board)
open EndgameCache (Entry Write InRange CacheRange StoreRange unset store probe)
open TranspositionTable (Bound Depth EntryData Selectivity BoundSound)

def selectivity : Selectivity := ⟨5, by decide⟩

def failLowScore (entry : Entry) (board : Board) (alpha : Int) : Option Int :=
  match probe entry board alpha with
  | some score => if score ≤ alpha then some score else none
  | none => none

def childAlpha (beta : ScoreRefresh.Score) : ScoreRefresh.DiscScore :=
  ScoreRefresh.toDiscDiff ⟨-beta.raw⟩

abbrev MoveBoard := Bitboard.Square × Board
abbrev Hit := Bitboard.Square × Int

def candidate (entryFor : Board → Entry) (beta : ScoreRefresh.Score) (m : MoveBoard) : Option Hit := do
  let score ← failLowScore (entryFor m.2) m.2 (childAlpha beta).raw.toInt
  some (m.1, -score)

def scan (entryFor : Board → Entry) (beta : ScoreRefresh.Score) (moves : List MoveBoard) : Option Hit :=
  moves.findSome? (candidate entryFor beta)

theorem failLowScore_sound (value : Board → Int) (bounded : ∀ b, InRange (value b))
    (writes : List Write) (valid : ∀ w ∈ writes, StoreRange w.score ∧ Solve2.NWS w.alpha (value w.board) w.score)
    (board : Board) (alpha score : Int)
    (hit : failLowScore (writes.foldl store (unset ⟨0, 0⟩)) board alpha = some score) :
    CacheRange score ∧ score ≤ alpha ∧ value board ≤ score ∧ (alpha ≤ 64 → InRange score) := by
  let entry := writes.foldl store (unset ⟨0, 0⟩)
  change failLowScore entry board alpha = some score at hit
  unfold failLowScore at hit
  cases probed : probe entry board alpha with
  | none =>
    rw [probed] at hit
    cases hit
  | some raw =>
    rw [probed] at hit
    dsimp only at hit
    split at hit
    next le =>
      cases Option.some.inj hit
      obtain ⟨range, nws⟩ := EndgameCache.anyWindowContract_proven value bounded writes valid board alpha score probed
      have upper : value board ≤ score := nws.1 le
      have valRange := bounded board
      refine ⟨range, le, upper, fun cap => ?_⟩
      unfold InRange at valRange ⊢
      omega
    next => cases hit

private theorem neg_score_toInt (beta : ScoreRefresh.Score)
    (lo : -32767 ≤ beta.raw.toInt) (hi : beta.raw.toInt ≤ 32767) :
    (-beta.raw).toInt = -beta.raw.toInt := by
  have noOverflow : beta.raw ≠ BitVec.intMin 32 := by
    intro h
    have v := congrArg BitVec.toInt h
    change beta.raw.toInt = -2147483648 at v
    omega
  exact BitVec.toInt_neg_of_ne_intMin noOverflow

private theorem childAlpha_toInt (beta : ScoreRefresh.Score)
    (lo : -32767 ≤ beta.raw.toInt) (hi : beta.raw.toInt ≤ 32767) :
    (childAlpha beta).raw.toInt = (-beta.raw.toInt) / 256 := by
  have floor := ScaledScore.disc_floor (ScoreRefresh.legacy ⟨-beta.raw⟩)
  change (childAlpha beta).raw.toInt = (-beta.raw).toInt / 256 at floor
  rw [floor, neg_score_toInt beta lo hi]

private theorem scaled_ge_beta {beta childScore : Int}
    (failLow : childScore ≤ (-beta) / 256) :
    beta ≤ (-childScore) * 256 := by
  have rem := Int.emod_nonneg (-beta) (by decide : (256 : Int) ≠ 0)
  have divEq := Int.mul_ediv_add_emod (-beta) 256
  omega

private theorem disc_word_toInt {d : Int} (lo : -127 ≤ d) (hi : d ≤ 127) :
    (BitVec.ofInt 32 d).toInt = d := by
  rw [BitVec.toInt_ofInt]
  exact Int.bmod_eq_of_le (by omega) (by omega)

def Contract : Prop :=
  ∀ (value : Board → Int), (∀ b, InRange (value b)) →
    ∀ (writesFor : Board → List Write),
    (∀ child w, w ∈ writesFor child → StoreRange w.score ∧ Solve2.NWS w.alpha (value w.board) w.score) →
    ∀ (beta : ScoreRefresh.Score), -32767 ≤ beta.raw.toInt → beta.raw.toInt ≤ 32767 →
    ∀ (moves : List MoveBoard) (parentValue : Int),
    (∀ m ∈ moves, -(256 * value m.2) ≤ parentValue) →
    ∀ (sq : Bitboard.Square) (negDisc : Int),
    scan (fun b => (writesFor b).foldl store (unset ⟨0, 0⟩)) beta moves = some (sq, negDisc) →
    (∃ before child after, moves = before ++ (sq, child) :: after ∧
      (∀ prev ∈ before, candidate (fun b => (writesFor b).foldl store (unset ⟨0, 0⟩)) beta prev = none) ∧
      value child ≤ -negDisc) ∧
    ∀ (checked : Bool),
      ∃ score : ScoreRefresh.Score,
        ScoreRefresh.negate checked beta = .ok ⟨-beta.raw⟩ ∧
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 negDisc⟩ = .ok score ∧
        score.raw.toInt = negDisc * 256 ∧
        -16640 ≤ score.raw.toInt ∧ score.raw.toInt ≤ 16640 ∧
        (-16384 ≤ beta.raw.toInt → -16384 ≤ score.raw.toInt ∧ score.raw.toInt ≤ 16384) ∧
        beta.raw.toInt ≤ score.raw.toInt ∧
        score.raw.toInt ≤ parentValue ∧
        ∀ (depth : Depth) (gen : BitVec 8) (ttPv : Bool)
          (parentKey prevKey : TranspositionTable.Save.Key) (prev written : EntryData),
          let incoming := TranspositionTable.Packing.newData ⟨score.raw⟩ (some sq) Bound.lower
            depth selectivity gen true ttPv
          TranspositionTable.Save.proposal parentKey prevKey incoming prev = some written →
          written.score.toInt = score.raw.toInt ∧ BoundSound written parentValue

theorem contract_proven : Contract := by
  intro value bounded writesFor valid beta betaLo betaHi moves parentValue dominates sq negDisc found
  obtain ⟨before, m, after, splitMoves, selected, earlier⟩ :=
    (List.findSome?_eq_some_iff (l := moves) (f := candidate _ beta) (b := (sq, negDisc))).mp found
  rcases m with ⟨msq, child⟩
  let entry := (writesFor child).foldl store (unset ⟨0, 0⟩)
  change (do let score ← failLowScore entry child (childAlpha beta).raw.toInt; some (msq, -score)) =
    some (sq, negDisc) at selected
  cases hit : failLowScore entry child (childAlpha beta).raw.toInt with
  | none =>
    rw [hit] at selected
    cases selected
  | some childScore =>
    rw [hit] at selected
    simp only [bind, Option.bind, Option.some.injEq, Prod.mk.injEq] at selected
    obtain ⟨rfl, rfl⟩ := selected
    obtain ⟨range, leAlpha, childUpper, tightRange⟩ :=
      failLowScore_sound value bounded (writesFor child) (valid child) child _ childScore hit
    have alphaEq := childAlpha_toInt beta betaLo betaHi
    rw [alphaEq] at leAlpha
    have geBeta : beta.raw.toInt ≤ (-childScore) * 256 := scaled_ge_beta leAlpha
    have inMem : (msq, child) ∈ moves := by rw [splitMoves]; simp
    have dom : -(256 * value child) ≤ parentValue := dominates (msq, child) inMem
    have negRange : -65 ≤ -childScore ∧ -childScore ≤ 65 := by
      unfold CacheRange at range
      omega
    have wordVal : (BitVec.ofInt 32 (-childScore)).toInt = -childScore :=
      disc_word_toInt (by omega) (by omega)
    have scaledVal : (BitVec.ofInt 32 (-childScore) <<< 8).toInt = (-childScore) * 256 := by
      rw [ScaledScore.bounded_scaled_value _ (by omega) (by omega), wordVal]
    refine ⟨⟨before, child, after, splitMoves, earlier, by omega⟩, ?_⟩
    intro checked
    let score : ScoreRefresh.Score := ⟨BitVec.ofInt 32 (-childScore) <<< 8⟩
    have negOk : ScoreRefresh.negate checked beta = .ok ⟨-beta.raw⟩ := by
      unfold ScoreRefresh.negate
      have noOvf : beta.raw.negOverflow = false := by
        rw [BitVec.negOverflow_eq]
        have neMin : beta.raw ≠ BitVec.intMin 32 := by
          intro h
          have v := congrArg BitVec.toInt h
          change beta.raw.toInt = -2147483648 at v
          omega
        simp [neMin]
      simp [noOvf]
    have fromOk : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 (-childScore)⟩ = .ok score := by
      unfold ScoreRefresh.fromDiscDiff
      have ok : ¬ ((BitVec.ofInt 32 (-childScore)).toInt < -127 ∨
          (BitVec.ofInt 32 (-childScore)).toInt > 127) := by omega
      simp only [ok, decide_false, Bool.and_false, Bool.false_eq_true, ↓reduceIte, score]
    have scoreEq : score.raw.toInt = (-childScore) * 256 := scaledVal
    have scoreLo : -16640 ≤ score.raw.toInt := by omega
    have scoreHi : score.raw.toInt ≤ 16640 := by omega
    have scoreTight : -16384 ≤ beta.raw.toInt → -16384 ≤ score.raw.toInt ∧ score.raw.toInt ≤ 16384 := by
      intro betaMin
      have alphaCap : (childAlpha beta).raw.toInt ≤ 64 := by
        rw [alphaEq]
        omega
      have r := tightRange alphaCap
      unfold InRange at r
      omega
    have scoreLeParent : score.raw.toInt ≤ parentValue := by omega
    refine ⟨score, negOk, fromOk, scoreEq, scoreLo, scoreHi, scoreTight, by omega, scoreLeParent, ?_⟩
    intro depth gen ttPv parentKey prevKey prev written incoming saved
    have narrow : (score.raw.setWidth 16).toInt = score.raw.toInt :=
      TranspositionTable.StoredBound.narrow_score ⟨score.raw⟩ (by change -32767 < score.raw.toInt; omega)
        (by change score.raw.toInt < 32767; omega)
    have incomingSound : BoundSound incoming parentValue := by
      refine ⟨fun _ => ?_, fun h => by simp [incoming, TranspositionTable.Packing.newData,
        TranspositionTable.Packing.boundByte] at h⟩
      simpa [incoming, TranspositionTable.Packing.newData, narrow] using scoreLeParent
    have prop := (TranspositionTable.Save.contract_proven parentKey prevKey incoming prev).1
    rw [prop] at saved
    split at saved
    · contradiction
    · have eq := Option.some.inj saved
      subst written
      have fields : (TranspositionTable.Save.payload parentKey prevKey incoming prev).score = incoming.score ∧
          (TranspositionTable.Save.payload parentKey prevKey incoming prev).bound = incoming.bound := by
        unfold TranspositionTable.Save.payload
        split <;> exact ⟨rfl, rfl⟩
      refine ⟨by rw [fields.1]; simpa [incoming, TranspositionTable.Packing.newData] using narrow, ?_⟩
      exact ⟨fun h => fields.1 ▸ incomingSound.1 (fields.2 ▸ h),
             fun h => fields.1 ▸ incomingSound.2 (fields.2 ▸ h)⟩

end NeuralReversi.EndgameCacheCutoff
