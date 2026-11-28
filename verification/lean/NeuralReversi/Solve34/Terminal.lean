module
public import NeuralReversi.Solve34.BoardBridge
public import NeuralReversi.Board.ScoreProperties

@[expose] public section

namespace NeuralReversi.Solve34

theorem emptyMask_count (ss : List Square) (h : ss.Nodup) :
    (Bitboard.count (emptyMask ss)).toNat = ss.length := by
  induction ss with
  | nil => simp [emptyMask, Bitboard.count]
  | cons s ss ih =>
    have hn := List.nodup_cons.mp h
    have hd : Bitboard.squareMask s &&& emptyMask ss = 0 := by
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      have hm := emptyMask_bit ss ⟨i,hi⟩
      by_cases hs : i = s.val
      · have he : (emptyMask ss).getLsbD i = false := by
          cases hb : (emptyMask ss).getLsbD i with
          | false => rfl
          | true =>
            have mem := hm.mp hb
            have eq : (⟨i,hi⟩ : Square) = s := Fin.ext hs
            exact False.elim (hn.1 (eq ▸ mem))
        simp [Bitboard.squareMask_bit, he, -BitVec.getLsbD_eq_getElem]
      · simp [Bitboard.squareMask_bit, hs, -BitVec.getLsbD_eq_getElem]
    have hc := Bitboard.count_or_of_disjoint _ _ hd
    rw [Board.squareMask_count_one, ih hn.2] at hc
    simpa [emptyMask, Nat.add_comm] using hc

theorem exactly_n_empty_nat (b : EmptyBoard n) :
    (Board.emptyCount b.asBoard).toNat = n := by
  change (Bitboard.count (~~~(b.player ||| b.opponent))).toNat = n
  rw [b.empty_exact, emptyMask_count b.squares b.nodup, b.length_eq]

theorem exactly_n_empty (b : EmptyBoard n) :
    Board.emptyCount b.asBoard = BitVec.ofNat 32 n := by
  have h := exactly_n_empty_nat b
  exact (BitVec.ofNat_toNat 32 (Board.emptyCount b.asBoard)).symm.trans (congrArg _ h)

def terminalScore (b : EmptyBoard n) : BitVec 32 :=
  Board.finishScore (Board.discScore b.asBoard) (BitVec.ofNat 32 n)

theorem terminal_solve (b : EmptyBoard n) (checked : Bool) :
    Board.solve checked b.asBoard (BitVec.ofNat 32 n) = .ok (terminalScore b) := by
  simp [Board.solve, exactly_n_empty b, terminalScore]

theorem terminal_awarded (b : EmptyBoard n) :
    (terminalScore b).toInt = Board.AwardedScore b.asBoard := by
  obtain ⟨result, release, _, policy, _⟩ := Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_n_empty b, terminal_solve b false] at release
  cases release
  exact policy

theorem terminal_bounds (b : EmptyBoard n) :
    -64 ≤ (terminalScore b).toInt ∧ (terminalScore b).toInt ≤ 64 := by
  obtain ⟨result, release, _, _, lower, upper, _⟩ :=
    Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_n_empty b, terminal_solve b false] at release
  cases release
  exact ⟨lower, upper⟩

theorem terminal_arithmetic_safe (b : EmptyBoard n) :
    BitVec.smulOverflow (Board.playerCount b.asBoard) 2 = false ∧
    BitVec.ssubOverflow (Board.playerCount b.asBoard * 2) 64 = false ∧
    BitVec.saddOverflow (Board.discScore b.asBoard) (BitVec.ofNat 32 n) = false ∧
    BitVec.saddOverflow (Board.discScore b.asBoard + BitVec.ofNat 32 n) (BitVec.ofNat 32 n) = false := by
  have base := Board.discScoreContract_proven b.asBoard
  obtain ⟨_, _, _, _, _, _, add1, add2⟩ :=
    Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_n_empty b] at add1 add2
  exact ⟨base.2.2.2.1, base.2.2.2.2, add1, add2⟩

end NeuralReversi.Solve34
