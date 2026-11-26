import NeuralReversi.PvsParallel.SerialTable.Store

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsBoardTree (Position)

private theorem rejected_tag (b : Position) (previous : Save.Key) (old e : EntryData)
    (score : Score) (move : Option Bitboard.Square) (bound : Bound) (depth : Depth) (sel : Selectivity)
    (g : BitVec 8) (ending pv : Bool) (other : ending = false ∨ sel.val < 5)
    (saved : Save.proposal (Cutoffs.key b) previous (Packing.newData score move bound depth sel g ending pv) old = some e) :
    ¬Cutoffs.ExactEndgame e := by
  let incoming := Packing.newData score move bound depth sel g ending pv
  have tag : ¬Cutoffs.ExactEndgame incoming := by
    rcases other with rfl | weak
    · have bit : incoming.flags.getLsbD 0 = false := by
        dsimp only [incoming,Packing.newData]
        cases pv <;> decide
      intro h
      exact Bool.noConfusion (bit.symm.trans h.1)
    · intro h
      have level := h.2
      change 5 ≤ (BitVec.ofNat 8 sel.val).toNat at level
      rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega : sel.val < 2^8)] at level
      omega
  rw [(Save.contract_proven ..).1] at saved
  split at saved
  · contradiction
  · cases Option.some.inj saved
    unfold Save.payload
    split <;> exact tag

theorem Table.store_other (t : Table l) (good : t.Good mc fc) (i : l.Index) (b : Position)
    (score : Score) (move : Option Bitboard.Square) (bound : Bound) (depth : Depth) (sel : Selectivity)
    (ending pv : Bool) (other : ending = false ∨ sel.val < 5) (legal : TableReads.LegalChoice b move) :
    (t.store i b score bound depth move sel ending pv).Good mc fc := by
  rw [Table.store,WriterMir.store_input]
  by_cases finite : score.raw.toInt.natAbs < 32767 <;> simp only [finite,ite_true,ite_false]
  · apply t.save_good good _ _ _ (TableReads.newHint legal)
    exact fun e hit => .other b e (rejected_tag b _ _ _ _ _ _ _ _ _ _ _ other hit)
  · exact good

/-- Store operands of a midgame or earlier selectivity level. The score may be any machine word. -/
structure EarlierStore (l : Layout) where
  slot : l.Index
  board : Position
  score : Score
  move : Option Bitboard.Square
  bound : Bound
  depth : Depth
  selectivity : Selectivity
  ending : Bool
  pv : Bool
  other : ending = false ∨ selectivity.val < 5
  legal : TableReads.LegalChoice board move

def EarlierStore.write (t : Table l) (e : EarlierStore l) : Table l :=
  t.store e.slot e.board e.score e.bound e.depth e.move e.selectivity e.ending e.pv

def EarlierContract : Prop := ∀ (l : Layout) mc fc (t : Table l) (stores : List (EarlierStore l)),
  t.Good mc fc → (stores.foldl EarlierStore.write t).Good mc fc

theorem earlier_good : EarlierContract := by
  intro l mc fc t stores
  induction stores generalizing t with
  | nil => exact id
  | cons e tail ih =>
    intro g
    exact ih _ (t.store_other g _ _ _ _ _ _ _ _ _ e.other e.legal)

end NeuralReversi.PvsParallel.SerialTable
