import Std

namespace NeuralReversi.Solve1.Projection

abbrev Word := BitVec 64
abbrev Byte := BitVec 8

def gather (cells : List Nat) (p : Word) : Byte :=
  (BitVec.ofBoolListLE (cells.map p.getLsbD)).setWidth 8

def deposit : List (Nat × Bool) → Word
  | [] => 0
  | (cell, bit) :: rest =>
      (if bit then (1 : Word) <<< cell else 0) ||| deposit rest

def scatter (cells : List Nat) (b : Byte) : Word :=
  deposit (cells.zipIdx.map fun (cell, i) => (cell, b.getLsbD i))

def mask : List Nat → Word
  | [] => 0
  | cell :: rest => ((1 : Word) <<< cell) ||| mask rest

private theorem single_bit (p : Word) (cell : Nat) :
    (if p.getLsbD cell then (1 : Word) <<< cell else 0) =
      p &&& ((1 : Word) <<< cell) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  by_cases he : i = cell
  · subst i
    by_cases hc : cell < 64
    · cases h : p.getLsbD cell <;>
        simp [h, hc, -BitVec.getLsbD_eq_getElem]
    · omega
  · have hd : ¬ (cell ≤ i ∧ i - cell = 0) := by omega
    cases h : p.getLsbD cell <;>
      simp [-BitVec.getLsbD_eq_getElem, Bool.and_eq_true] <;> omega

private theorem deposit_map (cells : List Nat) (p : Word) :
    deposit (cells.map fun cell => (cell, p.getLsbD cell)) = p &&& mask cells := by
  induction cells with
  | nil => simp [deposit, mask]
  | cons cell rest ih =>
      simp only [List.map_cons, deposit, mask, ih, single_bit,
        BitVec.and_or_distrib_left]

theorem gather_bit (cells : List Nat) (p : Word) (i : Nat) (hi : i < 8) :
    (gather cells p).getLsbD i = (cells.map p.getLsbD).getD i false := by
  simp [gather, hi, -BitVec.getLsbD_eq_getElem]

theorem scatter_gather (cells : List Nat) (p : Word) (hlen : cells.length ≤ 8) :
    scatter cells (gather cells p) = p &&& mask cells := by
  unfold scatter
  have hm : cells.zipIdx.map (fun (cell, i) => (cell, (gather cells p).getLsbD i)) =
      cells.map (fun cell => (cell, p.getLsbD cell)) := by
    calc
      _ = cells.zipIdx.map (fun item => (item.1, p.getLsbD item.1)) := by
        apply List.map_congr_left
        intro item hitem
        rcases item with ⟨cell, i⟩
        have hm := List.mem_zipIdx' hitem
        have hi : i < 8 := by omega
        simp only [gather_bit cells p i hi]
        have he := List.mk_mem_zipIdx_iff_getElem?.mp hitem
        simp [List.getD_eq_getElem?_getD, he]
      _ = _ := by
        change List.map ((fun cell => (cell, p.getLsbD cell)) ∘ Prod.fst) cells.zipIdx = _
        rw [← List.map_map, List.zipIdx_map_fst]
  rw [hm, deposit_map]

theorem masked_eq_scatter (cells : List Nat) (p : Word) (hlen : cells.length ≤ 8)
    (m : Word) (hmask : mask cells = m) :
    p &&& m = scatter cells (gather cells p) := by
  rw [scatter_gather cells p hlen, hmask]

end NeuralReversi.Solve1.Projection
