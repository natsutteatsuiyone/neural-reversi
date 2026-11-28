module
public import NeuralReversi.Stability.EdgePacking
public import NeuralReversi.Stability.EdgeModel

@[expose] public section

namespace NeuralReversi.Stability.Semantics
set_option linter.unusedSimpArgs false

def embedRow (x : Byte) (r : Fin 8) : Word := x.setWidth 64 <<< (8*r.val)

theorem embedRow_bit (x : Byte) (row r c : Fin 8) :
    (embedRow x row).getLsbD (8*r.val+c.val) =
      (decide (r = row) && x.getLsbD c.val) := by
  have hb : 8*r.val+c.val < 64 := by omega
  by_cases h : r = row
  · subst row
    have hn : ¬ 8*r.val+c.val < 8*r.val := by omega
    have hc : c.val < 64 := by omega
    simp [embedRow, hb, hn, hc, -BitVec.getLsbD_eq_getElem]
  · have hv : r.val ≠ row.val := by exact fun he => h (Fin.ext he)
    by_cases hlt : r.val < row.val
    · have hpos : ¬ 8*row.val ≤ 8*r.val+c.val := by omega
      simp [embedRow, h, hb, hpos, -BitVec.getLsbD_eq_getElem]
    · have hbig : 8 ≤ 8*r.val+c.val - 8*row.val := by omega
      have hzero := BitVec.getLsbD_of_ge x _ hbig
      simp [embedRow, h, hb, hzero, -BitVec.getLsbD_eq_getElem]

def getStableEdge (table : Edge.EdgeTable) (p o : Word) : Word :=
  embedRow (Edge.edgeStability table (rowBits p 0) (rowBits o 0)) 0 |||
  embedRow (Edge.edgeStability table (rowBits p 7) (rowBits o 7)) 7 |||
  unpackA2A7 (Edge.edgeStability table (packA1A8 p) (packA1A8 o)) |||
  unpackH2H7 (Edge.edgeStability table (packH1H8 p) (packH1H8 o))

theorem getStableEdge_bit (table : Edge.EdgeTable) (p o : Word) (r c : Fin 8) :
    (getStableEdge table p o).getLsbD (8*r.val+c.val) =
      ((((decide (r = 0) && (Edge.edgeStability table (rowBits p 0) (rowBits o 0)).getLsbD c.val) ||
       (decide (r = 7) && (Edge.edgeStability table (rowBits p 7) (rowBits o 7)).getLsbD c.val)) ||
       (decide (c.val = 0 ∧ 0 < r.val ∧ r.val < 7) &&
         (Edge.edgeStability table (packA1A8 p) (packA1A8 o)).getLsbD r.val)) ||
       (decide (c.val = 7 ∧ 0 < r.val ∧ r.val < 7) &&
         (Edge.edgeStability table (packH1H8 p) (packH1H8 o)).getLsbD r.val)) := by
  simp only [getStableEdge, BitVec.getLsbD_or, embedRow_bit, unpackA_bit, unpackH_bit]


theorem getStableEdge_sound_of_line_tables
    (table : Edge.EdgeTable) (s : State) (color : Color) (p o : Word)
    (repr : Represents s color p o)
    (rows : ∀ r : Fin 8, r = 0 ∨ r = 7 → ∀ i : Fin 8,
      (Edge.edgeStability table (encodeLine (fun j => s.board (r,j)) color)
        (encodeLine (fun j => s.board (r,j)) (Opposite color))).getLsbD i.val = true →
      Stable s color (r,i))
    (cols : ∀ c : Fin 8, c = 0 ∨ c = 7 → ∀ i : Fin 8,
      (Edge.edgeStability table (encodeLine (fun j => s.board (j,c)) color)
        (encodeLine (fun j => s.board (j,c)) (Opposite color))).getLsbD i.val = true →
      Stable s color (i,c)) :
    ∀ x, Marks (getStableEdge table p o) x → Stable s color x := by
  intro x hx
  rcases x with ⟨r,c⟩
  change (getStableEdge table p o).getLsbD (8*r.val+c.val) = true at hx
  rw [getStableEdge_bit] at hx
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hx
  have repr' := represents_swap repr
  rcases hx with ((⟨hr,hb⟩ | ⟨hr,hb⟩) | ⟨hc,hb⟩) | ⟨hc,hb⟩
  · subst r
    apply rows 0 (Or.inl rfl) c
    simpa only [encode_row s color p o repr 0,
      encode_row s (Opposite color) o p repr' 0] using hb
  · subst r
    apply rows 7 (Or.inr rfl) c
    simpa only [encode_row s color p o repr 7,
      encode_row s (Opposite color) o p repr' 7] using hb
  · have hc0 : c = 0 := Fin.ext hc.1
    subst c
    apply cols 0 (Or.inl rfl) r
    simpa only [encode_left s color p o repr,
      encode_left s (Opposite color) o p repr'] using hb
  · have hc7 : c = 7 := Fin.ext hc.1
    subst c
    apply cols 7 (Or.inr rfl) r
    simpa only [encode_right s color p o repr,
      encode_right s (Opposite color) o p repr'] using hb

end NeuralReversi.Stability.Semantics
