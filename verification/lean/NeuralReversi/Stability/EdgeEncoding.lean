import NeuralReversi.Stability.StructuralCoordinates

set_option linter.unusedSimpArgs false
namespace NeuralReversi.Stability.Semantics

abbrev Byte := BitVec 8
abbrev Word := BitVec 64

def encodeBits {n : Nat} (f : Fin n → Bool) : BitVec n :=
  (BitVec.ofBoolListLE (List.ofFn f)).setWidth n

@[simp] theorem encodeBits_bit {n : Nat} (f : Fin n → Bool) (i : Fin n) :
    (encodeBits f).getLsbD i.val = f i := by
  simp [encodeBits, i.isLt, List.getD_eq_getElem?_getD, -BitVec.getLsbD_eq_getElem]

def encodeColor {n : Nat} (line : Fin n → Option Color) (c : Color) : BitVec n :=
  encodeBits fun i => decide (line i = some c)

@[simp] theorem encodeColor_bit {n : Nat} (line : Fin n → Option Color)
    (c : Color) (i : Fin n) :
    (encodeColor line c).getLsbD i.val = decide (line i = some c) := by
  exact encodeBits_bit _ i

def decodeColor {n : Nat} (p o : BitVec n) (c : Color) : Fin n → Option Color :=
  fun i => if p.getLsbD i.val then some c
    else if o.getLsbD i.val then some (Opposite c) else none

theorem encodeColor_disjoint {n : Nat} (line : Fin n → Option Color) (c : Color) :
    encodeColor line c &&& encodeColor line (Opposite c) = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hp := encodeColor_bit line c ⟨i,hi⟩
  have ho := encodeColor_bit line (Opposite c) ⟨i,hi⟩
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero, hp, ho]
  cases h : line ⟨i,hi⟩ with
  | none => simp [h]
  | some d => cases d <;> cases c <;> simp [h, Opposite]

theorem decodeColor_encode {n : Nat} (line : Fin n → Option Color) (c : Color) :
    decodeColor (encodeColor line c) (encodeColor line (Opposite c)) c = line := by
  funext i
  simp only [decodeColor, encodeColor_bit]
  cases h : line i with
  | none => simp [h]
  | some d => cases d <;> cases c <;> simp [h, Opposite]

theorem decodeColor_bits {n : Nat} (p o : BitVec n) (c : Color) (i : Fin n)
    (hd : p &&& o = 0) :
    (decodeColor p o c i = some c ↔ p.getLsbD i.val = true) ∧
    (decodeColor p o c i = some (Opposite c) ↔ o.getLsbD i.val = true) := by
  have hdis := congrArg (fun v : BitVec n => v.getLsbD i.val) hd
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at hdis
  cases hp : p.getLsbD i.val <;> cases ho : o.getLsbD i.val <;>
    cases c <;> simp_all [decodeColor, Opposite]

theorem encodeColor_eq_of_bits {n : Nat} (line : Fin n → Option Color) (c : Color)
    (p : BitVec n) (h : ∀ i : Fin n, p.getLsbD i.val = true ↔ line i = some c) :
    encodeColor line c = p := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  apply Bool.eq_iff_iff.mpr
  rw [encodeColor_bit line c ⟨i,hi⟩]
  simpa using (h ⟨i,hi⟩).symm

theorem encodeColor_decode_player {n : Nat} (p o : BitVec n) (c : Color)
    (hd : p &&& o = 0) : encodeColor (decodeColor p o c) c = p := by
  apply encodeColor_eq_of_bits
  intro i
  exact (decodeColor_bits p o c i hd).1.symm

theorem encodeColor_decode_opponent {n : Nat} (p o : BitVec n) (c : Color)
    (hd : p &&& o = 0) : encodeColor (decodeColor p o c) (Opposite c) = o := by
  apply encodeColor_eq_of_bits
  intro i
  exact (decodeColor_bits p o c i hd).2.symm

abbrev encodeLine := @encodeColor 8
abbrev edgeLineColor := @decodeColor 8

theorem encodeLine_bit (line : Fin 8 → Option Color) (c : Color) (i : Fin 8) :
    (encodeLine line c).getLsbD i.val = true ↔ line i = some c := by
  rw [encodeColor_bit]
  simp

theorem encodeLine_disjoint (line : Fin 8 → Option Color) (c : Color) :
    encodeLine line c &&& encodeLine line (Opposite c) = 0 := encodeColor_disjoint line c

theorem encodeLine_eq_of_bits (line : Fin 8 → Option Color) (c : Color) (p : Byte)
    (h : ∀ i : Fin 8, p.getLsbD i.val = true ↔ line i = some c) :
    encodeLine line c = p := encodeColor_eq_of_bits line c p h

theorem edgeLineColor_bits (p o : Byte) (c : Color) (i : Fin 8) (hd : p &&& o = 0) :
    (edgeLineColor p o c i = some c ↔ p.getLsbD i.val = true) ∧
    (edgeLineColor p o c i = some (Opposite c) ↔ o.getLsbD i.val = true) :=
  decodeColor_bits p o c i hd

theorem edgeLineColor_encode (line : Fin 8 → Option Color) (c : Color) :
    edgeLineColor (encodeLine line c) (encodeLine line (Opposite c)) c = line :=
  decodeColor_encode line c

theorem encodeLine_decode_player (p o : Byte) (c : Color) (hd : p &&& o = 0) :
    encodeLine (edgeLineColor p o c) c = p := encodeColor_decode_player p o c hd

theorem encodeLine_decode_opponent (p o : Byte) (c : Color) (hd : p &&& o = 0) :
    encodeLine (edgeLineColor p o c) (Opposite c) = o := encodeColor_decode_opponent p o c hd


def encodeBoard (b : Board) (c : Color) : Word :=
  encodeColor (fun i => b (boardSquare i)) c

def decodeBoard (p o : Word) (c : Color) : Board :=
  fun x => decodeColor p o c (boardIndex x)

def Marks (p : Word) (x : Square) : Prop := p.getLsbD (boardIndex x).val = true

def RepresentsBoard (b : Board) (c : Color) (p o : Word) : Prop :=
  (∀ x, Marks p x ↔ b x = some c) ∧
  (∀ x, Marks o x ↔ b x = some (Opposite c))

def Represents (s : State) (c : Color) (p o : Word) : Prop :=
  RepresentsBoard s.board c p o

@[simp] theorem encodeBoard_bit (b : Board) (c : Color) (x : Square) :
    (encodeBoard b c).getLsbD (boardIndex x).val = decide (b x = some c) := by
  simp only [encodeBoard, encodeColor_bit, boardSquare_index]

theorem encodeBoard_disjoint (b : Board) (c : Color) :
    encodeBoard b c &&& encodeBoard b (Opposite c) = 0 :=
  encodeColor_disjoint _ c

theorem decodeBoard_encode (b : Board) (c : Color) :
    decodeBoard (encodeBoard b c) (encodeBoard b (Opposite c)) c = b := by
  funext x
  have h := congrFun (decodeColor_encode (fun i => b (boardSquare i)) c) (boardIndex x)
  simpa only [decodeBoard, encodeBoard, boardSquare_index] using h

theorem encodeBoard_decode_player (p o : Word) (c : Color) (hd : p &&& o = 0) :
    encodeBoard (decodeBoard p o c) c = p := by
  simp only [encodeBoard, decodeBoard, boardIndex_square]
  exact encodeColor_decode_player p o c hd

theorem encodeBoard_decode_opponent (p o : Word) (c : Color) (hd : p &&& o = 0) :
    encodeBoard (decodeBoard p o c) (Opposite c) = o := by
  simp only [encodeBoard, decodeBoard, boardIndex_square]
  exact encodeColor_decode_opponent p o c hd

theorem encodeBoard_represents (b : Board) (c : Color) :
    RepresentsBoard b c (encodeBoard b c) (encodeBoard b (Opposite c)) := by
  constructor <;> intro x <;> simp only [Marks, encodeBoard_bit, decide_eq_true_eq]

theorem decodeBoard_represents (p o : Word) (c : Color) (hd : p &&& o = 0) :
    RepresentsBoard (decodeBoard p o c) c p o := by
  constructor
  · intro x; exact (decodeColor_bits p o c (boardIndex x) hd).1.symm
  · intro x; exact (decodeColor_bits p o c (boardIndex x) hd).2.symm

theorem represents_own {s : State} {c : Color} {p o : Word}
    (h : Represents s c p o) : ∀ x, Marks p x → s.board x = some c :=
  fun x hx => (h.1 x).mp hx

theorem represents_occupied {s : State} {c : Color} {p o : Word}
    (h : Represents s c p o) : ∀ x, Marks (p ||| o) x → Occupied s.board x := by
  intro x hx
  simp only [Marks, BitVec.getLsbD_or, Bool.or_eq_true] at hx
  rcases hx with hp | ho
  · exact ⟨c, (h.1 x).mp hp⟩
  · exact ⟨Opposite c, (h.2 x).mp ho⟩

theorem represents_disjoint_at {s : State} {c : Color} {p o : Word}
    (h : Represents s c p o) (x : Square) : ¬ (Marks p x ∧ Marks o x) := by
  intro ⟨hp,ho⟩
  have hc := (h.1 x).mp hp
  have hc' := (h.2 x).mp ho
  rw [hc] at hc'
  cases c <;> simp [Opposite] at hc'

theorem represents_disjoint {s : State} {c : Color} {p o : Word}
    (h : Represents s c p o) : p &&& o = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hd := represents_disjoint_at h (boardSquare ⟨i,hi⟩)
  simp only [Marks, boardIndex_square] at hd
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero]
  cases hp : p.getLsbD i <;> cases ho : o.getLsbD i <;> simp_all

theorem represents_swap {s : State} {c : Color} {p o : Word}
    (h : Represents s c p o) : Represents s (Opposite c) o p := by
  refine ⟨h.2, ?_⟩
  intro x
  simpa [Opposite] using h.1 x

def rowBits (p : Word) (r : Fin 8) : Byte := (p >>> (8 * r.val)).setWidth 8
def columnBits (p : Word) (c : Fin 8) : Byte :=
  encodeBits fun r => p.getLsbD (boardIndex (r,c)).val

theorem rowBits_bit (p : Word) (r i : Fin 8) :
    (rowBits p r).getLsbD i.val = p.getLsbD (boardIndex (r,i)).val := by
  simp [rowBits, boardIndex, i.isLt, Nat.add_comm, -BitVec.getLsbD_eq_getElem]

theorem columnBits_bit (p : Word) (c i : Fin 8) :
    (columnBits p c).getLsbD i.val = p.getLsbD (boardIndex (i,c)).val :=
  encodeBits_bit _ i

theorem encode_row (s : State) (c : Color) (p o : Word) (repr : Represents s c p o)
    (r : Fin 8) : encodeLine (fun i => s.board (r,i)) c = rowBits p r := by
  apply encodeLine_eq_of_bits
  intro i
  rw [rowBits_bit]
  exact repr.1 (r,i)

theorem encode_column (s : State) (c : Color) (p o : Word) (repr : Represents s c p o)
    (col : Fin 8) : encodeLine (fun i => s.board (i,col)) c = columnBits p col := by
  apply encodeLine_eq_of_bits
  intro i
  rw [columnBits_bit]
  exact repr.1 (i,col)


theorem rowBits_disjoint (p o : Word) (hd : p &&& o = 0) (r : Fin 8) :
    rowBits p r &&& rowBits o r = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h := congrArg (fun v : Word => v.getLsbD (boardIndex (r,⟨i,hi⟩)).val) hd
  simpa [BitVec.getLsbD_and, BitVec.getLsbD_zero, -BitVec.getLsbD_eq_getElem, rowBits_bit p r ⟨i,hi⟩,
    rowBits_bit o r ⟨i,hi⟩] using h

theorem columnBits_disjoint (p o : Word) (hd : p &&& o = 0) (c : Fin 8) :
    columnBits p c &&& columnBits o c = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h := congrArg (fun v : Word => v.getLsbD (boardIndex (⟨i,hi⟩,c)).val) hd
  simpa [BitVec.getLsbD_and, BitVec.getLsbD_zero, -BitVec.getLsbD_eq_getElem, columnBits_bit p c ⟨i,hi⟩,
    columnBits_bit o c ⟨i,hi⟩] using h

theorem decode_row (s : State) (c : Color) (p o : Word) (repr : Represents s c p o)
    (r : Fin 8) : edgeLineColor (rowBits p r) (rowBits o r) c = fun i => s.board (r,i) := by
  rw [← encode_row s c p o repr r,
    ← encode_row s (Opposite c) o p (represents_swap repr) r]
  exact edgeLineColor_encode _ c

theorem decode_column (s : State) (c : Color) (p o : Word) (repr : Represents s c p o)
    (col : Fin 8) : edgeLineColor (columnBits p col) (columnBits o col) c = fun i => s.board (i,col) := by
  rw [← encode_column s c p o repr col,
    ← encode_column s (Opposite c) o p (represents_swap repr) col]
  exact edgeLineColor_encode _ c

end NeuralReversi.Stability.Semantics
