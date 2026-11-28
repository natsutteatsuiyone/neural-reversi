module
public import NeuralReversi.Bitboard.Model

@[expose] public section

/-! Model of `search::endgame::empties`: the empty squares of an endgame position as a
mask over the presorted square order, the odd-quadrant mask and the last-four ordering. -/

namespace NeuralReversi.Empties

open Bitboard (Bits Square squareMask)

/-- `PRESORTED` in `search/endgame/empties.rs`. -/
def order : Vector Square 64 := #v[0, 7, 56, 63, 2, 5, 16, 23, 40, 47, 58, 61, 18, 21, 42, 45, 3, 4, 24, 31, 32, 39, 59, 60, 19, 20, 26, 29, 34, 37, 43, 44, 11, 12, 25, 30, 33, 38, 51, 52, 10, 13, 17, 22, 41, 46, 50, 53, 1, 6, 8, 15, 48, 55, 57, 62, 9, 14, 49, 54, 27, 28, 35, 36]

/-- `PRESORTED_INDEX`: the position of each square in `order`. -/
def position : Vector (Fin 64) 64 := #v[0, 48, 4, 16, 17, 5, 49, 1, 50, 56, 40, 32, 33, 41, 57, 51, 6, 42, 12, 24, 25, 13, 43, 7, 18, 34, 26, 60, 61, 27, 35, 19, 20, 36, 28, 62, 63, 29, 37, 21, 8, 44, 14, 30, 31, 15, 45, 9, 52, 58, 46, 38, 39, 47, 59, 53, 2, 54, 10, 22, 23, 11, 55, 3]

def presorted : List Square := order.toList

theorem order_position : ∀ sq : Square, order[position[sq]] = sq := by decide

theorem position_order : ∀ i : Fin 64, position[order[i]] = i := by decide

theorem presorted_perm : presorted.Perm (List.finRange 64) := by
  apply List.perm_iff_count.mpr
  decide

theorem presorted_eq_map : presorted = (List.finRange 64).map (fun i => order[i]) := by decide

/-- Bit `i` of `p` is set exactly when `order[i]` is an empty square of `e`. -/
def Encodes (p : BitVec 64) (e : Bits) : Prop :=
  ∀ i : Fin 64, p.getLsbD i.val = e.getLsbD order[i].val

/-- The `Empties` value: the presorted mask and the number of empty squares. -/
structure State where
  mask : BitVec 64
  count : BitVec 32
  deriving DecidableEq, Repr

def State.Represents (s : State) (e : Bits) : Prop :=
  Encodes s.mask e ∧ s.count = Bitboard.count e

def bit (sq : Square) : BitVec 64 := (1 : BitVec 64) <<< position[sq].val

/-- `Empties::without`. -/
def State.without (s : State) (sq : Square) : State := ⟨s.mask ^^^ bit sq, s.count - 1⟩

theorem one_shift_bit (k i : Nat) (hi : i < 64) :
    ((1 : BitVec 64) <<< k).getLsbD i = decide (i = k) := by
  by_cases h : i = k
  · subst h
    simp [BitVec.getLsbD_shiftLeft, hi, -BitVec.getLsbD_eq_getElem]
  · simp only [BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and, h, decide_false]
    by_cases lt : i < k
    · simp [lt]
    · have : i - k ≠ 0 := by omega
      simp [lt, BitVec.getLsbD_one, this]

theorem encodes_without {p : BitVec 64} {e : Bits} (h : Encodes p e) (sq : Square)
    (empty : e.getLsbD sq.val = true) :
    Encodes (p ^^^ bit sq) (e &&& ~~~squareMask sq) := by
  intro i
  rw [BitVec.getLsbD_xor, BitVec.getLsbD_and, BitVec.getLsbD_not, h i, bit, squareMask,
    one_shift_bit _ _ i.isLt, one_shift_bit _ _ order[i].isLt]
  simp only [order[i].isLt, decide_true, Bool.true_and]
  by_cases same : i = position[sq]
  · subst same
    rw [order_position] at *
    simp [empty]
  · have other : order[i] ≠ sq := fun eq => same (by rw [← eq, position_order])
    have ne1 : i.val ≠ position[sq].val := fun eq => same (Fin.ext eq)
    have ne2 : order[i].val ≠ sq.val := fun eq => other (Fin.ext eq)
    simp only [Fin.getElem_fin] at ne1 ne2 ⊢
    simp [ne1, ne2]

/-- The empty squares of a mask in presorted order. -/
def squares (p : BitVec 64) : List Square :=
  (List.finRange 64).filterMap fun i => if p.getLsbD i.val then some order[i] else none

private theorem filterMap_encodes {p : BitVec 64} {e : Bits} (h : Encodes p e) (l : List (Fin 64)) :
    l.filterMap (fun i => if p.getLsbD i.val then some order[i] else none) =
      (l.map fun i => order[i]).filter (fun sq => e.getLsbD sq.val) := by
  induction l with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.filterMap_cons, List.map_cons, List.filter_cons, h i, ih]
    cases e.getLsbD order[i].val <;> rfl

theorem squares_of_encodes {p : BitVec 64} {e : Bits} (h : Encodes p e) :
    squares p = presorted.filter (fun sq => e.getLsbD sq.val) := by
  rw [presorted_eq_map]
  exact filterMap_encodes h _

/-- `Empties::first::<N>`. -/
def first (p : BitVec 64) (n : Nat) : List Square := (squares p).take n

/-- The quadrant index of a square: A1-D4, E1-H4, A5-D8, E5-H8. -/
def quad (sq : Square) : Fin 4 := ⟨2 * (sq.val / 32) + sq.val % 8 / 4, by omega⟩

/-- `quadrant` in `search/endgame/empties.rs`. -/
def quadrant (sq : Square) : Bits := 0x0F0F0F0F#64 <<< (sq.val &&& 36)

theorem quadrant_bit : ∀ sq j : Square, (quadrant sq).getLsbD j.val = (quad j == quad sq) := by
  decide +kernel

def fold (x : Bits) : Bits :=
  let x := x ^^^ (x >>> 8)
  let x := x ^^^ (x >>> 16)
  let x := x ^^^ (x >>> 2)
  x ^^^ (x >>> 1)

def origins : Bits := 0x0000001100000011#64

/-- `odd_quadrants` in `search/endgame/empties.rs`. -/
def oddQuadrants (x : Bits) : Bits := (fold x &&& origins) * 0x0F0F0F0F#64

private def xorAll (x : Bits) : List Nat → Bool
  | [] => false
  | i :: rest => x.getLsbD i ^^ xorAll x rest

private theorem xorAll_countP (x : Bits) (l : List Nat) :
    xorAll x l = (l.countP (fun i => x.getLsbD i) % 2 == 1) := by
  induction l with
  | nil => rfl
  | cons i rest ih =>
    simp only [xorAll, ih, List.countP_cons]
    rcases Nat.mod_two_eq_zero_or_one (rest.countP fun i => x.getLsbD i) with h | h <;>
      cases x.getLsbD i <;> simp [h, Nat.add_mod]

/-- The squares of quadrant `q` in the order `fold` combines them. -/
def quadSquares : Fin 4 → List Nat
  | 0 => [0,8,16,24,2,10,18,26,1,9,17,25,3,11,19,27]
  | 1 => [4,12,20,28,6,14,22,30,5,13,21,29,7,15,23,31]
  | 2 => [32,40,48,56,34,42,50,58,33,41,49,57,35,43,51,59]
  | 3 => [36,44,52,60,38,46,54,62,37,45,53,61,39,47,55,63]

private def origin : Fin 4 → Nat
  | 0 => 0 | 1 => 4 | 2 => 32 | 3 => 36

private theorem fold_origin (x : Bits) : ∀ q : Fin 4, (fold x).getLsbD (origin q) = xorAll x (quadSquares q) := by
  intro q
  match q with
  | 0 | 1 | 2 | 3 =>
    simp only [fold, origin, quadSquares, xorAll, BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight,
      Nat.reduceAdd, Bool.xor_false, Bool.xor_assoc]

private def spread (b0 b4 b32 b36 : Bool) : Bits :=
  (if b0 then 1#64 else 0) ||| (if b4 then 16#64 else 0) |||
    (if b32 then 0x100000000#64 else 0) ||| (if b36 then 0x1000000000#64 else 0)

private theorem getLsbD_ite_zero (b : Bool) (c : Bits) (j : Nat) :
    (if b = true then c else 0).getLsbD j = (b && c.getLsbD j) := by
  cases b <;> simp

private theorem origins_bit : ∀ j : Fin 64,
    origins.getLsbD j.val = decide (j.val = 0 ∨ j.val = 4 ∨ j.val = 32 ∨ j.val = 36) := by
  decide +kernel

private theorem spread_const_bits : ∀ j : Fin 64,
    (1#64).getLsbD j.val = decide (j.val = 0) ∧
    (16#64).getLsbD j.val = decide (j.val = 4) ∧
    (0x100000000#64).getLsbD j.val = decide (j.val = 32) ∧
    (0x1000000000#64).getLsbD j.val = decide (j.val = 36) := by
  decide +kernel

private theorem masked (y : Bits) :
    y &&& origins = spread (y.getLsbD 0) (y.getLsbD 4) (y.getLsbD 32) (y.getLsbD 36) := by
  apply BitVec.eq_of_getLsbD_eq
  intro j hj
  have hbits := spread_const_bits ⟨j, hj⟩
  have h1 : (1#64).getLsbD j = decide (j = 0) := hbits.1
  have h4 : (16#64).getLsbD j = decide (j = 4) := hbits.2.1
  have h32 : (0x100000000#64).getLsbD j = decide (j = 32) := hbits.2.2.1
  have h36 : (0x1000000000#64).getLsbD j = decide (j = 36) := hbits.2.2.2
  have ho : origins.getLsbD j = decide (j = 0 ∨ j = 4 ∨ j = 32 ∨ j = 36) := origins_bit ⟨j, hj⟩
  simp only [BitVec.getLsbD_and, ho, spread, BitVec.getLsbD_or,
    getLsbD_ite_zero, h1, h4, h32, h36]
  by_cases hj0 : j = 0
  · subst hj0; simp
  · by_cases hj4 : j = 4
    · subst hj4; simp
    · by_cases hj32 : j = 32
      · subst hj32; simp
      · by_cases hj36 : j = 36
        · subst hj36; simp
        · simp [hj0, hj4, hj32, hj36]

private def pick (b0 b4 b32 b36 : Bool) : Fin 4 → Bool
  | 0 => b0 | 1 => b4 | 2 => b32 | 3 => b36

private theorem spread_bits : ∀ (b0 b4 b32 b36 : Bool) (j : Fin 64),
    (spread b0 b4 b32 b36 * 0x0F0F0F0F#64).getLsbD j.val = pick b0 b4 b32 b36 (quad j) := by
  decide +kernel

private theorem pick_origin (y : Bits) (q : Fin 4) :
    pick (y.getLsbD 0) (y.getLsbD 4) (y.getLsbD 32) (y.getLsbD 36) q = y.getLsbD (origin q) := by
  match q with
  | 0 | 1 | 2 | 3 => rfl

theorem oddQuadrants_bit (x : Bits) (j : Square) :
    (oddQuadrants x).getLsbD j.val = (((quadSquares (quad j)).countP fun i => x.getLsbD i) % 2 == 1) := by
  rw [oddQuadrants, masked, spread_bits, pick_origin, fold_origin, xorAll_countP]

private theorem quadSquares_perm : ∀ q : Fin 4,
    (quadSquares q).Perm (((List.finRange 64).filter (fun k => quad k == q)).map Fin.val) := by
  intro q
  match q with
  | 0 | 1 | 2 | 3 => decide

/-- Bit `j` of `odd_quadrants` is set when the quadrant of `j` holds an odd number of
empty squares. -/
theorem oddQuadrants_spec (x : Bits) (j : Square) :
    (oddQuadrants x).getLsbD j.val =
      (((List.finRange 64).countP fun k => quad k == quad j && x.getLsbD k.val) % 2 == 1) := by
  rw [oddQuadrants_bit, (quadSquares_perm (quad j)).countP_eq, List.countP_map, List.countP_filter]
  congr 2
  apply List.countP_congr
  intro k _
  simp only [Function.comp_apply, Bool.and_comm]

/-- Whether `x` lies in a quadrant that holds an odd number of `cells`. -/
def oddIn (cells : List Square) (x : Square) : Bool :=
  cells.countP (fun y => quad y == quad x) % 2 == 1

/-- The branch structure of `sort_last4` after the presorted squares are read. -/
def fourOrder (even pa pb pc : Bool) (a b c d : α) : List α :=
  if even then [a, b, c, d]
  else if !pa then
    if pb then (if pc then [b, c, a, d] else [b, d, a, c])
    else if pc then [c, d, a, b] else [a, b, c, d]
  else if !pb then (if pc then [a, c, b, d] else [a, d, b, c])
  else [a, b, c, d]

/-- `sort_last4`. -/
def sortLastFour (s : State) (odd : Bits) : List Square :=
  match first s.mask 4 with
  | [a, b, c, d] => fourOrder (odd == 0) (odd.getLsbD a.val) (odd.getLsbD b.val) (odd.getLsbD c.val) a b c d
  | other => other

private def pick4 (a b c d : α) (i : Fin 4) : α :=
  match i.val with
  | 0 => a | 1 => b | 2 => c | _ => d

private theorem fourOrder_map (f : α → β) (e p q r : Bool) (a b c d : α) :
    (fourOrder e p q r a b c d).map f = fourOrder e p q r (f a) (f b) (f c) (f d) := by
  unfold fourOrder
  repeat' first | split | rfl

private theorem all_shapes : ∀ qa qb qc qd : Fin 4,
    let cells : List (Fin 4) := [0, 1, 2, 3]
    let odd : Fin 4 → Bool := fun i =>
      cells.countP (fun k => pick4 qa qb qc qd k == pick4 qa qb qc qd i) % 2 == 1
    fourOrder (!(odd 0 || odd 1 || odd 2 || odd 3)) (odd 0) (odd 1) (odd 2) (0 : Fin 4) 1 2 3 =
      cells.filter odd ++ cells.filter (fun i => !odd i) := by decide +kernel

theorem fourOrder_partition (a b c d : Square) :
    let cells := [a, b, c, d]
    let odd := oddIn cells
    fourOrder (!(odd a || odd b || odd c || odd d)) (odd a) (odd b) (odd c) a b c d =
      cells.filter odd ++ cells.filter (fun x => !odd x) := by
  let at4 := pick4 a b c d
  let indices : List (Fin 4) := [0, 1, 2, 3]
  have mapped : indices.map at4 = [a, b, c, d] := rfl
  have quads : (fun i => quad (at4 i)) = pick4 (quad a) (quad b) (quad c) (quad d) := by
    funext i
    unfold at4 pick4
    split <;> rfl
  have cert := congrArg (List.map at4) (all_shapes (quad a) (quad b) (quad c) (quad d))
  dsimp only at cert ⊢
  rw [fourOrder_map, List.map_append] at cert
  rw [← mapped, List.filter_map, List.filter_map]
  simp only [← quads] at cert
  simp only [oddIn, List.countP_map, Function.comp_def] at cert ⊢
  exact cert

/-- The cells of `e` are exactly the members of `cells`. -/
def Exactly (e : Bits) (cells : List Square) : Prop := ∀ sq : Square, e.getLsbD sq.val = decide (sq ∈ cells)

private theorem count_cells {e : Bits} {cells : List Square} (exact : Exactly e cells) (unique : cells.Nodup)
    (p : Square → Bool) :
    (List.finRange 64).countP (fun k => p k && e.getLsbD k.val) = cells.countP p := by
  have same : ((List.finRange 64).filter fun k => e.getLsbD k.val).Perm cells := by
    apply (List.perm_ext_iff_of_nodup ((List.nodup_finRange 64).filter _) unique).mpr
    intro sq
    have bit := exact sq
    simp only [List.mem_filter, List.mem_finRange, true_and]
    rw [bit]
    simp
  rw [← same.countP_eq, List.countP_filter]

theorem oddQuadrants_cells {e : Bits} {cells : List Square} (exact : Exactly e cells) (unique : cells.Nodup)
    (x : Square) : (oddQuadrants e).getLsbD x.val = oddIn cells x := by
  rw [oddQuadrants_spec, count_cells exact unique]
  rfl

private theorem odd_zero_iff {e : Bits} {cells : List Square} (exact : Exactly e cells) (unique : cells.Nodup) :
    (oddQuadrants e == 0) = !(cells.any (oddIn cells)) := by
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq, Bool.not_eq_true', List.any_eq_false]
  constructor
  · intro zero x _ odd
    have := oddQuadrants_cells exact unique x
    rw [zero, odd] at this
    simp at this
  · intro none
    apply BitVec.eq_of_getLsbD_eq
    intro j hj
    have bit := oddQuadrants_cells exact unique ⟨j, hj⟩
    rw [bit, show BitVec.getLsbD (0 : BitVec 64) j = false from BitVec.getLsbD_zero]
    cases parity : oddIn cells ⟨j, hj⟩ with
    | false => rfl
    | true =>
      exfalso
      have positive : 0 < cells.countP (fun y => quad y == quad ⟨j, hj⟩) := by
        simp only [oddIn, beq_iff_eq] at parity
        omega
      obtain ⟨y, mem, same⟩ := List.countP_pos_iff.mp positive
      have sameQuad : quad y = quad ⟨j, hj⟩ := by simpa using same
      apply none y mem
      simp only [oddIn] at parity ⊢
      rw [sameQuad]
      exact parity

theorem sortLastFour_partition (s : State) (e : Bits) (a b c d : Square)
    (mask : Encodes s.mask e) (exact : Exactly e [a, b, c, d]) (unique : [a, b, c, d].Nodup)
    (ordered : presorted.filter (fun sq => e.getLsbD sq.val) = [a, b, c, d]) :
    let cells := [a, b, c, d]
    let odd := oddIn cells
    sortLastFour s (oddQuadrants e) = cells.filter odd ++ cells.filter (fun x => !odd x) := by
  have read : first s.mask 4 = [a, b, c, d] := by
    rw [first, squares_of_encodes mask, ordered]
    rfl
  simp only [sortLastFour, read, odd_zero_iff exact unique, oddQuadrants_cells exact unique]
  have := fourOrder_partition a b c d
  simpa [List.any_cons, Bool.or_assoc] using this

/-- `Empties::new` reads the empty squares in presorted order. -/
def encode (e : Bits) : BitVec 64 :=
  (BitVec.ofBoolListLE ((List.finRange 64).map fun i => e.getLsbD order[i].val)).cast (by simp)

theorem encodes_encode (e : Bits) : Encodes (encode e) e := by
  intro i
  simp [encode, BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE, -BitVec.getLsbD_eq_getElem]

/-- Every square lies in the presorted order, so the squares of `e` are its filter. -/
theorem exactly_presorted (e : Bits) :
    Exactly e (presorted.filter fun sq => e.getLsbD sq.val) := by
  intro sq
  have mem : sq ∈ presorted := presorted_perm.mem_iff.mpr (List.mem_finRange sq)
  cases h : e.getLsbD sq.val <;> simp [List.mem_filter, mem, h, -BitVec.getLsbD_eq_getElem]

theorem presorted_filter_nodup (e : Bits) : (presorted.filter fun sq => e.getLsbD sq.val).Nodup :=
  (presorted_perm.nodup_iff.mpr (List.nodup_finRange 64)).filter _

private theorem countP_remove (f g : Square → Bool) (sq : Square)
    (drop : ∀ k, g k = (f k && decide (k ≠ sq))) :
    ∀ l : List Square, l.Nodup → sq ∈ l → l.countP f = l.countP g + (if f sq then 1 else 0)
  | [], _, mem => absurd mem List.not_mem_nil
  | k :: rest, unique, mem => by
    obtain ⟨absent, distinct⟩ := List.nodup_cons.mp unique
    by_cases here : k = sq
    · subst here
      have same : rest.countP f = rest.countP g := by
        apply List.countP_congr
        intro x inside
        have ne : x ≠ k := fun eq => absent (eq ▸ inside)
        rw [drop x]
        cases f x <;> simp [ne]
      rw [List.countP_cons, List.countP_cons, same, drop k]
      cases f k <;> simp
    · have inner := countP_remove f g sq drop rest distinct
        (by rcases List.mem_cons.mp mem with h | h; exact absurd h.symm here; exact h)
      rw [List.countP_cons, List.countP_cons, inner, drop k]
      cases f k <;> cases f sq <;> simp [here] <;> omega

private theorem without_bit (e : Bits) (sq x : Square) :
    (e &&& ~~~squareMask sq).getLsbD x.val = (e.getLsbD x.val && decide (x ≠ sq)) := by
  rw [BitVec.getLsbD_and, BitVec.getLsbD_not, squareMask, one_shift_bit _ _ x.isLt]
  by_cases same : x = sq
  · subst same
    simp
  · have ne : x.val ≠ sq.val := fun eq => same (Fin.ext eq)
    simp [ne, same, x.isLt]

private theorem parity_succ (n : Nat) : (((n + 1) % 2 == 1) ^^ true) = (n % 2 == 1) := by
  rcases Nat.mod_two_eq_zero_or_one n with h | h
  · have succ : (n + 1) % 2 = 1 := by omega
    simp [h, succ]
  · have succ : (n + 1) % 2 = 0 := by omega
    simp [h, succ]

private theorem quad_beq_comm : ∀ a b : Fin 4, (a == b) = (b == a) := by decide

/-- `node.odd_quadrants ^ quadrant(sq)` is the odd-quadrant mask of the child's empties. -/
theorem oddQuadrants_without (e : Bits) (sq : Square) (empty : e.getLsbD sq.val = true) :
    oddQuadrants e ^^^ quadrant sq = oddQuadrants (e &&& ~~~squareMask sq) := by
  apply BitVec.eq_of_getLsbD_eq
  intro j hj
  let q := quad ⟨j, hj⟩
  let count := fun x : Bits => (List.finRange 64).countP fun k => q == quad k && x.getLsbD k.val
  have spec : ∀ x : Bits, (oddQuadrants x).getLsbD j = (count x % 2 == 1) := by
    intro x
    have base := oddQuadrants_spec x ⟨j, hj⟩
    refine base.trans ?_
    congr 2
    apply List.countP_congr
    intro k _
    rw [quad_beq_comm]
  have split : count e = count (e &&& ~~~squareMask sq) + (if q == quad sq then 1 else 0) := by
    have := countP_remove (fun k => q == quad k && e.getLsbD k.val)
      (fun k => q == quad k && (e &&& ~~~squareMask sq).getLsbD k.val) sq
      (fun k => by rw [without_bit, Bool.and_assoc]) _ (List.nodup_finRange 64) (List.mem_finRange sq)
    simpa only [empty, Bool.and_true] using this
  have flip : (quadrant sq).getLsbD j = (q == quad sq) := by
    exact quadrant_bit sq ⟨j, hj⟩
  rw [BitVec.getLsbD_xor, spec, spec, flip, split]
  cases (q == quad sq)
  · simp
  · exact parity_succ _

/-- `Empties::new` lists the empty squares in presorted order; `without` removes one
    square and xoring the same bit back restores the mask. -/
def ScanContract : Prop :=
  ∀ (e : Bits) (p : BitVec 64) (sq : Square),
    squares (encode e) = presorted.filter (fun x => e.getLsbD x.val) ∧
    (Encodes p e → e.getLsbD sq.val = true →
      Encodes (p ^^^ bit sq) (e &&& ~~~squareMask sq) ∧ (p ^^^ bit sq) ^^^ bit sq = p)

theorem scanContract_proven : ScanContract := by
  intro e p sq
  refine ⟨squares_of_encodes (encodes_encode e), fun h empty => ⟨encodes_without h sq empty, ?_⟩⟩
  rw [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

/-- `quadrant` selects the square's four-by-four block, and `odd_quadrants` marks every
    block holding an odd number of empties; removing an empty square toggles its block. -/
def QuadrantContract : Prop :=
  ∀ (e : Bits) (sq j : Square),
    (quadrant sq).getLsbD j.val = (quad j == quad sq) ∧
    (oddQuadrants e).getLsbD j.val =
      (((List.finRange 64).countP fun k => quad k == quad j && e.getLsbD k.val) % 2 == 1) ∧
    (e.getLsbD sq.val = true → oddQuadrants e ^^^ quadrant sq = oddQuadrants (e &&& ~~~squareMask sq))

theorem quadrantContract_proven : QuadrantContract :=
  fun e sq j => ⟨quadrant_bit sq j, oddQuadrants_spec e j, oddQuadrants_without e sq⟩

/-- `sort_last4` returns the last four empties with the odd-quadrant squares first,
    each group in presorted order. -/
def SortLastFourContract : Prop :=
  ∀ (s : State) (e : Bits) (a b c d : Square), Encodes s.mask e → Exactly e [a, b, c, d] →
    [a, b, c, d].Nodup → presorted.filter (fun sq => e.getLsbD sq.val) = [a, b, c, d] →
    let cells := [a, b, c, d]
    sortLastFour s (oddQuadrants e) = cells.filter (oddIn cells) ++ cells.filter (fun x => !oddIn cells x)

theorem sortLastFourContract_proven : SortLastFourContract :=
  fun s e a b c d mask exact unique ordered => sortLastFour_partition s e a b c d mask exact unique ordered

end NeuralReversi.Empties
