module
public import NeuralReversi.Solve1.Bmi2
public import NeuralReversi.Board.Counting
public import NeuralReversi.Flip.Properties

@[expose] public section

namespace NeuralReversi.Solve1

open Flip (Before mask laneValue)
open Bitboard (count squareMask)
open Geometry (ray raySquares)

private def cells (sq : Fin 64) (d : Fin 8) : List Nat :=
  raySquares sq.val (Flip.delta d).1 (Flip.delta d).2

local instance (d : Fin 8) (i j : Nat) : Decidable (Before d i j) := by
  unfold Before
  split <;> infer_instance

private theorem cells_geometry : ∀ (sq : Fin 64) (d : Fin 8),
    (cells sq d).Pairwise (Before d) ∧ (∀ i ∈ cells sq d, i < 64 ∧ i ≠ sq.val) := by decide +kernel

private def cellsMask (xs : List Nat) : Geometry.Word :=
  xs.foldr (fun k acc => (1#64 <<< k) ||| acc) 0#64

private theorem cellsMask_bit (xs : List Nat) (i : Nat) (hi : i < 64) :
    (cellsMask xs).getLsbD i = decide (i ∈ xs) := by
  induction xs with
  | nil => simp [cellsMask]
  | cons k rest ih =>
    have h1 : ((1#64 <<< k).getLsbD i) = decide (i = k) := by
      simp only [BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
      by_cases heq : i = k
      · subst heq; simp
      · by_cases hlt : i < k
        · simp [hlt, heq]
        · have hne : i - k ≠ 0 := by omega
          simp [hlt, heq, BitVec.getLsbD_one, hne]
    change ((1#64 <<< k) ||| cellsMask rest).getLsbD i = decide (i ∈ k :: rest)
    rw [BitVec.getLsbD_or, h1, ih]
    by_cases heq : i = k <;> simp [heq]

private theorem mask_eq_cellsMask : ∀ (sq : Fin 64) (d : Fin 8),
    mask ⟨sq.val, by omega⟩ d = cellsMask (cells sq d) := by decide +kernel

private theorem cells_mask_fin : ∀ (sq : Fin 64) (d : Fin 8) (i : Fin 64),
    (mask ⟨sq.val, by omega⟩ d).getLsbD i.val = decide (i.val ∈ cells sq d) := by
  intro sq d i
  rw [mask_eq_cellsMask]
  exact cellsMask_bit (cells sq d) i.val i.isLt

private theorem cells_mask (sq : Fin 64) (d : Fin 8) (i : Nat) :
    (mask ⟨sq.val, by omega⟩ d).getLsbD i = true ↔ i ∈ cells sq d := by
  by_cases h : i < 64
  · simp only [cells_mask_fin sq d ⟨i, h⟩, decide_eq_true_eq]
  · rw [BitVec.getLsbD_of_ge _ _ (by omega)]
    exact ⟨by simp, fun found => False.elim (h ((cells_geometry sq d).2 i found).1)⟩

private theorem before_asymm {d : Fin 8} {i j : Nat} (h : Before d i j) :
    ¬ Before d j i := by
  by_cases forward : d.val < 4 <;> simp only [Before, forward, ↓reduceIte] at * <;> omega

private theorem before_irrefl (d : Fin 8) (i : Nat) : ¬ Before d i i := by
  simp [Before]

private theorem before_total {d : Fin 8} {i j : Nat} (h : i ≠ j) :
    Before d i j ∨ Before d j i := by
  unfold Before
  split <;> omega

private theorem ray_prefix (p : Geometry.Word) (near far : List Nat) (anchor acc : Nat)
    (clear : ∀ i ∈ near, p.getLsbD i = false) (own : p.getLsbD anchor = true) :
    ray p acc (near ++ anchor :: far) = acc + near.length := by
  induction near generalizing acc with
  | nil => simp [ray, own]
  | cons i rest ih =>
    have head := clear i (by simp)
    have tail : ∀ j ∈ rest, p.getLsbD j = false := fun j h => clear j (List.mem_cons_of_mem _ h)
    simp [ray, head, ih (acc + 1) tail]
    omega

private theorem ray_no_anchor (p : Geometry.Word) (xs : List Nat) (acc : Nat)
    (clear : ∀ i ∈ xs, p.getLsbD i = false) :
    ray p acc xs = 0 := by
  induction xs generalizing acc with
  | nil => rfl
  | cons i rest ih =>
    have head := clear i (by simp)
    simp [ray, head, ih (acc + 1) (fun j h => clear j (List.mem_cons_of_mem _ h))]

private theorem count_members (xs : List Nat) (distinct : xs.Nodup)
    (valid : ∀ i ∈ xs, i < 64) :
    (List.range 64).countP (fun i => decide (i ∈ xs)) = xs.length := by
  induction xs with
  | nil => simp
  | cons i rest ih =>
    obtain ⟨unique, distinct⟩ := List.nodup_cons.mp distinct
    have bit : (fun j => decide (j ∈ i :: rest)) =
        (fun j => (squareMask ⟨i, valid i (by simp)⟩).getLsbD j || decide (j ∈ rest)) := by
      funext j
      simp [List.mem_cons, Bitboard.squareMask_bit]
    rw [bit, Bitboard.countP_disjoint_or,
      ← (Bitboard.populationCountContract_proven (squareMask _)).1, Board.squareMask_count_one,
      ih distinct (fun j h => valid j (List.mem_cons_of_mem _ h))]
    · simp [Nat.add_comm]
    · intro j _
      simp only [Bitboard.squareMask_bit, Bool.and_eq_false_iff, decide_eq_false_iff_not]
      by_cases eq : j = i
      · exact Or.inr (eq ▸ unique)
      · exact Or.inl eq

private theorem earlier_iff_prefix (d : Fin 8) (near far : List Nat) (anchor i : Nat)
    (ordered : (near ++ anchor :: far).Pairwise (Before d)) :
    (i ∈ near ++ anchor :: far ∧ Before d i anchor) ↔ i ∈ near := by
  obtain ⟨_, suffix, cross⟩ := List.pairwise_append.mp ordered
  constructor
  · rintro ⟨member, earlier⟩
    rcases List.mem_append.mp member with found | later
    · exact found
    · rcases List.mem_cons.mp later with rfl | found
      · exact False.elim (before_irrefl d i earlier)
      · exact False.elim (before_asymm ((List.pairwise_cons.mp suffix).1 _ found) earlier)
  · intro member
    exact ⟨List.mem_append_left _ member, cross i member anchor (by simp)⟩

private theorem anchor_unique {x : Geometry.Word} {d : Fin 8} {i j : Nat}
    (first : Flip.Edge x d i) (second : Flip.Edge x d j) : i = j := by
  by_cases eq : i = j
  · exact eq
  · rcases before_total (d := d) eq with before | after
    · have no := second.2 i before
      rw [first.1] at no
      contradiction
    · have no := first.2 j after
      rw [second.1] at no
      contradiction

private theorem lane_count (p : Geometry.Word) (sq : Fin 64) (d : Fin 8) :
    (count (laneValue ⟨sq.val, by omega⟩ d p (~~~(p ||| squareMask sq)))).toNat =
      ray p 0 (cells sq d) := by
  let opp := ~~~(p ||| squareMask sq)
  let m := mask ⟨sq.val, by omega⟩ d
  let x := ~~~opp &&& m
  have setBit (i : Nat) : x.getLsbD i = true ↔ i ∈ cells sq d ∧ p.getLsbD i = true := by
    simp only [x, opp, BitVec.not_not, BitVec.getLsbD_and, Bool.and_eq_true,
      BitVec.getLsbD_or, Bool.or_eq_true, Bitboard.squareMask_bit, decide_eq_true_eq, m, cells_mask]
    constructor
    · rintro ⟨own | eq, member⟩
      · exact ⟨member, own⟩
      · exact False.elim (((cells_geometry sq d).2 i member).2 eq)
    · rintro ⟨member, own⟩
      exact ⟨Or.inl own, member⟩
  change (count (Flip.abstractLane m d p opp)).toNat = _
  rw [(Bitboard.populationCountContract_proven _).1]
  cases first : (cells sq d).find? p.getLsbD with
  | none =>
    have clear (i : Nat) (member : i ∈ cells sq d) : p.getLsbD i = false :=
      Bool.eq_false_iff.mpr (List.find?_eq_none.mp first i member)
    rw [ray_no_anchor p _ _ clear]
    apply List.countP_eq_zero.mpr
    intro i _ hit
    obtain ⟨j, edge, own, _⟩ := (Flip.abstract_lane_bits m d p opp i).mp hit
    rw [clear j ((setBit j).mp edge.1).1] at own
    contradiction
  | some anchor =>
    obtain ⟨own, near, far, decomposition, clear⟩ := List.find?_eq_some_iff_append.mp first
    have ordered := (cells_geometry sq d).1
    rw [decomposition] at ordered
    have beforeAnchor (i : Nat) : (i ∈ cells sq d ∧ Before d i anchor) ↔ i ∈ near := by
      rw [decomposition]
      exact earlier_iff_prefix d near far anchor i ordered
    have edge : Flip.Edge x d anchor := by
      refine ⟨(setBit _).mpr ⟨by simp [decomposition], own⟩, ?_⟩
      intro i earlier
      cases bit : x.getLsbD i
      · rfl
      · obtain ⟨member, on⟩ := (setBit i).mp bit
        have off := clear i ((beforeAnchor i).mp ⟨member, earlier⟩)
        simp [on] at off
    have bits (i : Nat) : (Flip.abstractLane m d p opp).getLsbD i = decide (i ∈ near) := by
      apply Bool.eq_iff_iff.mpr
      rw [decide_eq_true_eq, Flip.abstract_lane_bits]
      constructor
      · rintro ⟨j, first, _, inside, earlier⟩
        rw [anchor_unique first edge] at earlier
        exact (beforeAnchor i).mp ⟨(cells_mask sq d i).mp inside, earlier⟩
      · intro member
        obtain ⟨inside, earlier⟩ := (beforeAnchor i).mpr member
        exact ⟨anchor, edge, own, (cells_mask sq d i).mpr inside, earlier⟩
    rw [show (Flip.abstractLane m d p opp).getLsbD = _ from funext bits, decomposition,
      ray_prefix p near far anchor 0 (fun i mem => by simpa using clear i mem) own,
      Nat.zero_add]
    apply count_members
    · have strictly := (List.pairwise_append.mp ordered).1
      apply strictly.imp
      intro i j order eq
      subst j
      exact before_irrefl d i order
    · intro i found
      exact ((cells_geometry sq d).2 i (by rw [decomposition]; exact List.mem_append_left _ found)).1

private theorem count_union (masks : List Geometry.Word)
    (disjoint : masks.Pairwise (fun x y => x &&& y = 0)) :
    (count (masks.foldr (· ||| ·) 0)).toNat = (masks.map (fun x => (count x).toNat)).sum := by
  have apart (x : Geometry.Word) (xs : List Geometry.Word) (h : ∀ y ∈ xs, x &&& y = 0) :
      x &&& xs.foldr (· ||| ·) 0 = 0 := by
    induction xs with
    | nil => simp
    | cons y ys ih =>
      rw [List.foldr_cons, BitVec.and_or_distrib_left, h y (by simp),
        ih (fun z hmem => h z (List.mem_cons_of_mem _ hmem))]
      rfl
  induction masks with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨cross, tail⟩ := List.pairwise_cons.mp disjoint
    rw [List.foldr_cons, Bitboard.count_or_of_disjoint _ _ (apart x xs cross), ih tail]
    rfl

private def dirOk (sq d i : Nat) : Bool :=
  let r0 := sq / 8
  let c0 := sq % 8
  let r := i / 8
  let c := i % 8
  match d with
  | 0 => decide (r = r0 ∧ c > c0)
  | 1 => decide (r > r0 ∧ c = c0)
  | 2 => decide (r > r0 ∧ c > c0 ∧ r - r0 = c - c0)
  | 3 => decide (r > r0 ∧ c < c0 ∧ r - r0 = c0 - c)
  | 4 => decide (r = r0 ∧ c < c0)
  | 5 => decide (r < r0 ∧ c = c0)
  | 6 => decide (r < r0 ∧ c < c0 ∧ r0 - r = c0 - c)
  | _ => decide (r < r0 ∧ c > c0 ∧ r0 - r = c - c0)

private theorem cells_dir_spec : ∀ (sq : Fin 64) (d : Fin 8),
    (cells sq d).all (dirOk sq.val d.val) = true := by decide +kernel

private theorem disjoint_masks (sq : Fin 64) (d e : Fin 8) (hne : d ≠ e) :
    mask ⟨sq.val, by omega⟩ d &&& mask ⟨sq.val, by omega⟩ e = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_and, show (0 : Geometry.Word).getLsbD i = false from BitVec.getLsbD_zero]
  cases hd : (mask ⟨sq.val, by omega⟩ d).getLsbD i with
  | false => simp
  | true =>
    cases he : (mask ⟨sq.val, by omega⟩ e).getLsbD i with
    | false => simp
    | true =>
      exfalso
      have h1 := List.all_eq_true.mp (cells_dir_spec sq d) i ((cells_mask sq d i).mp hd)
      have h2 := List.all_eq_true.mp (cells_dir_spec sq e) i ((cells_mask sq e i).mp he)
      have hne' : d.val ≠ e.val := fun h => hne (Fin.ext h)
      rcases d with ⟨dv, hdv⟩
      rcases e with ⟨ev, hev⟩
      match dv, ev with
      | 0, 0 | 1, 1 | 2, 2 | 3, 3 | 4, 4 | 5, 5 | 6, 6 | 7, 7 => exact hne' rfl
      | 0, 1 | 0, 2 | 0, 3 | 0, 4 | 0, 5 | 0, 6 | 0, 7
      | 1, 0 | 1, 2 | 1, 3 | 1, 4 | 1, 5 | 1, 6 | 1, 7
      | 2, 0 | 2, 1 | 2, 3 | 2, 4 | 2, 5 | 2, 6 | 2, 7
      | 3, 0 | 3, 1 | 3, 2 | 3, 4 | 3, 5 | 3, 6 | 3, 7
      | 4, 0 | 4, 1 | 4, 2 | 4, 3 | 4, 5 | 4, 6 | 4, 7
      | 5, 0 | 5, 1 | 5, 2 | 5, 3 | 5, 4 | 5, 6 | 5, 7
      | 6, 0 | 6, 1 | 6, 2 | 6, 3 | 6, 4 | 6, 5 | 6, 7
      | 7, 0 | 7, 1 | 7, 2 | 7, 3 | 7, 4 | 7, 5 | 7, 6 =>
        simp only [dirOk, decide_eq_true_eq] at h1 h2; omega
      | _ + 8, _ | _, _ + 8 => omega

private def directions : List (Fin 8) := [0, 1, 2, 3, 4, 5, 6, 7]

private theorem reference_sum (p : Geometry.Word) (sq : Fin 64) :
    Bmi2.reference p sq.val = (directions.map (fun d => ray p 0 (cells sq d))).sum := by
  rw [Bmi2.reference_eq_geometry]
  simp [Geometry.reference, directions, cells, Flip.delta, Nat.add_assoc] <;> rfl

theorem reference_eq_flipCount (p : Geometry.Word) (sq : Fin 64) :
    Bmi2.reference p sq.val =
      (count (Flip.value ⟨sq.val, by omega⟩ p (~~~(p ||| squareMask sq)))).toNat := by
  let sq' : Flip.Square := ⟨sq.val, by omega⟩
  let opp := ~~~(p ||| squareMask sq)
  let lane := fun d => laneValue sq' d p opp
  have union : Flip.value sq' p opp = (directions.map lane).foldr (· ||| ·) 0 := by
    simp [directions, lane, Flip.value, Flip.laneValue, BitVec.or_assoc] <;> rfl
  have disjoint : (directions.map lane).Pairwise (fun x y => x &&& y = 0) := by
    rw [List.pairwise_map]
    apply (show directions.Nodup from by decide).imp
    intro d e ne
    apply BitVec.zero_iff_eq_false.mpr
    intro i
    have support (f : Fin 8) (bit : (lane f).getLsbD i = true) :
        (mask sq' f).getLsbD i = true := by
      obtain ⟨_, _, _, on, _⟩ := (Flip.abstract_lane_bits (mask sq' f) f p opp i).mp bit
      exact on
    have apart := congrArg (fun (w : Geometry.Word) => w.getLsbD i) (disjoint_masks sq d e ne)
    change (mask sq' d &&& mask sq' e).getLsbD i = (0#64).getLsbD i at apart
    simp only [BitVec.getLsbD_and] at apart ⊢
    cases hd : (lane d).getLsbD i <;> cases he : (lane e).getLsbD i <;> simp
    rw [support d hd, support e he] at apart
    simp at apart
  change Bmi2.reference p sq.val = (count (Flip.value sq' p opp)).toNat
  rw [union, count_union _ disjoint, reference_sum]
  rw [List.map_map]
  congr 1
  apply List.map_congr_left
  intro d _
  exact (lane_count p sq d).symm

theorem reference_reconstructedOpponent (p : Geometry.Word) (sq : Fin 64) :
    Bmi2.reference (~~~(p ||| squareMask sq)) sq.val = Bmi2.reference (~~~p) sq.val := by
  rw [reference_sum, reference_sum]
  congr 1
  apply List.map_congr_left
  intro d _
  have congrRay (xs : List Nat) (acc : Nat) (offTarget : ∀ i ∈ xs, i < 64 ∧ i ≠ sq.val) :
      ray (~~~(p ||| squareMask sq)) acc xs = ray (~~~p) acc xs := by
    induction xs generalizing acc with
    | nil => rfl
    | cons i rest ih =>
      obtain ⟨within, different⟩ := offTarget i (by simp)
      have same : (~~~(p ||| squareMask sq)).getLsbD i = (~~~p).getLsbD i := by
        simp only [BitVec.getLsbD_not, BitVec.getLsbD_or, Bitboard.squareMask_bit, within,
          different, decide_true, decide_false, Bool.or_false]
      simp only [ray, same, ih (acc + 1) (fun j h => offTarget j (List.mem_cons_of_mem _ h))]
  exact congrRay (cells sq d) 0 (cells_geometry sq d).2

end NeuralReversi.Solve1
