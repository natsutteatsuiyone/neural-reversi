import NeuralReversi.MoveList.Initialization
import NeuralReversi.MoveList.OrderWeights

namespace NeuralReversi.MoveList.Ordering
open Initialization

def swapMemory (mem : Memory α) (a b : Nat) : Memory α :=
  fun i =>
    if i = a then mem b
    else if i = b then mem a
    else mem i

def InitializedPrefix (mem : Memory α) (len : Nat) : Prop :=
  ∀ i, i < len → ∃ v, mem i = some v

theorem swapMemory_bounds_safe (mem : Memory α) (len a b : Nat)
    (init : InitializedPrefix mem len) (ha : a < len) (hb : b < len) :
    (∃ va, mem a = some va) ∧
    (∃ vb, mem b = some vb) ∧
    InitializedPrefix (swapMemory mem a b) len ∧
    (∀ i, len ≤ i → swapMemory mem a b i = mem i) := by
  refine ⟨init a ha, init b hb, ?_, ?_⟩
  · intro i hi
    unfold swapMemory
    split
    · exact init b hb
    · split
      · exact init a ha
      · exact init i hi
  · intro i hi
    have hna : i ≠ a := by omega
    have hnb : i ≠ b := by omega
    simp [swapMemory, hna, hnb]

theorem swapMemory_capacity_safe (len a b : Nat) (cap : len ≤ 35)
    (ha : a < len) (hb : b < len) : a < 35 ∧ b < 35 :=
  ⟨by omega, by omega⟩

def swapList (xs : List α) (a b : Nat) : List α :=
  match xs[a]?, xs[b]? with
  | some va, some vb => (xs.set a vb).set b va
  | _, _ => xs

theorem swapList_length (xs : List α) (a b : Nat) :
    (swapList xs a b).length = xs.length := by
  unfold swapList
  cases xs[a]? <;> cases xs[b]? <;> simp

theorem swapList_getElem?_eq (xs : List α) (a b i : Nat)
    (ha : a < xs.length) (hb : b < xs.length) :
    (swapList xs a b)[i]? =
      if i = b then xs[a]?
      else if i = a then xs[b]?
      else xs[i]? := by
  unfold swapList
  rw [List.getElem?_eq_getElem ha, List.getElem?_eq_getElem hb]
  dsimp only
  by_cases hib : i = b
  · subst i
    have hlen : b < (xs.set a xs[b]).length := by simpa using hb
    simp [List.getElem?_set_self hlen]
  · rw [List.getElem?_set_ne (by omega)]
    by_cases hia : i = a
    · subst i
      simp [hib, List.getElem?_set_self ha]
    · simp only [hib, hia, ↓reduceIte]
      exact List.getElem?_set_ne (by omega)

theorem swapList_symm (xs : List α) (a b : Nat) :
    swapList xs a b = swapList xs b a := by
  by_cases ha : a < xs.length
  · by_cases hb : b < xs.length
    · apply List.ext_getElem?
      intro i
      rw [swapList_getElem?_eq xs a b i ha hb, swapList_getElem?_eq xs b a i hb ha]
      by_cases hia : i = a <;> by_cases hib : i = b <;> simp_all
    · have hbn : xs[b]? = none := List.getElem?_eq_none_iff.mpr (by omega)
      simp [swapList, hbn]
  · have han : xs[a]? = none := List.getElem?_eq_none_iff.mpr (by omega)
    simp [swapList, han]

theorem swapList_involutive (xs : List α) (a b : Nat) :
    swapList (swapList xs a b) a b = xs := by
  by_cases ha : a < xs.length
  · by_cases hb : b < xs.length
    · apply List.ext_getElem?
      intro i
      have ha' : a < (swapList xs a b).length := by rw [swapList_length]; exact ha
      have hb' : b < (swapList xs a b).length := by rw [swapList_length]; exact hb
      rw [swapList_getElem?_eq (swapList xs a b) a b i ha' hb']
      rw [swapList_getElem?_eq xs a b a ha hb, swapList_getElem?_eq xs a b b ha hb,
        swapList_getElem?_eq xs a b i ha hb]
      by_cases hia : i = a <;> by_cases hib : i = b <;> by_cases hab : a = b <;> simp_all
    · have hbn : xs[b]? = none := List.getElem?_eq_none_iff.mpr (by omega)
      simp [swapList, hbn]
  · have han : xs[a]? = none := List.getElem?_eq_none_iff.mpr (by omega)
    simp [swapList, han]

private theorem list_split_two {xs : List α} {a b : Nat} (hab : a < b) (hb : b < xs.length) :
    ∃ p m s,
      p.length = a ∧
      (p ++ xs[b] :: m).length = b ∧
      xs = p ++ xs[a] :: m ++ xs[b] :: s := by
  have ha : a < xs.length := Nat.lt_trans hab hb
  let p := xs.take a
  let rest := xs.drop (a + 1)
  have hxs : xs = p ++ xs[a] :: rest := by
    have h1 := (List.take_append_drop a xs).symm
    have h2 : xs.drop a = xs[a] :: xs.drop (a + 1) := List.drop_eq_getElem_cons ha
    rw [h2] at h1
    exact h1
  have hp : p.length = a := by
    simp [p, Nat.min_eq_left (Nat.le_of_lt ha)]
  let d := b - (a + 1)
  have hd : d < rest.length := by
    simp [rest, d]
    omega
  let m := rest.take d
  let s := rest.drop (d + 1)
  have hrest : rest = m ++ rest[d] :: s := by
    have h1 := (List.take_append_drop d rest).symm
    have h2 : rest.drop d = rest[d] :: rest.drop (d + 1) := List.drop_eq_getElem_cons hd
    rw [h2] at h1
    exact h1
  have hrb : rest[d] = xs[b] := by
    have : (xs.drop (a + 1))[d] = xs[a + 1 + d] := List.getElem_drop
    have heq : a + 1 + d = b := by omega
    simp [rest, heq]
  have hm : m.length = d := by
    simp [m, Nat.min_eq_left (Nat.le_of_lt hd)]
  refine ⟨p, m, s, hp, by simp [hp, hm, d]; omega, ?_⟩
  calc xs = p ++ xs[a] :: rest := hxs
    _ = p ++ xs[a] :: (m ++ rest[d] :: s) := congrArg (fun r => p ++ xs[a] :: r) hrest
    _ = p ++ xs[a] :: (m ++ xs[b] :: s) := by rw [hrb]
    _ = p ++ xs[a] :: m ++ xs[b] :: s := by simp [List.append_assoc]

private theorem set_append_cons (p : List α) (x y : α) (s : List α) :
    (p ++ x :: s).set p.length y = p ++ y :: s := by
  induction p with
  | nil => rfl
  | cons h t ih =>
    simp [ih]

private theorem swapList_perm_lt {xs : List α} {a b : Nat} (hab : a < b) (hb : b < xs.length) :
    List.Perm (swapList xs a b) xs := by
  have ha : a < xs.length := Nat.lt_trans hab hb
  obtain ⟨p, m, s, hp, hpm, hxs⟩ := list_split_two hab hb
  have hgetA : xs[a]? = some xs[a] := List.getElem?_eq_getElem ha
  have hgetB : xs[b]? = some xs[b] := List.getElem?_eq_getElem hb
  unfold swapList
  rw [hgetA, hgetB]
  dsimp only
  generalize hva : xs[a] = va
  generalize hvb : xs[b] = vb
  rw [hvb] at hpm
  rw [hva, hvb] at hxs
  have h1 : xs = p ++ va :: (m ++ vb :: s) := by simpa [List.append_assoc] using hxs
  have hset1 : xs.set a vb = (p ++ vb :: m) ++ vb :: s := by
    rw [h1, ← hp, set_append_cons]
    simp [List.append_assoc]
  have hset2 : (xs.set a vb).set b va = p ++ vb :: (m ++ va :: s) := by
    rw [hset1, ← hpm, set_append_cons]
    simp [List.append_assoc]
  rw [hset2, h1]
  apply List.Perm.append_left p
  have p1 : List.Perm (vb :: (m ++ va :: s)) (vb :: va :: (m ++ s)) :=
    List.Perm.cons vb (List.perm_middle)
  have p2 : List.Perm (vb :: va :: (m ++ s)) (va :: vb :: (m ++ s)) :=
    List.Perm.swap va vb (m ++ s)
  have p3 : List.Perm (va :: vb :: (m ++ s)) (va :: (m ++ vb :: s)) :=
    List.Perm.cons va (List.perm_middle.symm)
  exact p1.trans (p2.trans p3)

theorem swapList_perm (xs : List α) (a b : Nat) :
    List.Perm (swapList xs a b) xs := by
  by_cases ha : a < xs.length
  · by_cases hb : b < xs.length
    · by_cases hlt : a < b
      · exact swapList_perm_lt hlt hb
      · by_cases hgt : b < a
        · rw [swapList_symm]
          exact swapList_perm_lt hgt ha
        · have heq : a = b := by omega
          subst heq
          have hself : swapList xs a a = xs := by
            apply List.ext_getElem?
            intro i
            rw [swapList_getElem?_eq xs a a i ha ha]
            by_cases hia : i = a <;> simp [hia]
          rw [hself]
    · have hbn : xs[b]? = none := List.getElem?_eq_none_iff.mpr (by omega)
      simp [swapList, hbn]
  · have han : xs[a]? = none := List.getElem?_eq_none_iff.mpr (by omega)
    simp [swapList, han]

theorem swapList_mem_iff (xs : List α) (a b : Nat) (m : α) :
    m ∈ swapList xs a b ↔ m ∈ xs :=
  (swapList_perm xs a b).mem_iff

theorem swapList_count [DecidableEq α] (xs : List α) (a b : Nat) (m : α) :
    (swapList xs a b).count m = xs.count m :=
  (swapList_perm xs a b).count_eq m

def postponeMove (xs : List α) (index : Nat) : List α :=
  match xs.drop index with
  | [] => xs
  | x :: rest => xs.take index ++ rest ++ [x]

theorem postponeMove_bounds_safe (len index : Nat) (cap : len ≤ 35) (hi : index < len) :
    index ≤ len ∧ 1 ≤ len - index ∧ index < 35 :=
  ⟨by omega, by omega, by omega⟩

theorem postponeMove_eq (xs : List α) (index : Nat) (hi : index < xs.length) :
    postponeMove xs index = xs.take index ++ xs.drop (index + 1) ++ [xs[index]] := by
  unfold postponeMove
  rw [List.drop_eq_getElem_cons hi]

theorem postponeMove_length (xs : List α) (index : Nat) :
    (postponeMove xs index).length = xs.length := by
  by_cases hi : index < xs.length
  · rw [postponeMove_eq xs index hi]
    simp only [List.length_append, List.length_take, List.length_drop, List.length_singleton]
    omega
  · have hdrop : xs.drop index = [] := List.drop_eq_nil_of_le (by omega)
    simp [postponeMove, hdrop]

theorem postponeMove_perm (xs : List α) (index : Nat) :
    List.Perm (postponeMove xs index) xs := by
  by_cases hi : index < xs.length
  · rw [postponeMove_eq xs index hi]
    have hxs : xs = xs.take index ++ (xs[index] :: xs.drop (index + 1)) := by
      have h1 := (List.take_append_drop index xs).symm
      rw [List.drop_eq_getElem_cons hi] at h1
      exact h1
    have hperm : List.Perm (xs.take index ++ (xs.drop (index + 1) ++ [xs[index]]))
        (xs.take index ++ (xs[index] :: xs.drop (index + 1))) :=
      List.Perm.append_left (xs.take index) (List.perm_append_singleton xs[index] (xs.drop (index + 1)))
    rw [List.append_assoc]
    exact hperm.trans (by rw [← hxs])
  · have hdrop : xs.drop index = [] := List.drop_eq_nil_of_le (by omega)
    simp [postponeMove, hdrop]

theorem postponeMove_mem_iff (xs : List α) (index : Nat) (m : α) :
    m ∈ postponeMove xs index ↔ m ∈ xs :=
  (postponeMove_perm xs index).mem_iff

theorem postponeMove_count [DecidableEq α] (xs : List α) (index : Nat) (m : α) :
    (postponeMove xs index).count m = xs.count m :=
  (postponeMove_perm xs index).count_eq m

theorem postponeMove_last (xs : List α) (index : Nat) (hi : index < xs.length) :
    (postponeMove xs index)[xs.length - 1]? = xs[index]? := by
  rw [postponeMove_eq xs index hi, List.getElem?_eq_getElem hi]
  have hlen : (xs.take index ++ xs.drop (index + 1)).length = xs.length - 1 := by
    simp only [List.length_append, List.length_take, List.length_drop]
    omega
  rw [List.getElem?_append_right (by omega), hlen]
  simp

theorem postponeMove_prefix (xs : List α) (index i : Nat)
    (hi : index < xs.length) (hpre : i < index) :
    (postponeMove xs index)[i]? = xs[i]? := by
  rw [postponeMove_eq xs index hi, List.append_assoc]
  have htake : i < (xs.take index).length := by
    simp [Nat.min_eq_left (Nat.le_of_lt hi), hpre]
  rw [List.getElem?_append_left htake, List.getElem?_take_of_lt hpre]

theorem postponeMove_shifted (xs : List α) (index i : Nat)
    (hlo : index ≤ i) (hhi : i + 1 < xs.length) :
    (postponeMove xs index)[i]? = xs[i + 1]? := by
  have hi : index < xs.length := by omega
  rw [postponeMove_eq xs index hi]
  have htakeLen : (xs.take index).length = index := by
    simp [Nat.min_eq_left (Nat.le_of_lt hi)]
  have hdropLen : (xs.drop (index + 1)).length = xs.length - (index + 1) := by
    simp
  have hpreLen : i < (xs.take index ++ xs.drop (index + 1)).length := by
    simp [htakeLen, hdropLen]
    omega
  rw [List.getElem?_append_left hpreLen, List.getElem?_append_right (by omega), htakeLen,
    List.getElem?_drop]
  have heq : index + 1 + (i - index) = i + 1 := by omega
  rw [heq]

def compareSwap (key : α → Int) (xs : List α) (a b : Nat) : List α :=
  match xs[a]?, xs[b]? with
  | some va, some vb =>
      if key va < key vb then swapList xs a b else xs
  | _, _ => xs

theorem compareSwap_length (key : α → Int) (xs : List α) (a b : Nat) :
    (compareSwap key xs a b).length = xs.length := by
  unfold compareSwap
  cases xs[a]? <;> cases xs[b]? <;> simp
  split <;> simp [swapList_length]

theorem compareSwap_perm (key : α → Int) (xs : List α) (a b : Nat) :
    List.Perm (compareSwap key xs a b) xs := by
  unfold compareSwap
  cases xs[a]? <;> cases xs[b]? <;> simp
  split
  · exact swapList_perm xs a b
  · exact List.Perm.refl xs

def SortedDesc (key : α → Int) (xs : List α) : Prop :=
  xs.Pairwise (fun a b => key b ≤ key a)

def insertDesc (key : α → Int) (x : α) : List α → List α
  | [] => [x]
  | y :: ys => if key y < key x then x :: y :: ys else y :: insertDesc key x ys

def insertionSortDesc (key : α → Int) : List α → List α
  | [] => []
  | x :: xs => insertDesc key x (insertionSortDesc key xs)

private theorem insertDesc_perm (key : α → Int) (x : α) (xs : List α) :
    List.Perm (insertDesc key x xs) (x :: xs) := by
  induction xs with
  | nil => exact List.Perm.refl [x]
  | cons y ys ih =>
    unfold insertDesc
    split
    · exact List.Perm.refl _
    · exact (List.Perm.cons y ih).trans (List.Perm.swap x y ys)

private theorem insertDesc_sorted (key : α → Int) (x : α) (xs : List α)
    (hxs : SortedDesc key xs) :
    SortedDesc key (insertDesc key x xs) := by
  induction xs with
  | nil => simp [insertDesc, SortedDesc]
  | cons y ys ih =>
    unfold insertDesc
    split
    · rename_i hlt
      simp only [SortedDesc, List.pairwise_cons] at hxs ⊢
      refine ⟨?_, hxs.1, hxs.2⟩
      intro z hz
      cases hz with
      | head => omega
      | tail _ hmem =>
        have hyz := hxs.1 z hmem
        omega
    · rename_i hge
      simp only [SortedDesc, List.pairwise_cons] at hxs ⊢
      refine ⟨?_, ih hxs.2⟩
      intro z hz
      have hmem : z ∈ x :: ys := (insertDesc_perm key x ys).mem_iff.mp hz
      cases hmem with
      | head => omega
      | tail _ hys => exact hxs.1 z hys

private theorem insertionSortDesc_perm (key : α → Int) (xs : List α) :
    List.Perm (insertionSortDesc key xs) xs := by
  induction xs with
  | nil => exact List.Perm.refl []
  | cons x xs ih =>
    exact (insertDesc_perm key x _).trans (List.Perm.cons x ih)

private theorem insertionSortDesc_sorted (key : α → Int) (xs : List α) :
    SortedDesc key (insertionSortDesc key xs) := by
  induction xs with
  | nil => simp [insertionSortDesc, SortedDesc]
  | cons x xs ih => exact insertDesc_sorted key x _ ih

def sortByKeyDesc (key : α → Int) (xs : List α) : List α :=
  match xs.length with
  | 0 | 1 => xs
  | 2 => compareSwap key xs 0 1
  | 3 =>
      let s1 := compareSwap key xs 0 1
      let s2 := compareSwap key s1 1 2
      compareSwap key s2 0 1
  | _ => insertionSortDesc key xs

theorem sortByKeyDesc_fastpath_bounds_safe (len : Nat) :
    (len = 2 → 0 < len ∧ 1 < len ∧ 0 < 35 ∧ 1 < 35) ∧
    (len = 3 → 0 < len ∧ 1 < len ∧ 2 < len ∧ 0 < 35 ∧ 1 < 35 ∧ 2 < 35) := by
  constructor <;> intro h <;> omega

theorem sortByKeyDesc_perm (key : α → Int) (xs : List α) :
    List.Perm (sortByKeyDesc key xs) xs := by
  unfold sortByKeyDesc
  cases h : xs.length with
  | zero => exact List.Perm.refl xs
  | succ n =>
    cases n with
    | zero => exact List.Perm.refl xs
    | succ n =>
      cases n with
      | zero => exact compareSwap_perm key xs 0 1
      | succ n =>
        cases n with
        | zero =>
          exact (compareSwap_perm key _ 0 1).trans
            ((compareSwap_perm key _ 1 2).trans (compareSwap_perm key xs 0 1))
        | succ _ => exact insertionSortDesc_perm key xs

theorem sortByKeyDesc_length (key : α → Int) (xs : List α) :
    (sortByKeyDesc key xs).length = xs.length :=
  (sortByKeyDesc_perm key xs).length_eq

theorem sortByKeyDesc_mem_iff (key : α → Int) (xs : List α) (m : α) :
    m ∈ sortByKeyDesc key xs ↔ m ∈ xs :=
  (sortByKeyDesc_perm key xs).mem_iff

theorem sortByKeyDesc_count [DecidableEq α] (key : α → Int) (xs : List α) (m : α) :
    (sortByKeyDesc key xs).count m = xs.count m :=
  (sortByKeyDesc_perm key xs).count_eq m

private theorem compareSwap_two (key : α → Int) (a b : α) :
    compareSwap key [a, b] 0 1 = if key a < key b then [b, a] else [a, b] := by
  simp [compareSwap, swapList]

private theorem compareSwap_three_01 (key : α → Int) (a b c : α) :
    compareSwap key [a, b, c] 0 1 = if key a < key b then [b, a, c] else [a, b, c] := by
  simp [compareSwap, swapList]

private theorem compareSwap_three_12 (key : α → Int) (a b c : α) :
    compareSwap key [a, b, c] 1 2 = if key b < key c then [a, c, b] else [a, b, c] := by
  simp [compareSwap, swapList]

theorem sortByKeyDesc_sorted (key : α → Int) (xs : List α) :
    SortedDesc key (sortByKeyDesc key xs) := by
  match xs with
  | [] => simp [sortByKeyDesc, SortedDesc]
  | [a] => simp [sortByKeyDesc, SortedDesc]
  | [a, b] =>
    simp only [sortByKeyDesc, List.length_cons, List.length_nil, compareSwap_two]
    split <;> simp [SortedDesc] <;> omega
  | [a, b, c] =>
    simp only [sortByKeyDesc, List.length_cons, List.length_nil, compareSwap_three_01]
    split
    · simp only [compareSwap_three_12]
      split
      · simp only [compareSwap_three_01]
        split <;> simp [SortedDesc] <;> omega
      · simp only [compareSwap_three_01]
        split <;> simp [SortedDesc] <;> omega
    · simp only [compareSwap_three_12]
      split
      · simp only [compareSwap_three_01]
        split <;> simp [SortedDesc] <;> omega
      · simp only [compareSwap_three_01]
        split <;> simp [SortedDesc] <;> omega
  | a :: b :: c :: d :: rest =>
    simp only [sortByKeyDesc, List.length_cons]
    exact insertionSortDesc_sorted key (a :: b :: c :: d :: rest)

example : postponeMove [10, 20, 30, 40] 1 = [10, 30, 40, 20] := rfl
example : postponeMove [10, 20, 30] 0 = [20, 30, 10] := rfl
example : swapList [10, 20, 30] 0 2 = [30, 20, 10] := rfl
def SortContract : Prop :=
  ∀ (α : Type) (key : α → Int) (xs : List α),
    List.Perm (sortByKeyDesc key xs) xs ∧ SortedDesc key (sortByKeyDesc key xs)

theorem sortContract_proven : SortContract := by
  intro α key xs
  exact ⟨sortByKeyDesc_perm key xs, sortByKeyDesc_sorted key xs⟩

def ttMoveValue : Int := 1073741824

def TtPresearchOrder [DecidableEq Label] (label : α → Label) (score : α → Int) (hint : Label)
    (moves ordered : List α) : Prop :=
  match moves with
  | [first, _] => ordered = if label first = hint then moves else swapList moves 0 1
  | _ :: _ :: _ :: _ =>
    ordered.Perm moves ∧ SortedDesc (fun move => if label move = hint then ttMoveValue else score move) ordered
  | _ => ordered = moves

/-- Every entry of the extracted square table lies between zero and its maximum. -/
theorem squareValue_range : ∀ i : Fin 64, 0 ≤ OrderWeights.squareValue[i] ∧ OrderWeights.squareValue[i] ≤ OrderWeights.squareMax := by
  decide

/-- The endgame ordering keys, with the extracted weights, fit in `i32` below the TT hint. -/
def EndgamePriorityContract : Prop :=
  ∀ square stability mobility potential evalScore : Int,
    0 ≤ square → square ≤ OrderWeights.squareMax → 0 ≤ stability → stability ≤ 64 →
    0 ≤ mobility → mobility ≤ 128 → 0 ≤ potential → potential ≤ 128 →
    -32767 ≤ evalScore → evalScore ≤ 32767 →
    let fast := square * OrderWeights.squareWeight + stability * OrderWeights.stabilityWeight +
      (36 - mobility) * OrderWeights.mobilityWeight
    let skip := -mobility
    let shallow := evalScore + stability * OrderWeights.searchStabilityWeight -
      mobility * OrderWeights.searchMobilityWeight - potential * OrderWeights.searchPotentialWeight
    ∀ value ∈ [fast, skip, shallow], -2147483648 ≤ value ∧ value < ttMoveValue

theorem endgamePriorityContract_proven : EndgamePriorityContract := by
  intro square stability mobility potential evalScore svLo svHi stLo stHi mLo mHi pLo pHi eLo eHi
  simp only [OrderWeights.squareMax, OrderWeights.squareWeight, OrderWeights.stabilityWeight,
    OrderWeights.mobilityWeight, OrderWeights.searchStabilityWeight, OrderWeights.searchMobilityWeight,
    OrderWeights.searchPotentialWeight] at svHi ⊢
  intro value member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;> unfold ttMoveValue <;> omega

def PresearchedOrderContract : Prop :=
  ∀ (α Label : Type) [DecidableEq Label] (label : α → Label) (moves ordered : List α)
    (hint : Label) (score : α → Int),
    (∃ selected ∈ moves, label selected = hint) →
    (∀ move ∈ moves, label move ≠ hint → score move < ttMoveValue) →
    TtPresearchOrder label score hint moves ordered →
    ordered.Perm moves ∧ ordered.head?.map label = some hint

theorem presearchedOrderContract_proven : PresearchedOrderContract := by
  rintro α Label eq label moves ordered hint score ⟨selected, present, matchHint⟩ lower observed
  cases moves with
  | nil => simp at present
  | cons first rest =>
    cases rest with
    | nil =>
      have same : selected = first := List.mem_singleton.mp present
      subst selected
      simp [TtPresearchOrder] at observed
      simp [observed, matchHint]
    | cons second rest =>
      cases rest with
      | nil =>
        by_cases same : label first = hint
        · simp [TtPresearchOrder, same] at observed
          simp [observed, same]
        · have other := (List.mem_cons.mp present).resolve_left (fun bad => same (bad ▸ matchHint))
          have secondHint := (List.mem_singleton.mp other) ▸ matchHint
          simp only [TtPresearchOrder, same, ↓reduceIte] at observed
          rw [observed]
          exact ⟨swapList_perm [first, second] 0 1, by simp [swapList, secondHint]⟩
      | cons third rest =>
        let moves := first :: second :: third :: rest
        let priority := fun move => if label move = hint then ttMoveValue else score move
        obtain ⟨perm, sorted⟩ : ordered.Perm moves ∧ SortedDesc priority ordered := observed
        have carried : selected ∈ ordered := perm.mem_iff.mpr present
        refine ⟨perm, ?_⟩
        cases ordered with
        | nil => simp at carried
        | cons top tail =>
          have priorityTop : priority selected ≤ priority top := by
            rcases List.mem_cons.mp carried with same | inside
            · rw [same]
              exact Int.le_refl _
            · exact (List.pairwise_cons.mp sorted).1 _ inside
          by_cases same : label top = hint
          · simp [same]
          · have low := lower top (perm.mem_iff.mp (by simp)) same
            simp [priority, matchHint, same] at priorityTop
            omega

def ReorderContract : Prop :=
  ∀ (α : Type) (xs : List α),
    (∀ a b, a < xs.length → b < xs.length →
      List.Perm (swapList xs a b) xs ∧ swapList (swapList xs a b) a b = xs) ∧
    ∀ i, i < xs.length → List.Perm (postponeMove xs i) xs

theorem reorderContract_proven : ReorderContract := by
  intro α xs
  exact ⟨fun a b _ _ => ⟨swapList_perm xs a b, swapList_involutive xs a b⟩,
    fun i _ => postponeMove_perm xs i⟩

example : sortByKeyDesc id [1, 3, 2] = [3, 2, 1] := rfl
example : sortByKeyDesc id [1, 2] = [2, 1] := rfl
example : sortByKeyDesc id [2, 5, 1, 4, 3] = [5, 4, 3, 2, 1] := rfl

end NeuralReversi.MoveList.Ordering
