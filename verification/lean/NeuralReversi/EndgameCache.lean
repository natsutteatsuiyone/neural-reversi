module
public import NeuralReversi.Board.Model
public import NeuralReversi.Solve2.Control
public import NeuralReversi.TranspositionTable.Clear

@[expose] public section

namespace NeuralReversi.EndgameCache

open Board (Board)

structure Entry where
  board : Board
  lower : Int8
  upper : Int8

def unset (board : Board) : Entry := ⟨board, -65, 65⟩

structure Write where
  board : Board
  alpha : Int
  score : Int

def store (e : Entry) (w : Write) : Entry :=
  let prior := if e.board = w.board then e else unset w.board
  let packed := Int8.ofInt w.score
  if w.score > w.alpha then ⟨w.board, max prior.lower packed, prior.upper⟩
  else ⟨w.board, prior.lower, min prior.upper packed⟩

def probe (e : Entry) (board : Board) (alpha : Int) : Option Int :=
  if e.board ≠ board then none
  else if e.lower.toInt > alpha then some e.lower.toInt
  else if e.upper.toInt ≤ alpha then some e.upper.toInt
  else none

def SlotContract : Prop :=
  ∀ key (len : BitVec 64), 0 < len.toNat →
    TranspositionTable.Clear.mulHi64 key len < len.toNat

theorem slotContract_proven : SlotContract := by
  intro key len positive
  exact TranspositionTable.Clear.mulHi64_lt key len positive

def InRange (n : Int) : Prop := -64 ≤ n ∧ n ≤ 64

def CacheRange (n : Int) : Prop := -65 ≤ n ∧ n ≤ 65

def StoreRange (n : Int) : Prop := -128 ≤ n ∧ n ≤ 127

def SearchRange (n : Int) : Prop := -127 ≤ n ∧ n ≤ 127

theorem searchRange_store {score : Int} (h : SearchRange score) : StoreRange score := by
  unfold SearchRange at h
  unfold StoreRange
  omega

theorem cacheRange_search {score : Int} (h : CacheRange score) : SearchRange score := by
  unfold CacheRange at h
  unfold SearchRange
  omega

private def Bounds (value : Board → Int) (e : Entry) : Prop :=
  e.lower.toInt ≤ value e.board ∧ value e.board ≤ e.upper.toInt ∧
    -65 ≤ e.lower.toInt ∧ e.upper.toInt ≤ 65

private theorem unset_bounds (value : Board → Int) (board : Board) (bounded : InRange (value board)) :
    Bounds value (unset board) := by
  change -65 ≤ value board ∧ value board ≤ 65 ∧ -65 ≤ (-65 : Int) ∧ (65 : Int) ≤ 65
  unfold InRange at bounded
  omega

private theorem store_bounds (value : Board → Int) (e : Entry) (w : Write)
    (prior : Bounds value e) (bounded : InRange (value w.board)) (range : StoreRange w.score)
    (sound : Solve2.NWS w.alpha (value w.board) w.score) : Bounds value (store e w) := by
  let base := if e.board = w.board then e else unset w.board
  have before : base.lower.toInt ≤ value w.board ∧ value w.board ≤ base.upper.toInt ∧
      -65 ≤ base.lower.toInt ∧ base.upper.toInt ≤ 65 := by
    by_cases same : e.board = w.board
    · simpa [base, same, Bounds] using prior
    · simpa [base, same, Bounds, unset] using unset_bounds value w.board bounded
  have cast : (Int8.ofInt w.score).toInt = w.score := by
    unfold StoreRange at range
    exact Int8.toInt_ofInt_of_le (by omega) (by omega)
  obtain ⟨baseLo, baseHi, loSentinel, hiSentinel⟩ := before
  unfold Bounds store
  by_cases high : w.alpha < w.score
  · have lower := sound.2 high
    simp only [gt_iff_lt, high, ↓reduceIte]
    change (max base.lower (Int8.ofInt w.score)).toInt ≤ _ ∧ _ ≤ base.upper.toInt ∧
      -65 ≤ (max base.lower (Int8.ofInt w.score)).toInt ∧ base.upper.toInt ≤ 65
    simp only [Int8.max_def, apply_ite Int8.toInt, Int8.le_iff_toInt_le, cast]
    split <;> omega
  · have upper := sound.1 (by omega)
    simp only [gt_iff_lt, high, ↓reduceIte]
    change base.lower.toInt ≤ _ ∧ _ ≤ (min base.upper (Int8.ofInt w.score)).toInt ∧
      -65 ≤ base.lower.toInt ∧ (min base.upper (Int8.ofInt w.score)).toInt ≤ 65
    simp only [Int8.min_def, apply_ite Int8.toInt, Int8.le_iff_toInt_le, cast]
    split <;> omega

def AnyWindowContract : Prop :=
  ∀ (value : Board → Int), (∀ board, InRange (value board)) →
    ∀ (writes : List Write),
    (∀ w ∈ writes, StoreRange w.score ∧ Solve2.NWS w.alpha (value w.board) w.score) →
    ∀ board alpha result,
      probe (writes.foldl store (unset ⟨0, 0⟩)) board alpha = some result →
      CacheRange result ∧ Solve2.NWS alpha (value board) result

theorem anyWindowContract_proven : AnyWindowContract := by
  intro value bounded writes valid
  have invariant (start : Entry) (init : Bounds value start) :
      Bounds value (writes.foldl store start) := by
    induction writes generalizing start with
    | nil => exact init
    | cons w ws ih =>
      obtain ⟨range, sound⟩ := valid w (by simp)
      exact ih (fun v mem => valid v (List.mem_cons_of_mem _ mem)) _
        (store_bounds value start w init (bounded _) range sound)
  intro board alpha result hit
  let e := writes.foldl store (unset ⟨0, 0⟩)
  have bounds : Bounds value e := invariant _ (unset_bounds value _ (bounded _))
  change probe e board alpha = some result at hit
  unfold probe at hit
  split at hit
  · contradiction
  next same =>
    have key : e.board = board := by simpa using same
    obtain ⟨lower, upper, loSentinel, hiSentinel⟩ := bounds
    rw [key] at lower upper
    have limits := bounded board
    unfold InRange at limits
    unfold CacheRange
    split at hit
    · cases hit
      exact ⟨by omega, by unfold Solve2.NWS; omega⟩
    · split at hit
      · cases hit
        exact ⟨by omega, by unfold Solve2.NWS; omega⟩
      · contradiction

def CutoffContract : Prop :=
  ∀ (value : Board → Int), (∀ board, InRange (value board)) →
    ∀ (writes : List Write),
    (∀ w ∈ writes, InRange w.score ∧ Solve2.NWS w.alpha (value w.board) w.score) →
    ∀ board alpha result, -65 ≤ alpha → alpha ≤ 64 →
    probe (writes.foldl store (unset ⟨0, 0⟩)) board alpha = some result →
    InRange result ∧ Solve2.NWS alpha (value board) result

theorem cutoffContract_proven : CutoffContract := by
  intro value bounded writes valid board alpha result alo ahi hit
  obtain ⟨_, sound⟩ := anyWindowContract_proven value bounded writes
    (fun w mem => by
      obtain ⟨range, sound⟩ := valid w mem
      exact ⟨by unfold InRange at range; unfold StoreRange; omega, sound⟩)
    board alpha result hit
  have valueRange := bounded board
  unfold InRange at valueRange ⊢
  refine ⟨?_, sound⟩
  by_cases low : result ≤ alpha
  · have := sound.1 low
    omega
  · have := sound.2 (by omega)
    omega

example (b : Board) :
    let e := store (store (unset b) ⟨b, -5, -4⟩) ⟨b, 4, 2⟩
    probe e b (-5) = some (-4) ∧ probe e b 2 = some 2 ∧ probe e b 0 = none := by
  simp [store, probe, unset] <;> decide

def enhancedCutoff (entryFor : Board → Entry) (alpha : Int) (children : List Board) : Option Int :=
  let childAlpha := -(alpha + 1)
  children.findSome? fun child => do
    let score ← probe (entryFor child) child childAlpha
    if score ≤ childAlpha then some (-score) else none

def EnhancedCutoffContract : Prop :=
  ∀ (value : Board → Int), (∀ b, InRange (value b)) →
    ∀ (writesFor : Board → List Write),
    (∀ child w, w ∈ writesFor child → InRange w.score ∧ Solve2.NWS w.alpha (value w.board) w.score) →
    ∀ children alpha score, -65 ≤ alpha → alpha ≤ 64 →
    enhancedCutoff (fun b => (writesFor b).foldl store (unset ⟨0, 0⟩)) alpha children = some score →
    InRange score ∧ alpha < score ∧
      ∃ best, (children.map (fun b => -(value b))).max? = some best ∧ score ≤ best

theorem enhancedCutoffContract_proven : EnhancedCutoffContract := by
  intro value bounded writesFor valid children alpha score low high cut
  obtain ⟨child, member, candidate⟩ := List.exists_of_findSome?_eq_some cut
  obtain ⟨upper, hit, selected⟩ := Option.bind_eq_some_iff.mp candidate
  by_cases qualifies : upper ≤ -(alpha + 1)
  · simp [qualifies] at selected
    obtain ⟨range, sound⟩ := cutoffContract_proven value bounded (writesFor child)
      (valid child) child (-(alpha + 1)) upper (by omega) (by omega) hit
    have childUpperBound := sound.1 qualifies
    have available : -(value child) ∈ children.map (fun b => -(value b)) :=
      List.mem_map.mpr ⟨child, member, rfl⟩
    refine ⟨?_, by omega, _, List.max?_eq_some_max (List.ne_nil_of_mem available), ?_⟩
    · unfold InRange at range ⊢
      omega
    · exact Int.le_trans (by omega) (List.le_max_of_mem available)
  · simp [qualifies] at selected

example : let first : Board := ⟨1, 2⟩
          let second : Board := ⟨2, 1⟩
          let entryFor := fun b => if b = first then store (unset b) ⟨b, -4, -3⟩
                                   else store (unset b) ⟨b, -4, -4⟩
          enhancedCutoff entryFor 3 [first, second] = some 4 := by decide

end NeuralReversi.EndgameCache
