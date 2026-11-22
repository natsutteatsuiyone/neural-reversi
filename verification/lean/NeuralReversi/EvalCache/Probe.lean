import NeuralReversi.EvalCache.Layout

namespace NeuralReversi.EvalCache

abbrev Bucket := Way → Word

private def maskFour (a b c d : Bool) : BitVec 32 :=
  (if a then 1 else 0) ||| (if b then 2 else 0) ||| (if c then 4 else 0) ||| (if d then 8 else 0)

private def flags (a b c d : Bool) (i : Way) : Bool :=
  if i = 0 then a else if i = 1 then b else if i = 2 then c else d

private theorem maskFour_bits :
    ∀ a b c d : Bool, ∀ i : Fin 32,
      (maskFour a b c d).getLsbD i =
        if h : i.val < 4 then flags a b c d ⟨i.val, h⟩ else false := by decide

def wayMask (f : Way → Bool) : BitVec 32 := maskFour (f 0) (f 1) (f 2) (f 3)

theorem wayMask_bit (f : Way → Bool) (i : Nat) :
    (wayMask f).getLsbD i = if h : i < 4 then f ⟨i, h⟩ else false := by
  have same : flags (f 0) (f 1) (f 2) (f 3) = f := by
    funext w
    have range : w.val = 0 ∨ w.val = 1 ∨ w.val = 2 ∨ w.val = 3 := by have := w.isLt; omega
    rcases range with z | o | t | u
    · rw [show w = 0 from Fin.ext z]; rfl
    · rw [show w = 1 from Fin.ext o]; rfl
    · rw [show w = 2 from Fin.ext t]; rfl
    · rw [show w = 3 from Fin.ext u]; rfl
  by_cases inside : i < 32
  · simpa only [wayMask, same] using maskFour_bits (f 0) (f 1) (f 2) (f 3) ⟨i, inside⟩
  · rw [BitVec.getLsbD_of_ge (x := wayMask f) (i := i) (by omega), dite_eq_right (by omega)]

def First (f : Way → Bool) (i : Way) : Prop :=
  f i = true ∧ ∀ j : Way, j.val < i.val → f j = false

theorem no_ways {f : Way → Bool} (none : wayMask f = 0) : ∀ i : Way, f i = false := by
  intro i
  have zero := wayMask_bit f i.val
  rw [none, dite_eq_left i.isLt] at zero
  simpa [BitVec.getLsbD] using zero.symm

private theorem ctz_bound {f : Way → Bool} (positive : wayMask f ≠ 0) :
    (wayMask f).ctz.toNat < 4 := by
  have one := BitVec.getLsbD_true_ctz_of_ne_zero positive
  rw [wayMask_bit] at one
  split at one
  · assumption
  · contradiction

def firstWay (f : Way → Bool) (nonzero : wayMask f ≠ 0) : Way :=
  ⟨(wayMask f).ctz.toNat, ctz_bound nonzero⟩

theorem firstWay_spec (f : Way → Bool) (nonzero : wayMask f ≠ 0) :
    First f (firstWay f nonzero) := by
  constructor
  · have hit := BitVec.getLsbD_true_ctz_of_ne_zero nonzero
    rwa [wayMask_bit, dite_eq_left (ctz_bound nonzero)] at hit
  · intro j lower
    have absent := BitVec.getLsbD_false_of_lt_ctz (x := wayMask f) lower
    rwa [wayMask_bit, dite_eq_left j.isLt] at absent

inductive ProbeResult where
  | hit (way : Way) (score : Int)
  | miss (way : Way)
  deriving DecidableEq, Repr

def hitFlags (key : Key) (entries : Bucket) (i : Way) : Bool := isHit key (entries i)

def emptyFlags (entries : Bucket) (i : Way) : Bool := entries i == emptyWord

def select (key : Key) (entries : Bucket) (replacement : Way) : ProbeResult :=
  if hit : wayMask (hitFlags key entries) ≠ 0 then
    let way := firstWay (hitFlags key entries) hit
    .hit way ((entries way).setWidth 16).toInt
  else if empty : wayMask (emptyFlags entries) ≠ 0 then .miss (firstWay (emptyFlags entries) empty)
  else .miss replacement

def SelectionSpec (key : Key) (entries : Bucket) (replacement : Way) : ProbeResult → Prop
  | .hit way score => First (hitFlags key entries) way ∧ read key (entries way) = some score
  | .miss way => (∀ i : Way, read key (entries i) = none) ∧
      (First (emptyFlags entries) way ∨ ((∀ i : Way, entries i ≠ emptyWord) ∧ way = replacement))

def ProbeContract : Prop :=
  ∀ (key : Key) (entries : Bucket) (replacement : Way),
    SelectionSpec key entries replacement (select key entries replacement)

theorem probeContract_proven : ProbeContract := by
  intro key entries replacement
  unfold select
  split
  · rename_i found
    have earliest := firstWay_spec (hitFlags key entries) found
    refine ⟨earliest, ?_⟩
    have hits : isHit key (entries (firstWay (hitFlags key entries) found)) = true := earliest.1
    simp only [read, hits, ite_true]
  · rename_i missed
    have zero : wayMask (hitFlags key entries) = 0 :=
      Classical.byContradiction missed
    have noHit := no_ways zero
    have none (i : Way) : read key (entries i) = none := by
      simp [read, show isHit key (entries i) = false from noHit i]
    split
    · exact ⟨none, Or.inl (firstWay_spec ..)⟩
    · rename_i full
      have zero : wayMask (emptyFlags entries) = 0 := Classical.byContradiction full
      have filled := no_ways zero
      refine ⟨none, Or.inr ⟨?_, rfl⟩⟩
      intro i eq
      have contradiction := filled i
      simp [emptyFlags, eq] at contradiction

private def duplicate : Bucket := fun i =>
  if i = 0 then pack 17 11 else if i = 1 then pack 17 22 else emptyWord

set_option maxRecDepth 4096 in
example : select 17 duplicate 3 = .hit 0 11 := by decide
set_option maxRecDepth 4096 in
example : select 0 (fun _ => emptyWord) 3 = .miss 0 := by decide
set_option maxRecDepth 4096 in
example : select 17 (fun _ => pack 19 12) 3 = .miss 3 := by decide

end NeuralReversi.EvalCache
