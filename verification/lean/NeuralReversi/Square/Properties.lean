import NeuralReversi.Square.Model

namespace NeuralReversi.Square

def DiscriminantRoundTrips (decode : Byte → Option Square) (encode : Square → Nat) : Prop :=
  (∀ value : Byte,
    (decode value).map encode = if value.toNat < 64 then some value.toNat else none) ∧
  (∀ sq : Square, decode (BitVec.ofNat 8 (encode sq)) = some sq)

def CheckedDiscriminantRoundTrips : Prop := DiscriminantRoundTrips fromU8 index

theorem checkedDiscriminantRoundTrips_proven : CheckedDiscriminantRoundTrips := by
  constructor
  · intro value
    simp [fromU8, index]
  · intro sq
    have bound : sq.val < 256 := by omega
    have accepted : sq.val < 64 := sq.isLt
    simp [fromU8, index, BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound, accepted]

def ExactDiscriminants (decode : Byte → Option Square) (encode : Square → Nat) : Prop :=
  (∀ sq : Square, encode sq = sq.val) ∧
  (∀ (value : Byte) (sq : Square), decode value = some sq ↔ sq.val = value.toNat) ∧
  (∀ value : Byte, decode value = none ↔ 64 ≤ value.toNat)

theorem exactDiscriminants_proven : ExactDiscriminants fromU8 index := by
  refine ⟨?_, ?_, ?_⟩
  · intro sq
    rfl
  · intro value sq
    by_cases accepted : value.toNat < 64
    · rw [fromU8, dite_eq_left accepted]
      constructor
      · intro equal
        exact (congrArg Fin.val (Option.some.inj equal)).symm
      · intro equal
        exact congrArg some (Fin.ext equal.symm)
    · rw [fromU8, dite_eq_right accepted]
      constructor
      · intro equal
        cases equal
      · intro equal
        have bound := sq.isLt
        omega
  · intro value
    by_cases accepted : value.toNat < 64
    · rw [fromU8, dite_eq_left accepted]
      constructor
      · intro equal
        cases equal
      · intro outside
        omega
    · rw [fromU8, dite_eq_right accepted]
      exact ⟨fun _ => by omega, fun _ => rfl⟩

def CheckedDiscriminants : Prop :=
  CheckedDiscriminantRoundTrips ∧ ExactDiscriminants fromU8 index

theorem checkedDiscriminants_proven : CheckedDiscriminants :=
  ⟨checkedDiscriminantRoundTrips_proven, exactDiscriminants_proven⟩

theorem exactDiscriminants_unique (decode : Byte → Option Square) (encode : Square → Nat)
    (contract : ExactDiscriminants decode encode) : decode = fromU8 ∧ encode = index := by
  constructor
  · funext value
    cases original : fromU8 value with
    | none =>
      exact (contract.2.2 value).mpr ((exactDiscriminants_proven.2.2 value).mp original)
    | some sq =>
      exact (contract.2.1 value sq).mpr ((exactDiscriminants_proven.2.1 value sq).mp original)
  · funext sq
    exact contract.1 sq

namespace DiscriminantRegression

def swap01 (sq : Square) : Square :=
  if sq.val = 0 then ⟨1, by decide⟩
  else if sq.val = 1 then ⟨0, by decide⟩ else sq

theorem swap01_involutive (sq : Square) : swap01 (swap01 sq) = sq := by
  by_cases zero : sq.val = 0
  · have equal : sq = ⟨0, by decide⟩ := Fin.ext zero
    subst sq
    rfl
  · by_cases one : sq.val = 1
    · have equal : sq = ⟨1, by decide⟩ := Fin.ext one
      subst sq
      rfl
    · simp [swap01, zero, one]

def swappedDecode (value : Byte) : Option Square := (fromU8 value).map swap01

def swappedEncode (sq : Square) : Nat := index (swap01 sq)

theorem swapped_roundTrips : DiscriminantRoundTrips swappedDecode swappedEncode := by
  constructor
  · intro value
    have old := checkedDiscriminantRoundTrips_proven.1 value
    cases original : fromU8 value with
    | none => simpa [swappedDecode, original] using old
    | some sq => simpa [swappedDecode, swappedEncode, original, swap01_involutive] using old
  · intro sq
    change (fromU8 (BitVec.ofNat 8 (index (swap01 sq)))).map swap01 = some sq
    rw [checkedDiscriminantRoundTrips_proven.2]
    change some (swap01 (swap01 sq)) = some sq
    rw [swap01_involutive]

theorem swapped_is_wrong :
    swappedDecode (0 : Byte) = some (⟨1, by decide⟩ : Square) ∧
    swappedEncode ⟨0, by decide⟩ = 1 := by
  constructor <;> rfl

theorem exact_rejects_swapped : ¬ ExactDiscriminants swappedDecode swappedEncode := by
  intro contract
  have impossible := contract.1 ⟨0, by decide⟩
  change (1 : Nat) = 0 at impossible
  cases impossible

theorem roundTrips_do_not_imply_exact :
    ¬ (∀ decode encode, DiscriminantRoundTrips decode encode → ExactDiscriminants decode encode) := by
  intro implication
  exact exact_rejects_swapped (implication swappedDecode swappedEncode swapped_roundTrips)

theorem boundaries :
    fromU8 (0 : Byte) = some (⟨0, by decide⟩ : Square) ∧
    fromU8 (1 : Byte) = some (⟨1, by decide⟩ : Square) ∧
    fromU8 (63 : Byte) = some (⟨63, by decide⟩ : Square) ∧
    fromU8 (64 : Byte) = none ∧ fromU8 (255 : Byte) = none ∧
    index (⟨63, by decide⟩ : Square) = 63 := by
  decide

end DiscriminantRegression

def CoordinateDecomposition : Prop :=
  ∀ sq : Square,
    file sq < 8 ∧ rank sq < 8 ∧ rank sq * 8 + file sq = index sq

theorem coordinateDecomposition_proven : CoordinateDecomposition := by
  intro sq
  have bound := sq.isLt
  simp only [file, rank, index]
  omega

def CoordinateConstruction : Prop :=
  ∀ f r : Byte,
    if f.toNat < 8 then
      if r.toNat < 8 then
        ∃ sq : Square, fromFileRank f r = .ok sq ∧
          file sq = f.toNat ∧ rank sq = r.toNat
      else fromFileRank f r = .error (.invalidRank r.toNat)
    else fromFileRank f r = .error (.invalidFile f.toNat)

theorem coordinateConstruction_proven : CoordinateConstruction := by
  intro f r
  by_cases hf : f.toNat < 8
  · by_cases hr : r.toNat < 8
    · simp only [hf, hr, ↓reduceIte]
      let sq : Square := ⟨r.toNat * 8 + f.toNat, by omega⟩
      refine ⟨sq, ?_, ?_, ?_⟩
      · simp [fromFileRank, hf, hr, sq]
      · simp only [file, index, sq]; omega
      · simp only [rank, index, sq]; omega
    · simp [hf, hr, fromFileRank]
  · simp [hf, fromFileRank]

def Enumeration : Prop :=
  iter.map index = List.range 64 ∧
  (∀ sq : Square, sq ∈ iter) ∧
  iter.Nodup

theorem enumeration_proven : Enumeration := by
  have indices : iter.map index = List.range 64 := by
    apply List.ext_getElem
    · simp [iter]
    · intro i hi hj
      simp [iter, index]
  refine ⟨indices, ?_, ?_⟩
  · intro sq
    exact List.mem_finRange sq
  · have mapped : (iter.map index).Nodup := by
      rw [indices]
      exact List.nodup_range
    exact List.Pairwise.of_map index (fun a b h equal => h (congrArg index equal)) mapped

end NeuralReversi.Square
