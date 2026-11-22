import NeuralReversi.Stability.StructuralBridge
import NeuralReversi.Stability.EdgeEncoding
import NeuralReversi.Stability.FullLines
import NeuralReversi.Stability.Propagation

namespace NeuralReversi.Stability.Semantics

def SoundMask (s : State) (c : Color) (mask : Word) : Prop :=
  ∀ sq : Fin 64, mask.getLsbD sq.val = true → Stable s c (boardSquare sq)

theorem source_contactStep_sound {s : State} {c : Color} {p o known : Word}
    (repr : Represents s c p o) (sound : SoundMask s c known) :
    SoundMask s c (contactStep (p &&& center) known (getFullLines (p ||| o))) := by
  intro sq marked
  cases old : known.getLsbD sq.val with
  | true => exact sound sq old
  | false =>
    obtain ⟨central, contacts⟩ := contactStep_new _ _ _ sq marked old
    simp only [BitVec.getLsbD_and, Bool.and_eq_true] at central
    have inside := center_interior sq central.2
    apply mask_contact_stable sq
    · change 1 ≤ sq.val / 8 ∧ sq.val / 8 ≤ 6 ∧ 1 ≤ sq.val % 8 ∧ sq.val % 8 ≤ 6
      omega
    · apply (repr.1 (boardSquare sq)).mp
      simpa only [Marks, boardIndex_square] using central.1
    · exact represents_occupied repr
    · exact sound
    · intro axis
      rcases contactAt_support _ _ axis sq (contacts axis) with full | neighbor
      · exact Or.inl ((fullLines_correct _ axis sq).mp full)
      · exact Or.inr neighbor

theorem source_contactClosure_sound {s : State} {c : Color} {p o seed : Word}
    (repr : Represents s c p o) (sound : SoundMask s c seed) :
    SoundMask s c (NeuralReversi.Stability.contactClosure (p &&& center) seed
      (getFullLines (p ||| o))) :=
  contactClosure_preserves _ _ _ (SoundMask s c)
    (fun _ h => source_contactStep_sound repr h) sound

theorem source_contact_sound {s : State} {c : Color} {p o seed : Word}
    (repr : Represents s c p o) (sound : SoundMask s c seed) :
    SoundMask s c (getStableByContact (p &&& center) seed (getFullLines (p ||| o))) := by
  unfold getStableByContact
  split
  · intro sq h
    simp at h
  · exact source_contactClosure_sound repr sound

theorem source_full_seed_sound {s : State} {c : Color} {p o : Word}
    (repr : Represents s c p o) :
    SoundMask s c ((getFullLines (p ||| o)).intersection &&& (p &&& center)) := by
  intro sq marked
  simp only [FullLines.intersection, BitVec.getLsbD_and, Bool.and_eq_true] at marked
  apply four_full_stable
  · apply (repr.1 (boardSquare sq)).mp
    simpa only [Marks, boardIndex_square] using marked.2.1
  · exact represents_occupied repr
  · intro axis
    rw [boardIndex_square]
    apply (fullLines_correct _ axis sq).mp
    rcases axis with ⟨a, ha⟩
    match a with
    | 0 => exact marked.1.1.1.1
    | 1 => exact marked.1.1.1.2
    | 2 => exact marked.1.1.2
    | 3 => exact marked.1.2
    | a + 4 => omega

def seededStable (p o edge : Word) : Word :=
  getStableByContact (p &&& center)
    (edge ||| ((getFullLines (p ||| o)).intersection &&& (p &&& center)))
    (getFullLines (p ||| o))

theorem seededStable_sound {s : State} {c : Color} {p o edge : Word}
    (repr : Represents s c p o) (edgeSound : SoundMask s c edge) :
    SoundMask s c (seededStable p o edge) := by
  apply source_contact_sound repr
  intro sq marked
  simp only [BitVec.getLsbD_or, Bool.or_eq_true] at marked
  rcases marked with edgeBit | fullBit
  · exact edgeSound sq edgeBit
  · exact source_full_seed_sound repr sq fullBit

theorem seededStable_never_flips {s t : State} {c : Color} {p o edge : Word}
    (repr : Represents s c p o) (edgeSound : SoundMask s c edge)
    (future : Reach s t) (sq : Fin 64)
    (marked : (seededStable p o edge).getLsbD sq.val = true) :
    t.board (boardSquare sq) = some c :=
  seededStable_sound repr edgeSound sq marked t future

end NeuralReversi.Stability.Semantics
