import NeuralReversi.Stability.EdgeGameBridge
import NeuralReversi.Stability.SingleEmptySafety

namespace NeuralReversi.Stability.Semantics

private def byAxis (f : Fin 4 → Word) : FullLines := ⟨f 0, f 1, f 2, f 3⟩

private theorem field_byAxis (f : Fin 4 → Word) (axis : Fin 4) : field (byAxis f) axis = f axis := by
  rcases axis with ⟨i, hi⟩
  match i with
  | 0 | 1 | 2 | 3 => rfl
  | _ + 4 => omega

private def noWrapL1 (axis : Fin 4) : Word :=
  match axis.val with
  | 0 => 0xfefefefefefefefe
  | 1 => 0xffffffffffffffff
  | 2 => 0x7f7f7f7f7f7f7f7f
  | _ => 0xfefefefefefefefe

private def noWrapR1 (axis : Fin 4) : Word :=
  match axis.val with
  | 0 => 0x7f7f7f7f7f7f7f7f
  | 1 => 0xffffffffffffffff
  | 2 => 0xfefefefefefefefe
  | _ => 0x7f7f7f7f7f7f7f7f

private def contactLines (known : Word) (lines : FullLines) : FullLines :=
  byAxis fun axis =>
    ((known <<< stride axis) &&& noWrapL1 axis) |||
    ((known >>> stride axis) &&& noWrapR1 axis) |||
    field lines axis

private def oneEmptyShield (disc : Word) (axis : Fin 4) (x : Fin 64) (blockers : Word) : Bool :=
  let cells := (List.finRange 64).filter fun z => decide (sameLine axis x.val z.val)
  match cells.filter (fun z => !disc.getLsbD z.val) with
  | [hole] => cells.any fun y =>
      blockers.getLsbD y.val && decide (Between hole.val y.val x.val)
  | _ => false

private def keepers (discs : Word) (lines : FullLines) (axis : Fin 4) : Word :=
  encodeBits fun sq => discs.getLsbD sq.val && (List.finRange 4).all fun b =>
    decide (b = axis) || (field lines b).getLsbD sq.val

private def shieldLines (disc : Word) (base : FullLines) (blockers : Word)
    (blockerLines : FullLines) : FullLines := byAxis fun axis =>
  let keep := keepers blockers blockerLines axis
  field base axis ||| encodeBits (fun sq => oneEmptyShield disc axis sq keep)

def singleEmptyStable (player opponent stable : Word) : Word :=
  let disc := player ||| opponent
  let full := getFullLines disc
  let playerContact := contactLines stable full
  let opponentEdge := getStableEdge Edge.initEdgeStability opponent player
  let opponentSafe := shieldLines disc (contactLines opponentEdge full) player playerContact
  let playerSafe := shieldLines disc playerContact opponent opponentSafe
  getStableByContact (player &&& center) stable playerSafe

private def SoundLines (s : State) (c : Color) (lines : FullLines) : Prop :=
  ∀ axis (x : Fin 64), (field lines axis).getLsbD x.val = true → SafeAxis s c axis (boardSquare x)

private theorem contact_sound {s : State} {c : Color} {known : Word} {lines : FullLines}
    (stable : SoundMask s c known) (safe : SoundLines s c lines)
    {axis : Fin 4} {x : Fin 64} (contact : ContactAt known lines axis x) :
    SafeAxis s c axis (boardSquare x) := by
  rcases contactAt_support _ _ _ _ contact with base | ⟨j, offset, bit⟩
  · exact safe axis x base
  · apply contact_safe_axis (y := boardSquare j) _ (stable j bit)
    have eq : ∀ b, stride b = axisStride b := by decide
    simpa only [boardIndex_square, ← eq] using offset

private theorem base_sound {s : State} {c col : Color} {p o known : Word}
    (repr : Represents s c p o) (stable : SoundMask s col known) :
    SoundLines s col (contactLines known (getFullLines (p ||| o))) := by
  intro axis x marked
  have contact : ContactAt known (getFullLines (p ||| o)) axis x := by
    simp only [ContactAt, BitVec.getLsbD_or, Bool.or_eq_true]
    simp only [contactLines, field_byAxis, BitVec.getLsbD_or, BitVec.getLsbD_and,
      Bool.or_eq_true, Bool.and_eq_true] at marked
    rcases marked with (h | h) | h
    · exact Or.inl (Or.inr h.1)
    · exact Or.inl (Or.inl h.1)
    · exact Or.inr h
  apply contact_sound stable _ contact
  intro b y bit
  apply full_safe_axis
  intro a line
  rw [boardIndex_square] at line
  exact represents_occupied repr a ((fullLines_correct _ b y).mp bit (boardIndex a) line)

private theorem shieldLines_sound {s : State} {c : Color} {p o : Word} {base other : FullLines}
    (repr : Represents s c p o) (safe : SoundLines s c base)
    (oppSafe : SoundLines s (Opposite c) other) :
    SoundLines s c (shieldLines (p ||| o) base o other) := by
  intro axis x marked
  simp only [shieldLines, field_byAxis, BitVec.getLsbD_or, Bool.or_eq_true, encodeBits_bit] at marked
  rcases marked with old | shield
  · exact safe _ _ old
  · unfold oneEmptyShield at shield
    dsimp only at shield
    split at shield
    next hole holes =>
      obtain ⟨y, member, blocks⟩ := List.any_eq_true.mp shield
      simp only [List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq] at member
      simp only [keepers, encodeBits_bit, Bool.and_eq_true, List.all_eq_true, List.mem_finRange,
        forall_const, Bool.or_eq_true, decide_eq_true_eq] at blocks
      apply shield_safe_axis (hole := boardSquare hole) (shield := boardSquare y)
      · intro z line empty
        rw [boardIndex_square] at line
        have free : (p ||| o).getLsbD (boardIndex z).val = false := by
          cases bit : (p ||| o).getLsbD (boardIndex z).val with
          | false => rfl
          | true =>
            obtain ⟨_, occ⟩ := represents_occupied repr z bit
            simp [empty] at occ
        have mem : boardIndex z ∈ [hole] := by
          rw [← holes]
          simp only [List.mem_filter, List.mem_finRange, line, decide_true, free, Bool.not_false,
            and_self]
        have eq : boardIndex z = hole := List.mem_singleton.mp mem
        simpa only [boardSquare_index] using congrArg boardSquare eq
      · simpa only [boardIndex_square] using member
      · simpa only [boardIndex_square] using blocks.2
      · apply (repr.2 _).mp
        simpa only [Marks, boardIndex_square] using blocks.1.1
      · intro b ne
        exact oppSafe b y ((blocks.1.2 b).resolve_left ne)
    next => contradiction

theorem singleEmptyStable_sound {s : State} {c : Color} {p o known : Word}
    (repr : Represents s c p o) (stable : SoundMask s c known) :
    SoundMask s c (singleEmptyStable p o known) := by
  have own := base_sound repr stable
  have edge := table_edge_sound _ initialized_table_agrees (represents_swap repr)
  have opp := shieldLines_sound (represents_swap repr) (base_sound repr edge)
    (by simpa [Opposite] using own)
  rw [BitVec.or_comm o p] at opp
  have safe := shieldLines_sound repr own opp
  apply getStableByContact_sound stable
  intro prev prevSound x bit
  cases old : prev.getLsbD x.val with
  | true => exact prevSound x old
  | false =>
    obtain ⟨central, contacts⟩ := contactStep_new _ _ _ x bit old
    apply stable_of_safe_axes
    · apply (repr.1 _).mp
      simp only [BitVec.getLsbD_and, Bool.and_eq_true] at central
      simpa only [Marks, boardIndex_square] using central.1
    · exact fun axis => contact_sound prevSound safe (contacts axis)

end NeuralReversi.Stability.Semantics
