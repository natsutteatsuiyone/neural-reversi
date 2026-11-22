import Std

namespace NeuralReversi.EmptyListLinks

abbrev Index := Fin 65
abbrev Square := Fin 64
def head : Index := 64
def slot (sq : Square) : Index := sq.castSucc

@[ext] structure Node where
  next : Index
  prev : Index
  deriving DecidableEq

abbrev Links := Index → Node

def writeNext (links : Links) (atNode value : Index) : Links :=
  fun i => { links i with next := if i = atNode then value else (links i).next }

def writePrev (links : Links) (atNode value : Index) : Links :=
  fun i => { links i with prev := if i = atNode then value else (links i).prev }

def remove (links : Links) (sq : Square) : Links :=
  let node := links (slot sq)
  writePrev (writeNext links node.prev node.next) node.next node.prev

def restore (links : Links) (sq : Square) : Links :=
  let node := links (slot sq)
  writePrev (writeNext links node.prev (slot sq)) node.next (slot sq)

def contains (links : Links) (sq : Square) : Prop :=
  (links (links (slot sq)).prev).next = slot sq

structure Linked (links : Links) (sq : Square) : Prop where
  prevOther : (links (slot sq)).prev ≠ slot sq
  nextOther : (links (slot sq)).next ≠ slot sq
  fromPrev : contains links sq
  fromNext : (links (links (slot sq)).next).prev = slot sq

theorem remove_next (links : Links) (sq : Square) (i : Index) :
    ((remove links sq) i).next =
      if i = (links (slot sq)).prev then (links (slot sq)).next else (links i).next := rfl

theorem remove_prev (links : Links) (sq : Square) (i : Index) :
    ((remove links sq) i).prev =
      if i = (links (slot sq)).next then (links (slot sq)).prev else (links i).prev := rfl

private theorem saved_node (links : Links) (sq : Square) (h : Linked links sq) :
    remove links sq (slot sq) = links (slot sq) := by
  simp [remove, writeNext, writePrev, Ne.symm h.prevOther, Ne.symm h.nextOther]

private theorem removed_absent (links : Links) (sq : Square) (h : Linked links sq) :
    ¬ contains (remove links sq) sq := by
  unfold contains
  rw [saved_node links sq h, remove_next]
  simpa using h.nextOther

private theorem restored (links : Links) (sq : Square) (h : Linked links sq) :
    restore (remove links sq) sq = links := by
  funext i
  have p := h.fromPrev
  unfold contains at p
  have n := h.fromNext
  unfold restore
  rw [saved_node links sq h]
  simp only [writeNext, writePrev]
  by_cases isP : i = (links (slot sq)).prev <;> by_cases isN : i = (links (slot sq)).next
  all_goals
    apply Node.ext <;> simp_all [remove_prev, remove_next]

def ValidRemovals : Links → List Square → Prop
  | _, [] => True
  | links, sq :: rest => Linked links sq ∧ ValidRemovals (remove links sq) rest

def LinksContract : Prop :=
  (∀ (links : Links) (sq : Square), Linked links sq →
    (∀ i, ((remove links sq) i).next =
      if i = (links (slot sq)).prev then (links (slot sq)).next else (links i).next) ∧
    (∀ i, ((remove links sq) i).prev =
      if i = (links (slot sq)).next then (links (slot sq)).prev else (links i).prev) ∧
    ¬ contains (remove links sq) sq ∧ restore (remove links sq) sq = links) ∧
  (∀ (links : Links) (moves : List Square), ValidRemovals links moves →
    moves.reverse.foldl restore (moves.foldl remove links) = links)

theorem linksContract_proven : LinksContract := by
  constructor
  · intro links sq h
    exact ⟨remove_next links sq, remove_prev links sq, removed_absent links sq h, restored links sq h⟩
  · intro links moves
    induction moves generalizing links with
    | nil => intro _; rfl
    | cons sq rest ih =>
      intro h
      obtain ⟨linked, valid⟩ := h
      simp only [List.reverse_cons, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [ih (remove links sq) valid, restored links sq linked]

example (sq : Square) :
    let singleton : Links := fun i => if i = head then ⟨slot sq, slot sq⟩ else ⟨head, head⟩
    Linked singleton sq ∧ remove singleton sq head = ⟨head, head⟩ := by
  have distinct : slot sq ≠ head := fun eq => by
    have := congrArg Fin.val eq
    change sq.val = 64 at this
    omega
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩ <;> simp [contains, remove, writePrev, writeNext, distinct, Ne.symm distinct]

end NeuralReversi.EmptyListLinks
