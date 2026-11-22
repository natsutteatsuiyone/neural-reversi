import NeuralReversi.EmptyListLinks
import NeuralReversi.EmptyListParity

namespace NeuralReversi.EmptyListInit
open EmptyListLinks

private def emptyNodes : Links := fun _ => ⟨head, head⟩

private def build (nodes : Links) (last : Index) : List Index → Links
  | [] => writePrev nodes head last
  | i :: rest => build (writePrev (writeNext nodes last i) i last) i rest

def scanOrder (empties : Bitboard.Bits) : List Index :=
  (EmptyListParity.presorted.filter (fun sq => empties.getLsbD sq.val)).map slot

def newLinks (empties : Bitboard.Bits) : Links := build emptyNodes head (scanOrder empties)

def edge (nodes : Links) (a b : Index) : Prop :=
  (nodes a).next = b ∧ (nodes b).prev = a

def chain (nodes : Links) (last : Index) : List Index → Prop
  | [] => edge nodes last head
  | i :: rest => edge nodes last i ∧ chain nodes i rest

private theorem outside_cons {i a : Index} {tail : List Index} (h : i ∉ a :: tail) :
    i ≠ a ∧ i ∉ tail := by simpa using h

private theorem next_untouched (nodes : Links) (last i : Index) (rest : List Index)
    (other : i ≠ last) (absent : i ∉ rest) :
    (build nodes last rest i).next = (nodes i).next := by
  induction rest generalizing last nodes with
  | nil => rfl
  | cons a tail ih =>
    obtain ⟨ne, outside⟩ := outside_cons absent
    rw [build, ih _ _ ne outside]
    simp [writeNext, writePrev, other]

private theorem prev_untouched (nodes : Links) (last i : Index) (rest : List Index)
    (other : i ≠ head) (absent : i ∉ rest) :
    (build nodes last rest i).prev = (nodes i).prev := by
  induction rest generalizing last nodes with
  | nil => simp [build, writePrev, other]
  | cons a tail ih =>
    obtain ⟨ne, outside⟩ := outside_cons absent
    rw [build, ih _ _ outside]
    simp [writePrev, writeNext, ne]

private theorem build_chain (nodes : Links) (last : Index) (rest : List Index)
    (unique : (last :: rest).Nodup) (noHead : head ∉ rest)
    (fresh : ∀ i ∈ last :: rest, (nodes i).next = head) :
    chain (build nodes last rest) last rest := by
  induction rest generalizing nodes last with
  | nil =>
    exact ⟨fresh last (by simp), by simp [build, writePrev]⟩
  | cons a tail ih =>
    obtain ⟨lastOutside, uniqueRest⟩ := List.nodup_cons.mp unique
    obtain ⟨different, lastNotTail⟩ := outside_cons lastOutside
    obtain ⟨headNe, headOutside⟩ := outside_cons noHead
    rw [build, chain]
    constructor
    · unfold edge
      rw [next_untouched _ _ _ _ different lastNotTail,
        prev_untouched _ _ _ _ headNe.symm (List.nodup_cons.mp uniqueRest).1]
      simp [writeNext, writePrev]
    · apply ih _ _ uniqueRest headOutside
      intro i mem
      have ne : i ≠ last := fun eq => lastOutside (eq ▸ mem)
      simp [writeNext, writePrev, ne, fresh i (List.mem_cons_of_mem _ mem)]

private theorem first_edge {nodes : Links} {last : Index} {rest : List Index}
    (path : chain nodes last rest) : edge nodes last (rest.headD head) := by
  cases rest with
  | nil => exact path
  | cons _ _ => exact path.1

theorem linked_of_mem {nodes : Links} {last : Index} {rest : List Index}
    {sq : Square} (path : chain nodes last rest) (unique : (last :: rest).Nodup)
    (notHead : slot sq ≠ head) (mem : slot sq ∈ rest) : Linked nodes sq := by
  induction rest generalizing last with
  | nil => cases mem
  | cons a tail ih =>
    obtain ⟨before, after⟩ := path
    obtain ⟨outside, uniqueTail⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp mem with eq | inTail
    · subst a
      obtain ⟨prevNext, selfPrev⟩ := before
      obtain ⟨selfNext, nextPrev⟩ := first_edge after
      have nextNe : tail.headD head ≠ slot sq := by
        cases tail with
        | nil => exact notHead.symm
        | cons b bs =>
          exact fun eq => (List.nodup_cons.mp uniqueTail).1 (by simp [← eq])
      exact ⟨selfPrev ▸ (outside_cons outside).1, selfNext ▸ nextNe,
        by unfold contains; rwa [selfPrev], by rwa [selfNext]⟩
    · exact ih after uniqueTail inTail

theorem slot_ne_head (sq : Square) : slot sq ≠ head := by
  intro eq
  have same := congrArg Fin.val eq
  change sq.val = 64 at same
  omega

theorem order_mem (empties : Bitboard.Bits) (sq : Square) :
    slot sq ∈ scanOrder empties ↔ empties.getLsbD sq.val = true := by
  have inOrder : sq ∈ EmptyListParity.presorted := EmptyListParity.scan_perm.mem_iff.mpr (by simp)
  constructor
  · intro h
    obtain ⟨a, ha, eq⟩ := List.mem_map.mp h
    have same : a = sq := Fin.ext (congrArg (fun i : Index => i.val) eq)
    exact same ▸ (List.mem_filter.mp ha).2
  · intro bit
    exact List.mem_map.mpr ⟨sq, List.mem_filter.mpr ⟨inOrder, bit⟩, rfl⟩

def InitContract : Prop :=
  ∀ empties,
    let nodes := newLinks empties
    chain nodes head (scanOrder empties) ∧
    ∀ sq, (contains nodes sq ↔ empties.getLsbD sq.val = true) ∧
      (empties.getLsbD sq.val = true → Linked nodes sq)

theorem scanOrder_unique (empties : Bitboard.Bits) : (head :: scanOrder empties).Nodup := by
  have noHead : head ∉ scanOrder empties := by
    rintro member
    obtain ⟨sq, _, eq⟩ := List.mem_map.mp member
    exact slot_ne_head sq eq
  have unique : (scanOrder empties).Nodup := by
    apply List.Pairwise.map (R := (· ≠ ·)) slot
      (fun _ _ ne eq => ne (Fin.ext (congrArg (fun i : Index => i.val) eq)))
    exact (EmptyListParity.scan_perm.nodup_iff.mpr (List.nodup_finRange 64)).filter _
  exact List.nodup_cons.mpr ⟨noHead, unique⟩

theorem initContract_proven : InitContract := by
  intro empties
  have unique := scanOrder_unique empties
  have noHead := (List.nodup_cons.mp unique).1
  have path := build_chain emptyNodes head (scanOrder empties)
    unique noHead (fun _ _ => rfl)
  refine ⟨path, ?_⟩
  intro sq
  have available (bit : empties.getLsbD sq.val = true) : Linked (newLinks empties) sq :=
    linked_of_mem path unique (slot_ne_head sq)
      ((order_mem empties sq).mpr bit)
  refine ⟨⟨?_, fun bit => (available bit).fromPrev⟩, available⟩
  intro present
  apply (order_mem empties sq).mp
  apply Classical.byContradiction
  intro absent
  have defaultPrev := prev_untouched emptyNodes head (slot sq) (scanOrder empties)
    (slot_ne_head sq) absent
  change (newLinks empties (slot sq)).prev = head at defaultPrev
  change (newLinks empties ((newLinks empties (slot sq)).prev)).next = slot sq at present
  rw [defaultPrev] at present
  have first := (first_edge path).1
  change (newLinks empties head).next = (scanOrder empties).headD head at first
  rw [first] at present
  cases h : scanOrder empties with
  | nil => exact slot_ne_head sq (by simpa [h] using present.symm)
  | cons a tail =>
    have eq : a = slot sq := by simpa [h] using present
    exact absent (by simp [h, eq])

end NeuralReversi.EmptyListInit
