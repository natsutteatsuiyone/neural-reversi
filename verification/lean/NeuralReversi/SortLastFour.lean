import NeuralReversi.EmptyListInit

namespace NeuralReversi.SortLastFour

def oneHot (q : Fin 4) : BitVec 8 := 1 <<< q.val

def parityOf (a b c d : Fin 4) : BitVec 8 :=
  oneHot a ^^^ oneHot b ^^^ oneHot c ^^^ oneHot d

def order (parity id1 id2 id3 : BitVec 8) (a b c d : α) : List α :=
  if parity = 0 then [a, b, c, d]
  else if parity &&& id1 = 0 then
    if parity &&& id2 ≠ 0 then
      if parity &&& id3 ≠ 0 then [b, c, a, d] else [b, d, a, c]
    else if parity &&& id3 ≠ 0 then [c, d, a, b] else [a, b, c, d]
  else if parity &&& id2 = 0 then
    if parity &&& id3 ≠ 0 then [a, c, b, d] else [a, d, b, c]
  else [a, b, c, d]

def oddQuadrant (quad : α → Fin 4) (cells : List α) (x : α) : Bool :=
  cells.countP (fun y => quad y == quad x) % 2 == 1

private def four (a b c d : α) (i : Fin 4) : α :=
  match i.val with
  | 0 => a | 1 => b | 2 => c | _ => d

private theorem order_map (f : α → β) (p i j k : BitVec 8) (a b c d : α) :
    (order p i j k a b c d).map f = order p i j k (f a) (f b) (f c) (f d) := by
  unfold order
  repeat' first | split | rfl

private theorem all_shapes : ∀ a b c d : Fin 4,
    let cells : List (Fin 4) := [0, 1, 2, 3]
    let odd := oddQuadrant (four a b c d) cells
    order (parityOf a b c d) (oneHot a) (oneHot b) (oneHot c) 0 1 2 3 =
      cells.filter odd ++ cells.filter (fun i => !odd i) := by decide

theorem order_eq_partition (a b c d : α) (quad : α → Fin 4) :
    let cells := [a, b, c, d]
    let odd := oddQuadrant quad cells
    order (parityOf (quad a) (quad b) (quad c) (quad d))
      (oneHot (quad a)) (oneHot (quad b)) (oneHot (quad c)) a b c d =
      cells.filter odd ++ cells.filter (fun x => !odd x) := by
  let atIndex := four a b c d
  let indices : List (Fin 4) := [0, 1, 2, 3]
  have mapped : indices.map atIndex = [a, b, c, d] := rfl
  have quadrants : (fun i => quad (atIndex i)) = four (quad a) (quad b) (quad c) (quad d) := by
    funext i
    unfold atIndex four
    split <;> rfl
  have cert := congrArg (List.map atIndex) (all_shapes (quad a) (quad b) (quad c) (quad d))
  dsimp only at cert ⊢
  rw [order_map, List.map_append] at cert
  rw [← mapped, List.filter_map, List.filter_map]
  simp only [← quadrants] at cert
  simp only [oddQuadrant, List.countP_map, Function.comp_def]
  exact cert

open EmptyListLinks (Links Index Square head slot)
open EmptyListInit (chain edge)

private def quadrant (sq : Square) : Fin 4 :=
  ⟨2 * (sq.val / 8 / 4) + sq.val % 8 / 4, by have := sq.isLt; omega⟩

private theorem id_eq : ∀ sq : Square,
    EmptyListParity.quadrantId sq = oneHot (quadrant sq) := by decide

private def squareAfter (nodes : Links) (i : Index) : Option Square :=
  let j := (nodes i).next
  if h : j.val < 64 then some ⟨j.val, h⟩ else none

private def nextSquare (nodes : Links) (sq : Square) : Option Square :=
  if (nodes (nodes (slot sq)).prev).next = slot sq then squareAfter nodes (slot sq) else none

def sortLastFour (nodes : Links) (byte : BitVec 8) : Option (List Square) := do
  let a ← squareAfter nodes head
  let b ← nextSquare nodes a
  let c ← nextSquare nodes b
  let d ← nextSquare nodes c
  return order byte (EmptyListParity.quadrantId a) (EmptyListParity.quadrantId b)
    (EmptyListParity.quadrantId c) a b c d

private theorem after_edge {nodes : Links} {i : Index} {sq : Square} (e : edge nodes i (slot sq)) :
    squareAfter nodes i = some sq := by
  simp [squareAfter, e.1, slot]

private theorem next_edges {nodes : Links} {i : Index} {a b : Square}
    (before : edge nodes i (slot a)) (after : edge nodes (slot a) (slot b)) :
    nextSquare nodes a = some b := by
  simp [nextSquare, before.2, before.1, after_edge after]

def OrderContract : Prop :=
  ∀ (nodes : Links) (byte : BitVec 8) (cells : List Square) (a b c d : Square),
    chain nodes head ([a, b, c, d].map slot) →
    EmptyListParity.Reachable cells byte → cells.Perm [a, b, c, d] →
    let xs := [a, b, c, d]
    let odd := oddQuadrant quadrant xs
    sortLastFour nodes byte = some (xs.filter odd ++ xs.filter (fun sq => !odd sq))

theorem orderContract_proven : OrderContract := by
  intro nodes byte cells a b c d links state same
  obtain ⟨ea, eb, ec, ed, _⟩ := links
  have parity := EmptyListParity.parity_of_perm state same
  rw [parity]
  simp only [sortLastFour, after_edge ea, next_edges ea eb, next_edges eb ec, next_edges ec ed,
    bind, Option.bind, pure, List.foldl_cons, List.foldl_nil, EmptyListParity.toggle,
    BitVec.zero_xor, id_eq]
  exact congrArg some (order_eq_partition a b c d quadrant)

theorem sort_initialized (empties : Bitboard.Bits) (a b c d : Square)
    (fourLeft : EmptyListParity.presorted.filter (fun sq => empties.getLsbD sq.val) = [a, b, c, d]) :
    let xs := [a, b, c, d]
    let odd := oddQuadrant quadrant xs
    sortLastFour (EmptyListInit.newLinks empties) (EmptyListParity.fromEmpties empties) =
      some (xs.filter odd ++ xs.filter (fun sq => !odd sq)) := by
  apply orderContract_proven _ _ _ a b c d
  · simpa only [EmptyListInit.scanOrder, fourLeft] using (EmptyListInit.initContract_proven empties).1
  · exact EmptyListParity.Reachable.initial empties
  · rw [← fourLeft]
    exact EmptyListParity.scan_perm.symm.filter _

end NeuralReversi.SortLastFour
