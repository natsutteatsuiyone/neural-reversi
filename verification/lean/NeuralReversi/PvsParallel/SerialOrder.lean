import NeuralReversi.PvsParallel.SerialRecursion

namespace NeuralReversi.PvsParallel.SerialOrder

open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open EndgameOrderMir (actualKey priority ordinary framePriority)
open MoveList
open PvsBoardTree (Position GeneratedSquare legalSquares)

private theorem checkedKey (mc : X86Config) (b : Position) (key : Bitboard.Square → Int)
    (hint : Option Bitboard.Square) (m : GeneratedSquare mc b.val) :
    actualKey (key ∘ Subtype.val) (GeneralNonPvMir.checkedHint mc b hint) m = actualKey key hint m.val := by
  cases hint with
  | none => simp [actualKey, GeneralNonPvMir.checkedHint]
  | some sq =>
    by_cases eq : sq = m.val
    · subst sq
      simp [actualKey, GeneralNonPvMir.checkedHint, m.property]
    · have ne : m.val ≠ sq := Ne.symm eq
      cases legal : (Board.getMoves mc b.val).getLsbD sq.val <;>
        simp [actualKey, GeneralNonPvMir.checkedHint, legal, ne, Subtype.ext_iff]

private theorem recover (f : α → β) (g : β → Option α) (left : ∀ x, g (f x) = some x)
    (ys : List β) (generated : ∀ y ∈ ys, ∃ x, f x = y) : (ys.filterMap g).map f = ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    obtain ⟨x, rfl⟩ := generated y List.mem_cons_self
    simp [left, ih (fun v h => generated v (.tail _ h))]

/-- Decode the actual initialized records returned by the unstable library sort.
    One tie order reproduces the square and flip mask of every iteration in
    either serial node type, including when different legal moves have equal keys. -/
theorem records (mc : X86Config) (fc : FlipConfig) (b : Position) (op : SerialRecursion.Operands mc fc b)
    (output : List Move.Move) :
    let xs := legalSquares mc b.val
    let hint := GeneralNonPvMir.probeHint b op.probe
    let hk := GeneralNonPvMir.checkedHint mc b hint
    let key := fun sq => BitVec.ofInt 32 (actualKey (priority mc fc b op.ordering) hint sq)
    let bytes := fun m : GeneratedSquare mc b.val => RootFrameMir.setKey key (Generation.entry fc b.val m.val)
    3 < xs.length →
    output.Perm ((Generation.records fc b.val (Board.getMoves mc b.val)).map (RootFrameMir.setKey key)) →
    MoveList.Ordering.SortedDesc (fun m : Move.Move => m.value.toInt) output →
    ∃ tie, (SerialRecursion.pvOrder b {op with tie}).map bytes = output ∧
      (GeneralNonPvMir.order (framePriority mc fc b op.ordering tie) hk xs).map bytes = output := by
  intro xs hint hk key bytes long perm sorted
  have layout : (Generation.records fc b.val (Board.getMoves mc b.val)).map (RootFrameMir.setKey key) =
      xs.map bytes := by
    dsimp [xs, bytes, Generation.records, Generation.squares, Function.comp_def]
    rw [← PvsBoardTree.legalSquares_values]
    simp [List.map_map, Function.comp_def]
  rw [layout] at perm
  let decode := RootFrameMir.asChoice mc b
  have left (m : GeneratedSquare mc b.val) : decode (bytes m) = some m := by
    simp [decode, bytes, RootFrameMir.asChoice, RootFrameMir.setKey, Generation.entry, m.property]
  let ys := output.filterMap decode
  have recovered : ys.map bytes = output :=
    recover bytes decode left output fun r mem => by
      obtain ⟨m, _, eq⟩ := List.mem_map.mp (perm.mem_iff.mp mem)
      exact ⟨m,eq⟩
  have order : ys.Perm xs := by
    simpa only [List.filterMap_map, Function.comp_def, left, List.filterMap_some] using perm.filterMap decode
  rw [← recovered] at sorted
  have desc : MoveList.Ordering.SortedDesc (actualKey (fun m : GeneratedSquare mc b.val =>
      priority mc fc b op.ordering m.val) hk) ys := by
    have read := List.pairwise_map.mp sorted
    apply read.imp
    intro u v cmp
    change (key v.val).toInt ≤ (key u.val).toInt at cmp
    change actualKey ((priority mc fc b op.ordering) ∘ Subtype.val) (GeneralNonPvMir.checkedHint mc b hint) v ≤
      actualKey ((priority mc fc b op.ordering) ∘ Subtype.val) (GeneralNonPvMir.checkedHint mc b hint) u
    rw [checkedKey mc b _ hint v, checkedKey mc b _ hint u]
    simpa only [key, EndgameOrderMir.key_bits_eq] using cmp
  obtain ⟨tie, sameOrder⟩ := EndgameOrderMir.compiled_library_order mc fc b op.ordering hk ys long order desc
  refine ⟨tie, ?_, sameOrder ▸ recovered⟩
  let k := fun m : GeneratedSquare mc b.val => priority mc fc b op.ordering m.val
  have source := EndgameOrderMir.concrete_order k tie hk xs
  have multiple : 1 < xs.length - (if xs.length == 1 || hk.isSome then 1 else 0) := by split <;> omega
  have notSmall : 1 < xs.length := by omega
  have pv : SerialRecursion.pvOrder b {op with tie} = ordinary k tie false hk xs := rfl
  rw [pv]
  have same : ordinary k tie false hk xs = ordinary k tie (xs.length == 1 || hk.isSome) hk xs := by
    unfold ordinary
    rw [ite_eq_left multiple]
    exact ite_eq_left (by simpa using notSmall)
  rw [same, source]
  exact sameOrder ▸ recovered

end NeuralReversi.PvsParallel.SerialOrder
