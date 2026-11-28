module
public import NeuralReversi.EndgameCache
public import NeuralReversi.Board.MovesProperties
public import NeuralReversi.Bitboard.IteratorProperties
public import NeuralReversi.ShallowMoveOrder
public import NeuralReversi.Solve34.Control

@[expose] public section

namespace NeuralReversi.ShallowCutoff

def scan (alpha : Int) (report : Fin 64 → Int) (next : Int → Int) :
    List (Fin 64) → Int → Int
  | [], best => next best
  | sq :: rest, best =>
    let score := report sq
    if alpha + 1 ≤ score then score else scan alpha report next rest (max best score)

def runGroups (alpha : Int) (report : Fin 64 → Int) (groups : List (List (Fin 64))) : Int :=
  groups.foldr (fun group next => scan alpha report next group) id (-127)

private theorem scan_append (a r k xs ys best) :
    scan a r k (xs ++ ys) best = scan a r (scan a r k ys) xs best := by
  induction xs generalizing best with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, scan]; split <;> simp_all

private theorem groups_eq_scan (a r) (gs : List (List (Fin 64))) :
    gs.foldr (fun group next => scan a r next group) id = scan a r id gs.flatten := by
  induction gs with
  | nil => rfl
  | cons g gs ih => funext best; simp only [List.foldr_cons, List.flatten_cons, ih, scan_append]

private theorem scan_result (alpha : Int) (report : Fin 64 → Int) (xs : List (Fin 64)) (acc : Int) :
    let result := scan alpha report id xs acc
    (∃ sq ∈ xs, result = report sq ∧ alpha < result) ∨
    ((∀ sq ∈ xs, report sq ≤ alpha ∧ report sq ≤ result) ∧ acc ≤ result ∧
      (result = acc ∨ ∃ sq ∈ xs, result = report sq)) := by
  induction xs generalizing acc with
  | nil => simp [scan]
  | cons x xs ih =>
    dsimp only
    rw [scan]
    split
    next high => exact Or.inl ⟨x, by simp, rfl, by omega⟩
    next low =>
      rcases ih (max acc (report x)) with ⟨sq, mem, eq, high⟩ | ⟨all, le, source⟩
      · exact Or.inl ⟨sq, List.mem_cons_of_mem _ mem, eq, high⟩
      · right
        refine ⟨?_, by omega, ?_⟩
        · intro s mem
          rcases List.mem_cons.mp mem with rfl | tail
          · exact ⟨by omega, by omega⟩
          · exact all s tail
        · rcases source with first | ⟨sq, mem, eq⟩
          · by_cases h : report x ≤ acc
            · exact Or.inl (by omega)
            · exact Or.inr ⟨x, by simp, by omega⟩
          · exact Or.inr ⟨sq, List.mem_cons_of_mem _ mem, eq⟩

theorem runGroups_sound {alpha : Int} {report value : Fin 64 → Int}
    {groups : List (List (Fin 64))} {best : Int}
    (attained : ∃ sq ∈ groups.flatten, value sq = best)
    (upper : ∀ sq ∈ groups.flatten, value sq ≤ best)
    (sound : ∀ sq ∈ groups.flatten, EndgameCache.InRange (report sq) ∧
      Solve2.NWS alpha (value sq) (report sq)) :
    EndgameCache.InRange (runGroups alpha report groups) ∧
      Solve2.NWS alpha best (runGroups alpha report groups) := by
  simp only [runGroups, groups_eq_scan]
  rcases scan_result alpha report groups.flatten (-127) with ⟨sq, mem, eq, high⟩ |
    ⟨all, _, init | hit⟩
  · rw [eq] at high ⊢
    obtain ⟨range, _, lower⟩ := sound sq mem
    have := upper sq mem
    exact ⟨range, fun _ => by omega, fun h => Int.le_trans (lower h) this⟩
  · obtain ⟨sq, mem, _⟩ := attained
    have := (all sq mem).2
    have := (sound sq mem).1.1
    omega
  · obtain ⟨sq, mem, eq⟩ := hit
    obtain ⟨winner, won, bestEq⟩ := attained
    have hi := (sound winner won).2.1 (all winner won).1
    obtain ⟨_, le⟩ := all winner won
    have := (all sq mem).1
    exact ⟨eq ▸ (sound sq mem).1, fun _ => by omega, fun _ => by omega⟩

def squares (m : LegalMoves.Raw) : List (Fin 64) :=
  (Bitboard.setIndices m.bits).attach.map fun ⟨i, member⟩ =>
    ⟨i, List.mem_range.mp (List.mem_filter.mp member).1⟩

private theorem mem_squares (m : LegalMoves.Raw) (sq : Fin 64) :
    sq ∈ squares m ↔ m.bits.getLsbD sq.val = true := by
  have indices := (Bitboard.iteratorEnumerationContract_proven m.bits false).2.2.1
  constructor
  · intro h
    obtain ⟨⟨_, member⟩, _, rfl⟩ := List.mem_map.mp h
    exact ((indices _).mp member).2
  · intro bit
    exact List.mem_map.mpr ⟨⟨sq.val, (indices _).mpr ⟨sq.isLt, bit⟩⟩, by simp, rfl⟩

open Bitboard.Moves (X86Config LegalMove)

def moveGroups (config : X86Config) (b : Board.Board) (parity : Fin 16) :=
  (ShallowMoveOrder.groups ⟨Board.getMoves config b⟩ (Board.emptySquares b) parity).map squares

private theorem legal_in_groups (config : X86Config) (b : Board.Board) (parity : Fin 16) (sq : Fin 64) :
    sq ∈ (moveGroups config b parity).flatten ↔ LegalMove b.player b.opponent sq.val := by
  simp only [moveGroups, ← List.flatMap_def, List.mem_flatMap, mem_squares]
  exact (ShallowMoveOrder.mem_groups_iff _ _ _ _).trans (Board.getMoves_exact config b sq.val)

def CutoffContract : Prop :=
  ∀ (config : X86Config) (b : Board.Board) (parity : Fin 16) (alpha : Int)
    (value report : Fin 64 → Int),
    (∃ sq : Fin 64, LegalMove b.player b.opponent sq.val) →
    (∀ sq : Fin 64, LegalMove b.player b.opponent sq.val →
      EndgameCache.InRange (report sq) ∧ Solve2.NWS (-(alpha + 1)) (value sq) (report sq)) →
    let result := runGroups alpha (fun sq => -(report sq)) (moveGroups config b parity)
    let values := (squares ⟨Board.getMoves config b⟩).map fun sq => -(value sq)
    ∃ best, values.max? = some best ∧ EndgameCache.InRange result ∧ Solve2.NWS alpha best result

theorem cutoffContract_proven : CutoffContract := by
  intro config b parity alpha value report nonempty sound
  dsimp only
  let moves : LegalMoves.Raw := ⟨Board.getMoves config b⟩
  let values := (squares moves).map fun sq => -(value sq)
  have legal (sq : Fin 64) : sq ∈ squares moves ↔ LegalMove b.player b.opponent sq.val :=
    (mem_squares moves sq).trans (Board.getMoves_exact config b sq.val)
  obtain ⟨sq, candidate⟩ := nonempty
  have member : -(value sq) ∈ values := List.mem_map.mpr ⟨sq, (legal sq).mpr candidate, rfl⟩
  have ne := List.ne_nil_of_mem member
  refine ⟨values.max ne, List.max?_eq_some_max ne, ?_⟩
  apply runGroups_sound (value := fun s => -(value s))
  · obtain ⟨s, hs, eq⟩ := List.mem_map.mp (List.max_mem ne : values.max ne ∈ values)
    exact ⟨s, (legal_in_groups config b parity s).mpr ((legal s).mp hs), eq⟩
  · intro s hs
    apply List.le_max_of_mem (l := values)
    exact List.mem_map.mpr ⟨s, (legal s).mpr ((legal_in_groups config b parity s).mp hs), rfl⟩
  · intro s hs
    obtain ⟨⟨lo, hi⟩, child⟩ := sound s ((legal_in_groups config b parity s).mp hs)
    exact ⟨⟨by omega, by omega⟩, Solve34.negate_nws child⟩

end NeuralReversi.ShallowCutoff
