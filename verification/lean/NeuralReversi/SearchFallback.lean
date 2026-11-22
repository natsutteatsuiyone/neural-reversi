import NeuralReversi.PvsBoardTree
import NeuralReversi.SearchPublication

namespace NeuralReversi.SearchFallback

open Bitboard (Square)
open Bitboard.Moves (LegalMove X86Config)
open PvsBoardTree (Position legalSquares play)
open SearchPublication (Row Snapshot Stamp)

structure CachedMove where
  square : Square
  score : Int
  stamp : Stamp
  endgame : Bool

structure Report where
  best : Row
  alternatives : List Row
  stamp : Stamp
  endgame : Bool
  nodes : Nat

def onePly (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (evaluate : Board.Board → Int) (sq : Square) : Int :=
  if legal : (Board.getMoves mc board.val).getLsbD sq.val = true then
    -evaluate (play mc fc board ⟨sq, legal⟩).val
  else 0

private def legalMoves (mc : X86Config) (board : Position) : List Square :=
  (legalSquares mc board.val).map Subtype.val

private theorem legal_moves (mc : X86Config) (board : Position) (sq : Square) :
    sq ∈ legalMoves mc board ↔ LegalMove board.val.player board.val.opponent sq.val := by
  rw [legalMoves, List.mem_map]
  exact PvsBoardTree.legalSquares_complete mc board.val sq

def bestFrom (score : Square → Int) (first : Square) (remaining : List Square) : Square :=
  remaining.foldl (fun best next => if score best < score next then next else best) first

private theorem bestFrom_maximal (score : Square → Int) (first : Square) (remaining : List Square) :
    let chosen := bestFrom score first remaining
    chosen ∈ first :: remaining ∧ ∀ sq ∈ first :: remaining, score sq ≤ score chosen := by
  induction remaining generalizing first with
  | nil =>
    refine ⟨by simp [bestFrom], ?_⟩
    intro sq found
    have same := List.mem_singleton.mp found
    subst sq
    exact Int.le_refl _
  | cons next tail ih =>
    simp only [bestFrom, List.foldl_cons]
    by_cases improves : score first < score next
    · rw [ite_eq_left improves]
      obtain ⟨chosen, maximal⟩ := ih next
      refine ⟨List.mem_cons_of_mem _ chosen, fun sq member => ?_⟩
      rcases List.mem_cons.mp member with rfl | later
      · exact Int.le_trans (Int.le_of_lt improves) (maximal next (by simp))
      · exact maximal sq later
    · rw [ite_eq_right improves]
      obtain ⟨chosen, maximal⟩ := ih first
      refine ⟨?_, fun sq member => ?_⟩
      · rcases List.mem_cons.mp chosen with same | later
        · exact List.mem_cons.mpr (.inl same)
        · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ later)
      · rcases List.mem_cons.mp member with same | later
        · exact maximal sq (List.mem_cons.mpr (.inl same))
        · rcases List.mem_cons.mp later with rfl | tailMem
          · exact Int.le_trans (Int.le_of_not_gt improves) (maximal first (by simp))
          · exact maximal sq (List.mem_cons_of_mem _ tailMem)

def fromSnapshot (snapshot : Snapshot) (nodes : Nat) : Option Report :=
  snapshot.rows.head?.map fun best => ⟨best, snapshot.rows, snapshot.stamp, snapshot.endgame, nodes⟩

def fromCache (cached : CachedMove) (nodes : Nat) : Report :=
  ⟨⟨cached.square, cached.score, [cached.square]⟩, [], cached.stamp, cached.endgame, nodes⟩

def quickMove (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (evaluate : Board.Board → Int) (nodes : Nat) : Option Report :=
  match legalMoves mc board with
  | [] => none
  | first :: others =>
    let score := onePly mc fc board evaluate
    let sq := bestFrom score first others
    some ⟨⟨sq, score sq, [sq]⟩, [], ⟨1, 5⟩, false, nodes + (first :: others).length⟩

def resolve (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (evaluate : Board.Board → Int) (saved : Option Snapshot) (cached : Option CachedMove)
    (nodes : Nat) : Option Report :=
  match saved with
  | some snapshot => fromSnapshot snapshot nodes
  | none =>
    match cached with
    | some move =>
      if (legalMoves mc board).contains move.square then some (fromCache move nodes)
      else quickMove mc fc board evaluate nodes
    | none => quickMove mc fc board evaluate nodes

def ResolveContract : Prop :=
  ∀ (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position) (evaluate : Board.Board → Int)
    (cached : Option CachedMove) (nodes : Nat),
    (∀ snapshot head tail, snapshot.rows = head :: tail →
      resolve mc fc board evaluate (some snapshot) cached nodes =
        some ⟨head, head :: tail, snapshot.stamp, snapshot.endgame, nodes⟩) ∧
    (∀ move, cached = some move → LegalMove board.val.player board.val.opponent move.square.val →
      resolve mc fc board evaluate none cached nodes = some (fromCache move nodes)) ∧
    ((∀ move, cached = some move → ¬LegalMove board.val.player board.val.opponent move.square.val) →
      let output := resolve mc fc board evaluate none cached nodes
      (output = none ↔ ∀ sq : Square, ¬LegalMove board.val.player board.val.opponent sq.val) ∧
      ∀ report, output = some report →
        LegalMove board.val.player board.val.opponent report.best.square.val ∧
        report.best.score = onePly mc fc board evaluate report.best.square ∧
        (∀ sq : Square, LegalMove board.val.player board.val.opponent sq.val →
          onePly mc fc board evaluate sq ≤ report.best.score) ∧
        report.best.pv = [report.best.square] ∧ report.alternatives = [] ∧
        report.stamp = ⟨1, 5⟩ ∧ report.endgame = false ∧ report.nodes = nodes + (legalMoves mc board).length)

theorem resolveContract_proven : ResolveContract := by
  intro mc fc board evaluate cached nodes
  refine ⟨?_, ?_, ?_⟩
  · intro snapshot head tail rows
    simp [resolve, fromSnapshot, rows]
  · intro move hit legal
    have member := (legal_moves mc board move.square).mpr legal
    simp [resolve, hit, member]
  · intro unavailable output
    have fallback : output = quickMove mc fc board evaluate nodes := by
      cases hit : cached with
      | none => simp [output, resolve, hit]
      | some move =>
        have missing : move.square ∉ legalMoves mc board :=
          fun member => unavailable move hit ((legal_moves mc board move.square).mp member)
        simp [output, resolve, hit, missing]
    rw [fallback, quickMove]
    cases moves : legalMoves mc board with
    | nil =>
      refine ⟨?_, fun _ produced => by cases produced⟩
      refine ⟨fun _ sq legal => ?_, fun _ => rfl⟩
      have member := (legal_moves mc board sq).mpr legal
      simp [moves] at member
    | cons first rest =>
      have firstLegal := (legal_moves mc board first).mp (by simp [moves])
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · intro impossible
        cases impossible
      · exact fun noMoves => False.elim (noMoves first firstLegal)
      · intro report found
        cases Option.some.inj found
        obtain ⟨member, maximal⟩ := bestFrom_maximal (onePly mc fc board evaluate) first rest
        refine ⟨(legal_moves mc board _).mp (by rw [moves]; exact member), rfl, ?_, rfl, rfl, rfl, rfl, rfl⟩
        intro sq legal
        have member := (legal_moves mc board sq).mpr legal
        rw [moves] at member
        exact maximal sq member

def InterruptedContract : Prop :=
  ∀ (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position) (evaluate : Board.Board → Int)
    (minimum nodes : Nat) (completed : SearchPublication.Completed) (pending : List SearchPublication.Attempt)
    (saved : Snapshot) (head : Row) (tail : List Row) (cached : Option CachedMove),
    completed.last = some saved → saved.rows = head :: tail →
    SearchPublication.Unpublished minimum completed pending →
    resolve mc fc board evaluate
      (SearchPublication.finish minimum (SearchPublication.run minimum completed pending) .interrupted)
      cached nodes = some ⟨head, head :: tail, saved.stamp, saved.endgame, nodes⟩

theorem interruptedContract_proven : InterruptedContract := by
  intro mc fc board evaluate minimum nodes completed pending saved head tail cached stored rows noCapture
  have retained := SearchPublication.unpublished_keeps_last minimum completed pending noCapture
  change resolve mc fc board evaluate (SearchPublication.run minimum completed pending).last cached nodes = _
  rw [retained, stored]
  exact (resolveContract_proven mc fc board evaluate cached nodes).1 saved head tail rows

end NeuralReversi.SearchFallback
