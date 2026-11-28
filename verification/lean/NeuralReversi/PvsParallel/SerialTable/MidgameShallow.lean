module
public import NeuralReversi.PvsParallel.SerialTable.MidgameValue
public import NeuralReversi.PvsParallel.SerialTable.MidgameSites
public import NeuralReversi.PvsParallel.SerialTable.MidgameShallow.Masks

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MidgameShallow

open SerialCalls (Request)
open MidgameChild (Plan)
open PvsBoardTree (GeneratedSquare Position legalSquares pass play terminalScore)
open Bitboard (Square Bits)
open TranspositionTable (EntryData)
open AspirationMir.SerialTable.InitialEstimate (static)

/-- Evaluator words and the visit order produced by fast move ordering may vary by path. -/
structure Inputs where
  raw : List Square → Position → Int
  order : List Square → Position → Option EntryData → Square → Nat

def wipe (fc : Cutoffs.FlipConfig) (b : Position) (sq : Square) : Bool :=
  Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent == b.val.opponent

def evalMoves (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (b : Position) (beta : Int) (best : Int) : List (GeneratedSquare mc b.val) → Int
  | [] => best
  | sq :: rest =>
    if wipe fc b sq.val then PvsMoveLoop.scoreMax
    else
      let v := -(static (ops.raw (sq.val :: path)) (play mc fc b sq).val).val
      if beta ≤ v then v else evalMoves mc fc ops path b beta (max best v) rest

def visitOne (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (b : Position) (beta : Int) :=
  let legal := legalSquares mc b.val
  evalMoves mc fc ops path b beta (-32767) (depthOneMasks.flatMap fun mask => legal.filter fun sq => mask.getLsbD sq.val.val)

def one (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (b : Position) (a z : Int) : Int :=
  if Board.hasLegalMoves mc b.val then visitOne mc fc ops path b z
  else if Board.hasLegalMoves mc (pass b).val then -(visitOne mc fc ops path (pass b) (-a))
  else (terminalScore b).val

/-- The mask loop visits each legal square exactly once, in increasing index within its tier. -/
theorem mask_partition : ∀ sq : Square, (depthOneMasks.filter fun mask => mask.getLsbD sq.val).length = 1 := by decide

private theorem evalMoves_range (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs)
    (path : List Square) (b : Position) (beta best : Int) (moves : List (GeneratedSquare mc b.val))
    (upper : best ≤ 16384) (lower : -16384 ≤ best ∨ moves ≠ []) :
    -16384 ≤ evalMoves mc fc ops path b beta best moves ∧ evalMoves mc fc ops path b beta best moves ≤ 16384 := by
  induction moves generalizing best with
  | nil => exact ⟨lower.resolve_right (fun h => h rfl),upper⟩
  | cons sq rest ih =>
    have ⟨lo,hi⟩ := (static (ops.raw (sq.val :: path)) (play mc fc b sq).val).property
    unfold PvsMoveLoop.scoreMax at lo hi
    rw [evalMoves]
    split
    · exact ⟨by decide,by decide⟩
    · dsimp only
      split
      · exact ⟨by omega,by omega⟩
      · exact ih _ (by omega) (.inl (by omega))

private theorem visitOne_range (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs)
    (path : List Square) (b : Position) (beta : Int) (mobile : Board.hasLegalMoves mc b.val = true) :
    -16384 ≤ visitOne mc fc ops path b beta ∧ visitOne mc fc ops path b beta ≤ 16384 := by
  apply evalMoves_range _ _ _ _ _ _ _ _ (by decide) (.inr ?_)
  obtain ⟨n,legal⟩ := (Board.hasLegalMoves_exact mc b.val).mp mobile
  obtain ⟨sq,found,_⟩ := (PvsBoardTree.legalSquares_complete mc b.val ⟨n,legal.1⟩).mpr legal
  obtain ⟨mask,selected⟩ := List.exists_mem_of_ne_nil (depthOneMasks.filter fun mask => mask.getLsbD sq.val.val) (by
    intro h
    have count := mask_partition sq.val
    rw [h] at count
    contradiction)
  have member := List.mem_filter.mp selected
  exact List.ne_nil_of_mem (List.mem_flatMap.mpr ⟨mask,member.1,List.mem_filter.mpr ⟨found,member.2⟩⟩)

theorem one_finite (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (b : Position) (a z : Int) : SerialCalls.ScoreWords.Finite (one mc fc ops path b a z) := by
  unfold one SerialCalls.ScoreWords.Finite
  split
  next active =>
    have ⟨lo,hi⟩ := visitOne_range mc fc ops path b z active
    omega
  next =>
    split
    next active =>
      have ⟨lo,hi⟩ := visitOne_range mc fc ops path (pass b) (-a) active
      omega
    next =>
      have ⟨lo,hi⟩ := (terminalScore b).property
      unfold PvsMoveLoop.scoreMax at lo hi
      omega

abbrev Child (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (q : Request) :=
  GeneratedSquare mc q.board.val → (r : Request) → r.pv = false → Aspiration.Window r.alpha r.beta → Plan mc fc r

def scan (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (q : Request) (sc : q.pv = false)
    (slot : Fin 2) (pv : Bool) (child : Child mc fc q) :
    List (GeneratedSquare mc q.board.val) → GeneratedSquare mc q.board.val → (a : Int) → Aspiration.Window a q.beta →
    Option MidgameValue.Value → Option (GeneratedSquare mc q.board.val) → Plan mc fc q
  | rest, sq, a, win, best, hint =>
    let r := MidgameSites.shallowMove {q with alpha := a} (play mc fc q.board sq).val
    let cw : Aspiration.Window r.alpha r.beta := MidgameScoreRange.reflected win
    MidgameValue.call r cw (child sq r rfl cw) fun word =>
      let v := word.neg
      let store (score : MidgameValue.Value) (hint : Option (GeneratedSquare mc q.board.val)) :=
        Plan.writeAt slot .classified 5 pv hint (MidgameValue.known sc score)
      if high : q.beta ≤ v.val then store v (some sq)
      else
        let nextBest := best.elim v (fun x => x.max v)
        let nextHint := if a < v.val then some sq else hint
        match rest with
        | [] => store nextBest nextHint
        | next :: more =>
          have raised : Aspiration.Window (max a v.val) q.beta :=
            ⟨Int.le_trans win.lower (Int.le_max_left ..),by have := win.ordered; omega,win.upper⟩
          scan mc fc q sc slot pv child more next (max a v.val) raised (some nextBest) nextHint

def withTable (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (q : Request) (sc : q.pv = false) (mobile : Board.hasLegalMoves mc q.board.val = true)
    (win : Aspiration.Window q.alpha q.beta) (child : Child mc fc q) : Plan mc fc q :=
  MidgameValue.probe q.board fun slot entry =>
    let moves := legalSquares mc q.board.val
    if let some hit := entry then
      if (TranspositionTable.canCut false hit.val (Writes.word q.beta) q.depth 5 false).toOption == some true then
        MidgameValue.known sc hit.word
      else ordered moves rfl slot (some hit.val)
    else ordered moves rfl slot none
where
  ordered (moves : List (GeneratedSquare mc q.board.val)) (eq : moves = legalSquares mc q.board.val)
      (slot : Fin 2) (entry : Option EntryData) : Plan mc fc q :=
    if moves.any (fun sq => wipe fc q.board sq.val) then .wipeout
    else
      let key := ops.order path q.board entry
      let cmp := fun (x y : GeneratedSquare mc q.board.val) => decide (key x.val ≤ key y.val)
      match found : moves.mergeSort cmp with
      | [] => False.elim (by
          have empty := (List.mergeSort_perm moves cmp).length_eq
          rw [found,List.length_nil,eq] at empty
          exact Bool.noConfusion (mobile.symm.trans (PvsBoardTree.stuck_of_nil (List.length_eq_zero_iff.mp empty.symm))))
      | first :: rest => scan mc fc q sc slot (entry.any TranspositionTable.Access.isPv) child rest first q.alpha win none none

def playable (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (active : (q : Request) → q.pv = false → Board.hasLegalMoves mc q.board.val = true → Aspiration.Window q.alpha q.beta → Plan mc fc q)
    (q : Request) (sc : q.pv = false) (win : Aspiration.Window q.alpha q.beta) : Plan mc fc q :=
  if here : Board.hasLegalMoves mc q.board.val then active q sc here win
  else if other : Board.hasLegalMoves mc (pass q.board).val then
    .pass (active (MidgameChild.passRequest q) sc other (MidgameScoreRange.reflected win))
  else .terminal

def oneWord (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (q : Request) (sc : q.pv = false) : Plan mc fc q :=
  MidgameValue.known sc ⟨_,one_finite mc fc ops path q.board q.alpha q.beta⟩

def two (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ops : Inputs) (path : List Square)
    (q : Request) (sc : q.pv = false) (win : Aspiration.Window q.alpha q.beta) : Plan mc fc q :=
  playable mc fc (fun parent scout mobile window => withTable mc fc ops path parent scout mobile window fun sq r s _ =>
    oneWord mc fc ops (sq.val :: path) r s) q sc win

/-- Scalar depth-zero/one leaves and the depth-two/three TT verifier reached by MidGameStrategy. -/
def run (ops : Inputs) (q : Request) (sc : q.pv = false) (win : Aspiration.Window q.alpha q.beta)
    (shallow : q.depth.toNat ≤ 3) : Plan mc fc q :=
  match depth : q.depth.toNat with
  | 0 => .static (ops.raw [] q.board)
  | 1 => oneWord mc fc ops [] q sc
  | 2 => two mc fc ops [] q sc win
  | 3 => playable mc fc (fun parent scout mobile window => withTable mc fc ops [] parent scout mobile window fun sq r s cw =>
      two mc fc ops [sq.val] r s cw) q sc win
  | _ + 4 => False.elim (by omega)

def plan (ops : Inputs) (q : Request) : Plan mc fc q :=
  if h : q.pv = false ∧ (-32767 ≤ q.alpha ∧ q.alpha < q.beta ∧ q.beta ≤ 32767) ∧ q.depth.toNat ≤ 3 then
    run ops q h.1 ⟨h.2.1.1,h.2.1.2.1,h.2.1.2.2⟩ h.2.2
  else .stopped

theorem plan_eq (ops : Inputs) (q : Request) (sc : q.pv = false) (win : Aspiration.Window q.alpha q.beta)
    (shallow : q.depth.toNat ≤ 3) : plan (mc := mc) (fc := fc) ops q = run ops q sc win shallow := by
  unfold plan
  rw [dite_eq_left ⟨sc,⟨win.lower,win.ordered,win.upper⟩,shallow⟩]

end NeuralReversi.PvsParallel.SerialTable.MidgameShallow
