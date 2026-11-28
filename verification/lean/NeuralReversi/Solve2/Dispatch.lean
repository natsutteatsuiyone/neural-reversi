module
public import NeuralReversi.Solve2.Control

@[expose] public section

namespace NeuralReversi.Solve2.Dispatch
set_option linter.unusedSimpArgs false

abbrev Word := BitVec 64
abbrev Square := Fin 64

structure Call where
  player : Word
  alpha : Score
  square : Square
  deriving DecidableEq, Repr

abbrev Oracle := Call → Score

structure Outcome where
  score : Score
  calls : List Call
  passed : Bool
  deriving DecidableEq, Repr

def Outcome.nodeDelta (out : Outcome) : Nat := 1 + out.passed.toNat + out.calls.length

def Outcome.counterAfter (out : Outcome) (initial : Word) : Word :=
  initial + BitVec.ofNat 64 out.nodeDelta

def passRun (oracle : Oracle) (alpha : Score) (first second : Option Call)
    (terminal : Score) : Outcome :=
  match first with
  | some c =>
      let best := oracle c
      if best.toInt ≤ alpha.toInt then ⟨best,[c],true⟩ else
      match second with
      | some d => ⟨minWord (oracle d) best,[c,d],true⟩
      | none => ⟨best,[c],true⟩
  | none => match second with
      | some d => ⟨oracle d,[d],true⟩
      | none => ⟨terminal,[],true⟩

def run (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call)
    (terminal : Score) : Outcome :=
  match p1 with
  | some c =>
      let best := -(oracle c)
      if (alpha+1).toInt ≤ best.toInt then ⟨best,[c],false⟩ else
      match p2 with
      | some d => ⟨maxWord (-(oracle d)) best,[c,d],false⟩
      | none => ⟨best,[c],false⟩
  | none => match p2 with
      | some d => ⟨-(oracle d),[d],false⟩
      | none => passRun oracle alpha o1 o2 terminal

def candidate (opponent flipped : Word) (alpha : Score) (remaining : Square) : Option Call :=
  if flipped = 0 then none else some ⟨opponent ^^^ flipped,alpha,remaining⟩

def guardedCandidate (adjacent : Bool) (opponent flipped : Word)
    (alpha : Score) (remaining : Square) : Option Call :=
  if adjacent then candidate opponent flipped alpha remaining else none

theorem guardedCandidate_eq (adjacent : Bool) (opponent flipped : Word)
    (alpha : Score) (remaining : Square)
    (guard : adjacent = false → flipped = 0) :
    guardedCandidate adjacent opponent flipped alpha remaining =
      candidate opponent flipped alpha remaining := by
  cases ha : adjacent
  · simp [guardedCandidate, ha, candidate, guard ha]
  · rfl

abbrev Flip := Square → Word → Word → Word
abbrev Flip2 := Square → Square → Word → Word → Word × Word
abbrev Adjacent := Word → Square → Bool

def eager (oracle : Oracle) (flip2 : Flip2) (player opponent : Word)
    (alpha : Score) (sq1 sq2 : Square) (terminal : Score) : Outcome :=
  let fp := flip2 sq1 sq2 player opponent
  let fo := flip2 sq1 sq2 opponent player
  run oracle alpha
    (candidate opponent fp.1 (childAlpha alpha) sq2)
    (candidate opponent fp.2 (childAlpha alpha) sq1)
    (candidate player fo.1 alpha sq2)
    (candidate player fo.2 alpha sq1) terminal

def fallback (oracle : Oracle) (flip : Flip) (adjacent : Adjacent)
    (player opponent : Word) (alpha : Score) (sq1 sq2 : Square) (terminal : Score) : Outcome :=
  run oracle alpha
    (guardedCandidate (adjacent opponent sq1) opponent (flip sq1 player opponent) (childAlpha alpha) sq2)
    (guardedCandidate (adjacent opponent sq2) opponent (flip sq2 player opponent) (childAlpha alpha) sq1)
    (guardedCandidate (adjacent player sq1) player (flip sq1 opponent player) alpha sq2)
    (guardedCandidate (adjacent player sq2) player (flip sq2 opponent player) alpha sq1) terminal

def eagerEnabled (x86_64 avx512cd avx512vl : Bool) : Bool := x86_64 && avx512cd && avx512vl

def dispatch (x86_64 avx512cd avx512vl : Bool) (oracle : Oracle) (flip : Flip)
    (flip2 : Flip2) (adjacent : Adjacent) (player opponent : Word)
    (alpha : Score) (sq1 sq2 : Square) (terminal : Score) : Outcome :=
  if eagerEnabled x86_64 avx512cd avx512vl then
    eager oracle flip2 player opponent alpha sq1 sq2 terminal
  else fallback oracle flip adjacent player opponent alpha sq1 sq2 terminal

theorem eager_eq_fallback (oracle : Oracle) (flip : Flip) (flip2 : Flip2)
    (adjacent : Adjacent) (player opponent : Word) (alpha : Score)
    (sq1 sq2 : Square) (terminal : Score)
    (playerFlips : flip2 sq1 sq2 player opponent =
      (flip sq1 player opponent, flip sq2 player opponent))
    (opponentFlips : flip2 sq1 sq2 opponent player =
      (flip sq1 opponent player, flip sq2 opponent player))
    (gp1 : adjacent opponent sq1 = false → flip sq1 player opponent = 0)
    (gp2 : adjacent opponent sq2 = false → flip sq2 player opponent = 0)
    (go1 : adjacent player sq1 = false → flip sq1 opponent player = 0)
    (go2 : adjacent player sq2 = false → flip sq2 opponent player = 0) :
    eager oracle flip2 player opponent alpha sq1 sq2 terminal =
      fallback oracle flip adjacent player opponent alpha sq1 sq2 terminal := by
  simp only [eager, fallback, playerFlips, opponentFlips,
    guardedCandidate_eq _ _ _ _ _ gp1, guardedCandidate_eq _ _ _ _ _ gp2,
    guardedCandidate_eq _ _ _ _ _ go1, guardedCandidate_eq _ _ _ _ _ go2]

theorem dispatch_eq_fallback (x86_64 avx512cd avx512vl : Bool)
    (oracle : Oracle) (flip : Flip) (flip2 : Flip2) (adjacent : Adjacent)
    (player opponent : Word) (alpha : Score) (sq1 sq2 : Square) (terminal : Score)
    (agreement : eager oracle flip2 player opponent alpha sq1 sq2 terminal =
      fallback oracle flip adjacent player opponent alpha sq1 sq2 terminal) :
    dispatch x86_64 avx512cd avx512vl oracle flip flip2 adjacent player opponent alpha sq1 sq2 terminal =
      fallback oracle flip adjacent player opponent alpha sq1 sq2 terminal := by
  unfold dispatch
  split
  · exact agreement
  · rfl

theorem run_score (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call) (terminal : Score) :
    (run oracle alpha p1 p2 o1 o2 terminal).score =
      solveWord alpha (p1.map oracle) (p2.map oracle) (o1.map oracle) (o2.map oracle) terminal := by
  cases p1 <;> cases p2 <;> cases o1 <;> cases o2 <;>
    simp only [run, passRun, solveWord, maxSearchWord, minSearchWord, Option.map]
  all_goals split <;> rfl

theorem run_passed (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call) (terminal : Score) :
    (run oracle alpha p1 p2 o1 o2 terminal).passed = (p1.isNone && p2.isNone) := by
  cases p1 <;> cases p2 <;> cases o1 <;> cases o2 <;>
    simp only [run, passRun, Option.isNone, Bool.false_and, Bool.true_and]
  all_goals split <;> rfl

theorem run_call_count (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call) (terminal : Score) :
    (run oracle alpha p1 p2 o1 o2 terminal).calls.length ≤ 2 := by
  cases p1 <;> cases p2 <;> cases o1 <;> cases o2 <;>
    simp only [run, passRun]
  all_goals (try split) <;> simp [Outcome.nodeDelta]

theorem run_node_bounds (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call) (terminal : Score) :
    2 ≤ (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta ∧
    (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta ≤ 4 := by
  cases p1 <;> cases p2 <;> cases o1 <;> cases o2 <;>
    simp only [run, passRun]
  all_goals (try split) <;> simp [Outcome.nodeDelta]

theorem both_pass_trace (oracle : Oracle) (alpha terminal : Score) :
    run oracle alpha none none none none terminal = ⟨terminal,[],true⟩ ∧
    (run oracle alpha none none none none terminal).nodeDelta = 2 := by
  exact ⟨rfl,rfl⟩

theorem run_nws (oracle : Oracle) (alpha terminal : Score) (p1 p2 o1 o2 : Option Call)
    (v1 v2 w1 w2 : Option Int)
    (windowSafe : alpha.toInt < 2147483647)
    (safe1 : NegSafe (p1.map oracle)) (safe2 : NegSafe (p2.map oracle))
    (hp1 : OptionalRel (OneEmpty (childAlpha alpha).toInt) v1 ((p1.map oracle).map BitVec.toInt))
    (hp2 : OptionalRel (OneEmpty (childAlpha alpha).toInt) v2 ((p2.map oracle).map BitVec.toInt))
    (ho1 : OptionalRel (OneEmpty alpha.toInt) w1 ((o1.map oracle).map BitVec.toInt))
    (ho2 : OptionalRel (OneEmpty alpha.toInt) w2 ((o2.map oracle).map BitVec.toInt)) :
    NWS alpha.toInt (value v1 v2 w1 w2 terminal.toInt)
      (run oracle alpha p1 p2 o1 o2 terminal).score.toInt := by
  rw [run_score]
  exact solveWord_nws windowSafe safe1 safe2 hp1 hp2 ho1 ho2


structure Compatible (flip : Flip) (flip2 : Flip2) (adjacent : Adjacent) : Prop where
  batch : ∀ a b p o, flip2 a b p o = (flip a p o, flip b p o)
  guard : ∀ sq p o, adjacent o sq = false → flip sq p o = 0

theorem eager_fallback_effects (oracle : Oracle) (flip : Flip) (flip2 : Flip2)
    (adjacent : Adjacent) (compatible : Compatible flip flip2 adjacent)
    (player opponent : Word) (alpha : Score) (sq1 sq2 : Square) (terminal : Score)
    (initial : Word) :
    let e := eager oracle flip2 player opponent alpha sq1 sq2 terminal
    let f := fallback oracle flip adjacent player opponent alpha sq1 sq2 terminal
    e.score = f.score ∧ e.calls = f.calls ∧ e.nodeDelta = f.nodeDelta ∧
      e.counterAfter initial = f.counterAfter initial := by
  have he := eager_eq_fallback oracle flip flip2 adjacent player opponent alpha sq1 sq2 terminal
    (compatible.batch _ _ _ _) (compatible.batch _ _ _ _)
    (compatible.guard _ _ _) (compatible.guard _ _ _) (compatible.guard _ _ _) (compatible.guard _ _ _)
  simp only [he, and_self]

theorem first_cut_trace (oracle : Oracle) (alpha terminal : Score) (first : Call)
    (p2 o1 o2 : Option Call) (cut : (alpha+1).toInt ≤ (-(oracle first)).toInt) :
    run oracle alpha (some first) p2 o1 o2 terminal = ⟨-(oracle first),[first],false⟩ ∧
    (run oracle alpha (some first) p2 o1 o2 terminal).nodeDelta = 2 := by
  simp only [run, cut, ↓reduceIte]
  simp [Outcome.nodeDelta]

theorem pass_first_cut_trace (oracle : Oracle) (alpha terminal : Score) (first : Call)
    (o2 : Option Call) (cut : (oracle first).toInt ≤ alpha.toInt) :
    run oracle alpha none none (some first) o2 terminal = ⟨oracle first,[first],true⟩ ∧
    (run oracle alpha none none (some first) o2 terminal).nodeDelta = 3 := by
  simp [run, passRun, cut, Outcome.nodeDelta]

theorem run_counter_no_overflow (oracle : Oracle) (alpha : Score) (p1 p2 o1 o2 : Option Call)
    (terminal : Score) (initial : Word) (room : initial.toNat ≤ 18446744073709551611) :
    BitVec.uaddOverflow initial (BitVec.ofNat 64 (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta) = false ∧
    ((run oracle alpha p1 p2 o1 o2 terminal).counterAfter initial).toNat =
      initial.toNat + (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta := by
  have bounds := run_node_bounds oracle alpha p1 p2 o1 o2 terminal
  have small : (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta < 2^64 := by omega
  have cast : (BitVec.ofNat 64 (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta).toNat =
      (run oracle alpha p1 p2 o1 o2 terminal).nodeDelta := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt small]
  constructor
  · simp only [BitVec.uaddOverflow, cast, decide_eq_false_iff_not]
    omega
  · simp only [Outcome.counterAfter, BitVec.toNat_add, cast]
    exact Nat.mod_eq_of_lt (by omega)

end NeuralReversi.Solve2.Dispatch
