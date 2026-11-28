module
public import NeuralReversi.Solve34.Nodes

@[expose] public section

namespace NeuralReversi.Solve34
open Dispatch
set_option linter.unusedSimpArgs false

private theorem maxRun_nodes_le (oracle : Oracle) (cap : Nat)
    (bounded : ∀ c, (oracle c).nodes ≤ cap) (alpha : Score)
    (first : Bool) (best : Score) (xs : List (Option Call)) (fallback : Outcome) :
    (Dispatch.maxRun oracle alpha first best xs fallback).nodes ≤ xs.length * cap + fallback.nodes := by
  induction xs generalizing first best with
  | nil => simp only [Dispatch.maxRun]; split <;> simp_all [finish]
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [Dispatch.maxRun]; split <;> simp_all [finish]
      | some c => simpa [Dispatch.maxRun, emit, finish, Event.cost] using
          Nat.le_add_right_of_le (bounded c)
    | cons y ys =>
      cases x with
      | none =>
        have h := ih false best
        simp only [Dispatch.maxRun, List.length_cons, Nat.add_mul] at *
        omega
      | some c =>
        have h := ih false (if first then -(oracle c).score else Solve2.maxWord (-(oracle c).score) best)
        have hc := bounded c
        simp only [Dispatch.maxRun]
        split <;> simp only [emit, finish, Event.cost, List.length_cons, Nat.add_mul] at * <;> omega

private theorem minRun_nodes_le (oracle : Oracle) (cap : Nat)
    (bounded : ∀ c, (oracle c).nodes ≤ cap) (alpha : Score)
    (first : Bool) (best : Score) (xs : List (Option Call)) (fallback : Outcome) :
    (Dispatch.minRun oracle alpha first best xs fallback).nodes ≤ xs.length * cap + fallback.nodes := by
  induction xs generalizing first best with
  | nil => simp only [Dispatch.minRun]; split <;> simp_all [finish]
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [Dispatch.minRun]; split <;> simp_all [finish]
      | some c => simpa [Dispatch.minRun, emit, finish, Event.cost] using
          Nat.le_add_right_of_le (bounded c)
    | cons y ys =>
      cases x with
      | none =>
        have h := ih false best
        simp only [Dispatch.minRun, List.length_cons, Nat.add_mul] at *
        omega
      | some c =>
        have h := ih false (if first then (oracle c).score else Solve2.minWord (oracle c).score best)
        have hc := bounded c
        simp only [Dispatch.minRun]
        split <;> simp only [emit, finish, Event.cost, List.length_cons, Nat.add_mul] at * <;> omega

theorem run_nodes_le (charge : Bool) (oracle : Oracle) (cap : Nat)
    (bounded : ∀ c, (oracle c).nodes ≤ cap) (alpha : Score)
    (ps os : List (Option Call)) (terminal : Score) :
    (run charge oracle alpha ps os terminal).nodes ≤
      2 * charge.toNat + (ps.length + os.length) * cap := by
  have ho := minRun_nodes_le oracle cap bounded alpha true 127 os (finish terminal)
  cases charge with
  | false =>
    have hp := maxRun_nodes_le oracle cap bounded alpha true (-127) ps
      (Dispatch.minRun oracle alpha true 127 os (finish terminal))
    simp only [run, Bool.false_eq_true, ↓reduceIte, Bool.toNat_false, finish, Nat.add_mul] at *
    omega
  | true =>
    have hp := maxRun_nodes_le oracle cap bounded alpha true (-127) ps
      (emit oracle .pass (Dispatch.minRun oracle alpha true 127 os (finish terminal)))
    simp only [run, ↓reduceIte, Bool.toNat_true, emit, Event.cost, finish] at *
    simp only [Nat.add_mul] at *
    omega

theorem sourceFallback_nodes_le (charge : Bool) (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (nodeCost : Call → Nat)
    (cap : Nat) (bounded : ∀ c, nodeCost c ≤ cap) (alpha : Score) :
    (sourceFallback charge b child nodeCost alpha).nodes ≤
      2 * charge.toNat + (n+n) * cap := by
  have h := run_nodes_le charge (verifiedOracle child nodeCost) cap
    (fun c => by rw [verifiedOracle_nodes]; exact bounded c) alpha
    (fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.player b.opponent (Solve2.childAlpha alpha) (boardSlots b))
    (fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.opponent b.player alpha (boardSlots b)) (terminalScore b)
  simpa only [sourceFallback, fallback, fallbackCandidates, List.length_map,
    boardSlots, b.length_eq] using h

theorem decodedNodeCost_le (cost : EmptyBoard n → Score → Nat) (cap : Nat)
    (bounded : ∀ b a, cost b a ≤ cap) (c : Call) : decodedNodeCost cost c ≤ cap := by
  unfold decodedNodeCost
  split
  · exact bounded _ _
  · omega

theorem twoNodeCount_le (b : EmptyBoard 2) (alpha : Score) : twoNodeCount b alpha ≤ 4 := by
  unfold twoNodeCount Solve2.sourceFallback Solve2.Dispatch.fallback
  exact (Solve2.Dispatch.run_node_bounds _ _ _ _ _ _ _).2

theorem source3Run_nodes_le (b : EmptyBoard 3) (alpha : Score) :
    (source3Run b alpha).nodes ≤ 26 :=
  sourceFallback_nodes_le true b twoSolver (decodedNodeCost twoNodeCount) 4
    (decodedNodeCost_le twoNodeCount 4 twoNodeCount_le) alpha

theorem source4Run_nodes_le (b : EmptyBoard 4) (alpha : Score) :
    (source4Run b alpha).nodes ≤ 208 :=
  sourceFallback_nodes_le false b solve3Concrete (decodedNodeCost threeNodeCount) 26
    (decodedNodeCost_le threeNodeCount 26 source3Run_nodes_le) alpha

def counterAfter (out : Outcome) (initial : Word) : Word :=
  initial + BitVec.ofNat 64 out.nodes

theorem counter_no_overflow (out : Outcome) (initial : Word)
    (room : initial.toNat + out.nodes < 2^64) :
    BitVec.uaddOverflow initial (BitVec.ofNat 64 out.nodes) = false ∧
      (counterAfter out initial).toNat = initial.toNat + out.nodes := by
  have small : out.nodes < 2^64 := by omega
  have cast : (BitVec.ofNat 64 out.nodes).toNat = out.nodes := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt small]
  constructor
  · simp only [BitVec.uaddOverflow, cast, decide_eq_false_iff_not]
    omega
  · simp only [counterAfter, BitVec.toNat_add, cast]
    exact Nat.mod_eq_of_lt room

theorem source3Run_counter_no_overflow (b : EmptyBoard 3) (alpha : Score) (initial : Word)
    (room : initial.toNat + 26 < 2^64) :
    BitVec.uaddOverflow initial (BitVec.ofNat 64 (source3Run b alpha).nodes) = false ∧
      (counterAfter (source3Run b alpha) initial).toNat =
        initial.toNat + (source3Run b alpha).nodes := by
  apply counter_no_overflow
  have h := source3Run_nodes_le b alpha
  omega

theorem source4Run_counter_no_overflow (b : EmptyBoard 4) (alpha : Score) (initial : Word)
    (room : initial.toNat + 208 < 2^64) :
    BitVec.uaddOverflow initial (BitVec.ofNat 64 (source4Run b alpha).nodes) = false ∧
      (counterAfter (source4Run b alpha) initial).toNat =
        initial.toNat + (source4Run b alpha).nodes := by
  apply counter_no_overflow
  have h := source4Run_nodes_le b alpha
  omega

end NeuralReversi.Solve34
