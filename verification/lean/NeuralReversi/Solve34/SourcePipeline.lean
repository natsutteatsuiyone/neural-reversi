import NeuralReversi.Solve34.Concrete
import NeuralReversi.Solve34.DispatchScore

namespace NeuralReversi.Solve34

def ValidCall (n : Nat) (c : Dispatch.Call) : Prop :=
  c.remaining.length = n ∧ c.remaining.Nodup ∧ c.player &&& c.opponent = 0 ∧
    ~~~(c.player ||| c.opponent) = emptyMask c.remaining

instance (n : Nat) (c : Dispatch.Call) : Decidable (ValidCall n c) := by
  unfold ValidCall
  infer_instance

def boardCall (b : EmptyBoard n) (alpha : Score) : Dispatch.Call :=
  ⟨b.player, b.opponent, alpha, b.squares⟩

def verifiedOracle (child : EmptyBoard n → Score → Score)
    (nodeCost : Dispatch.Call → Nat) (c : Dispatch.Call) : Dispatch.ChildResult :=
  if h : ValidCall n c then
    ⟨child ⟨c.player,c.opponent,c.remaining,h.1,h.2.1,h.2.2.1,h.2.2.2⟩ c.alpha, nodeCost c⟩
  else ⟨0, nodeCost c⟩

theorem boardCall_valid (b : EmptyBoard n) (alpha : Score) :
    ValidCall n (boardCall b alpha) :=
  ⟨b.length_eq,b.nodup,b.disjoint,b.empty_exact⟩

theorem verifiedOracle_boardCall (b : EmptyBoard n) (alpha : Score)
    (child : EmptyBoard n → Score → Score) (nodeCost : Dispatch.Call → Nat) :
    (verifiedOracle child nodeCost (boardCall b alpha)).score = child b alpha := by
  simp only [verifiedOracle, boardCall_valid, ↓reduceDIte]
  rfl

theorem verifiedOracle_nodes (child : EmptyBoard n → Score → Score)
    (nodeCost : Dispatch.Call → Nat) (c : Dispatch.Call) :
    (verifiedOracle child nodeCost c).nodes = nodeCost c := by
  unfold verifiedOracle
  split <;> rfl

def boardSlots (b : EmptyBoard n) : List Dispatch.Slot :=
  b.squares.map fun sq => (sq,b.squares.erase sq)

theorem candidate_verified (b : EmptyBoard n) (sq : {s // s ∈ b.squares})
    (alpha : Score) (child : EmptyBoard (n-1) → Score → Score)
    (nodeCost : Dispatch.Call → Nat) :
    (Dispatch.candidate b.player b.opponent (generatedFlip b sq.val) alpha sq.val
      (b.squares.erase sq.val)).map (fun c => (verifiedOracle child nodeCost c).score) =
      if generatedFlip b sq.val = 0 then none else some (child (moveChild b sq) alpha) := by
  unfold Dispatch.candidate
  split
  · rfl
  · change some (verifiedOracle child nodeCost (boardCall (moveChild b sq) alpha)).score = _
    rw [verifiedOracle_boardCall]

theorem fallbackCandidates_verified (b : EmptyBoard n) (alpha : Score)
    (child : EmptyBoard (n-1) → Score → Score) (nodeCost : Dispatch.Call → Nat) :
    (Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.player b.opponent alpha (boardSlots b)).map
        (Option.map (fun c => (verifiedOracle child nodeCost c).score)) =
      candidateScores b child alpha := by
  unfold Dispatch.fallbackCandidates boardSlots candidateScores
  simp only [List.map_map]
  rw [← List.attach_map_val]
  apply List.map_congr_left
  intro sq _
  dsimp
  rw [Dispatch.guardedCandidate_eq _ _ _ _ _ _ _
    (Solve2.Guard.scalar_compatible.guard sq.val b.player b.opponent)]
  exact candidate_verified b sq alpha child nodeCost

def sourceFallback (charge : Bool) (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) : Dispatch.Outcome :=
  Dispatch.fallback charge (verifiedOracle child nodeCost)
    Solve2.Guard.scalarFlip Solve2.Guard.adjacent b.player b.opponent alpha
      (boardSlots b) (terminalScore b)

theorem sourceFallback_score (charge : Bool) (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) :
    (sourceFallback charge b child nodeCost alpha).score = solveNext b child alpha := by
  unfold sourceFallback Dispatch.fallback
  rw [Dispatch.run_score, fallbackCandidates_verified b]
  change solveWord alpha _
    ((Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.pass.player b.pass.opponent alpha (boardSlots b.pass)).map
        (Option.map (fun c => (verifiedOracle child nodeCost c).score))) _ = _
  rw [fallbackCandidates_verified b.pass]
  rfl

theorem sourceFallback_nodes (charge : Bool) (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) :
    let out := sourceFallback charge b child nodeCost alpha
    out.nodes = charge.toNat + out.events.count .pass + (out.calls.map nodeCost).sum := by
  simpa only [sourceFallback, Dispatch.fallback, verifiedOracle_nodes] using
    (Dispatch.run_node_formula charge (verifiedOracle child nodeCost) alpha
      (Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
        b.player b.opponent (Solve2.childAlpha alpha) (boardSlots b))
      (Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
        b.opponent b.player alpha (boardSlots b)) (terminalScore b))

theorem boardSlots_three (b : EmptyBoard 3) (a c d : Square)
    (order : b.squares = [a,c,d]) : boardSlots b = Dispatch.slots3 a c d := by
  have distinct : [a,c,d].Nodup := order ▸ b.nodup
  obtain ⟨ha,hc,hd⟩ := erase_three a c d distinct
  simp only [boardSlots, order, List.map_cons, List.map_nil, ha, hc, hd, Dispatch.slots3]

theorem boardSlots_four (b : EmptyBoard 4) (a c d e : Square)
    (order : b.squares = [a,c,d,e]) : boardSlots b = Dispatch.slots4 a c d e := by
  have distinct : [a,c,d,e].Nodup := order ▸ b.nodup
  obtain ⟨ha,hc,hd,he⟩ := erase_four a c d e distinct
  simp only [boardSlots, order, List.map_cons, List.map_nil, ha, hc, hd, he, Dispatch.slots4]

def source3Fallback (b : EmptyBoard 3) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) : Dispatch.Outcome := sourceFallback true b twoSolver nodeCost alpha

def source4Fallback (b : EmptyBoard 4) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) : Dispatch.Outcome := sourceFallback false b solve3Concrete nodeCost alpha

theorem source3Fallback_score (b : EmptyBoard 3) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) : (source3Fallback b nodeCost alpha).score = solve3Concrete b alpha :=
  sourceFallback_score true b twoSolver nodeCost alpha

theorem source4Fallback_score (b : EmptyBoard 4) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) : (source4Fallback b nodeCost alpha).score = solve4Concrete b alpha :=
  sourceFallback_score false b solve3Concrete nodeCost alpha

theorem source3Fallback_nws (b : EmptyBoard 3) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactThreeValue b) (source3Fallback b nodeCost alpha).score.toInt := by
  rw [source3Fallback_score]
  exact solve3Concrete_nws b alpha safe

theorem source4Fallback_nws (b : EmptyBoard 4) (nodeCost : Dispatch.Call → Nat)
    (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactFourValue b) (source4Fallback b nodeCost alpha).score.toInt := by
  rw [source4Fallback_score]
  exact solve4Concrete_nws b alpha safe

end NeuralReversi.Solve34
