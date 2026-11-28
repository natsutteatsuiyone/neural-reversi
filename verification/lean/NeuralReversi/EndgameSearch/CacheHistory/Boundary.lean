module
public import NeuralReversi.EndgameSearch.RecursionCacheMir
public import NeuralReversi.PvsParallel.DeepLeafMir
public import NeuralReversi.PvsParallel.NonPvMir

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory

open PvsParallel
open EndgameTreeCache (Store Table Address soundHistory)
open Cancellation (EmptyState State Outcome)
open ScoreRefresh (Score toDiscDiff fromDiscDiff)
open TranspositionTable (Depth EntryData Selectivity)
open Flip.Dispatch (TargetConfig)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)

structure Input where
  board : Position
  empties : EmptyState
  aligned : empties.matches board
  alpha : Score
  beta : Score
  depth : Depth
  selectivity : Selectivity
  attempts : List TranspositionTable.ProbeRetry.RawAttempt
  polls : Threading.SplitCutoff.Polling
  parallel : Bool
  active : Bool
  aborted : Bool
  checked : Bool
  target : TargetConfig
  bmi2 : Bool
  paired : Bool

def Input.state (i : Input) (past : List Store) : State := ⟨i.empties,past,i.polls⟩

inductive Arm where
  | failed | fixed (arm : Direct.Mir.Arm) | table (entry : EntryData) | shallow (six : Bool) | deep (poll : Bool)

def Input.lookup (i : Input) : Option (Option EntryData) := do
  if i.depth.toNat = 12 ∧ i.parallel then
    match TranspositionTable.ProbeRetry.probe (i.attempts.map (TranspositionTable.ProbeRetry.observe (Cutoffs.key i.board))) with
    | .miss _ => return none
    | .hit _ entry =>
      let accepted ← (TranspositionTable.canCut i.checked entry ⟨i.beta.raw⟩ i.depth i.selectivity true).toOption
      return if accepted then some entry else none
  else return none

def Input.arm (i : Input) : Arm :=
  match i.lookup with
  | none => .failed
  | some (some e) => .table e
  | some none =>
    match NonPvMir.dispatch i.empties.count.toNat with
    | some (.fixed f) => .fixed f
    | some .shallow => .shallow (decide (i.empties.count.toNat = 6))
    | none => .deep i.active

structure Output where
  score : Option Int
  history : List Store

def Input.eval (i : Input) (mc : X86Config) (fc : FlipConfig) (address : Address) (past : List Store) : Arm → Output
  | .failed => ⟨none,past⟩
  | .table e => ⟨some e.score.toInt,past⟩
  | .fixed f =>
    let score : Option Score := do
      let raw ← Direct.Mir.discResult f i.target i.bmi2 i.board.val i.empties
        i.checked (toDiscDiff i.alpha).raw
      (fromDiscDiff i.checked raw).toOption
    ⟨score.map (·.raw.toInt),past⟩
  | .shallow _ =>
    let out := ShallowMir.run i.target i.bmi2 mc fc address i.board i.empties (toDiscDiff i.alpha).raw.toInt past
    ⟨((fromDiscDiff i.checked ⟨BitVec.ofInt 32 out.score⟩).toOption).map (·.raw.toInt),out.history⟩
  | .deep poll =>
    let state := DeepLeafMir.pollInput (i.state past) poll
    let out := DeepRecursionMir.run i.target i.bmi2 i.paired mc fc address i.board (toDiscDiff i.alpha).raw.toInt state
    let result := do
      let raw ← (fromDiscDiff i.checked ⟨BitVec.ofInt 32 (out.score.getD 0)⟩).toOption
      let score ← Cancellation.publish out i.aborted raw
      return score.raw.toInt
    ⟨result,out.state.history⟩

def Input.run (i : Input) (mc : X86Config) (fc : FlipConfig) (address : Address) (past : List Store) : Output :=
  i.eval mc fc address past i.arm

theorem Input.arm_of_score (i : Input) (value : (i.run mc fc addr past).score = some score) : i.arm ≠ .failed := by
  intro failed
  simp [Input.run,failed,Input.eval] at value

/-- In checked builds only the out-of-range packed score can prevent dispatch.
    The release lookup always selects a concrete search or table arm. -/
theorem Input.arm_of_lookup (i : Input)
    (bounds : ∀ slot e, TranspositionTable.ProbeRetry.probe
      (i.attempts.map (TranspositionTable.ProbeRetry.observe (Cutoffs.key i.board))) = .hit slot e →
      TranspositionTable.Eligible e i.depth i.selectivity true → i.checked = false ∨ e.score.toInt ≠ -32768) :
    i.arm ≠ .failed := by
  have read : ∃ found, i.lookup = some found := by
    unfold Input.lookup
    split
    · cases h : TranspositionTable.ProbeRetry.probe _ with
      | miss => exact ⟨_,rfl⟩
      | hit slot e =>
        dsimp only
        rw [TranspositionTable.canCut_contract]
        split
        · rename_i eligible
          rcases bounds slot e h eligible with release | safe <;> simp [*,Except.toOption]
        · exact ⟨_,rfl⟩
    · exact ⟨_,rfl⟩
  obtain ⟨found,accepted⟩ := read
  unfold Input.arm
  rw [accepted]
  cases found with
  | some => simp
  | none =>
    cases h : NonPvMir.dispatch i.empties.count.toNat with
    | none => simp
    | some a => cases a <;> simp

theorem shortcut (i : Input) (hit : i.lookup = some (some entry)) :
    i.run mc fc addr past = ⟨some entry.score.toInt,past⟩ := by
  simp [Input.run,Input.arm,hit,Input.eval]

theorem discarded_keeps_writes (i : Input) (deep : i.arm = .deep poll) (stopped : i.aborted = true) :
    let inner := DeepRecursionMir.run i.target i.bmi2 i.paired mc fc addr i.board (toDiscDiff i.alpha).raw.toInt
      (DeepLeafMir.pollInput (i.state past) poll)
    i.run mc fc addr past = ⟨none,inner.state.history⟩ := by
  simp [Input.run,deep,Input.eval,Cancellation.publish,stopped]

private theorem smallArm (n : Nat) (small : n ≤ 6) : ∃ a, NonPvMir.dispatch n = some a := by
  have : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 := by omega
  rcases this with h|h|h|h|h|h|h <;> subst n <;> exact ⟨_,rfl⟩

private theorem largeArm (n : Nat) (large : 6 < n) : NonPvMir.dispatch n = none := by
  obtain ⟨k,rfl⟩ : ∃ k, n = k+7 := ⟨n-7, by omega⟩
  rfl

theorem shallow_score (i : Input) (same : i.depth = i.empties.count) (short : i.empties.count.toNat ≤ 6) :
    (i.run mc fc addr past).score =
      (NonPvMir.sourceRun i.target i.bmi2 mc fc addr past i.board i.empties.mask
        i.depth i.empties.count i.alpha i.checked).map (·.raw.toInt) := by
  have noProbe : ¬ (i.depth.toNat = 12 ∧ i.parallel) := by rw [same]; omega
  have look : i.lookup = some none := by simp [Input.lookup,noProbe]
  obtain ⟨tag,branch⟩ := smallArm _ short
  change _ = ((if i.depth ≠ i.empties.count then none else (NonPvMir.dispatch i.empties.count.toNat).bind
    (fun | .fixed f => do
             let raw ← Direct.Mir.discResult f i.target i.bmi2 i.board.val i.empties
               i.checked (toDiscDiff i.alpha).raw
             (fromDiscDiff i.checked raw).toOption
         | .shallow => (fromDiscDiff i.checked ⟨BitVec.ofInt 32
             (ShallowMir.run i.target i.bmi2 mc fc addr i.board i.empties (toDiscDiff i.alpha).raw.toInt past).score⟩).toOption))).map _
  cases tag with
  | fixed f =>
    simp [Input.run,Input.arm,look,branch,Input.eval,same,Option.map_bind,Function.comp_def]
  | shallow => simp [Input.run,Input.arm,look,branch,Input.eval,same]

private def exactLookup (i : Input) := ({i with selectivity := Cutoffs.exactSelectivity} : Input).lookup

theorem deep_score (i : Input) (same : i.depth = i.empties.count)
    (large : 6 < i.empties.count.toNat) (within : i.empties.count.toNat ≤ 12)
    (exactTag : i.selectivity = Cutoffs.exactSelectivity) :
    (i.run mc fc addr past).score = DeepLeafMir.sourceRun i.target i.bmi2 i.paired mc fc addr i.board
      i.alpha i.beta i.depth (i.state past) i.attempts i.parallel i.active i.checked i.aborted := by
  have tag : exactLookup i = i.lookup := by simp [exactLookup,Input.lookup,exactTag]
  change _ = (if i.depth ≠ i.empties.count then none else (exactLookup i).bind (fun found =>
    (DeepLeafMir.dispatch i.depth.toNat i.parallel found.isSome i.active).bind fun
      | .probe => do return (← found).score.toInt
      | .activeNws => (i.eval mc fc addr past (.deep true)).score
      | .ignoreNws => (i.eval mc fc addr past (.deep false)).score))
  have noSmall := largeArm _ large
  simp only [ne_eq,same,not_true_eq_false,ite_false,tag]
  unfold Input.run Input.arm
  cases look : i.lookup with
  | none => rfl
  | some found =>
    cases found with
    | none =>
      cases i.active <;> simp [DeepLeafMir.dispatch,large,within,noSmall]
    | some e =>
      have allow : i.depth.toNat = 12 ∧ i.parallel := by
        by_cases h : i.depth.toNat = 12 ∧ i.parallel
        · exact h
        · simp [Input.lookup,h] at look
      simp [DeepLeafMir.dispatch,same ▸ allow.1,allow.2,Input.eval]

end NeuralReversi.EndgameSearch.CacheHistory
