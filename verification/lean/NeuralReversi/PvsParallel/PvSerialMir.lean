module
public import NeuralReversi.PvsParallel.PvSerialMir.Program
public import NeuralReversi.PvsParallel.RootOwnerMir.Slice
public import NeuralReversi.PvsParallel.RootSerialMir
public import NeuralReversi.PvsParallel.PvEntryMir
public import NeuralReversi.Threading.InFlight

@[expose] public section

namespace NeuralReversi.PvsParallel.PvSerialMir

open PvsMoveLoop (ChildSearch ScanCalls moveScore scan)
open PvsBoardTree (Position GeneratedSquare build play legalSquares)
open RootReplay (Snapshot)
open Bitboard (Square)

theorem smallClaim (depth : Nat) (small : depth ≤ 5) (slot : Option Threading.InFlight.Hash)
    (hash : Threading.InFlight.Hash) :
    Threading.InFlight.claimChild true depth minimumClaimDepth slot hash = (slot, .untracked) := by
  have shallow : ¬minimumClaimDepth ≤ depth := by unfold minimumClaimDepth; omega
  simp [Threading.InFlight.claimChild, shallow]

theorem smallHeader (idx count fresh depth : Nat) (small : depth ≤ 5) (free : Bool)
    (alpha scout wide : Int) :
    headerAction idx count fresh depth rootMinimumSplitDepth free false alpha scout wide =
      headerAction idx count fresh depth rootMinimumSplitDepth false false alpha scout wide := by
  have shallow : ¬rootMinimumSplitDepth ≤ depth := by unfold rootMinimumSplitDepth; omega
  simp [headerAction_eq, canHandoff, shallow]

structure Result where
  best : Int
  updates : List Square

def run (sq : Move → Square) (search : ChildSearch Move) (depth minSplit : Nat) (beta : Int) :
    List Move → Nat → Int → Int → Option Result
  | [], idx, alpha, best =>
    match action idx idx idx depth minSplit false false alpha 0 0 with
    | .finish => some ⟨best, []⟩
    | _ => none
  | move :: rest, idx, alpha, best => do
    let n := idx + (move :: rest).length
    let scout := search move false (-(alpha + 1)) (-alpha)
    let wide := search move true (-beta) (-alpha)
    let .child next _ _ score := action idx n n depth minSplit false false alpha scout wide | none
    let d := decision (decide (next = 1)) alpha beta best score
    let best' := if d.raisesBest then score else best
    let alpha' := if d.raisesAlpha then score else alpha
    let remaining ← if d.cuts then some ⟨best', []⟩ else run sq search depth minSplit beta rest next alpha' best'
    return ⟨remaining.best, if writesPv d then sq move :: remaining.updates else remaining.updates⟩

theorem agrees (sq : Move → Square) (search : ChildSearch Move) (depth minSplit : Nat) (beta : Int)
    (moves : List Move) (s : Snapshot) :
    ∃ res, run sq search depth minSplit beta moves s.seen.length s.state.alpha s.state.best = some res ∧
      res.best = (RootReplay.serial sq search beta moves s).state.best := by
  induction moves generalizing s with
  | nil => simp [run, action, headerAction_eq, RootReplay.serial]
  | cons m rest ih =>
    obtain ⟨_, _, header⟩ := RootSerialMir.ordinary_child search m rest s depth minSplit beta
    let score := moveScore true s.seen.isEmpty search m s.state.alpha beta
    let d := decision s.seen.isEmpty s.state.alpha beta s.state.best score
    let next := RootUpdateMir.after s (sq m) score d
    obtain ⟨same, cuts, _, _⟩ := RootUpdateMir.agrees s (sq m) beta score
    have first : decide (s.seen.length + 1 = 1) = s.seen.isEmpty := by cases s.seen <;> simp
    rw [run, action, header]
    simp only [first, ← show score = moveScore true s.seen.isEmpty search m s.state.alpha beta from rfl]
    change next = _ at same
    change d.cuts = _ at cuts
    rw [RootReplay.serial, ← show score = moveScore true s.seen.isEmpty search m s.state.alpha beta from rfl,
      ← same, ← cuts]
    cases cut : d.cuts with
    | true => exact ⟨_, rfl, rfl⟩
    | false =>
      obtain ⟨r, value, equals⟩ := ih next
      change run sq search depth minSplit beta rest (s.seen.length + 1)
        (if d.raisesAlpha then score else s.state.alpha) (if d.raisesBest then score else s.state.best) = some r at value
      refine ⟨⟨r.best, if writesPv d then sq m :: r.updates else r.updates⟩, ?_, equals⟩
      simp [d, value]

theorem execution (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (first : GeneratedSquare mc b.val) (rest moves : List (GeneratedSquare mc b.val))
    (depth minSplit : Nat) (alpha beta : Int) (search : ChildSearch (GeneratedSquare mc b.val))
    (members : legalSquares mc b.val = first :: rest) (order : moves.Perm (first :: rest))
    (calls : ScanCalls (fun m => children .serial (build mc fc (play mc fc b m).val))
      search true beta moves true alpha (-PvsMoveLoop.scoreInf)) :
    ∃ res, run Subtype.val search depth minSplit beta moves 0 alpha (-PvsMoveLoop.scoreInf) = some res ∧
      ExecutionWithUses exit children (build mc fc b) true alpha beta res.best := by
  obtain ⟨res, returns, value⟩ := agrees Subtype.val search depth minSplit beta moves (RootReplay.start alpha)
  rw [RootReplay.serial_best] at value
  refine ⟨res, returns, ?_⟩
  rw [build.eq_def]
  split
  next head tail h =>
    obtain ⟨rfl, rfl⟩ := List.cons.inj (members.symm.trans h)
    rw [value]
    exact .sequential moves (fun m => build mc fc (play mc fc b m).val) (order.map _) search calls
  next h => rw [members] at h; cases h

/-- A normal return from the PV frame, with a split callback only at a real handoff. -/
def scheduled (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (depth : ExactDepth.Depth) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (alpha beta : Int) (passCall : Position → ExactDepth.Depth → Int → Int → Option Int)
    (moves : List (GeneratedSquare mc b.val)) (replies : RootOwnerMir.Replies (GeneratedSquare mc b.val))
    (schedule : Nat → Bool × Bool)
    (splitCall : RootOwnerMir.Slice.Exit (GeneratedSquare mc b.val) → Option Int) : Option Int :=
  let generated := PvEntryMir.generated mc fc b paired mem
  match PvEntryMir.dispatch depth.toNat (generated.len.getD 0 != 0)
      (Board.hasLegalMoves mc (PvsBoardTree.pass b).val) generated.wipeout.join.isSome with
  | .continueSearch => do
    let owner ← RootOwnerMir.Slice.run Subtype.val replies schedule depth.toNat rootMinimumSplitDepth beta
      0 0 moves.length moves (RootReplay.start alpha)
    if owner.handoff then splitCall owner else some owner.snapshot.state.best
  | _ => PvEntryMir.run mc fc b depth paired checked mem alpha beta passCall

theorem scheduledExecution (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (depth : ExactDepth.Depth) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (alpha beta : Int) (passCall : Position → ExactDepth.Depth → Int → Int → Option Int)
    (moves : List (GeneratedSquare mc b.val)) (replies : RootOwnerMir.Replies (GeneratedSquare mc b.val))
    (schedule : Nat → Bool × Bool)
    (splitCall : RootOwnerMir.Slice.Exit (GeneratedSquare mc b.val) → Option Int)
    (full : ExactDepth.Full b depth) (order : moves.Perm (legalSquares mc b.val))
    (passes : legalSquares mc b.val = [] → Board.hasLegalMoves mc (PvsBoardTree.pass b).val = true →
      ∀ v, passCall (PvsBoardTree.pass b) depth (-beta) (-alpha) = some v →
        children .serial (build mc fc (PvsBoardTree.pass b)) true (-beta) (-alpha) v)
    (calls : ∀ m p a z v, replies m p a z = some v →
      children .serial (build mc fc (play mc fc b m).val) p a z v)
    (splits : ∀ r value, r.handoff = true → splitCall r = some value →
      let labels := r.moves.map (fun m => build mc fc (play mc fc b m).val)
      ∃ split : SplitExecution labels.length true alpha beta value,
        split.start.toNat = r.idx ∧ split.entry = r.snapshot.state ∧ ReportCalls children labels split)
    (answer : scheduled mc fc b depth paired checked mem alpha beta passCall moves replies schedule splitCall = some score) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b) true alpha beta score := by
  dsimp only [scheduled] at answer
  split at answer
  next branch =>
    cases available : legalSquares mc b.val with
    | nil =>
      simp only [PvEntryMir.generated_count, Option.getD_some, available, List.length_nil] at branch
      unfold PvEntryMir.dispatch at branch
      repeat' split at branch
      all_goals simp_all
    | cons first rest =>
      rw [RootOwnerMir.Slice.fromMoves_eq] at answer
      cases ran : RootOwnerMir.fromMoves Subtype.val replies schedule depth.toNat rootMinimumSplitDepth alpha beta moves with
      | none => simp [ran] at answer
      | some r =>
        obtain ⟨trace, _, _⟩ := RootOwnerMir.refines _ _ _ calls _ _ _ _ _ _ _ _ _ _ (by simp) rfl ran
        let child := fun m => build mc fc (play mc fc b m).val
        have ordered : ((⟨0, moves, []⟩ : PvsPostpone.Queue _).pending.map child).Perm (child first :: rest.map child) := by
          simpa [PvsPostpone.Queue.pending, available] using order.map child
        rw [build.eq_def]
        split
        next head tail h =>
          obtain ⟨rfl, rfl⟩ := List.cons.inj (available.symm.trans h)
          cases handed : r.handoff with
          | false =>
            have result : r.snapshot.state.best = score := by simpa [ran, RootOwnerMir.Slice.layout, handed] using answer
            have outcome : r.outcome = .returned score := by simp [RootOwnerMir.Result.outcome, handed, result]
            rw [outcome] at trace
            exact ExecutionWithUses.ofSerialRun child ordered rfl trace
          | true =>
            obtain ⟨split, cursor, entry, reports⟩ := splits (RootOwnerMir.Slice.layout [] r) score handed (by simpa [ran, RootOwnerMir.Slice.layout, handed] using answer)
            change split.entry = r.snapshot.state at entry
            have outcome : r.outcome = .handoff split.entry := by simp [RootOwnerMir.Result.outcome, handed, entry]
            rw [outcome] at trace
            exact ExecutionWithUses.ofSerialHandoff child first ordered rfl split cursor trace reports
        next h => rw [available] at h; cases h
  next =>
    exact PvEntryMir.exitExecutionWith mc fc b depth paired checked mem alpha beta passCall full passes answer
      (fun m legal wipes => .wipeout m legal wipes)

end NeuralReversi.PvsParallel.PvSerialMir
