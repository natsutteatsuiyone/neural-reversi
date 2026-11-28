module
public import NeuralReversi.PvsParallel.SerialPhysicalContext

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialPhysicalContext

open SerialCalls (MoveRequest Program Request)
open PvsBoardTree (GeneratedSquare Position)
open RootOwnerMir.Slice (Visit)
open SerialOwnerCalls (words)
open Bitboard.Moves (X86Config)

/-- Recover the intervals between a physical update and its undo by looking at
    the adjacent recursive calls of a single owner. -/
def groups : List (Request × Int) → List (List (Request × Int))
  | [] => []
  | [x] => [[x]]
  | x :: y :: rest =>
    if sameVisit x.1 y.1 then [x,y] :: groups rest else [x] :: groups (y :: rest)

theorem sameVisit_moves (b : Position) (mc : X86Config) (fc : Cutoffs.FlipConfig) (d : ExactDepth.Depth)
    (cut : Bool) (x y : MoveRequest (GeneratedSquare mc b.val)) :
    sameVisit (SerialPvCalls.moveRequest b mc fc d cut x) (SerialPvCalls.moveRequest b mc fc d cut y) = true ↔
    x.pv = false ∧ y.pv = true ∧ x.move = y.move := by
  simp only [sameVisit, SerialPvCalls.moveRequest, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq,
    and_true, and_assoc]
  constructor
  · rintro ⟨sc,wide,same⟩
    exact ⟨sc,wide,PvsBoardTree.play_injective mc fc b (congrArg Subtype.val same)⟩
  · rintro ⟨sc,wide,same⟩
    exact ⟨sc,wide,congrArg (fun m => (PvsBoardTree.play mc fc b m).val) same⟩

private theorem single (x : Request × Int) (rest : List (Request × Int))
    (separate : ∀ y ∈ rest, sameVisit x.1 y.1 = false) : groups (x :: rest) = [x] :: groups rest := by
  cases rest with
  | nil => rfl
  | cons y ys => rw [groups]; simp [separate _ List.mem_cons_self]

def block (adapt : MoveRequest Move → Request) (beta : Int) (v : Visit Move) :=
  (words beta v).map fun t => (adapt t.1, t.2)

private theorem word_move {v : Visit Move} (mem : x ∈ words beta v) : x.1.move = v.move := by
  cases sc : v.scout <;> cases w : v.wide <;> simp [words, sc, w] at mem <;> grind

private theorem split_blocks (adapt : MoveRequest Move → Request)
    (role : ∀ x y, sameVisit (adapt x) (adapt y) = true ↔ x.pv = false ∧ y.pv = true ∧ x.move = y.move)
    (vs : List (Visit Move)) (unique : (vs.map Visit.move).Nodup) (occupied : ∀ v ∈ vs, words beta v ≠ []) :
    groups (vs.flatMap (block adapt beta)) = vs.map (block adapt beta) := by
  induction vs with
  | nil => rfl
  | cons v rest ih =>
    obtain ⟨fresh, distinct⟩ := List.nodup_cons.mp unique
    have after := ih distinct fun t h => occupied t (.tail _ h)
    have cross (x : MoveRequest Move × Int) (current : x ∈ words beta v) (y) (later : y ∈ rest.flatMap (block adapt beta)) :
        sameVisit (adapt x.1) y.1 = false := by
      obtain ⟨v', vis, word⟩ := List.mem_flatMap.mp later
      obtain ⟨y, mem, rfl⟩ := List.mem_map.mp word
      apply Bool.eq_false_iff.mpr
      intro same
      have eq := (word_move current).symm.trans (((role _ _).mp same).2.2.trans (word_move mem))
      exact fresh (eq ▸ List.mem_map_of_mem vis)
    have has := occupied v List.mem_cons_self
    rw [List.flatMap_cons, List.map_cons, ← after]
    cases sc : v.scout <;> cases w : v.wide
    · simp [words, sc, w] at has
    · simp only [block, words, sc, w, Option.map_none, Option.map_some, Option.toList_none,
        Option.toList_some, List.nil_append, List.map_cons, List.map_nil, List.cons_append]
      rename_i word
      apply single
      exact cross (_,word) (by simp [words, sc, w])
    · simp only [block, words, sc, w, Option.map_none, Option.map_some, Option.toList_none,
        Option.toList_some, List.append_nil, List.map_cons, List.map_nil, List.cons_append, List.nil_append]
      rename_i word
      apply single
      exact cross (_,word) (by simp [words, sc, w])
    · simp only [block, words, sc, w, Option.map_some, Option.toList_some, List.cons_append, List.nil_append,
        List.map_cons, List.map_nil]
      rw [groups]
      simp [role]

/-- No two different source iterations can merge. Every visited move supplies
    one nonempty block and all callback words in that block are preserved. -/
theorem owner_groups {square : Move → Bitboard.Square} (adapt : MoveRequest Move → Request)
    (role : ∀ x y, sameVisit (adapt x) (adapt y) = true ↔ x.pv = false ∧ y.pv = true ∧ x.move = y.move)
    (ran : Program.Runs ((SerialOwnerCalls.run square beta moves s).mapRequests adapt) calls result) (unique : moves.Nodup) :
    groups calls = result.1.map (block adapt beta) ∧ (result.1.map Visit.move) <+: moves ∧
      (∀ v ∈ result.1, block adapt beta v ≠ []) := by
  obtain ⟨raw, owner, rfl⟩ := Program.runs_mapRequests ran
  obtain ⟨wordsEq, prefix_, _⟩ := SerialOwnerCalls.runs_visits owner
  have filled := SerialOwnerCalls.runs_nonempty owner
  refine ⟨?_, prefix_, ?_⟩
  · rw [wordsEq, List.map_flatMap]
    exact split_blocks adapt role _ (prefix_.sublist.nodup unique) filled
  · simpa only [block, ne_eq, List.map_eq_nil_iff] using filled

def Partition (adapt : MoveRequest Move → Request) (beta : Int) (moves : List Move) (calls : List (Request × Int)) : Prop :=
  ∃ visits : List (Visit Move), groups calls = visits.map (block adapt beta) ∧ (visits.map Visit.move) <+: moves ∧
    ∀ v ∈ visits, block adapt beta v ≠ []

theorem pv_groups {mc : X86Config} {fc : Cutoffs.FlipConfig} {b : Position}
    {moves : List (GeneratedSquare mc b.val)} (order : moves.Perm (PvsBoardTree.legalSquares mc b.val))
    (ran : Program.Runs (SerialPvCalls.frame mc fc b d paired checked cut mem a z moves) calls word) :
    match SerialPvCalls.action mc fc b d paired mem with
    | .continueSearch => Partition (SerialPvCalls.moveRequest b mc fc d cut) z moves calls
    | _ => groups calls = calls.map (fun t => [t]) := by
  unfold SerialPvCalls.frame at ran
  cases act : SerialPvCalls.action mc fc b d paired mem <;> rw [act] at ran
  case zero | terminal | wipeout =>
    obtain ⟨_,rfl⟩ := Program.runs_ofOption ran
    rfl
  case pass => cases ran with | call more => cases more; rfl
  case continueSearch =>
    obtain ⟨result, owner, _⟩ := Program.runs_map ran
    exact ⟨result.1, owner_groups _ (sameVisit_moves b mc fc d cut) owner
      (order.nodup_iff.mpr (EndgameOrderMir.generated_layout mc b).1)⟩

theorem root_groups {mc : X86Config} {fc : Cutoffs.FlipConfig} {b : Position}
    (ran : Program.Runs (SerialRootCalls.frame mc fc b d paired mem initial idx key sort pv a z) calls answer) :
    Partition (SerialRootCalls.request mc fc b d) z.toInt
      (AspirationMir.SmallRoot.candidates mc fc b (initial.map RootMoveStorage.Row.square) idx key sort) calls := by
  dsimp only [SerialRootCalls.frame] at ran
  split at ran
  · cases ran
  · cases ran
    exact ⟨[],rfl,List.nil_prefix,by simp⟩
  · obtain ⟨result, owner, _⟩ := Program.runs_map ran
    have ordered := (RootFrameMir.ordered_candidates mc fc b (initial.map RootMoveStorage.Row.square) idx key sort).1
    have unique := ordered.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc b).1.filter _)
    exact ⟨result.1, owner_groups _ (sameVisit_moves b mc fc d false) owner unique⟩

private theorem singletons (xs : List (Request × Int)) (scout : ∀ x ∈ xs, x.1.pv = false) :
    groups xs = xs.map (fun t => [t]) := by
  induction xs with
  | nil => rfl
  | cons x rest ih =>
    rw [single x rest, List.map_cons, ih (fun t h => scout t (.tail _ h))]
    intro y member
    simp [sameVisit, scout y (.tail _ member)]

/-- No PV call is issued in a non-PV frame, including a TT presearch and
    the subsequent reduced move list. Every callback is its own context update. -/
theorem nonPv_groups {mc : X86Config} {fc : Cutoffs.FlipConfig} (inputs : SerialExecution.Inputs mc fc)
    (c : SerialScout.Config) (i : Nat) (ctx : SerialExecution.Frame) (nonPv : ctx.request.pv = false)
    (ran : Program.Runs (SerialExecution.body c inputs i ctx) calls word) :
    groups calls = calls.map (fun t => [t]) := by
  by_cases small : ctx.request.depth.toNat ≤ 12
  · simp only [SerialExecution.body, nonPv, Bool.false_eq_true, ite_false, small, ite_true] at ran
    obtain ⟨_,rfl⟩ := Program.runs_ofOption ran
    rfl
  · simp only [SerialExecution.body, nonPv, Bool.false_eq_true, ite_false, small] at ran
    have low (m) (_ : m ∈ PvsBoardTree.legalSquares mc ctx.request.board.val) :=
      EndgameOrderMir.framePriority_low mc fc ctx.request.board (inputs i ctx).ordering (inputs i ctx).tie m
    have facts := SerialNonPvCalls.frame_children mc fc ctx.request.board ctx.request.depth _ _ c.paired c.checked
      ctx.request.cut (inputs i ctx).memory (inputs i ctx).probe (inputs i ctx).etc _ low ran
    apply singletons
    intro t h
    rcases facts.2 t h with ⟨m, eq⟩ | ⟨_, eq⟩ <;> rw [eq] <;> rfl

theorem root_segments (c : SerialScout.Config) (inputs : SerialExecution.Inputs mc fc) (b : Position)
    (root : SerialRecursion.RootInput) (initial : List RootMoveStorage.Row) (idx : Nat) (a z : BitVec 32)
    (fits : (Board.emptyCount b.val).toNat ≤ 60)
    (ran : SerialCalls.Executes (SerialExecution.body c inputs) (SerialExecution.push mc fc) i
      (SerialRootExecution.context b a z)
      ((SerialRootExecution.attempt c inputs i b root initial idx a z).map AspirationMir.SmallRoot.Answer.value)
      last word forest) :
    let ctx := SerialRootExecution.context b a z
    ∃ after, Walk mc fc ctx (ContextPathMir.start b) forest after ∧ ContextPathMir.Restored (ContextPathMir.start b) after ∧
      Partition (SerialRootCalls.request mc fc b ctx.request.depth) z.toInt
        (AspirationMir.SmallRoot.candidates mc fc b (initial.map RootMoveStorage.Row.square) idx root.key root.sort)
        (forest.map SerialCalls.Tree.direct) := by
  intro ctx
  obtain ⟨after, walk, restored⟩ := from_root b a z fits ran
  obtain ⟨_, frame, _⟩ := Program.runs_map ran.directCalls
  exact ⟨after,walk,restored,root_groups frame⟩

end NeuralReversi.PvsParallel.SerialPhysicalContext
