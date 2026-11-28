module
public import NeuralReversi.AspirationMir.SerialStack
public import NeuralReversi.PvsParallel.SerialPhysicalContext.Visits

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialContext

open PvsParallel
open ContextPathMir (Restored leafState)
open PvsBoardTree (Position)
open SerialStack (Visit)

open SerialCalls (Request Tree Executes)
open SerialExecution (Frame)
open SerialPhysicalContext (Partition groups)

/-- The same child-return word that resumes the caller also identifies the output
    of the state-threaded leaf. A longer search partitions its calls by visited move. -/
def FrameResult (c : SerialScout.Config) (op : SerialExecution.Inputs mc fc) (i : Nat) (f : Frame)
    (forest : List (Tree Request)) (w : Int) : Prop :=
  let r := f.request
  let calls := forest.map Tree.direct
  if r.pv then
    match SerialPvCalls.action mc fc r.board r.depth c.paired (op i f).memory with
    | .continueSearch => Partition (SerialPvCalls.moveRequest r.board mc fc r.depth r.cut) r.beta
        (SerialRecursion.pvOrder r.board (op i f)) calls
    | _ => groups calls = calls.map (fun x => [x])
  else groups calls = calls.map (fun x => [x]) ∧ (r.depth.toNat ≤ 12 →
    forest = [] ∧ ∃ history, EndgameSearch.SerialLeafRestore.leaf c mc fc r.board f.empties
      (SerialCalls.ScoreWords.encode r.alpha) (SerialCalls.ScoreWords.encode r.beta) (op i f).cache =
      some ((w,history),f.empties))

inductive Frames (c : SerialScout.Config) (op : SerialExecution.Inputs mc fc) :
    Frame → List (Tree Request) → Prop
  | nil : Frames c op f []
  | cons (pushed : SerialExecution.push mc fc f r = some child)
      (result : FrameResult c op i child inside w) (nested : Frames c op child inside) (tail : Frames c op f rest) :
      Frames c op f (.node i r w inside :: rest)

private theorem frames (c : SerialScout.Config) (op : SerialExecution.Inputs mc fc)
    (ran : Executes (SerialExecution.body c op) (SerialExecution.push mc fc) i f code k word forest)
    (aligned : f.empties.matches f.request.board) (full : ExactDepth.Full f.request.board f.request.depth) :
    Frames c op f forest := by
  induction ran with
  | pure => exact .nil
  | @call parent r child i middle w inside resume last v rest entered sub _ down more =>
    obtain ⟨s, found, eq⟩ := Option.map_eq_some_iff.mp entered
    subst child
    have edge := SerialChildState.edge_of_nextState found
    obtain ⟨t, hit, fits⟩ := SerialChildState.nextState_matches edge aligned
    have equal : t = s := Option.some.inj (hit.symm.trans found)
    subst t
    refine .cons entered ?_ (down fits (edge.full full)) (more aligned full)
    let f' : Frame := ⟨r,s⟩
    have direct := sub.directCalls
    cases role : r.pv with
    | true =>
      simp only [FrameResult, role, ite_true]
      simp only [SerialExecution.body, role, ite_true] at direct
      exact SerialPhysicalContext.pv_groups (EndgameOrderMir.ordinary_perm _ _ _ _ _) direct
    | false =>
      simp only [FrameResult, role, Bool.false_eq_true, ite_false]
      refine ⟨SerialPhysicalContext.nonPv_groups op c (i+1) f' role direct, ?_⟩
      intro size
      simp only [SerialExecution.body, role, Bool.false_eq_true, ite_false, size, ite_true] at direct
      obtain ⟨value,noneIssued⟩ := SerialCalls.Program.runs_ofOption direct
      have physicalSize : s.count.toNat ≤ 12 := by rw [fits.1, ← edge.full full]; exact size
      have id := EndgameSearch.SerialLeafRestore.leaf_refines c mc fc r.board s
        (SerialCalls.ScoreWords.encode r.alpha) (SerialCalls.ScoreWords.encode r.beta) (op (i+1) f').cache fits physicalSize
      refine ⟨List.map_eq_nil_iff.mp noneIssued, ?_⟩
      rw [id] at value ⊢
      obtain ⟨pair, read, returned⟩ := Option.map_eq_some_iff.mp value
      obtain ⟨result, observed, eq⟩ := Option.map_eq_some_iff.mp read
      cases eq
      exact ⟨result.2, returned ▸ read⟩

def RootGroups (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (roots : Nat → Word → Word → SerialRecursion.RootInput)
    (v : Visit) (forest : List (Tree Request)) : Prop :=
  let s := v.state
  let root := roots s.index s.regs.alpha s.regs.beta
  Partition (SerialRootCalls.request mc fc b (ContextPathMir.rootDepth (ContextPathMir.start b))) s.regs.beta.toInt
    (SmallRoot.candidates mc fc b (s.rows.map RootMoveStorage.Row.square) 0 root.key root.sort) (forest.map Tree.direct)

/-- Carry the projected worker context through successive aspiration attempts.
    Each attempt uses the same call forest for its row decodes and its context operations. -/
inductive Attempts (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) :
    SearchContextState.State → List Visit → SearchContextState.State → Prop
  | nil : Attempts c b inputs roots s [] s
  | cons (rows : SerialStack.OnForest c b inputs roots v forest)
      (ordered : RootGroups mc fc b roots v forest)
      (moves : SerialPhysicalContext.Walk mc fc
        (SerialRootExecution.context b v.state.regs.alpha v.state.regs.beta) s forest t)
      (children : Frames c (inputs v.state.index)
        (SerialRootExecution.context b v.state.regs.alpha v.state.regs.beta) forest)
      (later : Attempts c b inputs roots t rest u) :
      Attempts c b inputs roots s (v :: rest) u

private theorem carry {mc fc} (c : SerialScout.Config) (b : Position) (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (visits : List Visit)
    (seen : ∀ v ∈ visits, SerialStack.Projected c b inputs roots v) (s : SearchContextState.State)
    (matchesRoot : SearchContextState.Matches s b) :
    ∃ t, Attempts c b inputs roots s visits t ∧ ContextPathMir.Calls mc fc b s t := by
  induction visits generalizing s with
  | nil => exact ⟨s,.nil,.done⟩
  | cons v rest ih =>
    obtain ⟨forest,observed⟩ := seen v List.mem_cons_self
    obtain ⟨t,walk,restored⟩ := SerialPhysicalContext.of_execution observed.1 s rfl matchesRoot
    obtain ⟨u,more,later⟩ := ih (fun w h => seen w (.tail _ h)) t (restored.agrees matchesRoot)
    have full : ExactDepth.Full b (ContextPathMir.rootDepth (ContextPathMir.start b)) := ContextPathMir.At.initial.full (mc := mc) (fc := fc)
    have aligned : (leafState b).matches b := EndgameSearch.emptyState_initial b
    exact ⟨u,.cons observed (SerialPhysicalContext.root_groups observed.2.2.1) walk (frames c _ observed.1 aligned full) more,.next walk.calls later⟩

def PublicationContract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (rows : List RootMoveStorage.Row) (m : Machine)
    (inputs : Nat → SerialExecution.Inputs mc fc) (roots : Nat → Word → Word → SerialRecursion.RootInput)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    (Board.emptyCount b.val).toNat ≤ 60 →
    Root.Covers mc b (rows.map RootMoveStorage.Row.square) → 0 < rows.length →
    (∀ i, SerialExecution.ReadFacts (inputs i)) → (∀ i, SerialTermination.Hints (inputs i)) →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ prior last result t, let search := SerialStack.trial c b inputs roots
      SerialStack.run search 12 (SerialStack.first m rows) = some (prior,last) ∧
      Attempts c b inputs roots (ContextPathMir.start b) (prior ++ [last]) t ∧ Restored (ContextPathMir.start b) t ∧
      PublicationMir.completed sort (RootMoveStorage.sortRows last.answer.rows 0)
        (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem publication : PublicationContract := by
  intro mc fc c b rows m inputs roots sort nodes fallback fits covers positive valid hints pc window
  obtain ⟨prior,last,result,ran,_,_,seen,_,_,_,published,score⟩ :=
    SerialStack.publicScore mc fc c b rows m inputs roots sort nodes fallback covers positive valid hints pc window
  have aligned : SearchContextState.Matches (ContextPathMir.start b) b := by
    rw [ContextPathMir.start_eq]
    exact (SearchContextState.initialContract_proven.1 b fits).1
  obtain ⟨t,attempts,path⟩ := carry c b inputs roots (prior ++ [last]) seen _ aligned
  exact ⟨prior,last,result,t,ran,attempts,path.restores aligned,published,score⟩

end NeuralReversi.AspirationMir.SerialContext
