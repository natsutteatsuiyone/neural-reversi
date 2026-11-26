import NeuralReversi.PvsParallel.SerialTermination
import NeuralReversi.PvsParallel.NativeFrames.Returns

namespace NeuralReversi.PvsParallel.SerialNativeRows

open SerialCalls (Program Request Tree Executes)
open SerialCalls.ScoreWords (encode Finite)
open SerialExecution (Frame Inputs ReadFacts body push)
open SerialTermination (Entry Allowed Hints)
open NativeFrames (Args Row frameKey)
open Aspiration (Window)

def arguments (q : Request) (line : Option Nat) : Args :=
  ⟨q.board.val, q.depth, encode q.alpha, encode q.beta, true, Cutoffs.exactSelectivity, q.pv, q.cut, line⟩

def returnedFields (q : Request) (word : Int) : FrameReturns.OwnerCalls.ChildReturn :=
  ⟨q.board.val, q.depth.toNat, q.pv, q.cut, q.alpha, q.beta, word⟩

/-- The opening and score/leave steps here are positions in a projection of PVS events.
    The separate issue number in the execution tree labels the stack activation. -/
structure Recorded (q : Request) (word : Int) (line : Option Nat)
    (size worker opening parent scored left : Nat) (saved : Row) : Prop where
  create_finish : ∃ raw, (arguments q line).row size worker opening parent scored (encode word) = some raw ∧
    raw.finish left (encode word) = some saved
  scoreStep : saved.frame.scoreStep = scored
  decoded : NativeFrames.Returns.fromRow saved = some ⟨worker, opening, frameKey size worker opening,
    left, parent, line, returnedFields q word⟩

theorem record {q : Request} (full : ExactDepth.Full q.board q.depth) (w : Window q.alpha q.beta)
    (null : q.pv = false → q.beta = q.alpha+1) (finite : Finite word) (line : Option Nat)
    (size worker opening parent scored left : Nat) (later : scored < left) :
    ∃ saved, Recorded q word line size worker opening parent scored left saved := by
  have ⟨lo, ordered, hi⟩ := w
  have ⟨low, high⟩ := finite
  have alpha : (encode q.alpha).raw.toInt = q.alpha := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have beta : (encode q.beta).raw.toInt = q.beta := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have val : (encode word).raw.toInt = word := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have legal : q.board.val.player &&& q.board.val.opponent = 0 := q.board.property
  have created : ∃ r, (arguments q line).row size worker opening parent scored (encode word) = some r ∧
      r.frame.scoreStep = scored ∧ r.frame.returnStep = none ∧ r.frame.score = word ∧
      r.opening = opening ∧ r.frame.worker = worker ∧ r.frame.id = frameKey size worker opening ∧
      r.frame.parent = parent ∧ r.rootLine = line ∧ FrameReturns.OwnerCalls.fields r.frame = returnedFields q word := by
    unfold NativeFrames.Args.row
    dsimp only
    split
    · split
      · refine ⟨_,rfl,rfl,rfl,val,rfl,rfl,rfl,rfl,rfl,?_⟩
        simp [FrameReturns.OwnerCalls.fields, returnedFields, TraceCheck.FrameReturn.board, arguments, alpha, beta, val]
      next denied =>
        exfalso
        apply denied
        change q.depth = Board.emptyCount q.board.val ∧ -32767 ≤ (encode q.alpha).raw.toInt ∧
          (encode q.alpha).raw.toInt < (encode q.beta).raw.toInt ∧ (encode q.beta).raw.toInt ≤ 32767 ∧
          (q.pv = false → (encode q.beta).raw.toInt = (encode q.alpha).raw.toInt+1)
        rw [alpha,beta]
        exact ⟨full,lo,ordered,hi,null⟩
    next invalid => exact (invalid legal).elim
  obtain ⟨r, got, time, absent, score, opened, owner, key, caller, rootLine, fields⟩ := created
  have after : r.frame.scoreStep < left := time ▸ later
  have finished : ∃ saved, r.finish left (encode word) = some saved ∧ saved.frame.scoreStep = scored ∧
      NativeFrames.Returns.fromRow saved = some ⟨worker, opening, frameKey size worker opening,
        left, parent, line, returnedFields q word⟩ := by
    unfold NativeFrames.Row.finish
    rw [dite_eq_left after, ite_eq_left (by simp [absent, score, val])]
    exact ⟨_, rfl, time, by simpa [NativeFrames.Returns.fromRow, FrameReturns.OwnerCalls.fields, opened, owner, key, caller, rootLine] using fields⟩
  obtain ⟨saved, done_, time', returned⟩ := finished
  exact ⟨saved, ⟨⟨r,got,done_⟩,time',returned⟩⟩

/-- One entry, one score and one normal return per issued PVS call, in stack order.
    Rows are listed in return order and paired with their original issue numbers. -/
inductive Rows (size worker : Nat) : Nat → Nat → List (Tree Request) → Nat → List (Nat × Row) → Prop
  | nil : Rows size worker parent clock [] clock []
  | cons (inside : Rows size worker (frameKey size worker openStep) (openStep+1) children scored inner)
      (saved : Recorded q word none size worker openStep parent scored (scored+1) row)
      (later : Rows size worker parent (scored+2) rest stop remaining) :
      Rows size worker parent openStep (.node issue q word children :: rest) stop
        (inner ++ (issue,row) :: remaining)

private theorem all_rows (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs) (hints : Hints inputs)
    (ran : Executes (body c inputs) (push mc fc) i ctx code k word forest) (entry : Entry ctx)
    (live : Program.Ready (Allowed mc fc ctx) code) (size worker parent clock : Nat) :
    ∃ stop rows, Rows size worker parent clock forest stop rows := by
  induction ran generalizing parent clock with
  | pure => exact ⟨_,_,.nil⟩
  | @call ctx q child i k word children resume last out rest entered sub next ih tail =>
    cases live with
    | call allowed more =>
      obtain ⟨actual, pushed, fitted, _⟩ := SerialTermination.child_entry entry q allowed
      obtain rfl := Option.some.inj (pushed.symm.trans entered)
      obtain ⟨scored, inner, childRows⟩ := ih fitted (SerialTermination.frame_ready c inputs valid hints _ _ fitted)
        (frameKey size worker clock) (clock+1)
      have inv := SerialExecution.localRule c inputs valid _ _ _ _ sub.directCalls
        (sub.children _ (SerialExecution.localRule c inputs valid))
      have checked := inv fitted.1 fitted.2.1 fitted.2.2.1 fitted.2.2.2
      have finite := SerialCalls.ScoreWords.finite_of_replay fitted.2.2.1 fitted.2.2.2 checked
      obtain ⟨r, saved⟩ := record (allowed.1.full entry.2.1) allowed.2.1 allowed.2.2 finite none
        size worker clock parent scored (scored+1) (by omega)
      obtain ⟨stop, remaining, further⟩ := tail entry (more word) parent (scored+2)
      exact ⟨_,_,.cons childRows saved further⟩

theorem frame_rows (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs) (hints : Hints inputs)
    (ctx : Frame) (entry : Entry ctx)
    (ran : Executes (body c inputs) (push mc fc) i ctx (body c inputs i ctx) last word forest)
    (size worker parent clock : Nat) :
    ∃ stop rows, Rows size worker parent clock forest stop rows :=
  all_rows c inputs valid hints ran entry (SerialTermination.frame_ready c inputs valid hints i ctx entry)
    size worker parent clock

theorem Rows.elapsed (rows : Rows size worker parent clock forest stop emitted) :
    stop = clock + 3 * emitted.length := by
  induction rows with
  | nil => simp
  | cons _ _ _ inside later =>
    simp only [List.length_append, List.length_cons]
    omega

theorem Recorded.fields (rec : Recorded q word line size worker opening parent scored left row) :
    FrameReturns.OwnerCalls.fields row.frame = returnedFields q word := by
  obtain ⟨_, _, same⟩ := Option.map_eq_some_iff.mp rec.decoded
  exact congrArg NativeFrames.Returns.Returned.call same

mutual
  private def nodeReturns : Tree Request → List (Nat × FrameReturns.OwnerCalls.ChildReturn)
    | .node issue q word children => returnOrder children ++ [(issue, returnedFields q word)]

  /-- ABI fields in postorder: a nested result reaches its caller before the caller returns. -/
  def returnOrder : List (Tree Request) → List (Nat × FrameReturns.OwnerCalls.ChildReturn)
    | [] => []
    | t :: rest => nodeReturns t ++ returnOrder rest
end

theorem Rows.return_fields (written : Rows size worker parent clock forest stop rows) :
    (rows.map fun r => (r.1, FrameReturns.OwnerCalls.fields r.2.frame)) = returnOrder forest := by
  induction written with
  | nil => rfl
  | cons _ saved _ inside later =>
    simp only [List.map_append, List.map_cons, returnOrder, nodeReturns, inside, later, saved.fields,
      List.append_assoc, List.singleton_append]

theorem Rows.count (rows : Rows size worker parent clock forest stop emitted) :
    emitted.length = (SerialCalls.openingNumbers forest).length := by
  induction rows with
  | nil => rfl
  | cons _ _ _ inside later =>
    simp only [SerialCalls.openingNumbers, SerialCalls.Tree.openings, List.length_append, List.length_cons] at *
    omega

/-- The root row carries its Multi-PV line index. Nested PVS rows carry their own
    unfiltered score, and their leave steps precede the root score step. -/
theorem root_rows (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs) (hints : Hints inputs)
    (b : PvsBoardTree.Position) (root : SerialRecursion.RootInput) (initial : List RootMoveStorage.Row) (idx : Nat)
    (a z : BitVec 32) (covers : Root.Covers mc b (initial.map RootMoveStorage.Row.square))
    (within : idx < initial.length) (window : Window a.toInt z.toInt)
    (ran : Executes (body c inputs) (push mc fc) i (SerialRootExecution.context b a z)
      ((SerialRootExecution.attempt c inputs i b root initial idx a z).map AspirationMir.SmallRoot.Answer.value)
      last word forest) (size worker parent clock : Nat) :
    ∃ scored rows saved, Rows size worker (frameKey size worker clock) (clock+1) forest scored rows ∧
      Recorded (SerialRootExecution.context b a z).request word (some idx) size worker clock parent scored (scored+1) saved := by
  let ctx := SerialRootExecution.context b a z
  have entry : Entry ctx := by
    simp only [ctx, Entry, SerialRootExecution.context, ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b, rfl, window, by simp⟩
  have live := SerialRootCalls.frame_ready (fc := fc) (mem := (inputs i ctx).memory) (paired := c.paired)
    (key := root.key) (sort := root.sort) (pv := root.pvLine) covers within entry.2.1 window
  obtain ⟨scored, rows, written⟩ := all_rows c inputs valid hints ran entry (live.map AspirationMir.SmallRoot.Answer.value)
    size worker (frameKey size worker clock) (clock+1)
  obtain ⟨_, finiteSteps⟩ := ran.to_steps
  obtain ⟨_, _, _, _, _, _, _, _, _, sound⟩ :=
    SerialRootExecution.returns c inputs valid i b root initial idx a z covers within window finiteSteps
  obtain ⟨saved, returned⟩ := record entry.2.1 entry.2.2.1 entry.2.2.2 ⟨sound.1.lower,sound.1.upper⟩
    (some idx) size worker clock parent scored (scored+1) (by omega)
  exact ⟨_,_,_,written,returned⟩

end NeuralReversi.PvsParallel.SerialNativeRows
