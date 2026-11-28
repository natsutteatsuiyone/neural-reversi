module
public import NeuralReversi.PvsParallel.SerialRootExecution

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialPhysicalContext

open SerialCalls (Executes Request Tree)
open SerialExecution (Frame)
open PvsBoardTree (Position GeneratedSquare play pass)
open ContextPathMir (Argument Calls Restored leafState)
open SearchContextState (State Matches)

def flips (fc : Cutoffs.FlipConfig) (b : Position) (sq : Bitboard.Square) :=
  Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent

def childBoard (fc : Cutoffs.FlipConfig) {b : Position} : Argument mc b → Position
  | .move m => (play mc fc b m).val
  | .forcedPass _ => pass b

def childDepth (depth : ExactDepth.Depth) : Argument mc b → ExactDepth.Depth
  | .move _ => depth-1
  | .forcedPass _ => depth

def enter (fc : Cutoffs.FlipConfig) {b : Position} (s : State) : Argument mc b → State
  | .move m => ContextPathMir.place s m.val (flips fc b m.val)
  | .forcedPass _ => ContextPathMir.passTurn s

def leave (s : State) : Argument mc b → State
  | .move _ => ContextPathMir.restore s
  | .forcedPass _ => ContextPathMir.unpass s

def sameVisit (sc wide : Request) : Bool :=
  !sc.pv && wide.pv && decide (sc.board = wide.board ∧ sc.depth = wide.depth)

def retryNext (sc : Request) : List (Tree Request) → Bool
  | .node _ wide _ _ :: _ => sameVisit sc wide
  | [] => false

structure Opened (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (caller : Frame) (q : Request)
    (arg : Argument mc caller.request.board) (s : State) : Prop where
  board : q.board = childBoard fc arg
  depth : q.depth = childDepth caller.request.depth arg
  pushed : SerialExecution.push mc fc caller q = some ⟨q,leafState (childBoard fc arg)⟩

/-- A scout and wide re-search of one placed move use the same child context.
    The wide call starts from the state left by the scout's entire subtree. -/
inductive Walk (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    Frame → State → List (Tree Request) → State → Prop
  | nil : Walk mc fc caller s [] s
  | single {caller : Frame} {m : GeneratedSquare mc caller.request.board.val}
      (opening : Opened mc fc caller q (.move m) s) (alone : retryNext q rest = false)
      (inside : Walk mc fc ⟨q,leafState (childBoard fc (.move m))⟩ (enter fc s (.move m)) children t)
      (more : Walk mc fc caller (leave t (.move m)) rest u) :
      Walk mc fc caller s (.node issue q word children :: rest) u
  | pair {caller : Frame} {m : GeneratedSquare mc caller.request.board.val}
      (scout : Opened mc fc caller q (.move m) s) (wide : Opened mc fc caller r (.move m) s)
      (same : sameVisit q r = true)
      (first : Walk mc fc ⟨q,leafState (childBoard fc (.move m))⟩ (enter fc s (.move m)) firstTree t)
      (second : Walk mc fc ⟨r,leafState (childBoard fc (.move m))⟩ t secondTree u)
      (more : Walk mc fc caller (leave u (.move m)) rest v) :
      Walk mc fc caller s (.node i q w firstTree :: .node j r x secondTree :: rest) v
  | pass {caller : Frame} {forced : Board.ForcedPass mc caller.request.board.val}
      (opening : Opened mc fc caller q (.forcedPass forced) s)
      (inside : Walk mc fc ⟨q,leafState (pass caller.request.board)⟩ (ContextPathMir.passTurn s) children t)
      (more : Walk mc fc caller (ContextPathMir.unpass t) rest u) :
      Walk mc fc caller s (.node i q w children :: rest) u

private theorem under {b : Position} {arg : Argument mc b}
    (nested : Calls mc fc (childBoard fc arg) (enter fc s arg) t) : Calls mc fc b s (leave t arg) := by
  cases arg with
  | move m => exact .move m nested
  | forcedPass forced => exact .forcedPass forced nested

theorem Walk.calls (walk : Walk mc fc caller s forest t) : Calls mc fc caller.request.board s t := by
  induction walk with
  | nil => exact .done
  | single opening _ _ _ inner tail => exact .next (under (opening.board ▸ inner)) tail
  | pair sc pv _ _ _ _ first second tail =>
    exact .next (under (.next (sc.board ▸ first) (pv.board ▸ second))) tail
  | pass opening _ _ inner tail =>
    rw [opening.board] at inner
    exact .next (.forcedPass (by assumption) inner) tail

private theorem enter_matches {b : Position} (arg : Argument mc b) (aligned : Matches s b) :
    Matches (enter fc s arg) (childBoard fc arg) := by
  cases arg with
  | move m => exact (SearchContextState.placementContract_proven _ _ _ m _ aligned).2.1
  | forcedPass _ => exact (SearchContextState.passContract_proven _ _ aligned).1

/-- The accepted push determines the unique placed square or the legal turn switch. -/
private theorem opened {caller : Frame} (projection : leafState caller.request.board = caller.empties)
    (pushed : SerialExecution.push mc fc caller q = some child) :
    ∃ arg : Argument mc caller.request.board, Opened mc fc caller q arg s ∧ child = ⟨q,leafState (childBoard fc arg)⟩ := by
  obtain ⟨es, updated, rfl⟩ := Option.map_eq_some_iff.mp pushed
  have select {b next : Position} {d cd : ExactDepth.Depth} (edge : ExactDepth.RecursiveCall mc fc b d next cd)
      (accepted : SerialChildState.nextState mc fc b d (leafState b) next cd = some es) :
      ∃ arg : Argument mc b, next = childBoard fc arg ∧ cd = childDepth d arg ∧ es = leafState (childBoard fc arg) := by
    cases edge with
    | ply m d =>
      exact ⟨.move m, rfl,rfl, (Option.some.inj (accepted.symm.trans (SerialChildState.nextState_play _ _ _ _ _ m))).trans
        (ContextPathMir.leafState_play mc fc b m)⟩
    | forcedPass d forced =>
      exact ⟨.forcedPass forced, rfl,rfl, (Option.some.inj (accepted.symm.trans (SerialChildState.nextState_pass _ _ _ _ _ forced))).trans
        (ContextPathMir.leafState_pass b).symm⟩
  rw [← projection] at updated
  obtain ⟨arg, board, depth, rfl⟩ := select (SerialChildState.edge_of_nextState updated) updated
  exact ⟨arg,⟨board,depth,by simp [SerialExecution.push, ← projection, updated]⟩,rfl⟩

private inductive Children (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    Frame → List (Tree Request) → Prop
  | nil : Children mc fc caller []
  | cons (entered : SerialExecution.push mc fc caller q = some child) (inside : Children mc fc child sub)
      (tail : Children mc fc caller rest) : Children mc fc caller (.node i q w sub :: rest)

private theorem issued (ran : Executes body (SerialExecution.push mc fc) i caller code last word forest) :
    Children mc fc caller forest := by
  induction ran with
  | pure => exact .nil
  | call entered _ _ down next => exact .cons entered down next

private theorem Opened.again {caller : Frame} {arg : Argument mc caller.request.board}
    (opened : Opened mc fc caller q arg s) (board : q.board = r.board) (depth : q.depth = r.depth) :
    Opened mc fc caller r arg s := by
  obtain ⟨state, accepted, eq⟩ := Option.map_eq_some_iff.mp opened.pushed
  have stateEq := congrArg Frame.empties eq
  refine ⟨board.symm.trans opened.board, depth.symm.trans opened.depth, ?_⟩
  simp only [SerialExecution.push, ← board, ← depth, accepted, Option.map_some]
  exact congrArg (some ∘ Frame.mk r) stateEq

private theorem weave {mc fc} {caller : Frame} (forest : List (Tree Request)) (children : Children mc fc caller forest)
    (s : State) (projection : leafState caller.request.board = caller.empties) (aligned : Matches s caller.request.board) :
    ∃ t, Walk mc fc caller s forest t := by
  cases f : forest with
  | nil => exact ⟨_,.nil⟩
  | cons head rest =>
    cases h : head with
    | node i q w sub =>
      rw [f,h] at children
      cases children with
      | cons pushed below tail =>
        obtain ⟨arg, opening, rfl⟩ := opened projection pushed
        have childFit : Matches (enter fc s arg) q.board := opening.board ▸ enter_matches arg aligned
        obtain ⟨t, down⟩ := weave sub below (enter fc s arg) (by rw [opening.board]) childFit
        cases arg with
        | forcedPass forced =>
          have atPass := down.calls
          rw [opening.board] at atPass
          have block : Calls mc fc caller.request.board s (ContextPathMir.unpass t) := .forcedPass forced atPass
          have back := block.restores aligned
          obtain ⟨u, after⟩ := weave rest tail _ projection (back.agrees aligned)
          exact ⟨u,.pass opening down after⟩
        | move m =>
          by_cases paired : retryNext q rest = true
          · cases rem : rest with
            | nil => simp [rem,retryNext] at paired
            | cons next end_ =>
              cases n : next with
              | node j r x sub' =>
                have same : sameVisit q r = true := by simpa [rem,n,retryNext] using paired
                have ⟨board, depth⟩ : q.board = r.board ∧ q.depth = r.depth := by
                  simpa [sameVisit, Bool.and_eq_true] using (Bool.and_eq_true_iff.mp same).2
                rw [rem,n] at tail
                cases tail with
                | cons accepted inside after =>
                  have wide := opening.again board depth
                  obtain rfl := Option.some.inj (accepted.symm.trans wide.pushed)
                  have restored := down.calls.restores childFit
                  obtain ⟨u, searched⟩ := weave sub' inside t (by rw [wide.board]) (board ▸ restored.agrees childFit)
                  have block : Calls mc fc caller.request.board s (leave u (.move m)) :=
                    under (.next (opening.board ▸ down.calls) (wide.board ▸ searched.calls))
                  have back := block.restores aligned
                  obtain ⟨v, finished⟩ := weave end_ after _ projection (back.agrees aligned)
                  exact ⟨v,.pair opening wide same down searched finished⟩
          · have block : Calls mc fc caller.request.board s (leave t (.move m)) := under (opening.board ▸ down.calls)
            have back := block.restores aligned
            obtain ⟨u, after⟩ := weave rest tail _ projection (back.agrees aligned)
            exact ⟨u,.single opening (Bool.eq_false_iff.mpr paired) down after⟩
termination_by sizeOf forest
decreasing_by all_goals simp_all +arith <;> omega

/-- The chronological recursive calls reconstruct one physical context path with
    a single update and undo for each scout/PV pair. -/
theorem of_execution (ran : Executes body (SerialExecution.push mc fc) i caller code last word forest)
    (s : State) (projection : leafState caller.request.board = caller.empties) (aligned : Matches s caller.request.board) :
    ∃ t, Walk mc fc caller s forest t ∧ Restored s t := by
  obtain ⟨t, walked⟩ := weave forest (issued ran) s projection aligned
  exact ⟨t, walked, walked.calls.restores aligned⟩

theorem from_root (b : Position) (a z : BitVec 32) (fits : (Board.emptyCount b.val).toNat ≤ 60)
    (ran : Executes body (SerialExecution.push mc fc) i (SerialRootExecution.context b a z) code last word forest) :
    ∃ t, Walk mc fc (SerialRootExecution.context b a z) (ContextPathMir.start b) forest t ∧
      Restored (ContextPathMir.start b) t := by
  apply of_execution ran _ rfl
  rw [ContextPathMir.start_eq]
  exact (SearchContextState.initialContract_proven.1 b fits).1

end NeuralReversi.PvsParallel.SerialPhysicalContext
