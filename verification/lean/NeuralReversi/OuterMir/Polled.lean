module
public import NeuralReversi.OuterMir.PolledProgram
public import NeuralReversi.OuterMir.Completion
public import NeuralReversi.Threading.SearchAbort.Persistent
public import NeuralReversi.AspirationMir.SerialTable.InterruptedPublic
public import NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt.Mir

@[expose] public section

namespace NeuralReversi.OuterMir.Polled

open PublicationMir (Iteration SliceSort SearchResult)
open RootMoveStorage (Row)
open Threading.AtomicWord (Write Read)
open Threading.SearchAbort.Persistent (ReadPair Update observed)

/-- The fields observed by snapshot publication at the native loop cutpoints.
    Window arithmetic, callbacks and stop observations have no snapshot writes;
    their branch choices remain explicit in the generated transition tables. -/
structure View where
  completed : Completion.State
  rows : List Row
  pvIndex : Nat
  depth : Nat
  level : Nat
  result : Option SearchResult

/-- A query supplies its actual live rows, even when they are only partially
    searched. No numerical assumption is made about these rows on an abort.
    The fallback is resolved separately from the post-query physical table. -/
structure Inputs where
  sort : SliceSort
  queryRows : List Row
  pvIndex : Nat
  level : Nat
  nodes : Nat
  fallback : Unit → SearchResult

def effect (input : Inputs) (state : View) : Effect → Option View
  | .beginPhase => some {state with completed := Completion.beginPhase state.completed}
  | .prepareEndgame => some {state with level := 5}
  | .setPvIndex => some {state with pvIndex := input.pvIndex}
  | .setLevel => some {state with level := input.level}
  | .query => some {state with rows := input.queryRows}
  | .sortRows => some {state with rows := RootMoveStorage.sortRows state.rows state.pvIndex}
  | .lineCompleted => do
    let (saved, _) ← Completion.lineCompleted input.sort state.rows state.pvIndex state.depth state.level true state.completed
    return {state with completed := saved}
  | .iterationCompleted => do
    let (saved, _) ← Completion.iterationCompleted input.sort state.rows state.depth state.level true state.completed
    return {state with completed := saved}
  | .returnResult => do
    let result ← Completion.intoResult state.completed input.nodes input.fallback
    return {state with result := some result}
  | .nextWindow | .pollAbort | .recordAbort | .clockCheck | .timeCheck => some state

def execute (input : Inputs) : List Effect → View → Option View
  | [], state => some state
  | action :: rest, state => do
    let next ← effect input state action
    execute input rest next

def runStep (input : Inputs) (step : Option Transition) (state : View) : Option (Cutpoint × View) := do
  let transition ← step
  let after ← execute input transition.effects state
  return (transition.next, after)

/-- Compose exactly the two native cutpoint segments. A level which continues
    its inner loop has not reached the after-line poll and cannot take it here. -/
def afterLevel (input : Inputs) (reportable beforeAbort clockStop afterAbort timeout : Bool)
    (state : View) : Option (Cutpoint × View) := do
  let (next, middle) ← runStep input (levelStep true false reportable beforeAbort clockStop) state
  if next = .afterLine then runStep input (afterLineStep afterAbort timeout) middle else none

private theorem aborted_level (reportable clock : Bool) :
    levelStep true false reportable true clock =
      some ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .recordAbort]⟩ := by
  cases reportable <;> cases clock <;> rfl

private theorem aborted_after (timeout : Bool) :
    afterLineStep true timeout = some ⟨.returned, [.pollAbort, .returnResult]⟩ := by
  cases timeout <;> rfl

/-- Neither segment sorts or captures the live rows after the abort. The
    complete saved state is retained, and its extracted return body supplies
    the result. This equality permits arbitrary poisoned live query rows. -/
theorem stopped_path (input : Inputs) (state : View) (reportable clock timeout : Bool) :
    afterLevel input reportable true clock true timeout state =
      (Completion.intoResult state.completed input.nodes input.fallback).map fun result =>
        (.returned, {state with rows := input.queryRows, level := input.level, result := some result}) := by
  simp only [afterLevel, aborted_level, aborted_after, runStep, execute, effect,
    bind, Option.bind, pure, Pure.pure, ↓reduceIte]
  cases Completion.intoResult state.completed input.nodes input.fallback <;> rfl

/-- Same-request coherence of the global abort word justifies collapsing the
    two native abort reads to the immediate-stop branch of the serial model.
    This does not identify a helper-local abort or a split cutoff with a global
    abort-word observation. -/
def AbortContract : Prop :=
  ∀ (input : Inputs) (state : View) reportable clock timeout
    (writes : List Write) (early late : Read writes),
    ReadPair writes early late →
    (∀ (a b : Fin writes.length), early.source.val ≤ a.val → a.val + 1 = b.val → b.val ≤ late.source.val →
      Update (writes.get a).value (writes.get b).value) →
    observed (writes.get early.source).value = true →
    afterLevel input reportable (observed (writes.get early.source).value) clock
      (observed (writes.get late.source).value) timeout state =
        (Completion.intoResult state.completed input.nodes input.fallback).map fun result =>
          (.returned, {state with rows := input.queryRows, level := input.level, result := some result})

theorem abort_contract_proven : AbortContract := by
  intro input state reportable clock timeout writes early late reads active stopped
  have later := Threading.SearchAbort.Persistent.contract_proven writes early late reads active stopped
  rw [stopped, later]
  exact stopped_path input state reportable clock timeout

private theorem completed_level (clock : Bool) :
    levelStep true false true false clock = some ⟨if clock then .afterLine else .level,
      [.setLevel, .query, .nextWindow, .pollAbort, .sortRows, .lineCompleted, .clockCheck]⟩ := by
  cases clock <;> rfl

/-- The generated non-aborted path uses the actual suffix sort and stamp gate.
    With one PV line this is precisely the immediate capture used by the
    serial publication driver, rather than an assumed matching snapshot. -/
theorem single_completed (input : Inputs) (state : View) (clock : Bool)
    (single : state.completed.marks.length = 1) (index : state.pvIndex = 0)
    (nonempty : input.queryRows ≠ []) :
    let rows := RootMoveStorage.sortRows input.queryRows 0
    runStep input (levelStep true false true false clock) state =
      some (if clock then .afterLine else .level,
        {state with
          rows := rows
          level := input.level
          completed := ⟨[some (state.depth, input.level)],
            some (PublicationMir.capture input.sort rows state.depth input.level true)⟩}) := by
  intro rows
  have present : rows ≠ [] := by
    intro empty
    have size := (RootMoveStorage.sortRowsContract_proven input.queryRows 0).1.length_eq
    change rows.length = input.queryRows.length at size
    rw [empty] at size
    exact nonempty (List.length_eq_zero_iff.mp size.symm)
  obtain ⟨best,captured⟩ := Completion.single_line input.sort rows state.depth input.level true
    state.completed single present
  simp only [runStep,completed_level,execute,effect,bind,Option.bind,pure,Pure.pure,index]
  rw [captured]

private theorem final_line (reportable : Bool) :
    lineStep false reportable = some ⟨.returned,
      if reportable then [.iterationCompleted, .returnResult] else [.returnResult]⟩ := by
  cases reportable <;> rfl

/-- A normal loop exit may repeat the last line's capture. The live rows and
    metadata are identical, so every returned field remains the same. -/
theorem normal_recapture (input : Inputs) (state : View) (reportable : Bool)
    (saved : state.completed.last =
      some (PublicationMir.capture input.sort state.rows state.depth state.level true)) :
    (runStep input (lineStep false reportable) state).bind (fun out => out.2.result) =
      Completion.intoResult state.completed input.nodes input.fallback := by
  cases reportable with
  | false =>
    simp only [final_line,Bool.false_eq_true,↓reduceIte,runStep,execute,effect,bind,Option.bind,pure,Pure.pure]
    cases Completion.intoResult state.completed input.nodes input.fallback <;> rfl
  | true =>
    have same := Completion.recapture_same input.sort state.rows state.depth state.level true
      state.completed input.nodes input.fallback saved
    rw [← same]
    simp only [final_line,↓reduceIte,runStep,execute,effect,bind,Option.bind,pure,Pure.pure]
    cases Completion.iterationCompleted input.sort state.rows state.depth state.level true state.completed with
    | none => rfl
    | some out =>
      rcases out with ⟨next,best⟩
      dsimp only
      cases Completion.intoResult next input.nodes input.fallback <;> rfl

/-- Stop-path exactness uses the saved numerical invariant already produced by
    the serial driver and the post-query physical table. It never assumes an
    exact value for the current live rows. -/
theorem stopped_ensures (t : PvsParallel.SerialTable.Table l) (mc fc) (b : PvsBoardTree.Position)
    (evaluate : Board.Board → Int) (sort : SliceSort) (state : View) (queryRows : List Row)
    (level nodes : Nat) (reportable clock timeout : Bool) (result : SearchResult) (after : View)
    (good : t.Good mc fc)
    (saved : AspirationMir.SerialTable.InterruptedPublic.SavedReady mc fc b state.completed.last)
    (run : afterLevel ⟨sort,queryRows,state.pvIndex,level,nodes,
        fun _ => PublicationMir.SerialFallback.fallback t mc fc b evaluate nodes⟩
      reportable true clock true timeout state = some (.returned,after))
    (returned : after.result = some result) :
    after.completed = state.completed ∧ PublicationMir.ExactScore.Ensures mc fc b result := by
  rw [stopped_path] at run
  obtain ⟨actual,built,eq⟩ := Option.map_eq_some_iff.mp run
  have shape := (Prod.mk.inj eq).2
  rw [← shape] at returned
  simp only at returned
  have sameResult := Option.some.inj returned
  rw [← sameResult]
  refine ⟨by rw [← shape],?_⟩
  exact PublicationMir.SerialFallback.intoResult_ensures t mc fc b evaluate sort state.completed.last
    nodes actual good saved.2 built

/-- Reuse an executed serial loop's theorem to supply the saved invariant.
    There is no caller-supplied root-score bound or completion certificate.
    `table` is the valid physical table at the native return cutpoint. -/
theorem stopped_after_loop (p : EndgameSearch.CacheHistory.Pool) (c : PvsParallel.SerialScout.Config)
    (mc fc) (b : PvsBoardTree.Position) (ops : AspirationMir.SerialTable.FullLadder.Inputs mc)
    (minimum : Nat) (controls : AspirationMir.SerialTable.InterruptedPublic.Controls)
    (sort : SliceSort) (levels : List TranspositionTable.Selectivity)
    (start : AspirationMir.SerialTable.SelectiveLadder.Start (PvsParallel.SerialTable.Memory l))
    (out : AspirationMir.SerialTable.InterruptedPublic.Finished l)
    (address : c.address = p.address)
    (initial : start.valid mc b (AspirationMir.SerialTable.Repeated.ReadyMemory p mc fc))
    (searched : AspirationMir.SerialTable.InterruptedPublic.loop p c b ops fc minimum controls sort
      levels start none = some out)
    (table : PvsParallel.SerialTable.Table l) (good : table.Good mc fc)
    (evaluate : Board.Board → Int) (state : View) (retained : state.completed.last = out.last)
    (queryRows : List Row) (level nodes : Nat) (reportable clock timeout : Bool)
    (after : View) (result : SearchResult)
    (run : afterLevel ⟨sort,queryRows,state.pvIndex,level,nodes,
        fun _ => PublicationMir.SerialFallback.fallback table mc fc b evaluate nodes⟩
      reportable true clock true timeout state = some (.returned,after))
    (returned : after.result = some result) :
    after.completed = state.completed ∧ PublicationMir.ExactScore.Ensures mc fc b result := by
  obtain ⟨actual,ran,_,saved⟩ := AspirationMir.SerialTable.InterruptedPublic.loop_returns p c b ops fc
    minimum controls sort levels start none address initial AspirationMir.SerialTable.InterruptedPublic.saved_none
  rw [searched] at ran
  cases Option.some.inj ran
  exact stopped_ensures table mc fc b evaluate sort state queryRows level nodes reportable clock timeout
    result after good (retained ▸ saved) run returned

private theorem saved_returns (last : Option Iteration) (nodes : Nat)
    (fallback : Unit → SearchResult) (nonempty : ∀ it, last = some it → it.rows ≠ []) :
    ∃ result, PublicationMir.intoResult last nodes fallback = some result := by
  cases last with
  | none => exact ⟨fallback (),rfl⟩
  | some it =>
    obtain ⟨head,tail,rows⟩ := List.exists_cons_of_ne_nil (nonempty it rfl)
    exact ⟨PublicationMir.build it.rows head it.depth it.selectivity it.endgame nodes,
      by simp [PublicationMir.intoResult,PublicationMir.first,rows]⟩

/-- The saved snapshot comes from an executed serial loop. The next attempt
    starts with that loop's memory, constructs its own MIR witness, and leaves
    the very table used by fallback. This supplies both numerical invariants
    without an independent good-table or saved-row-score premise. The two
    true global observations are justified separately by `AbortContract`;
    a child-stop marker alone is not asserted to imply either observation. -/
theorem stopped_from_execution (p : EndgameSearch.CacheHistory.Pool) (c : PvsParallel.SerialScout.Config)
    (mc fc) (b : PvsBoardTree.Position) (ops : AspirationMir.SerialTable.FullLadder.Inputs mc)
    (minimum : Nat) (controls : AspirationMir.SerialTable.InterruptedPublic.Controls)
    (sort : SliceSort) (levels : List TranspositionTable.Selectivity)
    (start : AspirationMir.SerialTable.SelectiveLadder.Start (PvsParallel.SerialTable.Memory l))
    (out : AspirationMir.SerialTable.InterruptedPublic.Finished l)
    (address : c.address = p.address)
    (initial : start.valid mc b (AspirationMir.SerialTable.Repeated.ReadyMemory p mc fc))
    (searched : AspirationMir.SerialTable.InterruptedPublic.loop p c b ops fc minimum controls sort
      levels start none = some out)
    (current : TranspositionTable.Selectivity) (stops : AspirationMir.SerialTable.InterruptedAttempt.Stops)
    (machine : AspirationMir.Machine) (rows : List Row)
    (cover : PvsParallel.Root.Covers mc b (rows.map Row.square)) (positive : 0 < rows.length)
    (entry : machine.pc = 0)
    (window : Aspiration.Window machine.regs.alpha.toInt machine.regs.beta.toInt)
    (evaluate : Board.Board → Int) (marks : List (Option (Nat × Nat))) (queryRows : List Row)
    (nodes : Nat) (reportable clock timeout : Bool) :
    ∃ witness : AspirationMir.SerialTable.InterruptedAttempt.Mir.Witness
        p c b ops fc current stops machine rows out.memory,
      let state : View := ⟨⟨marks,out.last⟩,rows,0,(Board.emptyCount b.val).toNat,current.val,none⟩
      let input : Inputs := ⟨sort,queryRows,0,current.val,nodes,
        fun _ => PublicationMir.SerialFallback.fallback witness.last.after.tt mc fc b evaluate nodes⟩
      ∃ result, afterLevel input reportable true clock true timeout state =
        some (.returned,{state with rows := queryRows,level := current.val,result := some result}) ∧
          PublicationMir.ExactScore.Ensures mc fc b result := by
  obtain ⟨actual,ran,ready,saved⟩ := AspirationMir.SerialTable.InterruptedPublic.loop_returns p c b ops fc
    minimum controls sort levels start none address initial AspirationMir.SerialTable.InterruptedPublic.saved_none
  have same := Option.some.inj (ran.symm.trans searched)
  rw [same] at ready saved
  obtain ⟨witness⟩ := AspirationMir.SerialTable.InterruptedAttempt.Mir.returns p c b ops fc current stops
    machine rows out.memory address ready cover positive entry window
  let state : View := ⟨⟨marks,out.last⟩,rows,0,(Board.emptyCount b.val).toNat,current.val,none⟩
  let input : Inputs := ⟨sort,queryRows,0,current.val,nodes,
    fun _ => PublicationMir.SerialFallback.fallback witness.last.after.tt mc fc b evaluate nodes⟩
  obtain ⟨result,built⟩ := saved_returns out.last nodes input.fallback saved.1
  refine ⟨witness,result,?_,?_⟩
  · change afterLevel input reportable true clock true timeout state = _
    change Completion.intoResult state.completed input.nodes input.fallback = some result at built
    rw [stopped_path,built]
    rfl
  · exact PublicationMir.SerialFallback.intoResult_ensures witness.last.after.tt mc fc b evaluate sort
      out.last nodes result witness.safe.1.1.1 saved.2 built

/-- This branch is intentionally present in the generated table. If a reset
    is admitted between the polls, the after-line check can continue to the
    normal final capture; persistence is therefore a necessary boundary. -/
theorem cleared_abort_continues (input : Inputs) (state : View) (reportable clock : Bool) :
    afterLevel input reportable true clock false false state =
      some (.line, {state with rows := input.queryRows, level := input.level}) := by
  simp only [afterLevel,aborted_level,runStep,execute,effect,bind,Option.bind,pure,Pure.pure,↓reduceIte]
  rfl

/-- Complete that inconsistent-poll branch at the final native line cutpoint.
    This helper is used only to exhibit the boundary, not by the sound path. -/
def clearedExit (input : Inputs) (state : View) : Option (Cutpoint × View) := do
  let (next,middle) ← afterLevel input true true false false false state
  if next = .line then runStep input (lineStep false true) middle else none

/-- A concrete score-field counterexample if unrelated Boolean polls are
    accepted as one active native request: an aborted live +64 replaces the
    previously saved +4. It is not a counterexample to abort-word persistence. -/
theorem cleared_abort_publishes_live_score (sort : SliceSort) :
    let old : Row := ⟨10,1024,0,0,[10]⟩
    let live : Row := ⟨10,16384,0,0,[10]⟩
    let state : View := ⟨⟨[none],some ⟨[old],12,5,true⟩⟩,[old],0,12,5,none⟩
    let input : Inputs := ⟨sort,[live],0,5,77,fun _ => .noLegalMove⟩
    ((clearedExit input state).bind (fun out => out.2.result)).bind PublicationMir.ExactScore.scoreIfExact = some 16384 ∧
      (Completion.intoResult state.completed input.nodes input.fallback).bind
        PublicationMir.ExactScore.scoreIfExact = some 1024 := by
  intro old live state input
  have sorted : sort.byReverseKey PublicationMir.key [live] = [live] :=
    List.perm_singleton.mp (sort.permutation PublicationMir.key [live])
  simp [clearedExit,cleared_abort_continues,runStep,final_line,execute,effect,
    Completion.iterationCompleted,Completion.intoResult,PublicationMir.record,PublicationMir.capture,
    PublicationMir.snapshot,PublicationMir.first,PublicationMir.intoResult,PublicationMir.build,
    PublicationMir.ExactScore.scoreIfExact,input,state,live,old,sorted]

end NeuralReversi.OuterMir.Polled
