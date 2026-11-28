module
public import NeuralReversi.EndgameSearch.CacheOwner
public import NeuralReversi.EndgameSearch.CacheMir.BudgetProof
public import NeuralReversi.EndgameSearch.CacheHistory.Boundary

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory

open EndgameTreeCache (Address Store Table soundHistory)
open CacheState (Cells represents)
open RecursionCacheMir (Reads Steps)

structure Local where
  history : List Store
  cells : Cells

structure Good (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (width : Nat) (sizes : Table → Nat)
    (key : Board.Board → BitVec 64) (localState : Local) : Prop where
  valid : soundHistory mc fc localState.history
  shape : ∀ t, (localState.cells t).size = sizes t
  arrays : represents (CacheMir.address width sizes key) localState.history localState.cells

theorem Good.reads (g : Good mc fc width sizes key state) :
    Reads width key (CacheMir.address width sizes key) state.history state.cells := by
  intro t b a
  have atSlot := g.arrays t b
  simp only [CacheMir.address] at atSlot
  simp [CacheMir.probe,g.shape,atSlot,CacheMir.probeCell_eq]

private theorem finalCells (fit : CacheMir.Fits width sizes)
    (shape : ∀ t, (cells t).size = sizes t)
    (run : Steps width key (CacheMir.address width sizes key) past cells writes final) :
    final = writes.foldl (CacheState.write (CacheMir.address width sizes key)) cells := by
  induction run with
  | done => rfl
  | @more past cells item next ws final reads stored tail ih =>
    have h := CacheMir.write_eq (hash:=key) item shape fit
    rw [h] at stored
    cases Option.some.inj stored
    apply ih
    intro t
    by_cases same : t = item.table <;> simp [CacheState.write,same,shape]

private theorem localRuns (i : Input) (g : Good mc fc width sizes key state)
    (fit : CacheMir.Fits width sizes) :
    let addr := CacheMir.address width sizes key
    let out := i.run mc fc addr state.history
    soundHistory mc fc out.history ∧
      ∃ writes final, state.history ++ writes = out.history ∧ Steps width key addr state.history state.cells writes final := by
  let addr := CacheMir.address width sizes key
  have safe := ShallowMir.scoreThreshold i.alpha
  unfold Input.run
  cases arm : i.arm with
  | failed | table | fixed => exact ⟨g.valid,[],state.cells,List.append_nil _,.done g.reads⟩
  | shallow =>
    have eq := ShallowMir.agrees i.target i.bmi2 mc fc addr i.board i.empties _ state.history i.aligned safe g.valid
    have sound := (contract_proven mc fc (orderedOptions i.target i.bmi2 mc fc addr) .shallow i.board _
      state.history (orderedOptions_correct _ _ _ _ _) g.valid (fun _ _ => safe)).2.2.1
    refine ⟨?_,
      RecursionCacheMir.shallow i.target i.bmi2 mc fc width sizes key fit i.board i.empties _ state.history
        state.cells i.aligned safe g.valid g.shape g.arrays⟩
    change soundHistory mc fc (ShallowMir.run i.target i.bmi2 mc fc addr i.board i.empties _ state.history).history
    simpa only [eq] using sound
  | deep poll =>
    let input := PvsParallel.DeepLeafMir.pollInput (i.state state.history) poll
    have initial : input.history = state.history := by cases poll <;> rfl
    have aligned : input.empties.matches i.board := by cases poll <;> exact i.aligned
    have valid : soundHistory mc fc input.history := initial.symm ▸ g.valid
    have proj := DeepRecursionMir.returned_score i.target i.bmi2 i.paired mc fc addr i.board _ input aligned safe valid
    have trace := RecursionCacheMir.deep i.target i.bmi2 i.paired mc fc width sizes key fit i.board input _
      state.cells aligned safe valid g.shape (initial.symm ▸ g.arrays)
    rw [initial] at trace
    exact ⟨proj.2.1,trace⟩

def Input.after (i : Input) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (addr : Address) (s : Local) : Local :=
  let after := (i.run mc fc addr s.history).history
  ⟨after,(after.drop s.history.length).foldl (CacheState.write addr) s.cells⟩

theorem after_call (i : Input) (g : Good mc fc width sizes key state) (fit : CacheMir.Fits width sizes) :
    let addr := CacheMir.address width sizes key
    let final := i.after mc fc addr state
    Good mc fc width sizes key final ∧
      ∃ ws, state.history ++ ws = final.history ∧ Steps width key addr state.history state.cells ws final.cells := by
  let addr := CacheMir.address width sizes key
  obtain ⟨valid,ws,last,grows,steps⟩ := localRuns i g fit
  have same := finalCells fit g.shape steps
  have rest : ((i.run mc fc addr state.history).history.drop state.history.length) = ws := by
    rw [← grows]; simp
  have finalEq : (i.after mc fc addr state).cells = last := by
    change List.foldl _ _ ((i.run mc fc addr state.history).history.drop _) = _
    rw [rest,same]
  refine ⟨⟨valid,?_,?_⟩,ws,grows,finalEq.symm ▸ steps⟩
  · rw [finalEq,same]
    intro t
    have keep (xs : List Store) (start : Cells) : (xs.foldl (CacheState.write addr) start t).size = (start t).size := by
      induction xs generalizing start with
      | nil => rfl
      | cons w more ih =>
        simp only [List.foldl_cons, ih]
        by_cases eq : t = w.table <;> simp [CacheState.write,eq]
    exact (keep ws _).trans (g.shape t)
  · rw [finalEq,same]
    change represents addr (i.run mc fc addr state.history).history _
    rw [← grows]
    exact CacheState.replay_represents g.arrays ws

end NeuralReversi.EndgameSearch.CacheHistory
