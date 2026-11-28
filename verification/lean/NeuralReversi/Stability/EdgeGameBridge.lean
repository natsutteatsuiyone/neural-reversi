module
public import NeuralReversi.Stability.EdgeTableRefinement
public import NeuralReversi.Stability.EdgeEncodedMove
public import NeuralReversi.Stability.EdgeRecursiveSafety
public import NeuralReversi.Stability.EdgeAssembly
public import NeuralReversi.Stability.StructuralPipeline

@[expose] public section

namespace NeuralReversi.Stability.Semantics

theorem encodePair_actualMove (line : Line) (c : Color) (a : Fin 8)
    (empty : line a = none) :
    encodePair (linePlace line c a) c =
      Edge.applyEdgeMove a (encodePair line c).1 (encodePair line c).2 := by
  rw [Edge.applyEdgeMove_eq_coordinate_spec _ _ a
    (encodePair_disjoint line c) (encodePair_empty line c a empty)]
  exact encodePair_moveSpec line c a empty

theorem lineStep_encoded {a b : Line} (h : LineStep a b) (c : Color) :
    Edge.Step (encodePair a c) (encodePair b c) := by
  cases h with
  | place mover sq empty =>
    refine ⟨sq, encodePair_empty a c sq empty, ?_⟩
    by_cases same : mover = c
    · subst mover
      exact Or.inl (encodePair_actualMove a c sq empty)
    · have opposite : mover = Opposite c := by cases mover <;> cases c <;> simp_all [Opposite]
      subst mover
      right
      have moved := encodePair_actualMove a (Opposite c) sq empty
      rw [encodePair_swap] at moved
      have result := congrArg (fun q : Edge.State => (q.2,q.1)) moved
      simpa only [encodePair_swap, Edge.secondChild] using result

theorem trace_append {n m : Nat} {a b c : Edge.State}
    (ab : Edge.Trace n a b) (bc : Edge.Trace m b c) : Edge.Trace (n + m) a c := by
  induction ab with
  | refl => simpa using bc
  | step h rest ih => simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using Edge.Trace.step h (ih bc)

theorem lineReach_encoded {a b : Line} (h : LineReach a b) (c : Color) :
    ∃ n, Edge.Trace n (encodePair a c) (encodePair b c) := by
  induction h with
  | refl => exact ⟨0, .refl _⟩
  | tail _ step ih =>
    obtain ⟨n,hn⟩ := ih
    exact ⟨n+1, trace_append hn (.step (lineStep_encoded step c) (.refl _))⟩

theorem recursiveStable_line_safe {a b : Line} (h : LineReach a b) (c : Color) (i : Fin 8)
    (marked : (Edge.recursiveStable (encodePair a c).1 (encodePair a c).2).getLsbD i.val = true) :
    b i = some c := by
  obtain ⟨n,trace⟩ := lineReach_encoded h c
  have hb := Edge.recursiveStable_never_flips (encodePair_disjoint a c) trace i marked
  simpa [encodePair, -BitVec.getLsbD_eq_getElem] using hb

def TableAgrees (table : Edge.EdgeTable) : Prop :=
  ∀ p o, p &&& o = 0 → Edge.edgeStability table p o = Edge.recursiveStable p o

theorem table_row_stable {s : State} {c : Color} (table : Edge.EdgeTable)
    (agrees : TableAgrees table) (r : Fin 8) (edge : r = 0 ∨ r = 7) (i : Fin 8)
    (marked : (Edge.edgeStability table (encodeLine (fun j => s.board (r,j)) c)
      (encodeLine (fun j => s.board (r,j)) (Opposite c))).getLsbD i.val = true) :
    Stable s c (r,i) := by
  intro t future
  rw [agrees _ _ (encodeLine_disjoint _ c)] at marked
  apply recursiveStable_line_safe (reach_row_line future r ?_) c i marked
  rcases edge with h | h
  · left; subst r; rfl
  · right; subst r; rfl

theorem table_col_stable {s : State} {c : Color} (table : Edge.EdgeTable)
    (agrees : TableAgrees table) (r : Fin 8) (edge : r = 0 ∨ r = 7) (i : Fin 8)
    (marked : (Edge.edgeStability table (encodeLine (fun j => s.board (j,r)) c)
      (encodeLine (fun j => s.board (j,r)) (Opposite c))).getLsbD i.val = true) :
    Stable s c (i,r) := by
  intro t future
  rw [agrees _ _ (encodeLine_disjoint _ c)] at marked
  apply recursiveStable_line_safe (reach_col_line future r ?_) c i marked
  rcases edge with h | h
  · left; subst r; rfl
  · right; subst r; rfl

theorem table_edge_sound {s : State} {c : Color} {p o : Word}
    (table : Edge.EdgeTable) (agrees : TableAgrees table) (repr : Represents s c p o) :
    SoundMask s c (getStableEdge table p o) := by
  intro sq marked
  apply getStableEdge_sound_of_line_tables table s c p o repr
    (table_row_stable table agrees) (table_col_stable table agrees) (boardSquare sq)
  simpa only [Marks, boardIndex_square] using marked

theorem table_pipeline_never_flips {s t : State} {c : Color} {p o : Word}
    (table : Edge.EdgeTable) (agrees : TableAgrees table) (repr : Represents s c p o)
    (future : Reach s t) (sq : Fin 64)
    (marked : (seededStable p o (getStableEdge table p o)).getLsbD sq.val = true) :
    t.board (boardSquare sq) = some c :=
  seededStable_never_flips repr (table_edge_sound table agrees repr) future sq marked

def getStableDiscs (player opponent : Word) : Word :=
  seededStable player opponent (getStableEdge Edge.initEdgeStability player opponent)

theorem initialized_table_agrees : TableAgrees Edge.initEdgeStability :=
  Edge.initEdgeStability_eq_recursive

theorem returned_disc_never_flips {s t : State} {c : Color} {p o : Word}
    (repr : Represents s c p o) (future : Reach s t) (sq : Fin 64)
    (marked : (getStableDiscs p o).getLsbD sq.val = true) :
    t.board (boardSquare sq) = some c :=
  table_pipeline_never_flips Edge.initEdgeStability initialized_table_agrees repr future sq marked

end NeuralReversi.Stability.Semantics
