import NeuralReversi.PvsParallel.SerialTable.WeakRoot

namespace NeuralReversi.AspirationMir.SerialTable.MidgameRows

open PvsParallel
open RootMoveStorage (Row)
open MidgameAspiration (History FiniteScore)
open SerialCalls (Program)

def stats (r : Row) : History := ⟨r.average, r.meanSquared⟩
def Valid (rows : List Row) : Prop := ∀ r ∈ rows, (stats r).valid

theorem perm {before after : List Row} (order : after.Perm before) (ok : Valid before) : Valid after :=
  fun r member => ok r (order.mem_iff.mp member)

private theorem reset (rows : List Row) (idx : Nat) (ok : Valid rows) : Valid (RootMoveStorage.resetFrom rows idx) := by
  intro r member
  rcases List.mem_append.mp member with front | back
  · exact ok r (List.mem_of_mem_take front)
  · obtain ⟨old,found,rfl⟩ := List.mem_map.mp back
    exact ok old (List.mem_of_mem_drop found)

private theorem write (rows : List Row) (sq : Bitboard.Square) (score : Int) (visible : Bool) (pv : List Bitboard.Square)
    (finite : FiniteScore score) (ok : Valid rows) :
    Valid (RootSuffix.writeFirst sq (fun r => RootSuffix.updateRow r score visible pv) rows) := by
  induction rows with
  | nil => exact ok
  | cons r rs ih =>
    have head := ok r List.mem_cons_self
    have tail : Valid rs := fun old member => ok old (.tail _ member)
    rw [RootSuffix.writeFirst]
    split
    · intro old member
      rcases List.mem_cons.mp member with rfl | later
      · exact Or.inr (MidgameAspiration.history_updated head finite)
      · exact tail old later
    · intro old member
      rcases List.mem_cons.mp member with rfl | later
      · exact head
      · exact ih tail old later

private theorem visit_score (v : RootOwnerMir.Slice.Visit M) (beta : Int)
    (finite : ∀ t ∈ SerialOwnerCalls.words beta v, FiniteScore t.2) :
    FiniteScore (-(v.wide.or v.scout).getD 0) := by
  have neg (w : Int) (h : FiniteScore w) : FiniteScore (-w) := by
    obtain ⟨lo,hi⟩ := h
    exact ⟨by omega,by omega⟩
  rcases v with ⟨m,a,sc,wide⟩
  cases sc <;> cases wide <;> simp only [SerialOwnerCalls.words] at finite <;> dsimp
  · exact ⟨by decide,by decide⟩
  · exact neg _ (finite (⟨m,true,-beta,-a⟩,_) (by simp))
  · exact neg _ (finite (⟨m,false,-(a+1),-a⟩,_) (by simp))
  · exact neg _ (finite (⟨m,true,-beta,-a⟩,_) (by simp))

private theorem visits (vs : List (RootOwnerMir.Slice.Visit M)) (sq : M → Bitboard.Square)
    (pv : M → List Bitboard.Square) (s : Bool × List Row) (beta : Int) (ok : Valid s.2)
    (finite : ∀ t ∈ vs.flatMap (SerialOwnerCalls.words beta), FiniteScore t.2) :
    Valid (vs.foldl (SerialOwnerCalls.writeVisit sq pv) s).2 := by
  induction vs generalizing s with
  | nil => exact ok
  | cons v vs ih =>
    rw [List.foldl_cons]
    apply ih
    · exact write _ _ _ _ _ (visit_score v beta fun t mem => finite t (List.mem_flatMap.mpr ⟨v,.head _,mem⟩)) ok
    · exact fun t mem => finite t (List.mem_append_right _ mem)

theorem frame {mc fc b d paired mem rows idx root a z}
    (run : Program.Runs (PvsParallel.SerialTable.Root.frame mc fc b d paired mem rows idx root a z) cs out)
    (ok : Valid rows) (finite : ∀ t ∈ cs, FiniteScore t.2) : Valid out.answer.rows := by
  dsimp only [PvsParallel.SerialTable.Root.frame] at run
  split at run
  · cases run
  next sq _ =>
    cases run
    exact write _ sq 16384 true _ ⟨by decide,by decide⟩ (reset rows idx ok)
  next =>
    obtain ⟨owner,issued,rfl⟩ := Program.runs_map run
    obtain ⟨words,ran,rfl⟩ := Program.runs_mapRequests issued
    apply visits _ _ _ _ z.toInt (reset rows idx ok)
    rw [← (SerialOwnerCalls.runs_visits ran).1]
    exact fun t found => finite _ (List.mem_map_of_mem (f := fun x : SerialCalls.MoveRequest (PvsBoardTree.GeneratedSquare mc b.val) × Int =>
      (SerialRootCalls.request mc fc b d x.1,x.2)) found)

end NeuralReversi.AspirationMir.SerialTable.MidgameRows
