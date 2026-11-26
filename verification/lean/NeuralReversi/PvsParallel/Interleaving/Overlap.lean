import NeuralReversi.PvsParallel.Interleaving.Schedule

namespace NeuralReversi.PvsParallel.Interleaving.Overlap

open Threading
open Tickets (State Post Step)
open Schedule (Task Job)

/-- Captured by check_split_interleaving.py on hard-20.obf line 166. The owner
    took index 1 before the helper took 2. The helper reported first and took 3;
    the owner then cut, causing 3's zero child return to be discarded. -/
def task : Task := ⟨false,false,true,15,6144,⟨6143,5632⟩⟩

def child : PvsMoveLoop.ChildRelation Nat := fun i pv a b score =>
  pv = false ∧ a = -6144 ∧ b = -6143 ∧
    ((i = 1 ∧ score = -6144) ∨ (i = 2 ∧ score = -5632) ∨ (i = 3 ∧ score = 0))

private def helper : Job := ⟨2,0,0,fun _ _ _ _ => -5632⟩
private def owner : Job := ⟨1,0,0,fun _ _ _ _ => -6144⟩

def early : Post := ⟨⟨2,6143,5632⟩,false⟩
def cutoff : Post := ⟨⟨1,6143,6144⟩,true⟩
def mask : List SplitCompletion.Action := [.book 1 ⟨0,2⟩,.finish 0 ⟨0,7⟩,.finish 1 ⟨1,9⟩]
def finalState : State 64 := ⟨4,[],[],[],false,[early,cutoff],[3],false⟩

private abbrev Run := Schedule.RunWith False task child 8 (1 : BitVec 64) 0

private theorem work (prev : Run actions s) (edge : Step 8 s a t) (quiet : Retirement.effect a = none)
    (noStop : a ≠ .externalStop) (checks : ∀ p ∈ a.posts, ∃ j : Job, p = j.post task s.reports ∧ j.Calls child task s.reports) :
    Run actions t := .work prev edge quiet noStop checks

theorem run : Schedule.RunWith False task child 8 (1 : BitVec 64) 0 mask finalState := by
  have start : Run [] (Tickets.initial 1 0) := .initial
  have booked := Retirement.ExecutionWith.book 1 ⟨0,2⟩ start (.book rfl (by simp [Tickets.initial]) (by simp [Tickets.State.available,Tickets.initial]))
  have one := work booked (.take (worker := 0) (left := [1]) (right := []) rfl rfl) rfl (by intro h; cases h) (by simp [Tickets.Action.posts])
  have two := work one (.take (worker := 1) (left := []) (right := []) rfl rfl) rfl (by intro h; cases h) (by simp [Tickets.Action.posts])
  have published := work two (.report (post := early) (worker := 1) (left := []) (right := [(0,1)]) rfl) rfl (by intro h; cases h) (by
    simp only [Tickets.Action.posts,List.mem_singleton]
    rintro _ rfl
    exact ⟨helper,rfl,by constructor <;> unfold child <;> decide +kernel⟩)
  have three := work published (.take (worker := 1) (left := []) (right := []) rfl rfl) rfl (by intro h; cases h) (by simp [Tickets.Action.posts])
  have cut := work three (.report (post := cutoff) (worker := 0) (left := [(1,3)]) (right := []) rfl) rfl (by intro h; cases h) (by
    simp only [Tickets.Action.posts,List.mem_singleton]
    rintro _ rfl
    exact ⟨owner,rfl,by constructor <;> unfold child <;> decide +kernel⟩)
  have left := Retirement.ExecutionWith.finish 0 ⟨0,7⟩ cut (.leave (left := []) (right := []) rfl)
  have discarded := work left (.discard (worker := 1) (index := 3) (left := []) (right := []) rfl (Or.inr ⟨cutoff,by simp, rfl⟩)) rfl
    (by intro h; cases h) (by simp [Tickets.Action.posts])
  exact Retirement.ExecutionWith.finish 1 ⟨1,9⟩ discarded (.leave (left := []) (right := []) rfl)

theorem answer : finalState.retired ∧ finalState.indices = [2,1] ∧ finalState.discarded = [3] ∧
    finalState.outside = false ∧ (task.locked finalState.reports).best = 6144 := by
  exact ⟨⟨rfl,rfl,rfl⟩,rfl,rfl,rfl,rfl⟩

end NeuralReversi.PvsParallel.Interleaving.Overlap
