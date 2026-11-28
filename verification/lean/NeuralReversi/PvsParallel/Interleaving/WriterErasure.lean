module
public import NeuralReversi.PvsParallel.Interleaving.CallJournal

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving

namespace CallJournal

def Event.erase : Event α → Event Unit
  | .other w => .other w
  | .localCall w e a => .localCall w e a
  | .boundaryScore w p t b a v => .boundaryScore w p t b a v
  | .beginWrite w k _ => .beginWrite w k ()
  | .atomic w k => .atomic w k

def Writer.erase (w : Writer α) : Writer Unit := ⟨w.invocation,w.next,w.owner,()⟩

def Issued.erase : Issued α → Issued Unit
  | .localCall w e => .localCall w e
  | .atomic w pc => .atomic w.erase pc

def State.erase (s : State α) : State Unit :=
  ⟨s.stacks,s.localClock,fun w => (s.writing w).map Writer.erase,s.started,s.choices,s.closed,s.issued.map Issued.erase⟩

end CallJournal

namespace WriterErasure

open CallJournal

theorem records (s : State α) : s.erase.records = s.records := by
  simp only [State.records,State.erase,List.map_map]
  congr 1
  funext i
  cases i <;> rfl

private theorem mapAfter (input : Option α) {f : α → Option β} {g : α → Option γ}
    (h : ∀ a, g a = (f a).map k) : input.bind g = (input.bind f).map k := by
  cases input with
  | none => rfl
  | some a => exact h a

private theorem localStep_eq (s : State α) (t w : Nat) (e : NativeFrames.Event) (a : Option NativeFrames.Args) :
    localStep s.erase t w e a = (localStep s t w e a).map State.erase := by
  cases e <;> simp only [localStep,State.erase]
  all_goals repeat' first | rfl | (apply mapAfter; intro _) | split

private theorem boundary_eq (s : State α) (p : EndgameSearch.CacheHistory.Pool) (t : EndgameSearch.CacheHistory.Timeline p) :
    boundaryStep s.erase mc fc time worker p t before after word =
      (boundaryStep s mc fc time worker p t before after word).map State.erase := by
  simp only [boundaryStep, State.erase, Option.isNone_map]
  repeat' first | rfl | (apply mapAfter; intro _) | split

private theorem erasePut (f : Nat → Option (Writer α)) (i : Nat) (v : Option (Writer α)) :
    put (fun j => (f j).map Writer.erase) i (v.map Writer.erase) = fun j => ((put f i v) j).map Writer.erase := by
  funext j
  by_cases eq : i = j <;> simp [put,eq]

theorem step (s : State α) (time : Nat) (e : Event α) :
    CallJournal.step mc fc c endian s.erase time e.erase =
      (CallJournal.step mc fc c endian s time e).map State.erase := by
  cases e with
  | other => rfl
  | boundaryScore => exact boundary_eq _ _ _
  | localCall w e a =>
    simp only [Event.erase,CallJournal.step]
    rw [localStep_eq]
    simp only [State.erase,Option.isNone_map]
    apply mapAfter
    intro _
    cases outcome : localStep s time w e a with
    | none => rfl
    | some t => simp [State.erase,Issued.erase]
  | beginWrite w k raw =>
    simp only [Event.erase,CallJournal.step,State.erase,Option.isNone_map]
    apply mapAfter
    intro _
    congr 2
    exact erasePut s.writing w (some ⟨k,0,(s.stacks w).head?.bind Activation.picked,raw⟩)
  | atomic w k =>
    simp only [Event.erase,CallJournal.step,State.erase]
    cases held : s.writing w with
    | none => rfl
    | some wr =>
      dsimp only [Option.map_some,Option.bind_eq_bind,Option.bind_some,Writer.erase]
      apply mapAfter
      intro _
      apply mapAfter
      intro pc
      apply mapAfter
      intro _
      simp only [Pure.pure,State.erase,Option.map_some,List.map_append,List.map_cons,List.map_nil,Issued.erase]
      split
      · congr 2
        exact erasePut s.writing w (some {wr with next := wr.next+1})
      · congr 2
        exact erasePut s.writing w none

/-- Erasing operands preserves both failure and success on every prefix. -/
theorem run (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : NativeTableWrites.Journal.Calls)
    (endian : Bool) (events : List (Event α)) :
    CallJournal.run mc fc c endian (events.map Event.erase) = (CallJournal.run mc fc c endian events).map State.erase := by
  have fold (xs : List (Event α × Nat)) (s : State α) :
      xs.foldlM (fun t e => CallJournal.step mc fc c endian t e.2 e.1.erase) s.erase =
      (xs.foldlM (fun t e => CallJournal.step mc fc c endian t e.2 e.1) s).map State.erase := by
    induction xs generalizing s with
    | nil => rfl
    | cons e rest ih =>
      rw [List.foldlM_cons, List.foldlM_cons,step]
      cases issued : CallJournal.step mc fc c endian s e.2 e.1 with
      | none => rfl
      | some next => exact ih next
  have empty : (start : State α).erase = start := rfl
  simpa only [empty,CallJournal.run,List.zipIdx_map,List.foldlM_map,Prod.map_fst,Prod.map_snd,id_eq] using fold events.zipIdx CallJournal.start

theorem decoded (accepted : CallJournal.decode mc fc c endian seen events = some s) :
    CallJournal.run mc fc c endian (events.map Event.erase) = some s.erase := by
  obtain ⟨done_,ran,ok⟩ := Option.bind_eq_some_iff.mp accepted
  split at ok <;> try contradiction
  cases ok
  rw [run,ran,Option.map_some]

theorem sameRecords (typed : CallJournal.decode mc fc c endian seen events = some s)
    (physical : CallJournal.run mc fc c endian (events.map Event.erase) = some raw) :
    s.records = raw.records := by
  rw [decoded typed] at physical
  cases physical
  exact (records s).symm

end WriterErasure
end NeuralReversi.PvsParallel.Interleaving
