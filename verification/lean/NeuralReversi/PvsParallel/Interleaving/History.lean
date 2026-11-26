import NeuralReversi.PvsParallel.Interleaving.BoardFrame

namespace NeuralReversi.PvsParallel.Interleaving.History

open Aspiration (SoundResult Window)
open CausalTable (Values Lookups Rule)
open PvsBoardTree (Position minimax)
open TranspositionTable
open NativeTableWrites.Journal (Attempt)
open FrameTableReads (Address)
open TranspositionTable.SeqLock (Payload)

structure Score where
  board : Position
  pv : Bool
  alpha : Int
  beta : Int
  value : Int

def Score.Valid (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (s : Score) : Prop :=
  Window s.alpha s.beta → (s.pv = false → s.beta = s.alpha+1) → SoundResult (minimax mc fc s.board) s.alpha s.beta s.value

def values (xs : List Score) : Values := fun b pv lo hi v => ⟨b,pv,lo,hi,v⟩ ∈ xs

structure State where
  pending : List Score
  returned : List Score
  writes : List (Address × Payload)

/-- At a successful reader boundary the payload may come from an older release
    of this slot, including its initial contents. A writer still inside its critical
    section contributes no version. Coherent native reads refine this choice. -/
def seen (seed : Address → Payload) (s : State) (a : Address) (p : Payload) : Prop :=
  p = seed a ∨ (a,p) ∈ s.writes

def reads (seed : Address → Payload) (s : State) (endian : Bool) : Lookups :=
  fun b e => e.bound ≠ 0 ∧ ∃ a, seen seed s a (SlotJournal.packed endian b e)

def empty : State := ⟨[],[],[]⟩

/-- A frame decides its value before its TT release; the caller can consume the
    value only after return. Unrelated workers can act at any of these boundaries. -/
inductive Run (seed : Address → Payload) (endian : Bool) (rule : Rule) : State → Prop where
  | initial : Run seed endian rule empty
  | choose (f : Score) : Run seed endian rule s →
      rule (values s.returned) (reads seed s endian) f.board f.pv f.alpha f.beta f.value →
      Run seed endian rule {s with pending := f :: s.pending}
  | finish (f : Score) : Run seed endian rule s → s.pending = left ++ f :: right →
      Run seed endian rule {s with pending := left ++ right, returned := f :: s.returned}
  | publish (w : Attempt) : Run seed endian rule s →
      TranspositionTable.WriterMir.publication w.event (w.program endian) = some (w.call endian) →
      Table.Operands endian (CausalTable.Input (values s.pending)) (seen seed s w.address) w.origin →
      Run seed endian rule {s with writes := (w.address,(w.call endian).payload) :: s.writes}

structure Safe (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool)
    (seed : Address → Payload) (s : State) : Prop where
  pending : ∀ f ∈ s.pending, f.Valid mc fc
  returned : ∀ f ∈ s.returned, f.Valid mc fc
  visible : ∀ a p, seen seed s a p → Table.Safe mc fc endian p

theorem Safe.lookup (good : Safe mc fc endian seed s) (found : reads seed s endian b e) : Table.EntrySafe mc fc b e := by
  obtain ⟨occupied,a,used⟩ := found
  have ok := good.visible a _ used
  simpa only [SlotJournal.packed,Packing.unpack_pack] using
    ok b rfl (by simpa only [SlotJournal.packed,Packing.unpack_pack] using occupied)

theorem Run.safe (run : Run seed endian rule s) (kernel : CausalTable.Preserves mc fc rule)
    (initial : ∀ a, Table.Safe mc fc endian (seed a)) : Safe mc fc endian seed s := by
  induction run with
  | initial => exact ⟨by simp [empty],by simp [empty],by simpa [seen,empty] using initial⟩
  | choose f _ frame ih =>
    refine ⟨?_,ih.returned,ih.visible⟩
    intro other member
    rcases List.mem_cons.mp member with rfl | old
    · exact fun win narrow => kernel _ _ (fun b pv lo hi v w n member => ih.returned _ member w n)
        (fun _ _ found => ih.lookup found) _ _ _ _ _ win narrow frame
    · exact ih.pending other old
  | finish f _ held ih =>
    have valid := ih.pending f (held ▸ List.mem_append_right _ List.mem_cons_self)
    refine ⟨?_,fun _ mem => (List.mem_cons.mp mem).elim (fun eq => eq ▸ valid) (ih.returned _),ih.visible⟩
    intro g remains
    apply ih.pending g
    rw [held]
    exact (List.mem_append.mp remains).elim (List.mem_append_left _)
      (fun present => List.mem_append_right _ (List.mem_cons_of_mem _ present))
  | publish w _ _ args ih =>
    have picked : CausalTable.Correct mc fc (values _) := fun b pv a z v win n mem => ih.pending _ mem win n
    have good := Table.Produced.safe (w := w) (args.mapInputs (fun _ _ input => input.safe picked))
      (ih.visible w.address)
    refine ⟨ih.pending,ih.returned,?_⟩
    intro a p found
    rcases found with first | released
    · exact ih.visible a p (.inl first)
    · rcases List.mem_cons.mp released with same | old
      · have eq : p = (w.call endian).payload := congrArg Prod.snd same
        exact eq.symm ▸ good
      · exact ih.visible a p (.inr old)

def zero (_ : Address) : Payload := ⟨0,0,0⟩

theorem zero_safe : Table.Safe mc fc endian (zero a) := by
  intro b key occupied
  apply False.elim
  apply occupied
  cases endian <;> simp [zero, Packing.unpack]

/-- A normal exact frame may be consumed at any point in a mixed interleaving
    of frame choices, returns, and writer releases. Earlier weak entries may be used as hints. -/
def Contract : Prop := ∀ mc fc endian s, Run zero endian (BoardFrame.Frame mc fc) s →
    (∀ f ∈ s.returned, f.Valid mc fc) ∧
    (∀ b e, reads zero s endian b e → Table.EntrySafe mc fc b e)

theorem correct : Contract := by
  intro mc fc endian s run
  have safe := run.safe BoardFrame.preserves (fun _ => zero_safe)
  exact ⟨safe.returned,fun _ _ found => safe.lookup found⟩

end NeuralReversi.PvsParallel.Interleaving.History
