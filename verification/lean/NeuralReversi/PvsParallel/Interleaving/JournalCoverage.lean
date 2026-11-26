import NeuralReversi.PvsParallel.Interleaving.JournalPvSplit

namespace NeuralReversi.PvsParallel.Interleaving.JournalCoverage

open JournalBodies (Check)
open CausalTable (Frame)
open EndgameSearch.CacheHistory (Pool)

private instance (test : Check P) (rest : Prop) [Decidable rest] : Decidable (test.requires rest) :=
  match test with
  | .outside => inferInstanceAs (Decidable rest)
  | .rejected => isFalse id
  | .proved _ => isTrue True.intro

private theorem through (test : Check P) (convert : A → B) : test.requires A → test.requires B := by
  cases test with | outside => exact convert | rejected => exact id | proved _ => exact id

/-- A declined branch lets the next instruction path try. A malformed
    recognized branch rejects the tape, even if a later checker accepts. -/
def next (first : Check P) (later : Unit → Check P) : Check P :=
  match first with | .outside => later () | .rejected => .rejected | .proved p => .proved p

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

structure Owners (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op)
    (s : CallJournal.State Op) where
  pv : JournalPv.Options mc
  scout : JournalScoutOwner.Inputs mc
  pvSplit : JournalPvSplit.Feed es s mc fc
  scoutSplit : (f : Frame) → JournalScoutOwner.SplitFor es mc fc f.board

abbrev Final (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (s : CallJournal.State Op) :=
  (f : Frame) → Check (JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen))

variable (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (look : JournalScout.ChildReads m) (pool : Pool) (checked : Bool) (s : CallJournal.State Op)
    (op : Owners mc fc es s) (root : Final mc fc init s)

def last (f : Frame) :=
  next (JournalScoutOwner.checkWith mc fc init probes look checked s f (op.scout f) (some (op.scoutSplit f))) (fun _ => root f)

def recognized (f : Frame) (prior : List Frame) :=
  JournalPvSplit.pending mc fc init probes look pool checked s op.pv op.scout op.pvSplit f prior
    ((last mc fc init probes look checked s op root f).requires False)

instance (f : Frame) (prior : List Frame) : Decidable (recognized mc fc init probes look pool checked s op root f prior) := by
  unfold recognized JournalPvSplit.pending JournalScoutOwner.pending JournalProbCut.pending JournalScout.pending JournalPv.pending JournalPass.pending
  infer_instance

/-- Exhaust the actual three finite registries. There is no success case for a
    score that fell outside all of the modeled instruction paths. -/
def covers : Bool :=
  let accepted := fun f prior => decide (recognized mc fc init probes look pool checked s op root f prior)
  s.frames.all (fun f => accepted f (JournalPass.seedRows s f)) &&
    s.weakChildren.all (fun f => accepted f s.weakChildren) && s.exactFrames.all (fun f => accepted f s.exactFrames)

theorem covers_of_recognized
    (hSeed : ∀ f ∈ s.frames, recognized mc fc init probes look pool checked s op root f (JournalPass.seedRows s f))
    (hWeak : ∀ f ∈ s.weakChildren, recognized mc fc init probes look pool checked s op root f s.weakChildren)
    (hExact : ∀ f ∈ s.exactFrames, recognized mc fc init probes look pool checked s op root f s.exactFrames) :
    covers mc fc init probes look pool checked s op root = true := by
  simp only [covers,Bool.and_eq_true,List.all_eq_true,decide_eq_true_eq]
  exact ⟨⟨hSeed,hWeak⟩,hExact⟩

private theorem discharge {f : Frame} {prior : List Frame}
    (use : JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen) → Q)
    (closed : recognized mc fc init probes look pool checked s op root f prior) :
    JournalPvSplit.pending mc fc init probes look pool checked s op.pv op.scout op.pvSplit f prior Q :=
  through _ (through _ (through _ (through _ (through _ (through _ (through _ (through _
    (fun h => Check.derive _ use (through _ False.elim h))))))))) closed

theorem checkedFrames (ok : covers mc fc init probes look pool checked s op root = true) :
    JournalPvSplit.Remaining mc fc init probes look pool checked s op.pv op.scout op.pvSplit := by
  obtain ⟨prior,full⟩ := Bool.and_eq_true_iff.mp ok
  obtain ⟨seed,weak⟩ := Bool.and_eq_true_iff.mp prior
  constructor
  · intro f member
    exact discharge mc fc init probes look pool checked s op root (fun r => r.seed) (of_decide_eq_true (List.all_eq_true.mp seed f member))
  · intro f member
    exact discharge mc fc init probes look pool checked s op root (fun r => r.weak (bne_iff_ne.mp (Bool.and_eq_true_iff.mp (List.mem_filter.mp member).2).1))
      (of_decide_eq_true (List.all_eq_true.mp weak f member))
  · intro f member
    exact discharge mc fc init probes look pool checked s op root (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp member).2))
      (of_decide_eq_true (List.all_eq_true.mp full f member))

private def accepts (test : Check True) := decide (test.requires False)
#guard accepts (next .outside (fun _ => .proved trivial))
#guard !(accepts (next .rejected (fun _ => .proved trivial)))
#guard !(accepts (next .outside (fun _ => .outside)))

end NeuralReversi.PvsParallel.Interleaving.JournalCoverage
