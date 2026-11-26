import NeuralReversi.PvsParallel.FrameTableReads
import NeuralReversi.TranspositionTable.ReaderMir.Cluster

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations

open TranspositionTable
open SeqLock (Word)
open ReaderMir (AtomicAction)

abbrev Code := (Nat → Word) → List AtomicAction

structure Site where
  address : Address
  opening : Nat
  deriving DecidableEq

inductive Op where
  | slot (site : Site) (action : AtomicAction)
  | generation (allocation : Nat)
  | spin
  deriving DecidableEq

structure Supply where
  word : Address → Nat → Word
  generationByte : Nat → Nat → BitVec 8

def Site.words (r : Site) (s : Supply) := fun i => s.word r.address (r.opening+i)

def Site.emit (r : Site) (s : Supply) (code : Code) :=
  (code (r.words s)).map (Op.slot r)

/-- Indices count the executed operations of this worker. A fence remains at its
    own position. A different worker may execute any number of operations between them. -/
def Site.next (r : Site) (s : Supply) (code : Code) := r.opening + (r.emit s code).length

structure Round where
  front : Site
  back : Option Site
  generation : BitVec 8

def Round.read (r : Round) (s : Supply) (endian : Bool) : ProbeRetry.RawAttempt :=
  ⟨fun i => if i = 0 then snapshot endian (r.front.words s) else r.back.bind (fun b => snapshot endian (b.words s)),r.generation⟩

def Round.pass (r : Round) (s : Supply) (endian : Bool) (query : ProbeRetry.Key) :=
  let a := ProbeRetry.observe query (r.read s endian)
  ReaderMir.probePass a.generation a.first a.second

def Round.emit (r : Round) (s : Supply) (endian : Bool) (query : ProbeRetry.Key) :=
  (r.pass s endian query).effects.flatMap fun
    | .read i => if i = 0 then r.front.emit s (snapCode endian) else r.back.toList.flatMap (fun b => b.emit s (snapCode endian))
    | .generation => [.generation r.front.address.allocation]
    | .spin => [.spin]

def roundAt (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) : Round :=
  let code := snapCode endian
  let front : Site := ⟨slotAt allocation clusters key 0,start⟩
  let first := snapshot endian (front.words s)
  if ReaderMir.probeSecond query first then
    let back : Site := ⟨slotAt allocation clusters key 1,front.next s code⟩
    let second := snapshot endian (back.words s)
    let g := if ReaderMir.readsGeneration (ProbeRetry.classify query first) (ProbeRetry.classify query second) then
        s.generationByte allocation (back.next s code) &&& 127 else 0
    ⟨front,some back,g⟩
  else ⟨front,none,0⟩

structure Probed where
  rounds : List Round
  result : ProbeRetry.Result

def sweep (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) :
    Nat → Nat → Fin 2 → Probed
  | 0,_,last => ⟨[],.miss last⟩
  | fuel+1,start,last =>
    let r := roundAt s allocation clusters key endian query start
    match (r.pass s endian query).exit with
    | .done result => ⟨[r],result⟩
    | .retry chosen =>
      let tail := sweep s allocation clusters key endian query fuel (start+(r.emit s endian query).length) (chosen.getD last)
      ⟨r::tail.rounds,tail.result⟩

def probe (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) :=
  sweep s allocation clusters key endian query ReaderMir.probeRetries start 0

def Probed.emit (result : Probed) (s : Supply) (endian : Bool) (query : ProbeRetry.Key) :=
  result.rounds.flatMap (fun r => r.emit s endian query)

private theorem sweep_scan (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key)
    (fuel start : Nat) (last : Fin 2) :
    let r := sweep s allocation clusters key endian query fuel start last
    r.rounds.length ≤ fuel ∧ ProbeRetry.scan last (r.rounds.map (fun a => ProbeRetry.observe query (a.read s endian))) = r.result := by
  induction fuel generalizing start last with
  | zero => exact ⟨Nat.le_refl _,rfl⟩
  | succ fuel ih =>
    let r := roundAt s allocation clusters key endian query start
    have one := fun rest => ReaderMir.pass_scan (ProbeRetry.observe query (r.read s endian)) rest last
    simp only [sweep]
    cases h : (r.pass s endian query).exit with
    | done result =>
      have step := one []
      change _ = (match (r.pass s endian query).exit with | .done v => v | .retry x => ProbeRetry.scan (x.getD last) []) at step
      rw [h] at step
      simp only [←show r = roundAt s allocation clusters key endian query start from rfl,List.length_singleton,List.map_cons,List.map_nil]
      exact ⟨by omega,step⟩
    | retry chosen =>
      let next := start+(r.emit s endian query).length
      let tail := sweep s allocation clusters key endian query fuel next (chosen.getD last)
      obtain ⟨len,res⟩ := ih next (chosen.getD last)
      have step := one (tail.rounds.map (fun a => ProbeRetry.observe query (a.read s endian)))
      change _ = (match (r.pass s endian query).exit with | .done v => v | .retry x => ProbeRetry.scan (x.getD last) _) at step
      rw [h] at step
      simp only [←show r = roundAt s allocation clusters key endian query start from rfl,List.length_cons,List.map_cons]
      exact ⟨Nat.succ_le_succ len,step.trans res⟩

theorem probe_result (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) :
    let actual := probe s allocation clusters key endian query start
    ProbeRetry.probe (actual.rounds.map (fun r => ProbeRetry.observe query (r.read s endian))) = actual.result := by
  obtain ⟨len,result⟩ := sweep_scan s allocation clusters key endian query ReaderMir.probeRetries start 0
  dsimp only
  unfold probe ProbeRetry.probe
  rw [List.take_of_length_le (by simpa only [List.length_map] using (show _ ≤ 3 from len))]
  exact result

end NeuralReversi.PvsParallel.FrameTableReads.Operations
