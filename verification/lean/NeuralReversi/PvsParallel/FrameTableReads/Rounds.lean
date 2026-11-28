module
public import NeuralReversi.PvsParallel.FrameTableReads.Instructions

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations

open TranspositionTable
open FrameReturns (Log Frame)
open SeqLock (Word)

structure Placed (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) (r : Round) : Prop where
  front : r.front.address = slotAt allocation clusters key 0
  opening : r.front.opening = start
  branch : r.back.isSome = ReaderMir.probeSecond query (snapshot endian (r.front.words s))
  back : ∀ b, r.back = some b → b.address = slotAt allocation clusters key 1 ∧ b.opening = r.front.next s (snapCode endian)

theorem placed (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (query : ProbeRetry.Key) (start : Nat) :
    Placed s allocation clusters key endian query start (roundAt s allocation clusters key endian query start) := by
  unfold roundAt
  dsimp only
  split
  · exact ⟨rfl,rfl,by simp_all,fun _ h => by cases h; exact ⟨rfl,rfl⟩⟩
  · exact ⟨rfl,rfl,by simp_all,fun _ h => by contradiction⟩

theorem Round.emit_slots (r : Round) (s : Supply) (endian : Bool) (query : ProbeRetry.Key)
    (branch : r.back.isSome = ReaderMir.probeSecond query (snapshot endian (r.front.words s))) :
    let obs := ProbeRetry.observe query (r.read s endian)
    r.emit s endian query = r.front.emit s (snapCode endian) ++ r.back.toList.flatMap (fun b => b.emit s (snapCode endian)) ++
      (if ReaderMir.readsGeneration obs.first obs.second then [.generation r.front.address.allocation] else []) ++
      (if (ProbeRetry.decisive obs).isNone then [.spin] else []) := by
  let obs := ProbeRetry.observe query (r.read s endian)
  have request : ReaderMir.readsSecond obs.first = r.back.isSome := (ReaderMir.second_of_snapshot _ _).trans branch.symm
  unfold Round.emit Round.pass
  rw [ReaderMir.pass_effects]
  change (([.read 0] : List ReaderMir.ProbeEffect) ++ (if ReaderMir.readsSecond obs.first then _ else _) ++ (if ReaderMir.readsGeneration obs.first obs.second then _ else _) ++
    (if (ProbeRetry.decisive obs).isNone then _ else _)).flatMap _ = _
  rw [request]
  cases hb : r.back <;> cases ReaderMir.readsGeneration obs.first obs.second <;> cases (ProbeRetry.decisive obs).isNone <;>
    simp [hb,obs] at * <;> rfl

theorem at_expected {log : Log mc fc} {t : Table log} {f : Frame} (r : Invocation t addr f code) :
    Packet.atSlot (Packet.mk addr r) addr = some r := by simp [Packet.atSlot]

theorem transported_words {log : Log mc fc} {t : Table log} {a b : Address} (eq : a = b)
    (r : InvocationAt t a consumer code) : (eq ▸ r : InvocationAt t b consumer code).words = r.words := by cases eq; rfl

/-- Construct the calls consumed by the production probe decoder from raw
    atomic instructions. A first-slot shortcut never needs a second invocation. -/
theorem Round.decode {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (r : Round) (pos : Placed s allocation clusters key endian query start r) (ran : Runs issued start (r.emit s endian query)) :
    ∃ packets actual, decodeRound t allocation clusters key f endian query r.generation packets = some actual ∧
      actual.raw = r.read s endian := by
  have prefix_ := (r.emit_slots s endian query pos.branch) ▸ ran
  obtain ⟨frontRan,backRan⟩ := prefix_.append.1.append.1.append
  let front := m.invocation r.front (snapshot_path endian) (pos.opening ▸ frontRan)
  have value : snapshot endian front.words = snapshot endian (r.front.words s) := m.value ..
  have frontAddr : r.front.address = ⟨allocation,(ReaderMir.probeSlots clusters key).1⟩ := pos.front
  let first : Invocation t ⟨allocation,(ReaderMir.probeSlots clusters key).1⟩ f (snapCode endian) := frontAddr ▸ front
  have firstVal : snapshot endian first.words = snapshot endian (r.front.words s) := by
    exact (congrArg (snapshot endian) (transported_words frontAddr front)).trans value
  have branch := pos.branch
  cases back : r.back with
  | none =>
    have skip : ReaderMir.probeSecond query (snapshot endian first.words) = false := by simpa [back,firstVal] using branch.symm
    let actual : ProbeRound t allocation clusters key f endian query := ⟨first,none,skip.symm,r.generation⟩
    refine ⟨[⟨_,first⟩],actual,?_,?_⟩
    · simp [decodeRound,at_expected,skip,actual]
    · cases r with
      | mk a b g =>
        cases back
        unfold ProbeRound.raw Round.read
        congr 1
        funext i
        split <;> first | exact firstVal | rfl
  | some b =>
    have takeSecond : ReaderMir.probeSecond query (snapshot endian first.words) = true := by simpa [back,firstVal] using branch.symm
    obtain ⟨addr,opens⟩ := pos.back b back
    have ranB : Runs issued b.opening (b.emit s (snapCode endian)) := by simpa [back,opens,Site.next,pos.opening] using backRan
    let backRead := m.invocation b (snapshot_path endian) ranB
    have atB : b.address = ⟨allocation,(ReaderMir.probeSlots clusters key).2⟩ := addr
    let second : Invocation t ⟨allocation,(ReaderMir.probeSlots clusters key).2⟩ f (snapCode endian) := atB ▸ backRead
    have secondVal : snapshot endian second.words = snapshot endian (b.words s) :=
      (congrArg (snapshot endian) (transported_words atB backRead)).trans (m.value ..)
    let actual : ProbeRound t allocation clusters key f endian query := ⟨first,some second,takeSecond.symm,r.generation⟩
    refine ⟨[⟨_,first⟩,⟨_,second⟩],actual,?_,?_⟩
    · simp [decodeRound,at_expected,takeSecond,actual]
    · unfold ProbeRound.raw Round.read
      congr 1
      funext i
      split
      · exact firstVal
      · simpa [actual,back] using secondVal

end NeuralReversi.PvsParallel.FrameTableReads.Operations
