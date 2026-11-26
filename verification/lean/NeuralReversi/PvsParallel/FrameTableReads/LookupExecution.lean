import NeuralReversi.PvsParallel.FrameTableReads.Rounds
import NeuralReversi.PvsParallel.NativeOwners

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations

open TranspositionTable
open SeqLock (Word)
open FrameReturns (Log Frame)

structure Looked where
  front : Site
  back : Option Site

def lookup (s : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (board : PvsBoardTree.Position)
    (start : Nat) : Looked :=
  let r : Site := ⟨slotAt allocation clusters key 0,start⟩
  match lookupValue endian board (r.words s) with
  | some _ => ⟨r,none⟩
  | none => ⟨r,some ⟨slotAt allocation clusters key 1,r.next s (lookupCode endian board)⟩⟩

def Looked.emit (out : Looked) (s : Supply) (endian : Bool) (b : PvsBoardTree.Position) :=
  let code := lookupCode endian b
  out.front.emit s code ++ out.back.toList.flatMap (fun r => r.emit s code)

def Looked.pair (out : Looked) (s : Supply) (endian : Bool) (b : PvsBoardTree.Position) :=
  (lookupValue endian b (out.front.words s),out.back.bind fun r => lookupValue endian b (r.words s))

theorem lookup_packets {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (b : PvsBoardTree.Position) (start : Nat) :
    let out := lookup s allocation clusters key endian b start
    Runs issued start (out.emit s endian b) →
    ∃ packets answer, decodeLookup t allocation clusters key f endian b packets = some answer ∧ answer.pair = out.pair s endian b := by
  let first : Site := ⟨slotAt allocation clusters key 0,start⟩
  let second : Site := ⟨slotAt allocation clusters key 1,first.next s (lookupCode endian b)⟩
  cases before : lookupValue endian b (first.words s) with
  | some d =>
    have shape : lookup s allocation clusters key endian b start = ⟨first,none⟩ := by
      change (match lookupValue endian b (first.words s) with | some _ => _ | none => _) = _
      rw [before]
    dsimp only
    rw [shape]
    intro ran
    have one : Runs issued first.opening (first.emit s (lookupCode endian b)) := by simpa [Looked.emit,first] using ran
    let front : Invocation t (slotAt allocation clusters key 0) f (lookupCode endian b) := m.invocation first (lookup_path endian b) one
    have value : lookupValue endian b front.words = some d := (m.value first (lookup_path ..) one).trans before
    let answer : Lookup t allocation clusters key f endian b := .front front (by simp [value])
    exact ⟨[⟨slotAt allocation clusters key 0,front⟩],answer,by
      simp [decodeLookup,at_expected,answer]
      split <;> first | rfl | (rename_i h; exact nomatch value.symm.trans h),
      by simp [Lookup.pair,Looked.pair,answer,value,before]⟩
  | none =>
    have shape : lookup s allocation clusters key endian b start = ⟨first,some second⟩ := by
      change (match lookupValue endian b (first.words s) with | some _ => _ | none => _) = _
      rw [before]
    dsimp only
    rw [shape]
    intro ran
    obtain ⟨one,two⟩ := ran.append
    let front : Invocation t (slotAt allocation clusters key 0) f (lookupCode endian b) := m.invocation first (lookup_path endian b) one
    have miss : lookupValue endian b front.words = none := (m.value first (lookup_path ..) one).trans before
    have more : Runs issued second.opening (second.emit s (lookupCode endian b)) := by simpa [second,Site.next,first] using two
    let back : Invocation t (slotAt allocation clusters key 1) f (lookupCode endian b) := m.invocation second (lookup_path endian b) more
    have value : lookupValue endian b back.words = lookupValue endian b (second.words s) := m.value ..
    let answer : Lookup t allocation clusters key f endian b := .back front miss back
    exact ⟨[⟨slotAt allocation clusters key 0,front⟩,⟨slotAt allocation clusters key 1,back⟩],answer,by
      simp [decodeLookup,at_expected,answer]
      split <;> first | rfl | (rename_i h; exact nomatch miss.symm.trans h),
      by simp [Lookup.pair,Looked.pair,answer,miss,before,value]⟩

theorem cluster_published {log : Log mc fc} {t : Table log} {f : Frame} (m : Machine t (log.identify f.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (b : PvsBoardTree.Position) (start : Nat) :
    let out := lookup s allocation clusters key endian b start
    let pair := out.pair s endian b
    Runs issued start (out.emit s endian b) → pair.1.or pair.2 = some d →
    Published log (log.identify f.use) endian (Cutoffs.key b) d := by
  intro out pair ran hit
  obtain ⟨_,answer,_,eq⟩ := lookup_packets m allocation clusters key endian b start ran
  change answer.pair = pair at eq
  rw [←eq] at hit
  cases answer with
  | front front _ =>
    simp only [Lookup.pair,Option.or_none] at hit
    exact lookup_published b front hit
  | back front missed back =>
    simp only [Lookup.pair,missed,Option.none_or] at hit
    exact lookup_published b back hit

/-- The child board and both optional ETC results come from this move's
    actual lookup, including a fast key mismatch followed by a second-slot hit. -/
theorem child_packets {log : Log mc fc} {t : Table log} {row : NativeFrames.Row} (m : Machine t (log.identify row.frame.use) issued s)
    (allocation : Nat) (clusters key : Word) (endian : Bool) (move : PvsBoardTree.GeneratedSquare mc row.frame.position.val)
    (start : Nat) :
    let b := (PvsBoardTree.play mc fc row.frame.position move).val
    let out := lookup s allocation clusters key endian b start
    Runs issued start (out.emit s endian b) →
    ∃ (query : NativeOwners.Lookup t row endian) (receipt : TableReads.ChildLookupFrom mc fc row.frame.position (FrameReturns.Samples log endian row.frame)), NativeOwners.Lookup.read query = some receipt ∧
      receipt.move = move ∧ (receipt.front,receipt.back) = out.pair s endian b := by
  intro b out ran
  obtain ⟨packets,actual,read,values⟩ := lookup_packets m allocation clusters key endian b start ran
  dsimp only [b] at read
  exact ⟨⟨move,allocation,clusters,key,packets⟩,childLookup _ move actual,
    by simp only [NativeOwners.Lookup.read,decodedChild,read,Option.map_some],rfl,values⟩

end NeuralReversi.PvsParallel.FrameTableReads.Operations
