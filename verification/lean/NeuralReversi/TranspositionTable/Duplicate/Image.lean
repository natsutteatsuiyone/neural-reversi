module
public import NeuralReversi.TranspositionTable.Duplicate.Program
public import NeuralReversi.Board.Hash

@[expose] public section

namespace NeuralReversi.TranspositionTable.Duplicate

def image (bytes : Bytes) (base index : Nat) : Fin 32 → BitVec 8 :=
  fun offset => bytes (slotAddress base index offset.val)

theorem copied_image (bits mib count : Nat) (before : Bytes) (source dest : Nat)
    (width : bits = 32 ∨ bits = 64) (admitted : Size.clusterCount bits mib = some count)
    (sAlign : source % 64 = 0) (dAlign : dest % 64 = 0)
    (sEnd : source + count*2*32 ≤ 2^bits) (dEnd : dest + count*2*32 ≤ 2^bits)
    (apart : source + count*2*32 ≤ dest ∨ dest + count*2*32 ≤ source) (valid : index < count*2) :
    image (copy before source dest (count*2*32)) dest index = image before source index := by
  have bytes := (copyContract_proven bits mib count before source dest width admitted
    sAlign dAlign sEnd dEnd apart).2.2.2.1
  funext offset
  exact (bytes index offset.val valid offset.isLt).2.2.1

def readWord (endian : Bool) (slot : Fin 32 → BitVec 8) (loc : SeqLock.Location) : SeqLock.Word :=
  let bytes := List.ofFn (fun i : Fin 8 =>
    slot ⟨8*(wordIndex loc).val+i.val,by have := (wordIndex loc).isLt; have := i.isLt; omega⟩)
  Board.Hash.readLittle (if endian then bytes.reverse else bytes) 0

private theorem location_wordIndex (loc : SeqLock.Location) : locationAt (wordIndex loc) = loc := by
  cases loc with
  | sequence => rfl
  | payload f => cases f <;> rfl

/-- Native object representation of the source table at the copy. Each atomic
    write supplies all eight bytes of its u64, including the packed metadata. -/
def materialize (endian : Bool) (base : Nat) (words : Nat → SeqLock.Location → SeqLock.Word) : Bytes := fun addr =>
  let offset := addr-base
  let loc := locationAt ⟨offset%32/8,by omega⟩
  (Board.Hash.nativeBytes endian (words (offset/32) loc))[offset%8]?.getD 0

private theorem materialized_byte (base index : Nat) (words : Nat → SeqLock.Location → SeqLock.Word)
    (loc : SeqLock.Location) (i : Fin 8) :
    image (materialize endian base words) base index ⟨8*(wordIndex loc).val+i.val,by
      have := (wordIndex loc).isLt; have := i.isLt; omega⟩ =
    (Board.Hash.nativeBytes endian (words index loc))[i.val]?.getD 0 := by
  let k := 8*(wordIndex loc).val+i.val
  have inside : k < 32 := by dsimp [k]; have := (wordIndex loc).isLt; have := i.isLt; omega
  have diff : slotAddress base index k - base = index*32+k := by unfold slotAddress; omega
  have locAt : (⟨(index*32+k)%32/8,by omega⟩ : Fin 4) = wordIndex loc := by
    apply Fin.ext
    dsimp [k]
    have := i.isLt
    have := (wordIndex loc).isLt
    omega
  change materialize endian base words (slotAddress base index k) = _
  simp only [materialize,diff,locAt,location_wordIndex,show (index*32+k)/32 = index by omega,
    show (index*32+k)%8 = i.val by dsimp [k]; have := i.isLt; omega]

theorem readMaterialized (endian : Bool) (base index : Nat) (words : Nat → SeqLock.Location → SeqLock.Word)
    (loc : SeqLock.Location) :
    readWord endian (image (materialize endian base words) base index) loc = words index loc := by
  let bytes := Board.Hash.nativeBytes endian (words index loc)
  have length : bytes.length = 8 := by cases endian <;> simp [bytes,Board.Hash.nativeBytes,Board.Hash.littleBytes]
  have allBytes : List.ofFn (fun i : Fin 8 => bytes[i.val]?.getD 0) = bytes := by
    apply List.ext_getElem
    · simpa only [List.length_ofFn] using length.symm
    · intro i _ bound
      simp only [List.getElem_ofFn,List.getElem?_eq_getElem bound,Option.getD_some]
  simp only [readWord,materialized_byte]
  change Board.Hash.readLittle (if endian then _ else List.ofFn (fun i : Fin 8 => bytes[i.val]?.getD 0)) 0 = _
  rw [allBytes]
  cases endian <;> simpa [bytes,Board.Hash.nativeBytes] using Board.Hash.readLittle_exact (words index loc)

theorem zeroImage (prior : Bytes) (base len index : Nat) (valid : index < len) :
    image (zeroEntries prior base len) base index = fun _ => 0 := by
  funext i
  have := i.isLt
  apply ite_eq_left
  unfold slotAddress
  constructor <;> omega

inductive Blank (endian : Bool) (index : Nat) (initial : SeqLock.Location → SeqLock.Word) : Prop
  | allocated (prior : Bytes) (base len : Nat) (valid : index < len)
      (imageAt : ∀ loc, initial loc = readWord endian (image (zeroEntries prior base len) base index) loc)
  | cleared (old : SeqLock.Location → SeqLock.Word) (imageAt : ∀ loc, initial loc = clearEntry old loc)

theorem Blank.zero (blank : Blank endian index initial) (loc : SeqLock.Location) : initial loc = 0 := by
  cases blank with
  | allocated prior base len valid words =>
    rw [words,zeroImage prior base len index valid]
    simp [readWord,Board.Hash.readLittle]
  | cleared old words =>
    rw [words]
    cases loc with
    | sequence => rfl
    | payload f => cases f <;> rfl

/-- Actual buffers of a returning duplicate, with the length measured in TTEntry elements. -/
structure Buffers where
  bits : Nat
  mib : Nat
  clusters : Nat
  source : Nat
  dest : Nat
  width : bits = 32 ∨ bits = 64
  admitted : Size.clusterCount bits mib = some clusters
  sourceAlign : source % 64 = 0
  destAlign : dest % 64 = 0
  sourceEnd : source + clusters*2*32 ≤ 2^bits
  destEnd : dest + clusters*2*32 ≤ 2^bits
  apart : source + clusters*2*32 ≤ dest ∨ dest + clusters*2*32 ≤ source

def Buffers.after (b : Buffers) (prior : Bytes) := copyEntries prior b.source b.dest (b.clusters*2)

theorem Buffers.image_eq (b : Buffers) (prior : Bytes) (valid : index < b.clusters*2) :
    image (b.after prior) b.dest index = image prior b.source index := by
  unfold Buffers.after copyEntries
  rw [copied_image b.bits b.mib b.clusters _ b.source b.dest b.width b.admitted b.sourceAlign b.destAlign
    b.sourceEnd b.destEnd b.apart valid]
  funext i
  simp only [image,zeroEntries]
  have outside : ¬ (b.dest ≤ slotAddress b.source index i.val ∧
      slotAddress b.source index i.val < b.dest + b.clusters*2*32) := by
    have := i.isLt
    unfold slotAddress
    rcases b.apart with h | h <;> omega
  rw [ite_eq_right outside]

end NeuralReversi.TranspositionTable.Duplicate
