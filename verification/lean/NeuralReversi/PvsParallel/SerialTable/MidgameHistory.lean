module
public import NeuralReversi.AspirationMir.SerialTable.Repeated
public import NeuralReversi.TranspositionTable.Duplicate.Image

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MidgameHistory

open AspirationMir.SerialTable.Repeated (ReadyMemory)
open TranspositionTable
open EndgameSearch.CacheHistory (Pool)

def sharedSlot (s : Clear.SlotState) :=
  Clear.afterAttempt s (Clear.clearSharedAttempt s.seq s.seq s.seq s.data)

private theorem shared_even (s : Clear.SlotState) (even : s.seq &&& 1 = 0) : (sharedSlot s).seq &&& 1 = 0 := by
  have step := Clear.sharedAttemptContract_proven.2.2 s even
  by_cases empty : s.data = 0
  · rw [sharedSlot,step.2.1 empty]; exact even
  · rw [sharedSlot,step.2.2 empty]
    change (s.seq + 2) &&& 1 = 0
    have base : SeqLock.seqWord s.seq.toNat = s.seq := by simp [SeqLock.seqWord]
    have evenIndex := (SeqLock.seqWord_even_iff _).mp (base ▸ even)
    change SeqLock.publish s.seq &&& 1 = 0
    rw [← base,SeqLock.publish_seqWord,SeqLock.seqWord_even_iff]
    omega

def clearedImage (im : Image l) (shared : Bool) : Image l :=
  ⟨im.slots.map (fun s => if shared then sharedSlot s else Clear.emptySlot),by simp [im.size],im.generation⟩

theorem supported (im : Image l) (old : im.Supported hist) (shared : Bool) :
    (clearedImage im shared).Supported [] := by
  have eq (i : l.Index) : (clearedImage im shared).get i = (if shared then sharedSlot (im.get i) else Clear.emptySlot) := by
    simp [clearedImage,Image.get]
  constructor <;> intro i <;> rw [eq] <;> cases shared
  · simp [Clear.emptySlot]
  · exact shared_even _ (old.even i)
  · simp [Resident,Clear.emptySlot,Packing.unpack]
  · have zero := (Clear.sharedAttemptContract_proven.2.2 _ (old.even i)).1
    change (sharedSlot (im.get i)).data = 0 at zero
    simp [Resident,zero,Packing.unpack]

/-- Clear only the TT. Its generation and the worker's two local arrays survive. In a
    quiescent shared clear each occupied slot advances its own sequence number. -/
def clear (m : Memory l) (shared : Bool) : Memory l :=
  ⟨⟨clearedImage m.tt.image shared,[]⟩,supported _ m.supported shared,m.cache⟩

theorem clear_ready (m : Memory l) (shared : Bool) (good : ReadyMemory p mc fc m) :
    ReadyMemory p mc fc (clear m shared) := by
  refine ⟨⟨⟨⟨(clear m shared).supported,.initial,?_⟩,good.1.1.2⟩,?_⟩,?_⟩
  all_goals intro b e present; cases present

/-- Search::init resets the generation after clearing the TT, retaining the pool's caches. -/
def reset (m : Memory l) (shared : Bool) : Memory l :=
  let cleared := clear m shared
  ⟨{cleared.tt with image := {cleared.tt.image with generation := 0}},
    ⟨cleared.supported.even,cleared.supported.resident⟩,cleared.cache⟩

theorem reset_ready (m : Memory l) (shared : Bool) (good : ReadyMemory p mc fc m) :
    ReadyMemory p mc fc (reset m shared) := by
  have ok := clear_ready m shared good
  exact ⟨⟨⟨⟨(reset m shared).supported,ok.1.1.1.publications,ok.1.1.1.hints⟩,ok.1.1.2⟩,ok.1.2⟩,ok.2⟩

/-- Attaching an engine-owned TT keeps the receiving pool's cache arrays.
    A resize to a different allocation supplies a fresh source table. -/
def attach (src : Memory layout) (dst : Memory old) : Memory layout :=
  {src with cache := dst.cache}

theorem attach_ready (src : Memory layout) (dst : Memory old) (source : ReadyMemory p mc fc src)
    (localState : ReadyMemory p mc fc dst) : ReadyMemory p mc fc (attach src dst) :=
  ⟨⟨⟨source.1.1.1,localState.1.1.2⟩,source.1.2⟩,source.2⟩

def word (slot : Clear.SlotState) : SeqLock.Location → SeqLock.Word
  | .sequence => slot.seq
  | .payload .player => slot.player
  | .payload .opponent => slot.opponent
  | .payload .metadata => slot.data

def words (im : Image l) (i : Nat) := word (im.slots[i]?.getD Clear.emptySlot)

def copiedSlot (im : Image l) (buffers : Duplicate.Buffers) (i : l.Index) : Clear.SlotState :=
  let bytes := Duplicate.materialize l.bigEndian buffers.source (words im)
  let read := Duplicate.readWord l.bigEndian (Duplicate.image (buffers.after bytes) buffers.dest i.val)
  ⟨read .sequence,read (.payload .player),read (.payload .opponent),read (.payload .metadata)⟩

private theorem copied_eq (im : Image l) (buffers : Duplicate.Buffers) (size : buffers.clusters = l.clusters.toNat)
    (i : l.Index) : copiedSlot im buffers i = im.get i := by
  have valid : i.val < buffers.clusters * 2 := by rw [size]; exact i.isLt
  have inside : i.val < im.slots.size := by rw [im.size]; exact i.isLt
  simp [copiedSlot,buffers.image_eq _ valid,Duplicate.readMaterialized,words,inside,Image.get,word]

def copiedImage (im : Image l) (buffers : Duplicate.Buffers) : Image l :=
  ⟨Array.ofFn (copiedSlot im buffers),by simp,im.generation⟩

theorem copied_image_eq (im : Image l) (buffers : Duplicate.Buffers) (size : buffers.clusters = l.clusters.toNat) :
    copiedImage im buffers = im := by
  have same : Array.ofFn (copiedSlot im buffers) = im.slots := by
    apply Array.ext
    · simp [im.size]
    · intro i hi _
      have inLayout : i < l.clusters.toNat * Clear.clusterSize := by simpa using hi
      simpa [Image.get] using copied_eq im buffers size ⟨i,inLayout⟩
  cases im
  simp_all [copiedImage]

/-- A duplicate receives the copied source entries and generation. The thread pool supplies
    the local cache independently of which TT the caller attached to the next search. -/
def duplicate (src dst : Memory l) (buffers : Duplicate.Buffers) (size : buffers.clusters = l.clusters.toNat) : Memory l :=
  ⟨{src.tt with image := copiedImage src.tt.image buffers},copied_image_eq _ _ size ▸ src.supported,dst.cache⟩

theorem duplicate_ready (src dst : Memory l) (buffers : Duplicate.Buffers) (size : buffers.clusters = l.clusters.toNat)
    (source : ReadyMemory p mc fc src) (localState : ReadyMemory p mc fc dst) :
    ReadyMemory p mc fc (duplicate src dst buffers size) := by
  have tt : (duplicate src dst buffers size).tt = src.tt := by
    change {src.tt with image := copiedImage src.tt.image buffers} = src.tt
    rw [copied_image_eq _ _ size]
  exact ⟨⟨⟨tt ▸ source.1.1.1,localState.1.1.2⟩,tt ▸ source.1.2⟩,tt ▸ source.2⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameHistory
