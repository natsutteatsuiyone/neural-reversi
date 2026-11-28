module
public import NeuralReversi.PvsParallel.SerialTable.Image
public import NeuralReversi.PvsParallel.TableReads

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.StableSample

open TranspositionTable SeqLock
open Cutoffs (RawSample)

/-- One serial read interval, beginning after the slot's last completed write. -/
def location : Location → Nat
  | .sequence => 0
  | .payload .player => 1
  | .payload .opponent => 2
  | .payload .metadata => 3

def write (loc : Location) (k : Nat) := if k = 0 then location loc else 10

def reads (w r : Nat) : Prop := (w < 4 ∧ r = w + 4) ∨ (w = 0 ∧ r = 9)

theorem initial {loc : Location} (h : reads (write loc i) r) : i = 0 := by
  by_cases other : i = 0
  · exact other
  · simp [write, other, reads] at h

def value (s : Clear.SlotState) : Location → Word
  | .sequence => s.seq
  | .payload f => WriterTrace.fieldWord (payload s) f

def eventValue (s : Clear.SlotState) : Nat → Word
  | 5 => s.player
  | 6 => s.opponent
  | 7 => s.data
  | _ => s.seq

def memory (s : Clear.SlotState) : Memory Nat where
  sb := (· < ·)
  hb := (· < ·)
  rf := reads
  write := write
  writeValue := fun loc _ => value s loc
  readValue := eventValue s
  rf_value := by
    intro loc i r rf
    obtain rfl := initial rf
    cases loc with
    | sequence => simp [reads,write,location] at rf; rcases rf with rfl|rfl <;> rfl
    | payload f =>
      cases f <;> simp [reads,write,location] at rf <;> subst r <;> rfl
  release := fun _ => False
  acquire := fun _ => True
  releaseFence := fun _ => False
  acquireFence := fun _ => True
  sb_hb := id
  sb_trans := Nat.lt_trans
  hb_trans := Nat.lt_trans
  hb_irrefl := Nat.lt_irrefl
  release_acquire := fun impossible _ _ => impossible.elim
  fence_fence := fun impossible _ _ _ _ => impossible.elim
  write_write := by
    intro loc i j h
    cases loc with
    | sequence => simp [write,location] at h; split at h <;> split at h <;> omega
    | payload f => cases f <;> simp [write,location] at h <;> split at h <;> split at h <;> omega
  write_read := by
    intro loc i j r before rf
    have j0 := initial rf
    by_cases zero : i = 0
    · omega
    · simp [write,zero] at before
      rcases rf with ⟨hw,hr⟩|⟨hw,hr⟩ <;> omega
  read_read := by intro loc i j a b _ h _; rw [initial h]; omega

def slot (s : Clear.SlotState) (even : s.seq &&& 1 = 0) : WriterTrace.Slot (memory s) where
  history := []
  unique := by simp
  initialized := even
  source := fun _ _ h => Nat.le_of_eq (initial h)
  modified := by simp
  cas := by simp
  publish := by simp
  fields := fun _ => {
    history := []
    unique := by simp
    source := fun _ _ h => Nat.le_of_eq (initial h)
    store := by simp
    finish := by simp }

def reader (s : Clear.SlotState) : RawReader (memory s) 0 0 :=
  ⟨4,8,9,True.intro,True.intro,by change 4 < 8; decide,by change 8 < 9; decide,Or.inl ⟨by change 0 < 4; decide,rfl⟩,Or.inr ⟨rfl,rfl⟩⟩

def raw (endian : Bool) (s : Clear.SlotState) (even : s.seq &&& 1 = 0) : RawSample where
  memory := memory s
  firstIndex := 0
  finalIndex := 0
  slot := slot s even
  reader := reader s
  pe := 0
  oe := 0
  de := 0
  player := ⟨5,by change 4 < 5; decide,by change 5 < 8; decide,Or.inl ⟨by change 1 < 4; decide,rfl⟩⟩
  opponent := ⟨6,by change 4 < 6; decide,by change 6 < 8; decide,Or.inl ⟨by change 2 < 4; decide,rfl⟩⟩
  metadata := ⟨7,by change 4 < 7; decide,by change 7 < 8; decide,Or.inl ⟨by change 3 < 4; decide,rfl⟩⟩
  noWrap := by decide
  bigEndian := endian

theorem published {l : Layout} {s : Clear.SlotState} (even : s.seq &&& 1 = 0) (resident : Resident l h s) :
    (raw l.bigEndian s even).publication h := resident

theorem snapshot {l : Layout} {s : Clear.SlotState} (even : s.seq &&& 1 = 0) (resident : Resident l h s) :
    TableReads.SnapshotRead h (sample l.bigEndian s) := by
  cases result : Snapshot.read l.bigEndian s.seq s.seq (payload s) with
  | none => simpa [sample,result] using (TableReads.SnapshotReadFrom.unavailable (source := (·.publication h)))
  | some snap =>
    simp only [sample,result,Option.map_some]
    exact .sampled (raw l.bigEndian s even) snap (published even resident) result

theorem lookup {l : Layout} {s : Clear.SlotState} (even : s.seq &&& 1 = 0) (resident : Resident l h s)
    (b : PvsBoardTree.Position) :
    TableReads.SlotLookup h b (LookupCut.readOccupied l.bigEndian s.seq s.seq (payload s) b.val.player b.val.opponent) := by
  cases result : LookupCut.readOccupied l.bigEndian s.seq s.seq (payload s) b.val.player b.val.opponent with
  | none => exact .miss
  | some e => exact .hit (raw l.bigEndian s even) e (published even resident) result

end NeuralReversi.PvsParallel.SerialTable.StableSample
