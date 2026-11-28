module
public import NeuralReversi.PvsParallel.SerialTable.Image
public import NeuralReversi.TranspositionTable.WriterMir.StoreProof

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsBoardTree (Position minimax)
open Cutoffs (History key exactSelectivity)
open ExactTableHistory (Publications Admission)

def Image.set (im : Image l) (i : l.Index) (s : Clear.SlotState) : Image l :=
  {im with slots := im.slots.set i.val s (im.size.symm ▸ i.isLt), size := by simp [im.size]}

theorem get_set (im : Image l) (i j : l.Index) (s : Clear.SlotState) :
    (im.set i s).get j = if i = j then s else im.get j := by
  by_cases h : i = j
  · subst j; simp [Image.get,Image.set]
  · have ne : i.val ≠ j.val := fun eq => h (Fin.ext eq)
    simp [Image.get,Image.set,h,Array.getElem_set,ne]

structure Table (l : Layout) where
  image : Image l
  history : History

structure Table.Good (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (t : Table l) : Prop where
  supported : t.image.Supported t.history
  publications : Publications mc fc t.history
  hints : TableReads.LegalHints t.history

def Table.empty (l : Layout) (generation : BitVec 8) : Table l :=
  ⟨⟨Array.replicate (l.clusters.toNat * Clear.clusterSize) Clear.emptySlot, by simp, generation⟩,[]⟩

theorem Table.empty_supported (l : Layout) (g : BitVec 8) :
    (Table.empty l g).image.Supported [] := by
  constructor <;> intro i
  · simp [Table.empty,Image.get,Clear.emptySlot]
  · simp [Resident,Table.empty,Image.get,Clear.emptySlot,Packing.unpack]

theorem Table.empty_good (l : Layout) (generation : BitVec 8) (mc fc) :
    (Table.empty l generation).Good mc fc :=
  ⟨Table.empty_supported l generation,.initial,by simp [TableReads.LegalHints,Table.empty]⟩

def observation (old : Clear.SlotState) : Save.Observation :=
  ⟨old.seq,old.seq,payload old,old.seq⟩

def Table.put (t : Table l) (i : l.Index) (b : Position) (seq : SeqLock.Word) (e : EntryData) : Table l :=
  ⟨t.image.set i ⟨seq,b.val.player,b.val.opponent,Packing.pack l.bigEndian e⟩,t.history ++ [(b,e)]⟩

/-- The parent keeps the index from its entry probe. After nested calls,
    save reads the slot now at that index, including a different board. -/
def Table.save (t : Table l) (i : l.Index) (b : Position) (incoming : EntryData) : Table l :=
  match Save.attempt l.bigEndian (key b) incoming (observation (t.image.get i)) with
  | .written seq entry => t.put i b seq entry
  | _ => t

theorem quiet_attempt (endian : Bool) (old : Clear.SlotState) (b : Position) (d : EntryData)
    (even : old.seq &&& 1 = 0) :
    Save.attempt endian (key b) d (observation old) =
      match Save.proposal (key b) (old.player,old.opponent) d (Packing.unpack endian old.data) with
      | none => .rejected
      | some written => .written (SeqLock.publish old.seq) written := by
  by_cases exactBound : d.bound = Packing.boundByte .exact
  · rw [Save.exactAttemptContract_proven _ _ _ _ exactBound]
    simp only [observation,even,true_and,ite_true,Save.proposal,exactBound]
  · let snap := SeqLock.snapshotValue old.seq (payload old) (Snapshot.empty endian)
    have read : Snapshot.read endian old.seq old.seq (payload old) = some snap :=
      (SeqLock.snapshot_success_iff ..).2 ⟨even,rfl,rfl⟩
    unfold Save.attempt
    simp only [exactBound,ite_false,observation,read]
    cases vacant : Snapshot.empty endian old.data with
    | true =>
      have zero : (Packing.unpack endian old.data).bound = 0 := beq_iff_eq.mp vacant
      simp only [snap,SeqLock.snapshotValue,payload,vacant,ite_true]
      simp only [Save.proposal,exactBound,zero,ite_false,ite_true,Save.claimWrite,SeqLock.claim]
    | false =>
      simp only [snap,SeqLock.snapshotValue,payload,vacant,Bool.false_eq_true,ite_false]
      cases Save.proposal (key b) (old.player,old.opponent) d (Packing.unpack endian old.data) <;>
        simp [Save.claimWrite,SeqLock.claim]

theorem Table.save_eq (t : Table l) (i : l.Index) (b : Position) (d : EntryData)
    (supported : t.image.Supported t.history) :
    let old := t.image.get i
    t.save i b d = match Save.proposal (key b) (old.player,old.opponent) d (Packing.unpack l.bigEndian old.data) with
      | some e => t.put i b (SeqLock.publish old.seq) e
      | none => t := by
  dsimp only
  rw [Table.save,quiet_attempt _ _ _ _ (supported.even i)]
  cases Save.proposal (key b) ((t.image.get i).player,(t.image.get i).opponent) d
    (Packing.unpack l.bigEndian (t.image.get i).data) <;> rfl

theorem Table.put_supported (t : Table l) (i : l.Index) (b : Position) (e : EntryData)
    (supported : t.image.Supported t.history) :
    let out := t.put i b (SeqLock.publish (t.image.get i).seq) e
    out.image.Supported out.history := by
  let old := t.image.get i
  have even : SeqLock.publish old.seq &&& 1 = 0 := by
      have init := supported.even i
      change old.seq &&& 1 = 0 at init
      have base : SeqLock.seqWord old.seq.toNat = old.seq := by simp [SeqLock.seqWord]
      have epoch := SeqLock.even_index_epoch old.seq.toNat (by rw [base]; exact init)
      have repr : SeqLock.seqWord (2 * (old.seq.toNat / 2)) = old.seq := (congrArg SeqLock.seqWord epoch).symm.trans base
      simpa only [repr] using SeqLock.publish_even (old.seq.toNat / 2)
  constructor <;> intro j
  · change ((t.image.set i _).get j).seq &&& 1 = 0
    rw [get_set]; split
    · exact even
    · exact supported.even j
  · change Resident l (t.history ++ [(b,e)]) ((t.image.set i _).get j)
    rw [get_set]; split
    · simp only [Resident,Packing.unpack_pack]; exact fun _ => ⟨b,by simp,rfl,rfl⟩
    · intro occupied
      obtain ⟨board,mem,p,o⟩ := supported.resident j occupied
      exact ⟨board,List.mem_append_left _ mem,p,o⟩

theorem Table.save_supported (t : Table l) (i : l.Index) (b : Position) (d : EntryData)
    (supported : t.image.Supported t.history) : (t.save i b d).image.Supported (t.save i b d).history := by
  rw [t.save_eq i b d supported]
  split
  · exact t.put_supported i b _ supported
  · exact supported

theorem Table.save_good (t : Table l) (good : t.Good mc fc) (i : l.Index) (b : Position) (d : EntryData)
    (fresh : TableReads.LegalHint b d)
    (accepted : ∀ e, Save.proposal (key b) ((t.image.get i).player,(t.image.get i).opponent) d
          (Packing.unpack l.bigEndian (t.image.get i).data) = some e → Admission mc fc t.history b e) :
    (t.save i b d).Good mc fc := by
  let old := t.image.get i
  let stored := Packing.unpack l.bigEndian old.data
  rw [t.save_eq i b d good.supported]
  change (match Save.proposal (key b) (old.player,old.opponent) d stored with
    | some e => t.put i b (SeqLock.publish old.seq) e | none => t).Good mc fc
  cases prop : Save.proposal (key b) (old.player,old.opponent) d stored with
  | none => exact good
  | some e =>
    have oldHint : stored.bound ≠ 0 → stored.bestMove ≠ Packing.noMove →
        Save.legal (old.player,old.opponent) stored.bestMove := by
      intro occupied present
      obtain ⟨board,mem,p,o⟩ := good.supported.resident i occupied
      change old.player = _ at p
      change old.opponent = _ at o
      rw [p,o]
      exact good.hints board stored mem present
    have writtenHint := (Save.contract_proven (key b) (old.player,old.opponent) d stored).2 fresh oldHint e prop
    refine ⟨t.put_supported i b e good.supported,.append good.publications (accepted e prop),?_⟩
    intro board entry mem
    rcases List.mem_append.mp mem with earlier | hit
    · exact good.hints board entry earlier
    · obtain ⟨rfl,rfl⟩ := Prod.mk.inj (List.mem_singleton.mp hit)
      exact writtenHint

def Table.store (t : Table l) (i : l.Index) (b : Position) (score : Score) (bound : Bound) (depth : Depth)
    (move : Option Bitboard.Square) (sel : Selectivity) (endgame ttPv : Bool) : Table l :=
  match WriterMir.storeInput i b score bound depth move sel t.image.generation endgame ttPv with
  | some (slot,board,d) => t.save slot board d
  | none => t

theorem Table.store_supported (t : Table l) (support : t.image.Supported t.history) (i : l.Index)
    (b : Position) (score : Score) (bound : Bound) (depth : Depth) (move : Option Bitboard.Square)
    (sel : Selectivity) (endgame ttPv : Bool) :
    let out := t.store i b score bound depth move sel endgame ttPv
    out.image.Supported out.history := by
  unfold Table.store
  split
  · exact t.save_supported _ _ _ support
  · exact support

theorem Table.store_searched (t : Table l) (good : t.Good mc fc) (i : l.Index) (b : Position)
    (pv ttPv : Bool) (a z score : Score) (depth : Depth) (move : Option Bitboard.Square)
    (window : Aspiration.Window a.raw.toInt z.raw.toInt)
    (sound : Aspiration.SoundResult (minimax mc fc b) a.raw.toInt z.raw.toInt score.raw.toInt)
    (legal : TableReads.LegalChoice b move) :
    (t.store i b score (classify pv score a z) depth move exactSelectivity true ttPv).Good mc fc := by
  rw [Table.store,WriterMir.store_input]
  by_cases finite : score.raw.toInt.natAbs < 32767 <;> simp only [finite,ite_true,ite_false]
  · apply t.save_good good _ _ _ (TableReads.newHint legal)
    exact fun e proposed => .searched b pv a z score move depth _ ttPv _ _ e window sound proposed
  · exact good

end NeuralReversi.PvsParallel.SerialTable
