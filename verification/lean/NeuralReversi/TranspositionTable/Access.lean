import NeuralReversi.TranspositionTable.Packing
import NeuralReversi.TranspositionTable.ProbeRetry
import NeuralReversi.TranspositionTable.Snapshot

namespace NeuralReversi.TranspositionTable.Access

open Packing

def isPv (entry : EntryData) : Bool := entry.flags &&& 2#8 != 0#8

def MetadataContract : Prop :=
  ∀ score move bound depth sel generation endgame pv endian,
    let info := unpack endian (pack endian (newData score move bound depth sel generation endgame pv))
    decodeBestMove info.bestMove = move ∧ info.bound = boundByte bound ∧
    (info.depth.setWidth 32).toNat = depth.toNat % 256 ∧
    selectivityRank info.selectivity = sel.val ∧ info.generation = generation &&& 127#8 ∧
    isEndgame info = endgame ∧ isPv info = pv

theorem metadataContract_proven : MetadataContract := by
  intro score move bound depth sel generation endgame pv endian
  rw [unpack_pack]
  refine ⟨decodeBestMove_encodeBestMove move, rfl, ?_, ?_, rfl, ?_, ?_⟩
  · exact depth_widen (depth.setWidth 8)
  · change selectivityRank (BitVec.ofNat 8 sel.val) = sel.val
    rw [selectivityRank_eq_min, BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (show sel.val < 2^8 by have := sel.isLt; omega),
      Nat.min_eq_left (show sel.val ≤ 5 by have := sel.isLt; omega)]
  · change decide ((((if endgame then 1#8 else 0#8) ||| (if pv then 2#8 else 0#8)) &&& 1#8) ≠ 0#8) = endgame
    cases endgame <;> cases pv <;> decide
  · change (((if endgame then 1#8 else 0#8) ||| (if pv then 2#8 else 0#8)) &&& 2#8 != 0#8) = pv
    cases endgame <;> cases pv <;> decide

def RawAccessorContract : Prop :=
  ∀ info, (info.depth.setWidth 32).toNat = info.depth.toNat ∧
    selectivityRank info.selectivity = min info.selectivity.toNat 5 ∧
    isEndgame info = info.flags.getLsbD 0 ∧ isPv info = info.flags.getLsbD 1

theorem rawAccessorContract_proven : RawAccessorContract := by
  intro info
  exact ⟨depth_widen _, selectivityRank_eq_min _, (byte_masks _).1, (byte_masks _).2⟩

def resultSlot : ProbeRetry.Result → Fin 2
  | .hit slot _ | .miss slot => slot

def resultData : ProbeRetry.Result → Option EntryData
  | .hit _ data => some data
  | .miss _ => none

def resultMove (result : ProbeRetry.Result) : Option (Fin 64) :=
  (resultData result).bind fun data => decodeBestMove data.bestMove

def ProbeAccessorContract : Prop :=
  (∀ slot data, let hit := ProbeRetry.Result.hit slot data
    resultSlot hit = slot ∧ resultData hit = some data ∧ resultMove hit = decodeBestMove data.bestMove) ∧
  (∀ slot, let miss := ProbeRetry.Result.miss slot
    resultSlot miss = slot ∧ resultData miss = none ∧ resultMove miss = none) ∧
  ∀ key (attempts : List ProbeRetry.RawAttempt) returned,
    let result := ProbeRetry.probe (attempts.map (ProbeRetry.observe key))
    resultData result = some returned →
    ∃ observation ∈ attempts.take 3,
      observation.readSlot (resultSlot result) = some (key, returned) ∧ returned.bound ≠ 0

theorem probeAccessorContract_proven : ProbeAccessorContract := by
  refine ⟨fun _ _ => ⟨rfl, rfl, rfl⟩, fun _ => ⟨rfl, rfl, rfl⟩, ?_⟩
  intro key attempts returned
  dsimp only
  cases probe : ProbeRetry.probe (attempts.map (ProbeRetry.observe key)) with
  | miss slot => simp [resultData]
  | hit slot data =>
    intro read
    have eq : data = returned := Option.some.inj read
    subst returned
    exact ProbeRetry.contract_proven.2 key attempts slot data probe

def KeyEqualityContract : Prop :=
  ∀ lp lo rp ro, sameBoardBits lp lo rp ro = true ↔ lp = rp ∧ lo = ro

theorem keyEqualityContract_proven : KeyEqualityContract := sameBoardBits_exact

def InitialSlotContract : Prop :=
  ∀ endian, unpack endian 0 = zeroData ∧
    Snapshot.read endian 0 0 ⟨0, 0, 0⟩ = some ⟨0, 0, 0, 0⟩

theorem initialSlotContract_proven : InitialSlotContract := by
  intro endian
  cases endian <;> exact ⟨rfl, rfl⟩

end NeuralReversi.TranspositionTable.Access
