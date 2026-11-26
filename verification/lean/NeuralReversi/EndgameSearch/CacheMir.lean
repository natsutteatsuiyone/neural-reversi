import NeuralReversi.EndgameSearch.CacheMir.Program
import NeuralReversi.EndgameSearch.CacheState

namespace NeuralReversi.EndgameSearch.CacheMir

open EndgameCache (Entry Write)
open EndgameTreeCache (Table Address Store)
open CacheState (Cells represents)

theorem index_eq (width length : Nat) (key : BitVec 64) (positive : 0 < length)
    (word : length < 2^64) (fits : length ≤ 2^width) :
    slot width length key = TranspositionTable.Clear.mulHi64 key (BitVec.ofNat 64 length) ∧
      slot width length key < length := by
  let count := BitVec.ofNat 64 length
  have value : count.toNat = length := Nat.mod_eq_of_lt word
  have high := MulHi.wordContract_proven key count
  have below : (highWord key count).toNat < length := by
    exact value ▸ high.2 (value.symm ▸ positive)
  have wide : (highWord key count).toNat < 2^width := Nat.lt_of_lt_of_le below fits
  change ((highWord key count).setWidth width).toNat = _ ∧ _
  have reduced : slot width length key = (highWord key count).toNat := by
    simp only [slot, BitVec.toNat_setWidth, count, Nat.mod_eq_of_lt wide]
  exact ⟨reduced.trans high.1,reduced ▸ below⟩

theorem newCache_eq (bytes : Nat) (positive : 0 < bytes / 18) :
    newCache bytes = some (Array.replicate (bytes / 18) (EndgameCache.unset ⟨0,0⟩)) := by
  simp only [newCache, positive, ↓reduceIte]
  rfl

theorem too_small (bytes : Nat) (small : bytes < 18) : newCache bytes = none := by
  simp [newCache, Nat.div_eq_of_lt small]

private theorem same_board (a b : Board.Board) :
    a = b ↔ a.player = b.player ∧ a.opponent = b.opponent := by
  cases a
  cases b
  simp

theorem probeCell_eq (e : Entry) (b : Board.Board) (alpha : Int) :
    probeCell e b alpha = EndgameCache.probe e b alpha := by
  by_cases player : e.board.player = b.player <;> by_cases opponent : e.board.opponent = b.opponent <;>
    simp [probeCell, scoreForAlpha, EndgameCache.probe, same_board, player, opponent]

theorem storeCell_eq (e : Entry) (w : Write) :
    storeCell e w.board w.alpha w.score = EndgameCache.store e w := by
  by_cases player : e.board.player = w.board.player <;>
    by_cases opponent : e.board.opponent = w.board.opponent <;>
    by_cases high : w.score > w.alpha <;>
    simp [storeCell, overwrite, merge, EndgameCache.store, EndgameCache.unset, lowerUnset,
      upperUnset, same_board, player, opponent, high]

theorem store_at (xs : Array Entry) (i : Nat) (inside : i < xs.size) (w : Write) :
    store xs i w.board w.alpha w.score =
      some (xs.modify i (fun old => EndgameCache.store old w)) := by
  simp only [store, inside, ↓reduceDIte, storeCell_eq, Option.some.injEq]
  apply Array.ext_getElem?
  intro j
  rw [Array.getElem?_set, Array.getElem?_modify]
  by_cases same : i = j
  · subst j
    simp only [↓reduceIte, Array.getElem?_eq_getElem inside, Option.map_some]
  · simp only [same, ↓reduceIte]

theorem owned_slot_in_bounds (width length : Nat) (xs : Array Entry) (key : BitVec 64)
    (shape : xs.size = length) (positive : 0 < length) (word : length < 2^64) (fits : length ≤ 2^width) :
    slot width xs.size key < xs.size := by
  simpa only [shape] using (index_eq width length key positive word fits).2

def address (width : Nat) (length : Table → Nat) (hash : Board.Board → BitVec 64) : Address :=
  fun table board => slot width (length table) (hash board)

def Fits (width : Nat) (length : Table → Nat) : Prop :=
  ∀ t, 0 < length t ∧ length t < 2^64 ∧ length t ≤ 2^width

theorem new_represents (width : Nat) (bytes : Table → Nat) (hash : Board.Board → BitVec 64)
    (fit : Fits width (fun t => bytes t / 18)) :
    let length := fun t => bytes t / 18
    (∀ t, newCache (bytes t) = some (CacheState.empty length t)) ∧
      represents (address width length hash) [] (CacheState.empty length) := by
  dsimp only
  refine ⟨fun t => newCache_eq _ (fit t).1, CacheState.empty_represents _ _ ?_⟩
  intro t b
  exact (index_eq _ _ _ (fit t).1 (fit t).2.1 (fit t).2.2).2

def writeCells (width : Nat) (hash : Board.Board → BitVec 64) (cells : Cells)
    (e : Store) : Option Cells := do
  let xs := cells e.table
  let idx := slot width xs.size (hash e.write.board)
  let after ← store xs idx e.write.board e.write.alpha e.write.score
  return fun t => if t = e.table then after else cells t

variable {hash : Board.Board → BitVec 64} {length : Table → Nat} {cells : Cells}

theorem write_eq (e : Store) (shape : ∀ t, (cells t).size = length t) (fit : Fits width length) :
    writeCells width hash cells e = some (CacheState.write (address width length hash) cells e) := by
  have inside := (index_eq width (length e.table) (hash e.write.board)
    (fit e.table).1 (fit e.table).2.1 (fit e.table).2.2).2
  unfold writeCells
  dsimp only
  rw [shape, store_at _ _ (by simpa only [shape] using inside)]
  apply congrArg some
  funext t
  by_cases eq : t = e.table
  · subst t
    rfl
  · simp [CacheState.write, eq]

theorem written_represents (e : Store) (shape : ∀ t, (cells t).size = length t) (fit : Fits width length)
    (before : represents (address width length hash) history cells)
    (written : writeCells width hash cells e = some after) :
    represents (address width length hash) (history ++ [e]) after ∧
      ∀ t, (after t).size = length t := by
  rw [write_eq e shape fit] at written
  cases written
  refine ⟨CacheState.write_represents e before,?_⟩
  intro t
  by_cases eq : t = e.table <;> simp [CacheState.write, eq, shape]

theorem replay_eq (writes : List Store) (shape : ∀ t, (cells t).size = length t)
    (fit : Fits width length) :
    writes.foldlM (writeCells width hash) cells =
      some (writes.foldl (CacheState.write (address width length hash)) cells) := by
  induction writes generalizing cells with
  | nil => rfl
  | cons w rest ih =>
    rw [List.foldlM_cons, write_eq w shape fit]
    apply ih
    intro t
    by_cases same : t = w.table <;> simp [CacheState.write, same, shape]

theorem after_calls (calls : List CacheState.Call) (correct : CorrectOptions mc fc opts)
    (addr : opts.address = address width length hash) (fit : Fits width length)
    (shape : ∀ t, (cells t).size = length t)
    (valid : EndgameTreeCache.soundHistory mc fc history)
    (before : represents opts.address history cells) :
    let after := calls.foldl (fun h c => c.historyAfter mc fc opts h) history
    ∃ updated, (after.drop history.length).foldlM (writeCells width hash) cells = some updated ∧
      EndgameTreeCache.soundHistory mc fc after ∧ represents opts.address after updated := by
  obtain ⟨sound,arrays⟩ := CacheState.after_calls calls correct valid before
  refine ⟨_,replay_eq _ shape fit,sound,?_⟩
  simpa only [addr] using arrays

theorem probe_bound (t : Table) (b : PvsBoardTree.Position)
    (shape : (cells t).size = length t) (before : represents (address width length hash) history cells)
    (valid : EndgameTreeCache.soundHistory mc fc history)
    (hit : probe (cells t) (slot width (cells t).size (hash b.val)) b.val alpha = some (some score)) :
    EndgameCache.CacheRange score ∧ Solve2.NWS alpha (EndgameTreeCache.discValue mc fc b.val) score := by
  apply CacheState.probe_bound before valid t b alpha score
  rw [probe, shape] at hit
  cases read : (cells t)[address width length hash t b.val]? with
  | none =>
    simp [address] at read
    simp [read] at hit
  | some e =>
    simp only [address] at read
    simpa [read, probeCell_eq] using hit

abbrev OwnedCells := Nat → Cells

/-- `owner` denotes the exclusive owner of the two cache allocations. -/
structure OwnedWrite where
  owner : Nat
  event : Store

def writeOwned (width : Nat) (key : Board.Board → BitVec 64) (all : OwnedCells) (w : OwnedWrite) :
    Option OwnedCells := do
  let updated ← writeCells width key (all w.owner) w.event
  return fun owner => if owner = w.owner then updated else all owner

def appendOwned (histories : Nat → List Store) (w : OwnedWrite) : Nat → List Store :=
  fun owner => if owner = w.owner then histories owner ++ [w.event] else histories owner

def Aligned (width : Nat) (sizes : Nat → Table → Nat) (key : Board.Board → BitVec 64)
    (histories : Nat → List Store) (all : OwnedCells) : Prop :=
  ∀ owner, (∀ t, (all owner t).size = sizes owner t) ∧
    represents (address width (sizes owner) key) (histories owner) (all owner)

theorem allocated_owned (bytes : Nat → Table → Nat) (width : Nat) (key : Board.Board → BitVec 64)
    (fit : ∀ n, Fits width (fun t => bytes n t / 18)) :
    Aligned width (fun n t => bytes n t / 18) key (fun _ => [])
      (fun n => CacheState.empty (fun t => bytes n t / 18)) := by
  intro owner
  exact ⟨fun t => by simp [CacheState.empty],(new_represents width (bytes owner) key (fit owner)).2⟩

theorem owned_write (w : OwnedWrite) (fit : ∀ owner, Fits width (sizes owner))
    (aligned : Aligned width sizes hash histories all) :
    ∃ after, writeOwned width hash all w = some after ∧
      Aligned width sizes hash (appendOwned histories w) after ∧
      ∀ other, other ≠ w.owner → after other = all other := by
  obtain ⟨shape,old⟩ := aligned w.owner
  let address' := address width (sizes w.owner) hash
  let update := CacheState.write address' (all w.owner) w.event
  have written : writeCells width hash (all w.owner) w.event = some update :=
    write_eq _ shape (fit w.owner)
  obtain ⟨replays,size⟩ := written_represents w.event shape (fit w.owner) old written
  refine ⟨fun n => if n = w.owner then update else all n, ?_, ?_, ?_⟩
  · simp [writeOwned, written]
  · intro owner
    by_cases same : owner = w.owner
    · subst owner
      simpa only [appendOwned, ↓reduceIte] using And.intro size replays
    · simpa only [appendOwned, same, ↓reduceIte] using aligned owner
  · intro other ne
    simp [ne]

theorem interleaved (events : List OwnedWrite) (fit : ∀ owner, Fits width (sizes owner))
    (aligned : Aligned width sizes hash histories all) :
    ∃ after, events.foldlM (writeOwned width hash) all = some after ∧
      Aligned width sizes hash (events.foldl appendOwned histories) after := by
  induction events generalizing all histories with
  | nil => exact ⟨all,rfl,aligned⟩
  | cons w rest ih =>
    obtain ⟨next,step,keeps,_⟩ := owned_write w fit aligned
    obtain ⟨after,writes,good⟩ := ih keeps
    exact ⟨after,by simpa [List.foldlM_cons,step] using writes,good⟩

end NeuralReversi.EndgameSearch.CacheMir

