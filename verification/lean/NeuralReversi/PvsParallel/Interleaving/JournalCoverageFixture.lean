import NeuralReversi.PvsParallel.Interleaving.PhysicalCalls
import NeuralReversi.Board.InitializationModel

namespace NeuralReversi.PvsParallel.Interleaving.JournalCoverageFixture

open TranspositionTable
open NativeTableWrites.Journal (Calls Record Execution Even mods)
open PvsBoardTree (Position)
open EndgameSearch.CacheHistory (Pool)

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
def board : Position := ⟨⟨0xffffffdfbfffffff,0x40000000⟩,rfl⟩

/-- Native square numbering. A pass is 64 and is accepted only if the opponent can move. -/
private def play (b : Board.Board) (index : Nat) : Option Board.Board := do
  if index = 64 then
    guard (!(Board.hasLegalMoves mc b) && Board.hasLegalMoves mc (Board.switchPlayers b))
    return Board.switchPlayers b
  else if h : index < 64 then
    guard ((Board.getMoves mc b).getLsbD index)
    let sq := (⟨index,h⟩ : Fin 64)
    return Board.place b (Flip.Dispatch.resultValue fc sq b.player b.opponent) sq
  else none

private def game := [44,43,42,51,58,45,26,18,53,61,46,41,33,54,10,34,63,55,40,29,47,59,60,
  32,62,49,30,22,50,21,24,31,20,57,48,25,38,13,12,3,56,11,14,2,17,16,9,4,52,6,7,15,5,39,23,8,0,64,1,64,19,64]

#guard game.foldlM play Board.newBoard = some board.val
#guard Board.emptyCount board.val = 1
#guard play board.val 37 = some ⟨0,0xffffffffffffffff⟩

def calls : Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩
private def events (score : Int) : List NativeFrames.Event :=
  [.entry 0 0 .pvs .none none,.scored 1 ⟨BitVec.ofInt 32 score⟩,.leave 2 0 .pvs none (some ⟨BitVec.ofInt 32 score⟩)]
private def records (score : Int) := (events score).map (Record.localCall 0)

/-- Event numbers below the bound follow tape order. A positive modification
    index denotes an absent write and is outside every reads-from edge. -/
private def zeroMemory (limit : Nat) : SeqLock.Memory Nat where
  sb a b := a < b ∧ b < limit + 1
  hb a b := a < b ∧ b < limit + 1
  rf w _ := w = 0
  write _ i := if i = 0 then 0 else limit + 1
  writeValue := fun _ _ => 0
  readValue := fun _ => 0
  rf_value := fun _ => rfl
  release := fun _ => False
  acquire := fun _ => True
  releaseFence := fun _ => False
  acquireFence := fun _ => True
  sb_hb := id
  sb_trans h k := ⟨Nat.lt_trans h.1 k.1,k.2⟩
  hb_trans h k := ⟨Nat.lt_trans h.1 k.1,k.2⟩
  hb_irrefl a h := Nat.lt_irrefl a h.1
  release_acquire := fun h => h.elim
  fence_fence := fun h => h.elim
  write_write := by intro loc i j h; split at h <;> split at h <;> omega
  write_read := by intro loc i j r h _; split at h <;> omega
  read_read := by intro loc i j a b _ h _; split at h <;> omega

private theorem absentMods (rs : List Record) (noWriter : ∀ s, .writer s ∉ rs) (a loc) :
    mods calls rs a loc = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  exact fun _ mem => noWriter _ (NativeTableWrites.Journal.mem_mods.mp mem).1

def emptyExecution (rs : List Record) (noWriter : ∀ s, .writer s ∉ rs) (limit : Nat := 4096) :
    NativeTableWrites.Journal.Execution calls false rs where
  memory := fun _ => zeroMemory limit
  distinct := by intro a loc; simp [absentMods rs noWriter a loc]
  source := by
    intro a loc i r h
    change (if i = 0 then 0 else limit + 1) = 0 at h
    split at h <;> omega
  modified := by intro a loc i s h; simp [absentMods rs noWriter a loc] at h
  rmw := by intro a i k h; simp [absentMods rs noWriter a .sequence] at h
  ordering := fun s h => (noWriter s h).elim
  earlier := fun s h => (noWriter s h).elim

theorem emptyEven (rs : List Record) (noWriter : ∀ s, .writer s ∉ rs) (limit : Nat := 4096) :
    ∀ a, Even ((emptyExecution rs noWriter limit).memory a) := by
  intro a
  change (0 : BitVec 64) &&& 1 = 0
  decide

theorem emptyAtomics {rs : List Record} {noWriter : ∀ s, .writer s ∉ rs} {limit : Nat}
    (es : Joint.Tape Op) (bound : es.length ≤ limit)
    (zero : ∀ i a loc mode, Joint.opcode es i = some (.load a loc mode) →
      Joint.origin es i = 0 ∧ Joint.word es i = 0) :
    Joint.Atomics (emptyExecution rs noWriter limit) es where
  reads i a loc mode h := by
    obtain ⟨src,value⟩ := zero _ _ _ _ h
    change (if Joint.origin es i = 0 then 0 else limit + 1) = 0 ∧ 0 = Joint.word es i
    simp [src,value]
  acquire := fun _ _ _ _ => trivial
  fence := fun _ _ _ => trivial
  sb := fun _ _ _ ij je _ => ⟨ij,by omega⟩

theorem emptyOrder {rs : List Record} {noWriter : ∀ s, .writer s ∉ rs} {limit : Nat} :
    ∀ a i j, ((emptyExecution rs noWriter limit).memory a).hb i j → i < j :=
  fun _ _ _ edge => edge.1

theorem emptyResident {rs : List Record} {noWriter : ∀ s, .writer s ∉ rs} {limit : Nat} :
    ∀ a, Entry.Resident mc fc false
      (((emptyExecution rs noWriter limit).slot a (emptyEven rs noWriter limit a)).payloadAt 0) :=
  fun _ => .zero

theorem zeroNoWrap {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op}
    (m : Joint.Atomics x es) (zero : ∀ i, Joint.origin es i = 0) : PhysicalCalls.NoWrap m := by
  intro finish consumer opening args use _ _ call site _ _
  change Joint.origin es _ < Joint.origin es _ + SeqLock.modulus
  simp [zero,SeqLock.modulus]

private theorem noWriters (score : Int) : ∀ s, .writer s ∉ records score := by
  intro site
  simp [records,events]
private abbrev memory (score : Int) := emptyExecution (records score) (noWriters score)
private abbrev even (score : Int) := emptyEven (records score) (noWriters score)
private abbrev Operand (score : Int) := CallJournal.Operand calls false (CallJournal.observed (memory score) (even score))

private def tape (level : Selectivity) (score : Int) (stray : Bool) : Joint.Tape (Operand score) :=
  let args : NativeFrames.Args := ⟨board.val,1,⟨BitVec.ofInt 32 (-512)⟩,⟨512⟩,true,level,true,false,none⟩
  [.engine (.localCall 0 (.entry 0 0 .pvs .none none) (some args))] ++
    (if stray then [.enter 0 ⟨.probe (Cutoffs.key board),0,16,Board.Hash.hash false board.val⟩] else []) ++
    [.engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 score⟩) none),
     .engine (.localCall 0 (.leave 2 0 .pvs none (some ⟨BitVec.ofInt 32 score⟩)) none)]

private theorem atomics (level score stray) : Joint.Atomics (memory score) (tape level score stray) := by
  have noOps (i : Nat) : Joint.opcode (tape level score stray) i = none := by
    rcases i with _|(_|(_|(_|i))) <;> cases stray <;> rfl
  apply emptyAtomics (tape level score stray)
  · cases stray <;> simp [tape]
  · intro i a loc mode h
    rw [noOps] at h
    contradiction

private def pool : Pool := ⟨4,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩

private def decoded (level : Selectivity) (score : Int) (stray : Bool) :=
  CallJournal.decode mc fc calls false (CallJournal.observed (memory score) (even score))
    ((tape level score stray).map Joint.Event.engineEvent)

example : (decoded 5 16384 false).map CallJournal.State.records = some (records 16384) := by rfl
example : (decoded 2 16384 false).map CallJournal.State.records = some (records 16384) := by rfl

private def checked (level : Selectivity) (score : Int) (stray : Bool) : Option (Nat × Nat × Nat × Bool) := do
  let es := tape level score stray
  let s ← decoded level score stray
  let op : JournalCoverage.Owners mc fc es s := {
    pv := fun _ => ⟨fun _ => .skip,fun m => m.val,fun _ => (false,false)⟩
    scout := fun _ => ⟨fun _ => .skip,fun m => m.val,⟨fun _ _ => false,fun _ _ => false⟩⟩
    pvSplit := fun _ _ _ _ _ => ⟨7,fun _ _ _ => none⟩
    scoutSplit := fun _ _ _ _ _ _ => none }
  let passed := JournalCoverage.covers (m := atomics level score stray) mc fc (even score)
    (fun _ _ => none) (fun _ _ _ => none) pool true s op (fun _ => .outside)
  return (s.frames.length,s.weakChildren.length,s.exactFrames.length,passed)

#guard checked 5 16384 false = some (1,0,1,true)
#guard checked 2 16384 false = some (1,1,0,true)
#guard checked 5 16128 false = some (1,0,1,false)
#guard checked 5 16384 true = some (1,0,1,false)

end NeuralReversi.PvsParallel.Interleaving.JournalCoverageFixture
