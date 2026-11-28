module
public import NeuralReversi.PvsParallel.Interleaving.PublicEndgame
public import NeuralReversi.PvsParallel.Interleaving.JournalCoverageFixture

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.PublicEndgameFixture

open TranspositionTable
open NativeTableWrites.Journal (Calls Record Even)
open JournalAtomicGraph (Graph Rules SB HB)
open PvsBoardTree (Position)
open EndgameSearch.CacheHistory (Pool)
open ParallelOperational (CutPrefix OperationalRun OperationalBranch BranchComplete)

def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
def board : Position := JournalCoverageFixture.board
def calls : Calls := JournalCoverageFixture.calls
def pool : Pool := ⟨4,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩

def args : NativeFrames.Args :=
  ⟨board.val,1,⟨BitVec.ofInt 32 (-512)⟩,⟨512⟩,true,5,true,false,none⟩

def unitTape : Joint.Tape Unit :=
  [.engine (.localCall 0 (.entry 0 0 .pvs .none none) (some args)),
   .engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 16384⟩) none),
   .engine (.localCall 0 (.leave 2 0 .pvs none (some ⟨BitVec.ofInt 32 16384⟩)) none)]

private def records : List Record :=
  [.localCall 0 (.entry 0 0 .pvs .none none),
   .localCall 0 (.scored 1 ⟨BitVec.ofInt 32 16384⟩),
   .localCall 0 (.leave 2 0 .pvs none (some ⟨BitVec.ofInt 32 16384⟩))]

def graph : Graph := ⟨fun _ _ => 0,fun _ _ => none,SB unitTape⟩

private theorem noWriters : ∀ s, Record.writer s ∉ records := by
  intro s h; simp [records] at h

theorem unitRunSome : (CallJournal.run mc fc calls false (unitTape.map Joint.Event.engineEvent)).isSome = true := by
  decide

def uState : CallJournal.State Unit :=
  (CallJournal.run mc fc calls false (unitTape.map Joint.Event.engineEvent)).get unitRunSome

theorem uRan : CallJournal.run mc fc calls false (unitTape.map Joint.Event.engineEvent) = some uState :=
  (Option.some_get unitRunSome).symm

private theorem uRecords : uState.records = records := by rfl

private theorem unitRules : Rules graph calls false uState.records unitTape where
  trans h1 h2 := ⟨h1.1,h2.2.1,Nat.lt_trans h1.2.2.1 h2.2.2.1,h2.2.2.2.1,h1.2.2.2.2.trans h2.2.2.2.2⟩
  irrefl _ h := Nat.lt_irrefl _ h.2.2.1
  program sb := ⟨Nat.lt_trans sb.1 sb.2.1,sb.2.1,sb⟩
  sync rel := by
    rw [uRecords] at rel
    rcases rel with ⟨_,m,_⟩ | ⟨_,m,_⟩ <;> exact (noWriters _ m).elim
  fences f := by
    rw [uRecords] at f
    exact (noWriters _ f.choose_spec.1).elim
  ww {a l u v i j} du dv hb := by
    exfalso
    have facts : WriterSteps.Facts calls unitTape uState.records := WriterSteps.run uRan
    have timed : ModificationIndex.Timed calls uState.records unitTape.length :=
      ⟨fun _ => facts.bound,fun _ _ => facts.injective,facts.distinct⟩
    obtain ⟨inside,stamp⟩ := (ModificationIndex.decode_eq_some_iff timed).mp du
    have zeroMods : NativeTableWrites.Journal.mods calls uState.records a l = [] := by
      rw [uRecords]
      apply List.eq_nil_iff_forall_not_mem.mpr
      exact fun _ mem => noWriters _ (NativeTableWrites.Journal.mem_mods.mp mem).1
    rw [zeroMods] at inside
    simp only [List.length_nil] at inside
    have i0 : i = 0 := by omega
    subst i0
    change unitTape.length + ModificationIndex.tag l = u at stamp
    have uLt := hb.1
    omega
  wr _ _ _ rf := by cases rf.1
  rr _ _ _ rf := by cases rf.1
  rw _ _ rf := by cases rf.1
  writer s m := by
    rw [uRecords] at m
    exact (noWriters s m).elim
  load r _ _ _ op := by rcases r with _|(_|(_|r)) <;> cases op

/-- Concrete Task 3 RF+HB-closed `CutPrefix` on `board` (3 events, `cut = 3`). -/
def unitCut : CutPrefix mc fc where
  calls := calls
  endian := false
  fullTape := unitTape
  cut := 3
  cutLe := Nat.le_refl _
  preState := uState
  preRan := uRan
  postRecords := []
  fullState := uState
  fullRan := uRan
  fullRecEq := by simp
  postAfter := fun _ h => by cases h
  fullGraph := graph
  fullRules := by simpa using unitRules
  hbClosed := fun _ _ hb _ => hb.1
  rfClosed := fun _ _ _ rf _ => by cases rf.1
  ordered := fun _ _ hb => hb.2.2.1

theorem unitInitEven : ∀ a, Even (unitCut.toPrefix.memory.memory a) := fun _ => by
  change (0 : BitVec 64) &&& 1 = 0
  decide

theorem unitOrigin : ∀ a,
    MultiSessionCarry.SlotOrigin mc fc unitCut.endian
      ((unitCut.toPrefix.memory.slot a (unitInitEven a)).payloadAt 0) :=
  fun _ => .empty

abbrev UnitOp := CallJournal.Operand unitCut.calls unitCut.endian
  (CallJournal.observed unitCut.toPrefix.memory unitInitEven)

def scoredTape : Joint.Tape UnitOp :=
  [.engine (.localCall 0 (.entry 0 0 .pvs .none none) (some args)),
   .engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 16384⟩) none),
   .engine (.localCall 0 (.leave 2 0 .pvs none (some ⟨BitVec.ofInt 32 16384⟩)) none)]

theorem scoredErases : scoredTape.map PhysicalReadback.rawEvent = unitCut.toPrefix.events := rfl

theorem scoredDecodedSome :
    (CallJournal.decode mc fc unitCut.calls unitCut.endian
      (CallJournal.observed unitCut.toPrefix.memory unitInitEven)
      (scoredTape.map Joint.Event.engineEvent)).isSome = true := by
  decide

def scoredState : CallJournal.State UnitOp :=
  (CallJournal.decode mc fc unitCut.calls unitCut.endian
    (CallJournal.observed unitCut.toPrefix.memory unitInitEven)
    (scoredTape.map Joint.Event.engineEvent)).get scoredDecodedSome

theorem scoredDecoded :
    CallJournal.decode mc fc unitCut.calls unitCut.endian
      (CallJournal.observed unitCut.toPrefix.memory unitInitEven)
      (scoredTape.map Joint.Event.engineEvent) = some scoredState :=
  (Option.some_get scoredDecodedSome).symm

def unitOwners :
    JournalCoverage.Owners mc fc (unitCut.toPrefix.attach scoredTape) scoredState where
  pv := fun _ => ⟨fun _ => .skip,fun m => m.val,fun _ => (false,false)⟩
  scout := fun _ => ⟨fun _ => .skip,fun m => m.val,⟨fun _ _ => false,fun _ _ => false⟩⟩
  pvSplit := fun _ _ _ _ _ => ⟨7,fun _ _ _ => none⟩
  scoutSplit := fun _ _ _ _ _ _ => none

def unitRootOptions : CausalTable.Frame → JournalRootStore.Options := fun _ =>
  ⟨fun _ => none,false,⟨fun _ => 0,fun xs => ⟨xs,.refl _⟩,fun _ => []⟩,fun _ => (false,false)⟩

/-- Concrete `OperationalRun` built WITHOUT assuming `raw.covered = true` or child scores. -/
def unitOpRun : OperationalRun mc fc where
  prefix_ := unitCut
  initEven := unitInitEven
  slotOrigin := unitOrigin
  events := scoredTape
  erases := scoredErases
  state := scoredState
  decoded := scoredDecoded
  nowrap := unitCut.toPrefix.noWrap_of_length scoredTape scoredErases (by decide)
  pool := pool
  checked := true
  poolHeaderOk := rfl
  owners := unitOwners
  rootOptions := unitRootOptions
  rootSplits := fun _ _ _ _ => none

-- Verify the exact minimax value of `board` (+64 discs = +16384 raw score).
example : 256 * EndgameTreeCache.discValue mc fc board.val = 16384 := by decide +kernel

private def singleFrame : CausalTable.Frame :=
  { chosen := 1, returned := some 2, board := board, pv := true, alpha := -512, beta := 512, score := 16384 }

-- Prove `BranchComplete unitOpRun` directly from Branch 2 (`OperationalBranch.prefixCut`)
-- and derive `unitOpRun.toRaw.covered = true` via `ParallelOperational.covers_proven`!
private theorem unitBranchComplete : BranchComplete unitOpRun := by
  have fSeed : unitOpRun.toRaw.state.frames = [singleFrame] := rfl
  have fWeak : unitOpRun.toRaw.state.weakChildren = [] := rfl
  have fExact : unitOpRun.toRaw.state.exactFrames = [singleFrame] := rfl
  have hOut : JournalBodies.check mc fc unitOpRun.toRaw.init
      (PhysicalCalls.probes unitOpRun.toRaw.atomics unitOpRun.toRaw.nowrap unitOpRun.toRaw.state)
      unitOpRun.toRaw.pool unitOpRun.toRaw.state singleFrame = .outside := rfl
  have hPref : ∃ p, JournalPrefix.check mc fc unitOpRun.toRaw.init
      (PhysicalCalls.probes unitOpRun.toRaw.atomics unitOpRun.toRaw.nowrap unitOpRun.toRaw.state)
      unitOpRun.toRaw.checked unitOpRun.toRaw.state singleFrame = .proved p := by
    have isOk : (match JournalPrefix.check mc fc unitOpRun.toRaw.init
        (PhysicalCalls.probes unitOpRun.toRaw.atomics unitOpRun.toRaw.nowrap unitOpRun.toRaw.state)
        unitOpRun.toRaw.checked unitOpRun.toRaw.state singleFrame with
      | .proved _ => true | _ => false) = true := by rfl
    generalize JournalPrefix.check mc fc unitOpRun.toRaw.init
        (PhysicalCalls.probes unitOpRun.toRaw.atomics unitOpRun.toRaw.nowrap unitOpRun.toRaw.state)
        unitOpRun.toRaw.checked unitOpRun.toRaw.state singleFrame = res at isOk
    cases res with
    | proved p => exact ⟨p,rfl⟩
    | outside | rejected => cases isOk
  obtain ⟨p,hPrefEq⟩ := hPref
  refine {
    seed := fun f mem => by
      rw [fSeed,List.mem_singleton] at mem; subst f; exact .prefixCut hOut hPrefEq
    weak := fun f mem => by
      rw [fWeak] at mem; cases mem
    exact := fun f mem => by
      rw [fExact,List.mem_singleton] at mem; subst f; exact .prefixCut hOut hPrefEq }

-- Derived (not assumed!): `unitOpRun.toRaw.covered = true`!
theorem unitCovered : unitOpRun.toRaw.covered = true :=
  ParallelOperational.covers_proven mc fc unitOpRun unitBranchComplete

-- End-to-end public guarantee via `PublicEndgame.exact_endgame_sound`:
theorem unitPublicSound (eval : Board.Board → Int) (sort : PublicationMir.SliceSort)
    (nodes : Nat) (sq : Bitboard.Square) :
    PublicationMir.ExactScore.Ensures mc fc board
      (.bestMove ⟨⟨sq,0,[]⟩,[],⟨0,5⟩,false,0⟩) ∧
    PublicationMir.ExactScore.Ensures mc fc board .noLegalMove := by
  have h1 := (PublicEndgame.exact_endgame_sound mc fc unitOpRun unitBranchComplete board _ false
    (.nonExactExit eval sort none none nodes (.randomOpening sq) (fun h => by cases h) _ rfl)).1
  have h2 := (PublicEndgame.exact_endgame_sound mc fc unitOpRun unitBranchComplete board _ false
    (.nonExactExit eval sort none none nodes .noLegalMove (fun h => by cases h) _ rfl)).1
  exact ⟨h1,h2⟩

end NeuralReversi.PvsParallel.Interleaving.PublicEndgameFixture
