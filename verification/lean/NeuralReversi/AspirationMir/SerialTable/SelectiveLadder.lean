import NeuralReversi.AspirationMir.SerialTable.FromBoard
import NeuralReversi.AspirationMir.SerialEntry
import NeuralReversi.PvsParallel.SerialTable.Finite

namespace NeuralReversi.AspirationMir.SerialTable.SelectiveLadder

open PvsParallel
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open SerialCache (Trial Receipt Path)

/-- Earlier root calls may return a different approximate score at every window. -/
def FiniteTrial (mc : Bitboard.Moves.X86Config) (b : Position) (good : M → Prop) (t : Trial M) : Prop :=
  ∀ s before, good before → Root.Covers mc b (s.rows.map Row.square) → 0 < s.rows.length →
    Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt →
    ∃ k ans after, t s before = some (k,ans,after) ∧ good after ∧
      -32767 < ans.value ∧ ans.value < 32767 ∧ (ans.rows.map Row.square).Perm (s.rows.map Row.square)

structure Accepted (mc : Bitboard.Moves.X86Config) (b : Position) (good : M → Prop)
    (last : Receipt M) : Prop where
  safe : good last.after
  cover : Root.Covers mc b (last.answer.rows.map Row.square)
  positive : 0 < last.answer.rows.length
  window : Aspiration.Window last.state.regs.alpha.toInt last.state.regs.beta.toInt
  inside : last.state.regs.alpha.toInt < last.call.score.toInt ∧ last.call.score.toInt < last.state.regs.beta.toInt
  value : last.call.score.toInt = last.answer.value

theorem level {M : Type} {t : Trial M} {good : M → Prop} (works : FiniteTrial mc b good t)
    (fuel : Nat) (s : SerialStack.State) (before : M)
    (initial : good before) (cover : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
    (ready : Ready n s.regs) (budget : n + fuel = 12) :
    ∃ prior last, SerialCache.run t fuel s before = some (prior,last) ∧ Path t s before prior last ∧
      prior.length < fuel ∧ Accepted mc b good last := by
  induction fuel generalizing n s before with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    obtain ⟨k,ans,after,returned,safe,lo,hi,names⟩ := works s before initial cover positive ready.window
    let v : Receipt M := ⟨⟨s,k,ans⟩,before,after⟩
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    have finite : -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ (BitVec.ofInt 32 ans.value).toInt < 32767 := by
      simpa only [signed] using And.intro lo hi
    have covered := names.trans cover
    have size : ans.rows.length = s.rows.length := by simpa using names.length_eq
    have pos : 0 < ans.rows.length := size ▸ positive
    rcases ready_widen ready _ finite with stop | ⟨regs,wide,nextReady⟩
    · exact ⟨[],v,by simp [SerialCache.run,v,returned,SerialStack.Visit.call,stop],.accept returned stop,
        Nat.zero_lt_succ _,⟨safe,covered,pos,ready.window,widen_false stop,signed⟩⟩
    · obtain ⟨prior,last,ran,path,short,accepted⟩ :=
        ih (SerialStack.next v.toVisit regs) after safe covered pos nextReady (by omega)
      refine ⟨v::prior,last,?_,.retry returned wide path,by simpa using short,accepted⟩
      dsimp only [v] at ran ⊢
      simp [SerialCache.run,returned,SerialStack.Visit.call,wide,ran]

structure Start (M : Type) where
  rows : List Row
  machine : Machine
  memory : M

def Start.valid (s : Start M) (mc : Bitboard.Moves.X86Config) (b : Position) (good : M → Prop) : Prop :=
  Root.Covers mc b (s.rows.map Row.square) ∧ 0 < s.rows.length ∧ good s.memory ∧ s.machine.pc = 0 ∧
    Aspiration.Window s.machine.regs.alpha.toInt s.machine.regs.beta.toInt

def successor (last : Receipt M) : Start M :=
  ⟨RootMoveStorage.sortRows last.answer.rows 0,SerialEntry.nextMachine last.call.score,last.after⟩

theorem Accepted.successor_valid (accepted : Accepted mc b good last) : (successor last).valid mc b good := by
  have sorted := (RootMoveStorage.sortRowsContract_proven last.answer.rows 0).1
  exact ⟨(sorted.map Row.square).trans accepted.cover,sorted.length_eq ▸ accepted.positive,accepted.safe,rfl,
    SerialEntry.accepted_next_window accepted.window (widen_inside _ _ accepted.inside.1 accepted.inside.2)⟩

def advance (trials : Nat → Trial M) (s : Start M) (lvl : Nat) : Option (Start M) := do
  let (_,last) ← SerialCache.run (trials lvl) 12 (SerialStack.first s.machine s.rows) s.memory
  return successor last

/-- Each step records the actual retries, verifies the accepted window, sorts its output rows,
    and passes the same mutable state to the next selectivity. -/
inductive Levels (mc : Bitboard.Moves.X86Config) (b : Position) (good : M → Prop) (trials : Nat → Trial M) :
    List Nat → Start M → Start M → Prop where
  | done : Levels mc b good trials [] s s
  | more (ran : SerialCache.run (trials lvl) 12 (SerialStack.first s.machine s.rows) s.memory = some (prior,last))
      (path : Path (trials lvl) (SerialStack.first s.machine s.rows) s.memory prior last) (limit : prior.length < 12)
      (accepted : Accepted mc b good last)
      (rest : Levels mc b good trials tail (successor last) out) : Levels mc b good trials (lvl::tail) s out

theorem ladder (levels : List Nat) (s : Start M) (initial : s.valid mc b good)
    (works : ∀ lvl ∈ levels, FiniteTrial mc b good (trials lvl)) :
    ∃ out, levels.foldlM (advance trials) s = some out ∧ Levels mc b good trials levels s out ∧ out.valid mc b good := by
  induction levels generalizing s with
  | nil => exact ⟨s,rfl,.done,initial⟩
  | cons lvl rest ih =>
    obtain ⟨cover,pos,safe,_,window⟩ := initial
    obtain ⟨prior,last,ran,path,bound,accepted⟩ := level (works lvl List.mem_cons_self) 12
      (SerialStack.first s.machine s.rows) s.memory safe cover pos (first_ready _ window) rfl
    obtain ⟨out,fold,more,valid⟩ := ih (successor last) accepted.successor_valid (fun x h => works x (.tail _ h))
    exact ⟨out,by simpa [List.foldlM_cons,advance,ran] using fold,.more ran path bound accepted more,valid⟩

open EndgameSearch.CacheHistory (Pool)
open PvsParallel.SerialTable (Layout Memory)

/-- The earlier stages share the current TT and both worker arrays with the exact fifth stage. -/
def ColdContract : Prop := ∀ (l : Layout) (g : BitVec 8) (p : Pool) mc fc (c : SerialScout.Config) (b : Position)
    (trials : Nat → Trial (Memory l)) (orders : SerialStack.State → MoveInputs mc)
    (birth : MoveList.Initialization.Memory Move.Move) (base : Word) certificate sort nodes fallback,
    c.address = p.address →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc b.val = true → -PvsMoveLoop.scoreMax ≤ base.toInt → base.toInt ≤ PvsMoveLoop.scoreMax →
    (∀ lvl ∈ OuterMir.ladder.take 4, FiniteTrial mc b (Memory.Valid p mc fc) (trials lvl)) →
    let rows := NativeRootBirth.initial mc b
    let start : Start (Memory l) := ⟨rows,SerialEntry.firstMachine base,Memory.empty l g p⟩
    NativeRootBirth.started mc fc b c.paired birth = some rows ∧
    ∃ final, (OuterMir.ladder.take 4).foldlM (advance trials) start = some final ∧
      Levels mc b (Memory.Valid p mc fc) trials (OuterMir.ladder.take 4) start final ∧
      Published mc fc b (search p c b orders fc) (Memory.Good p mc fc) (recorded (search p c b orders fc))
        final.rows final.machine final.memory sort nodes fallback

theorem cold : ColdContract := by
  intro l g p mc fc c b trials orders birth base check sort nodes fallback addr verified mobile lower upper works rows s
  have started := started_of_certificate mc fc b c.paired birth verified mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers started
  obtain ⟨final,ran,path,squares,nonempty,good,pc,window⟩ := ladder _ s
    ⟨cover,List.length_pos_iff.mpr pos,Memory.cold_valid ..,rfl,SerialEntry.firstMachine_window base lower upper⟩ works
  exact ⟨started,final,ran,path,publication _ (SerialTable.works p c b orders fc addr) good.1 squares nonempty
    _ pc window sort nodes fallback⟩

end NeuralReversi.AspirationMir.SerialTable.SelectiveLadder
