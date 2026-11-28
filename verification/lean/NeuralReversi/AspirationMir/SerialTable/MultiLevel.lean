module
public import NeuralReversi.PvsParallel.SerialTable.MultiExact
public import NeuralReversi.PvsParallel.SerialTable.MultiWeak

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiLevel

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open PvsParallel
open PvsParallel.SerialTable (Memory Layout)
open Repeated (ReadyMemory)
open SerialCache (Receipt)
open SelectiveLadder (Start)
open RootMoveStorage (Row sortRows)

def trial (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Nat) : SerialCache.Trial (Memory l) := fun s m =>
  let f := ops.order level s
  if weak : level < 5 then
    PvsParallel.SerialTable.MultiWeak.trial idx p c b (f.base (fc := fc)) f.root ⟨level,by omega⟩ (ops.screen level s) s m
  else PvsParallel.SerialTable.MultiExact.trial idx p c b (f.base (fc := fc)) f.root s m

def Certificate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (idx level : Nat)
    (s : SerialStack.State) (ans : SmallRoot.Answer) : Prop :=
  level = 5 → PvsParallel.SerialTable.MultiExact.Certified mc fc b idx s ans

theorem returns (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Nat) (s : SerialStack.State) (m : Memory l) (addr : c.address = p.address)
    (safe : ReadyMemory p mc fc m) (cover : Root.Covers mc b (s.rows.map Row.square)) (index : idx < s.rows.length)
    (win : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans after, trial idx p c b ops fc level s m = some (k,ans,after) ∧ ReadyMemory p mc fc after ∧
      SerialCalls.ScoreWords.Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square ∧
      ans.rows.take idx = s.rows.take idx ∧ Certificate mc fc b idx level s ans := by
  let f := ops.order level s
  by_cases weak : level < 5
  · obtain ⟨k,ans,after,ran,good,finite,keys,front⟩ := PvsParallel.SerialTable.MultiWeak.trial_returns
      idx p c addr b (f.base (fc := fc)) f.root ⟨level,by omega⟩ weak (ops.screen level s) s m safe cover index win
    exact ⟨k,ans,after,by simpa [trial,weak,f] using ran,good,finite,keys,front,fun eq => by omega⟩
  · obtain ⟨k,ans,after,ran,cert,good⟩ := PvsParallel.SerialTable.MultiExact.trial_returns idx p c b
      (f.base (fc := fc)) f.root s m addr safe cover index win
    have certificate := cert
    obtain ⟨snap,_,writes,sound⟩ := cert
    have rowKeys := writes.scores (Root.roots_nodup cover)
    exact ⟨k,ans,after,by simpa [trial,weak,f] using ran,good,⟨sound.1.lower,sound.1.upper⟩,
      rowKeys.1,rowKeys.2.1,fun _ => certificate⟩

structure Accepted (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (idx level : Nat) (start : List Row) (good : M → Prop) (last : Receipt M) : Prop where
  memory : good last.after
  keys : last.state.rows.map Row.square = start.map Row.square
  front : last.state.rows.take idx = start.take idx
  afterKeys : last.answer.rows.map Row.square = start.map Row.square
  afterPrefix : last.answer.rows.take idx = start.take idx
  window : Aspiration.Window last.state.regs.alpha.toInt last.state.regs.beta.toInt
  inside : last.state.regs.alpha.toInt < last.answer.value ∧ last.answer.value < last.state.regs.beta.toInt
  signed : last.call.score.toInt = last.answer.value
  certificate : Certificate mc fc b idx level last.state last.answer

theorem retries (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level fuel : Nat) (s : SerialStack.State) (m : Memory l) (addr : c.address = p.address)
    (good : ReadyMemory p mc fc m) (cover : Root.Covers mc b (s.rows.map Row.square)) (index : idx < s.rows.length)
    (ready : Ready n s.regs) (budget : n + fuel = 12) :
    ∃ prior last, SerialCache.run (trial idx p c b ops fc level) fuel s m = some (prior,last) ∧
      Accepted mc fc b idx level s.rows (ReadyMemory p mc fc) last := by
  induction fuel generalizing n s m with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    obtain ⟨k,ans,after,ran,valid,fin,names,front,cert⟩ := returns idx p c b ops fc level s m addr good cover index ready.window
    let v : Receipt _ := ⟨⟨s,k,ans⟩,m,after⟩
    obtain ⟨lo,hi⟩ := fin
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    have bounded : -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ (BitVec.ofInt 32 ans.value).toInt < 32767 :=
      by rw [signed]; exact ⟨lo,hi⟩
    rcases ready_widen ready _ bounded with done_ | ⟨regs,wide,nextReady⟩
    · refine ⟨[],v,by simp [v,SerialCache.run,ran,SerialStack.Visit.call,done_],
        valid,rfl,rfl,names,front,ready.window,?_,signed,cert⟩
      simpa only [signed] using widen_false done_
    · have size : ans.rows.length = s.rows.length := by simpa using congrArg List.length names
      obtain ⟨prior,last,rest,ok⟩ := ih (SerialStack.next v.toVisit regs) after valid (names ▸ cover)
        (size ▸ index) nextReady (by omega)
      refine ⟨v::prior,last,by dsimp only [v] at rest; simp [SerialCache.run,ran,SerialStack.Visit.call,wide,rest,v],
        ok.memory,ok.keys.trans names,ok.front.trans front,ok.afterKeys.trans names,
        ok.afterPrefix.trans front,ok.window,ok.inside,ok.signed,ok.certificate⟩

def successor (idx : Nat) (v : Receipt M) : Start M :=
  ⟨sortRows v.answer.rows idx,SerialEntry.nextMachine v.call.score,v.after⟩

def advance (idx : Nat) (t : Nat → SerialCache.Trial M) (s : Start M) (lvl : Nat) : Option (Start M) := do
  let (_,last) ← SerialCache.run (t lvl) 12 (SerialStack.first s.machine s.rows) s.memory
  return successor idx last

structure Valid (mc : Bitboard.Moves.X86Config) (b : Position) (idx : Nat) (good : M → Prop) (s : Start M) : Prop where
  cover : Root.Covers mc b (s.rows.map Row.square)
  index : idx < s.rows.length
  safe : good s.memory
  pc : s.machine.pc = 0
  window : Aspiration.Window s.machine.regs.alpha.toInt s.machine.regs.beta.toInt

theorem stage (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Nat) (s : Start (Memory l)) (addr : c.address = p.address)
    (ok : Valid mc b idx (ReadyMemory p mc fc) s) :
    ∃ last, advance idx (trial idx p c b ops fc) s level = some (successor idx last) ∧
      Accepted mc fc b idx level s.rows (ReadyMemory p mc fc) last ∧
      Valid mc b idx (ReadyMemory p mc fc) (successor idx last) ∧
      ((successor idx last).rows.map Row.square).Perm (s.rows.map Row.square) ∧
      (successor idx last).rows.take idx = s.rows.take idx := by
  obtain ⟨prior,last,ran,post⟩ := retries idx p c b ops fc level 12 (SerialStack.first s.machine s.rows)
    s.memory addr ok.safe ok.cover ok.index (first_ready _ ok.window) rfl
  have sorted := RootMoveStorage.sortRowsContract_proven last.answer.rows idx
  have perm := (sorted.1.map Row.square).trans (List.Perm.of_eq post.afterKeys)
  have size : (successor idx last).rows.length = s.rows.length := by simpa [successor] using (show ((sortRows last.answer.rows idx).map Row.square).length = (s.rows.map Row.square).length from perm.length_eq)
  have interior := post.inside
  rw [← post.signed] at interior
  exact ⟨last,by simp [advance,ran],post,
    ⟨perm.trans ok.cover,size ▸ ok.index,post.memory,rfl,
      SerialEntry.accepted_next_window post.window (widen_inside _ _ interior.1 interior.2)⟩,
    perm,sorted.2.1.trans post.afterPrefix⟩

end NeuralReversi.AspirationMir.SerialTable.MultiLevel
