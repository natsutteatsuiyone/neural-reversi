import NeuralReversi.PvsParallel.Interleaving.RootFinite
import NeuralReversi.AspirationMir.SerialTable.MultiPublic

namespace NeuralReversi.PvsParallel.Interleaving.RootLadder

open AspirationMir
open AspirationMir.SerialTable
open PvsBoardTree (Position GeneratedSquare)
open PvsMoveLoop (ChildRelation scoreMax)
open RootMoveStorage (Row sortRows)
open SerialCalls.ScoreWords (Finite)
open Aspiration (SoundResult Window)
open SelectiveLadder (Start)
open SerialCache (Receipt)

abbrev Children (mc : Bitboard.Moves.X86Config) (b : Position) := Nat → ChildRelation (GeneratedSquare mc b.val)

/-- A scheduler may supply any completed run for this call. A missing return ends
    the driver, including when the next action would block or the owner was stopped. -/
inductive Attempt (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position) (idx : Nat)
    (s : SerialStack.State) (done_ : ChildRelation (GeneratedSquare mc b.val)) : SmallRoot.Answer → Prop where
  | searched (run : RootSchedule.Trial mc b s.rows idx s.regs.alpha.toInt s.regs.beta.toInt done_ score rows) :
      Attempt mc fc b idx s done_ ⟨score,rows⟩
  | wiped (pv : Bitboard.Square → List Bitboard.Square)
      (cached : (MoveList.Generation.fill false paired fc b.val
        (Root.remainingMask mc b.val (s.rows.map Row.square) idx) mem).wipeout = some (some sq)) :
      Attempt mc fc b idx s done_ ⟨scoreMax,FrameReturns.wipeRows s.rows idx sq pv⟩

structure Reply (M : Type) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (idx : Nat) (s : SerialStack.State) (done_ : ChildRelation (GeneratedSquare mc b.val)) where
  closing : Nat
  answer : SmallRoot.Answer
  after : M
  ran : Window s.regs.alpha.toInt s.regs.beta.toInt → Attempt mc fc b idx s done_ answer

abbrev Source (M : Type) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position) (done_ : Children mc b) :=
  (idx level : Nat) → (s : SerialStack.State) → M → Option (Reply M mc fc b idx s (done_ level))

def trial (source : Source M mc fc b done_) (idx level : Nat) : SerialCache.Trial M := fun s m =>
  (source idx level s m).map fun r => (r.closing,r.answer,r.after)

structure Laws (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position) (done_ : Children mc b) : Prop where
  finite : ∀ level, level ≠ 5 → ∀ m pv a z v, done_ level m pv a z v → Finite v
  exact : ∀ m pv a z v, Window a z → (pv = false → z = a+1) → done_ 5 m pv a z v →
    SoundResult (-(Root.squareValue mc fc b m.val)) a z v

theorem Attempt.properties (run : Attempt mc fc b idx s done_ ans)
    (weak : lvl ≠ 5 → ∀ m p a z v, done_ m p a z v → Finite v)
    (exact5 : lvl = 5 → ∀ m p a z v, Window a z → (p = false → z = a+1) → done_ m p a z v →
      SoundResult (-(Root.squareValue mc fc b m.val)) a z v)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (win : Window s.regs.alpha.toInt s.regs.beta.toInt) :
    Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square ∧ ans.rows.take idx = s.rows.take idx ∧
      MultiLevel.Certificate mc fc b idx lvl s ans := by
  cases run with
  | searched path =>
    by_cases full : lvl = 5
    · subst lvl
      obtain ⟨snap,best,writes,sound⟩ := path.certified cover pos win (exact5 rfl)
      have rows := writes.scores (Root.roots_nodup cover)
      exact ⟨⟨sound.1.lower,sound.1.upper⟩,rows.1,rows.2.1,fun _ => ⟨snap,best.symm,writes,sound⟩⟩
    · obtain ⟨fin,keys,front⟩ := path.finite cover pos (weak full)
      exact ⟨fin,keys,front,fun equal => (full equal).elim⟩
  | @wiped paired mem sq pv cached =>
    obtain ⟨move,legal,eq,wipes⟩ := ConcurrentReplay.generatedWipeout mc fc b (s.rows.map Row.square) idx paired mem sq cover cached
    subst sq
    have kept := (Root.choices_squares_mem (idx := idx) cover move.val).mp (List.mem_map_of_mem legal)
    have clear := RootSuffix.Writes.reset (initial := s.rows) (idx := idx) (alpha := s.regs.alpha.toInt) (beta := s.regs.beta.toInt)
    have names := (clear.scores (Root.roots_nodup cover)).1
    have pos : move.val ∈ ((RootMoveStorage.resetFrom s.rows idx).drop idx).map Row.square := by
      rw [List.map_drop,names]
      exact kept
    have written := RootSuffix.Writes.record clear ⟨move.val,s.regs.alpha.toInt,scoreMax⟩ true (pv move.val) pos rfl
    obtain ⟨keys,front,_⟩ := written.scores (Root.roots_nodup cover)
    refine ⟨⟨by change -32767 < (16384 : Int); decide,by change (16384 : Int) < 32767; decide⟩,keys,front,fun _ => ⟨_,by change (16384 : Int) = max (-32767) 16384; decide,written,?_⟩⟩
    exact RootReplay.wipeoutContract_proven mc fc b _ idx _ _ move cover (by simpa using ‹idx < s.rows.length›) win legal wipes
theorem properties (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (hit : trial source idx lvl s m = some (k,ans,after)) (cover : Root.Covers mc b (s.rows.map Row.square))
    (pos : idx < s.rows.length) (win : Window s.regs.alpha.toInt s.regs.beta.toInt) :
    Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square ∧ ans.rows.take idx = s.rows.take idx ∧
      MultiLevel.Certificate mc fc b idx lvl s ans := by
  obtain ⟨reply,_,eq⟩ := Option.map_eq_some_iff.mp hit
  cases eq
  exact (reply.ran win).properties (laws.finite _) (fun h => h ▸ laws.exact) cover pos win

theorem retries (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (s : SerialStack.State) (cover : Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (ready : Ready n s.regs)
    (ran : SerialCache.run (trial source idx lvl) fuel s m = some (prior,last)) :
    MultiLevel.Accepted mc fc b idx lvl s.rows (fun _ : M => True) last := by
  induction fuel generalizing n s m prior last with
  | zero => simp [SerialCache.run] at ran
  | succ fuel ih =>
    obtain ⟨⟨k,ans,after⟩,ret,tail⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨finite,names,front,cert⟩ := properties source laws ret cover pos ready.window
    let v : Receipt M := ⟨⟨s,k,ans⟩,m,after⟩
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by have := finite.1; omega) (by have := finite.2; omega)
    have bounded : Finite (BitVec.ofInt 32 ans.value).toInt := by rw [signed]; exact finite
    rcases ready_widen ready _ bounded with accept | ⟨regs,wide,next⟩
    · simp only [SerialStack.Visit.call,accept,bind,Option.bind,Bool.false_eq_true,ite_false,pure,Pure.pure,Option.some.injEq] at tail
      cases tail
      exact ⟨trivial,rfl,rfl,names,front,ready.window,by simpa only [signed] using widen_false accept,signed,cert⟩
    · simp only [SerialStack.Visit.call,wide,bind,Option.bind,ite_true] at tail
      obtain ⟨⟨ps,final⟩,more,eq⟩ := Option.bind_eq_some_iff.mp tail
      cases eq
      have len : ans.rows.length = s.rows.length := by simpa using congrArg List.length names
      have ok := ih (SerialStack.next v.toVisit regs) (names ▸ cover) (len ▸ pos) next more
      exact ⟨trivial,ok.keys.trans names,ok.front.trans front,ok.afterKeys.trans names,
        ok.afterPrefix.trans front,ok.window,ok.inside,ok.signed,ok.certificate⟩

theorem stage (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (ok : MultiLevel.Valid mc b idx (fun _ : M => True) s)
    (ran : MultiLevel.advance idx (trial source idx) s lvl = some out) :
    ∃ last, out = MultiLevel.successor idx last ∧ MultiLevel.Accepted mc fc b idx lvl s.rows (fun _ : M => True) last ∧
      MultiLevel.Valid mc b idx (fun _ : M => True) out ∧ (out.rows.map Row.square).Perm (s.rows.map Row.square) ∧
      out.rows.take idx = s.rows.take idx := by
  obtain ⟨⟨prior,last⟩,steps,eq⟩ := Option.bind_eq_some_iff.mp ran
  cases eq
  have post := retries source laws _ ok.cover ok.index (first_ready _ ok.window) steps
  have sorted := RootMoveStorage.sortRowsContract_proven last.answer.rows idx
  have perm := (sorted.1.map Row.square).trans (List.Perm.of_eq post.afterKeys)
  have size : (MultiLevel.successor idx last).rows.length = s.rows.length := by simpa [MultiLevel.successor, SerialStack.first] using perm.length_eq
  have interior := post.inside
  rw [← post.signed] at interior
  exact ⟨last,rfl,post,⟨perm.trans ok.cover,size ▸ ok.index,trivial,rfl,
    SerialEntry.accepted_next_window post.window (widen_inside _ _ interior.1 interior.2)⟩,
    perm,sorted.2.1.trans post.afterPrefix⟩

theorem stages (source : Source M mc fc b done_) (laws : Laws mc fc b done_) (levels : List Nat)
    (ok : MultiLevel.Valid mc b idx (fun _ : M => True) s)
    (ran : levels.foldlM (MultiLevel.advance idx (trial source idx)) s = some out) :
    MultiLevel.Valid mc b idx (fun _ : M => True) out ∧ (out.rows.map Row.square).Perm (s.rows.map Row.square) ∧
      out.rows.take idx = s.rows.take idx := by
  induction levels generalizing s with
  | nil =>
    cases ran
    exact ⟨ok,.refl _,rfl⟩
  | cons lvl rest ih =>
    obtain ⟨middle,first,tail⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨_,_,_,good,keys,front⟩ := stage source laws ok first
    obtain ⟨valid,names,pre⟩ := ih good tail
    exact ⟨valid,names.trans keys,pre.trans front⟩

theorem accepted_line (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (cover : Root.Covers mc b (rows.map Row.square)) (within : idx < rows.length)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (prev : ∀ v ∈ MultiLine.previous idx rows, -scoreMax ≤ v)
    (ran : MultiLine.run idx base (trial source idx) OuterMir.ladder rows m = some out) :
    ∃ active last, out = MultiLevel.successor idx last ∧ MultiLevel.Valid mc b idx (fun _ : M => True) active ∧
      MultiLevel.Accepted mc fc b idx 5 active.rows (fun _ : M => True) last ∧
      MultiLevel.Valid mc b idx (fun _ : M => True) out ∧
      (out.rows.map Row.square).Perm (active.rows.map Row.square) ∧
      (active.rows.map Row.square).Perm (rows.map Row.square) ∧ active.rows.take idx = rows.take idx := by
  let s : Start _ := ⟨rows,MultiLine.seed idx base rows,m⟩
  have win := SerialEntry.windowMachine_valid (RootAspiration.seedWindowsContract_proven.1 idx base.toInt _ range.1 range.2 prev)
  change ((OuterMir.ladder.take 4) ++ [5]).foldlM _ s = some out at ran
  rw [List.foldlM_append] at ran
  obtain ⟨active,early,late⟩ := Option.bind_eq_some_iff.mp ran
  obtain ⟨ok,keys,front⟩ := stages source laws (OuterMir.ladder.take 4) (s := s) ⟨cover,within,trivial,rfl,win⟩ early
  have ran5 : MultiLevel.advance idx (trial source idx) active 5 = some out := by simpa using late
  obtain ⟨last,rfl,accepted,post,names,_⟩ := stage source laws ok ran5
  exact ⟨active,last,rfl,ok,accepted,post,names,keys,front⟩

theorem full_line (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (cover : Root.Covers mc b (rows.map Row.square)) (within : idx < rows.length)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (prev : ∀ v ∈ MultiLine.previous idx rows, -scoreMax ≤ v)
    (ran : MultiLine.run idx base (trial source idx) OuterMir.ladder rows m = some out) :
    Root.Covers mc b (out.rows.map Row.square) ∧ out.rows.length = rows.length ∧
      (∀ v ∈ MultiLine.previous (idx+1) out.rows, -scoreMax ≤ v) ∧
      ∀ final, RootSuffix.Lines mc fc b (idx+1) out.rows final → RootSuffix.Lines mc fc b idx rows final := by
  obtain ⟨active,last,rfl,ok,accepted,post,names,keys,front⟩ := accepted_line source laws cover within range prev ran
  obtain ⟨snap,best,written,sound⟩ := accepted.certificate rfl
  have before : Root.Covers mc b (last.state.rows.map Row.square) := accepted.keys ▸ ok.cover
  have stateLen : last.state.rows.length = active.rows.length := by simpa using congrArg List.length accepted.keys
  have inside : last.state.regs.alpha.toInt < snap.state.best ∧ snap.state.best < last.state.regs.beta.toInt := best ▸ accepted.inside
  rw [best] at sound
  obtain ⟨row,chosen,_,bounded⟩ := MultiRows.accepted_previous before (stateLen ▸ ok.index) written inside sound
  have ordered : ((MultiLevel.successor idx last).rows.map Row.square).Perm (rows.map Row.square) := names.trans keys
  refine ⟨post.cover,by simpa using ordered.length_eq,?_,?_⟩
  · intro v mem
    change ((sortRows last.answer.rows idx)[idx]?.map Row.score) = some v at mem
    rw [chosen] at mem
    exact Option.some.inj mem ▸ bounded.1
  · intro final rest
    apply RootSuffix.Lines.acceptedWrites before ((List.Perm.of_eq accepted.keys).trans keys)
      (accepted.front.trans front) (stateLen ▸ ok.index) written inside sound rest

theorem exact_lines (source : Source M mc fc b done_) (laws : Laws mc fc b done_)
    (levels : Nat → List Nat) (count idx : Nat) (rows : List Row)
    (cover : Root.Covers mc b (rows.map Row.square)) (size : idx + count = rows.length)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax)
    (prev : ∀ v ∈ MultiLine.previous idx rows, -scoreMax ≤ v)
    (all : ∀ i, idx ≤ i → i < idx + count → levels i = OuterMir.ladder)
    (ran : MultiBatch.lines (trial source) levels base count idx rows m = some (final,after)) :
    RootSuffix.Lines mc fc b idx rows final := by
  induction count generalizing idx rows m with
  | zero =>
    cases ran
    exact (show idx = final.length by omega) ▸ .done final
  | succ count ih =>
    obtain ⟨out,first,tail⟩ := Option.bind_eq_some_iff.mp ran
    rw [all idx (by omega) (by omega)] at first
    obtain ⟨covers,len,seed,step⟩ := full_line source laws cover (by omega) range prev first
    exact step _ (ih _ _ covers (by omega) seed (fun i start stop => all i (by omega) (by omega)) tail)

end NeuralReversi.PvsParallel.Interleaving.RootLadder
