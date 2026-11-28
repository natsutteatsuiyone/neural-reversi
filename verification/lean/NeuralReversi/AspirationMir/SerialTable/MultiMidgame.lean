module
public import NeuralReversi.AspirationMir.SerialTable.MidgameSession

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiMidgame

open PvsParallel
open PvsParallel.SerialTable (Memory Layout)
open PvsBoardTree (Position)
open EndgameSearch.CacheHistory (Pool)
open TranspositionTable (Selectivity)
open Repeated (ReadyMemory)
open RootMoveStorage (Row sortRows)
open MidgameIterate (State Clock Outcome Valid Complete Completed)
open MidgameAspiration (Attempt)
open PvsMoveLoop (scoreMax)

abbrev Inputs (l : Layout) := Nat → MidgameSession.Inputs l

def trial {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (level : Selectivity) (ops : Inputs l) :
    MidgameIterate.Trial (Memory l) := fun d s w =>
  let f := MidgameSession.frames (mc := mc) (fc := fc) (ops idx) c level b false d s w
  MidgameRoot.runAt idx mc fc b (BitVec.ofNat 32 d) c.paired f.scratch f.order level
    (PvsParallel.SerialTable.MidgameChild.run p f.children) s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta) s.memory

def Works (idx : Nat) (mc : Bitboard.Moves.X86Config) (b : Position) (safe : M → Prop) (t : MidgameIterate.Trial M) :=
  ∀ d s w, Valid mc b safe s → idx < s.rows.length → w.window →
    ∃ ans m, t d s w = some (ans,m) ∧ Valid mc b safe ⟨ans.rows,m⟩ ∧
      ans.rows.length = s.rows.length ∧ ans.rows.take idx = s.rows.take idx ∧ MidgameAspiration.FiniteScore ans.value ∧
      PvInterior.InsideRange w.alpha w.beta ans.value ∧ (w.alpha < ans.value → MidgameRoot.RowMaximum (ans.rows.drop idx) ans.value)

theorem works (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (level : Selectivity) (ops : Inputs l) :
    Works idx mc b (ReadyMemory p mc fc) (trial (mc := mc) (fc := fc) idx p c b level ops) := by
  intro d s w ok within win
  have ⟨a,az,z⟩ := win
  have repr {x : Int} (lo : -32767 ≤ x) (hi : x ≤ 32767) : (BitVec.ofInt 32 x).toInt = x :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have low := repr a (by omega)
  have high := repr (by omega) z
  let f := MidgameSession.frames (fc := fc) (mc := mc) (ops idx) c level b false d s w
  have children : MidgameRoot.ChildReturns (l := l) p mc fc (PvsParallel.SerialTable.MidgameChild.run p f.children) :=
    fun q m window _ ready => PvsParallel.SerialTable.MidgameChild.returns p f.children q m window ready
  obtain ⟨ans,m,ran,ready,finite,inside,keys,hist,prefixEq,highest⟩ := MidgameRoot.returnsAt idx p mc fc b
    (BitVec.ofNat 32 d) c.paired f.scratch f.order level _ s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta)
    s.memory children ok.2.2.2 ok.1 within (by rw [low,high]; exact ⟨a,az,z⟩)
  have count : ans.rows.length = s.rows.length := by simpa using congrArg List.length keys
  rw [low,high] at inside
  rw [low] at highest
  exact ⟨ans,m,ran,⟨keys ▸ ok.1,count ▸ ok.2.1,hist ok.2.2.1,ready⟩,count,prefixEq,finite,inside,highest⟩

def Bounded (row : Row) : Prop := -scoreMax ≤ row.score ∧ row.score ≤ scoreMax

private theorem retry_returns {t : MidgameIterate.Trial M} (works : Works idx mc b safe t) (s : State M)
    (ok : Valid mc b safe s) (within : idx < s.rows.length) (w : Attempt) (win : w.window) (pos : 0 < w.delta)
    (budget : MidgameAspiration.HasFuel fuel w) (d : Nat) :
    ∃ ans m, MidgameIterate.retry t d fuel s w = some (ans,m) ∧ Valid mc b safe ⟨ans.rows,m⟩ ∧
      ans.rows.length = s.rows.length ∧ ans.rows.take idx = s.rows.take idx ∧
      (-scoreMax ≤ ans.value ∧ ans.value ≤ scoreMax) ∧ MidgameRoot.RowMaximum (ans.rows.drop idx) ans.value := by
  induction fuel generalizing s w with
  | zero => contradiction
  | succ fuel ih =>
    obtain ⟨ans,m,ran,valid,len,front,finite,range,highest⟩ := works d s w ok within win
    cases wide : w.retry ans.value with
    | none =>
      obtain ⟨lo,hi⟩ := MidgameAspiration.retry_none wide
      exact ⟨ans,m,by simp [MidgameIterate.retry,ran,wide],valid,len,front,range lo hi,highest lo⟩
    | some next =>
      obtain ⟨window,width⟩ := MidgameAspiration.retry_window win finite pos wide
      obtain ⟨ans',after,more,good,size,prefixEq,limits,peak⟩ := ih _ valid (len ▸ within) _ window
        (by rw [width]; unfold MidgameAspiration.grow; omega) (budget _ finite _ wide)
      exact ⟨ans',after,by simpa [MidgameIterate.retry,ran,wide] using more,good,size.trans len,prefixEq.trans front,limits,peak⟩

theorem selected_score (rows : List Row) (idx : Nat) (pos : idx < rows.length)
    (pk : MidgameRoot.RowMaximum (rows.drop idx) v) (h : idx < (sortRows rows idx).length) :
    ((sortRows rows idx)[idx]).score = v := by
  let sorted := sortRows rows idx
  have order := RootMoveStorage.sortRowsContract_proven rows idx
  have inside : idx < sorted.length := h
  have rest : sorted.drop idx = (rows.drop idx).mergeSort (fun a b => decide (b.score ≤ a.score)) :=
    List.drop_left' (by simp [Nat.min_eq_left (Nat.le_of_lt pos)])
  have perm : (sorted.drop idx).Perm (rows.drop idx) := rest ▸ List.mergeSort_perm ..
  have headEq : (sorted.drop idx).head? = some sorted[idx] := by simpa only [List.head?_drop] using List.getElem?_eq_getElem inside
  obtain ⟨tail,cons⟩ := List.head?_eq_some_iff.mp headEq
  obtain ⟨⟨winner,seen,found⟩,upper⟩ := pk
  have eq : sorted[idx].score = v := by
    have bestMem : sorted[idx] ∈ sorted.drop idx := cons ▸ List.mem_cons_self
    apply Int.le_antisymm (upper _ (perm.mem_iff.mp bestMem))
    have ranked := order.2.2.1
    change (sorted.drop idx).Pairwise _ at ranked
    have present := perm.mem_iff.mpr seen
    rw [cons] at ranked present
    rw [← found]
    rcases List.mem_cons.mp present with rfl | later
    · exact Int.le_refl _
    · exact (List.pairwise_cons.mp ranked).1 _ later
  exact eq

private theorem choose (rows : List Row) (idx : Nat) (pos : idx < rows.length) (pk : MidgameRoot.RowMaximum (rows.drop idx) v)
    (bound : -scoreMax ≤ v ∧ v ≤ scoreMax) (earlier : ∀ r ∈ rows.take idx, Bounded r) :
    ∀ r ∈ (sortRows rows idx).take (idx+1), Bounded r := by
  let sorted := sortRows rows idx
  have order := RootMoveStorage.sortRowsContract_proven rows idx
  have inside : idx < sorted.length := order.1.length_eq ▸ pos
  have eq := selected_score rows idx pos pk inside
  intro r mem
  rw [List.take_succ_eq_append_getElem inside] at mem
  rcases List.mem_append.mp mem with old | added
  · exact earlier r (order.2.1 ▸ old)
  · have same := List.mem_singleton.mp added
    rw [same]
    unfold Bounded
    rw [eq]
    exact bound

def line (t : MidgameIterate.Trial M) (depth idx : Nat) (s : State M) : Option (State M) := do
  let w := MidgameAspiration.initial depth (s.rows[idx]?.map MidgameRows.stats) 0
  let (ans,m) ← MidgameIterate.retry t depth 16 s w
  return ⟨sortRows ans.rows idx,m⟩

theorem line_returns {t : MidgameIterate.Trial M} (works : Works idx mc b safe t) (depth : Nat)
    (s : State M) (ok : Valid mc b safe s) (within : idx < s.rows.length) :
    ∃ next, line t depth idx s = some next ∧ Valid mc b safe next ∧ next.rows.length = s.rows.length ∧
      ((∀ r ∈ s.rows.take idx, Bounded r) → ∀ r ∈ next.rows.take (idx+1), Bounded r) := by
  let hist := s.rows[idx]?.map MidgameRows.stats
  let w := MidgameAspiration.initial depth hist 0
  have initial := MidgameAspiration.initialContract_proven depth 0 hist (by
    intro h member
    obtain ⟨r,eq,rfl⟩ := Option.map_eq_some_iff.mp member
    exact ok.2.2.1 r (List.mem_of_getElem? eq))
  obtain ⟨ans,after,ran,valid,size,front,bounds,peak⟩ := retry_returns works s ok within w initial.1
    (Int.lt_of_lt_of_le (by decide : (0:Int) < 512) initial.2.1) (MidgameAspiration.enough_retries initial.1 initial.2.1) depth
  have order := RootMoveStorage.sortRowsContract_proven ans.rows idx
  let next : State M := ⟨sortRows ans.rows idx,after⟩
  have nextOk : Valid mc b safe next := ⟨(order.1.map Row.square).trans valid.1,order.1.length_eq ▸ valid.2.1,
    MidgameRows.perm order.1 valid.2.2.1,valid.2.2.2⟩
  refine ⟨next,?_,nextOk,order.1.length_eq.trans size,?_⟩
  · change (MidgameIterate.retry t depth 16 s w).bind _ = _
    rw [ran]; rfl
  · intro prior
    exact choose ans.rows idx (size ▸ within) peak bounds (front ▸ prior)

def lines (trials : Nat → MidgameIterate.Trial M) (depth : Nat) :
    Nat → Nat → State M → Option (State M)
  | 0, _, s => some s
  | n+1, idx, s => do
    let next ← line (trials idx) depth idx s
    lines trials depth n (idx+1) next

private theorem lines_return {trials : Nat → MidgameIterate.Trial M} (works : ∀ i, Works i mc b safe (trials i))
    (depth fuel idx : Nat) (s : State M) (ok : Valid mc b safe s) (count : idx + fuel = s.rows.length)
    (prefixOk : ∀ r ∈ s.rows.take idx, Bounded r) :
    ∃ final, lines trials depth fuel idx s = some final ∧ Valid mc b safe final ∧ ∀ r ∈ final.rows, Bounded r := by
  induction fuel generalizing idx s with
  | zero =>
    have last : idx = s.rows.length := by omega
    exact ⟨s,rfl,ok,by simpa [last] using prefixOk⟩
  | succ n ih =>
    obtain ⟨next,ran,nextOk,size,bounds⟩ := line_returns (works idx) depth s ok (by omega)
    obtain ⟨final,more,good,bounded⟩ := ih (idx+1) next nextOk (by omega) (bounds prefixOk)
    exact ⟨final,by simpa only [lines,ran,bind,Option.bind] using more,good,bounded⟩

def iteration (trials : Nat → MidgameIterate.Trial M) (sort : PublicationMir.SliceSort) (level depth : Nat)
    (s : State M) : Option (Completed M) := do
  let done_ ← lines trials depth s.rows.length 0 s
  let saved := PublicationMir.capture sort done_.rows depth level false
  let base ← PublicationMir.first saved
  return ⟨⟨sortRows done_.rows 0,done_.memory⟩,saved,base⟩

theorem iteration_works {trials : Nat → MidgameIterate.Trial M} (works : ∀ i, Works i mc b safe (trials i))
    (sort : PublicationMir.SliceSort) (level : Nat) : MidgameIterate.DepthWorks mc b safe level (iteration trials sort level) := by
  intro depth s ok
  obtain ⟨final,ran,good,bounded⟩ := lines_return works depth s.rows.length 0 s ok (by simp) (by simp)
  let saved := PublicationMir.capture sort final.rows depth level false
  have rows : saved.rows.Perm final.rows := sort.permutation _ _
  have present : saved.rows ≠ [] := by
    intro empty
    have h := rows.length_eq
    rw [empty] at h
    have pos := good.2.1
    simp at h; omega
  obtain ⟨base,rest,eq⟩ := List.exists_cons_of_ne_nil present
  have read : PublicationMir.first saved = some base := by simp [PublicationMir.first,eq]
  have sorted := (RootMoveStorage.sortRowsContract_proven final.rows 0).1
  refine ⟨⟨⟨sortRows final.rows 0,final.memory⟩,saved,base⟩,?_,⟨⟨(sorted.map Row.square).trans good.1,
    sorted.length_eq ▸ good.2.1,MidgameRows.perm sorted good.2.2.1,good.2.2.2⟩,read,rfl,rfl,rfl,?_⟩⟩
  · change (lines trials depth s.rows.length 0 s).bind _ = _
    rw [ran]
    change (PublicationMir.first saved).bind _ = _
    rw [read]; rfl
  · exact bounded base (rows.mem_iff.mp (eq ▸ List.mem_cons_self))

def prepareAt (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l) : Option (Outcome (Memory l)) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let depth := min (max requested 1) (Board.emptyCount q.board.val).toNat
  let step := iteration (fun i => trial (mc := mc) (fc := fc) i p c q.board level ops) q.sort level.val
  MidgameIterate.loopWith step depth clock (depth+2) (MidgameIterate.startDepth depth) ⟨rows,Repeated.increment m⟩

def prepare (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (clock : Option (Clock (Memory l))) (m : Memory l) : Option (Outcome (Memory l)) :=
  prepareAt q p c fc level ops (Board.emptyCount q.board.val).toNat clock m

theorem initial_valid (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (m : Memory l) (ok : ReadyMemory p mc fc m)
    (born : NativeRootBirth.started mc fc q.board c.paired q.scratch = some (NativeRootBirth.initial mc q.board)) :
    Valid mc q.board (ReadyMemory p mc fc) ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩ := by
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers born
  refine ⟨cover,List.length_pos_iff.mpr pos,?_,Repeated.begin_valid ok⟩
  intro r mem
  obtain ⟨_,_,rfl⟩ := List.mem_map.mp mem
  exact Or.inl rfl

theorem prepareAt_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ out, prepareAt q p c fc level ops requested clock m = some out ∧
      MidgameIterate.Ensures mc q.board (ReadyMemory p mc fc) level.val (min (max requested 1) (Board.emptyCount q.board.val).toNat) out := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  let depth := min (max requested 1) (Board.emptyCount q.board.val).toNat
  simpa only [prepareAt,born,bind,Option.bind] using MidgameIterate.loop_returns _
    (iteration_works (fun i => works (mc := mc) (fc := fc) i p c q.board level ops) q.sort level.val)
    ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩ (initial_valid q p c fc m ok born)
    depth clock (depth+2) (MidgameIterate.startDepth depth) (by omega)

theorem prepare_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (clock : Option (Clock (Memory l))) (m : Memory l) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ out, prepare q p c fc level ops clock m = some out ∧
      MidgameIterate.Ensures mc q.board (ReadyMemory p mc fc) level.val (Board.emptyCount q.board.val).toNat out := by
  simpa only [prepare,Nat.min_eq_right (Nat.le_max_left ..)] using prepareAt_returns q p c fc level ops (Board.emptyCount q.board.val).toNat clock m ok checked mobile

theorem reaches_endgame (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (m : Memory l) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ out, prepare q p c fc level ops (some MidgameIterate.continueToEnd) m = some (.endgame out) ∧
      Complete mc q.board (ReadyMemory p mc fc) level.val (Board.emptyCount q.board.val).toNat out := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨i,legal⟩ := (Board.hasLegalMoves_exact mc q.board.val).mp mobile
  let sq : PvsBoardTree.GeneratedSquare mc q.board.val := ⟨⟨i,legal.1⟩,(Board.getMoves_exact mc q.board.val _).mpr legal⟩
  let depth := (Board.emptyCount q.board.val).toNat
  have positive : 0 < depth := (ExactDepth.move_depth mc fc q.board _ sq rfl).1
  simp only [prepare,prepareAt,born,bind,Option.bind,Nat.min_eq_right (Nat.le_max_left ..)]
  apply MidgameIterate.loop_to_end _ (iteration_works (fun i => works (mc := mc) (fc := fc) i p c q.board level ops) q.sort level.val) _
    (initial_valid q p c fc m ok born)
  · unfold MidgameIterate.startDepth; change 0 < (Board.emptyCount q.board.val).toNat at positive; split <;> omega
  · unfold MidgameIterate.startDepth; split <;> omega
  · omega

end NeuralReversi.AspirationMir.SerialTable.MultiMidgame
