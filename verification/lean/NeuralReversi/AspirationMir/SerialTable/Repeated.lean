module
public import NeuralReversi.AspirationMir.SerialTable.FullLadder
public import NeuralReversi.AspirationMir.SerialTable.WarmEstimate
public import NeuralReversi.PvsParallel.SerialTable.ExactRetention
public import NeuralReversi.PvsParallel.SerialTable.WeakTrial.Preserves
public import NeuralReversi.TranspositionTable.Generation

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.Repeated

open PvsParallel
open PvsParallel.SerialTable (Memory Layout ExactScores)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open FullLadder (Inputs)

def ReadyMemory (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (m : Memory l) : Prop :=
  m.Valid p mc fc ∧ ExactScores m.tt.history

/-- A new request advances the seven-bit age used for TT replacement. -/
def increment (m : Memory l) : Memory l :=
  ⟨{m.tt with image := {m.tt.image with generation := TranspositionTable.Generation.increment m.tt.image.generation}},
    ⟨m.supported.even,m.supported.resident⟩,m.cache⟩

theorem begin_valid {m : Memory l} (ready : ReadyMemory p mc fc m) :
    ReadyMemory p mc fc (increment m) :=
  ⟨⟨⟨⟨(increment m).supported,ready.1.1.1.publications,ready.1.1.1.hints⟩,ready.1.1.2⟩,ready.1.2⟩,ready.2⟩

theorem earlier (p : Pool) (c : SerialScout.Config) (b : Position) (ops : Inputs mc)
    (fc : Cutoffs.FlipConfig) (addr : c.address = p.address) (lvl : Nat) (selected : lvl ∈ OuterMir.ladder.take 4) :
    SelectiveLadder.FiniteTrial mc b (ReadyMemory (l := l) p mc fc) (FullLadder.trial p c b ops fc lvl) := by
  have weak : lvl < 5 := by simp [OuterMir.ladder] at selected; omega
  intro s m ready cover pos window
  obtain ⟨k,ans,next,ran,valid,low,high,perm⟩ :=
    FullLadder.earlier_returns p c b ops fc addr lvl selected s m ready.1 cover pos window
  have actual := ran
  simp only [FullLadder.trial,weak,↓reduceDIte] at actual
  have kept := PvsParallel.SerialTable.WeakTrial.preserves p c addr b ((ops.order lvl s).base (fc := fc))
    (ops.order lvl s).root ⟨lvl,by omega⟩ weak (ops.screen lvl s) s m ready.1 ready.2 cover pos window actual
  exact ⟨k,ans,next,ran,⟨valid,kept.1⟩,low,high,perm⟩

theorem exact_trial (p : Pool) (c : SerialScout.Config) (b : Position)
    (orders : SerialStack.State → MoveInputs mc) (fc : Cutoffs.FlipConfig) (addr : c.address = p.address) :
    let t := search (l := l) p c b orders fc
    Works mc fc b t (ReadyMemory p mc fc) (recorded t) := by
  intro t s m ready cover pos window
  obtain ⟨k,ans,next,ran,cert,good,clock,ev⟩ := works p c b orders fc addr s m ready.1.1 cover pos window
  obtain ⟨finite,bounded⟩ := PvsParallel.SerialTable.ExactRetention.trial ready.1.2 ready.2 ran good
  exact ⟨k,ans,next,ran,cert,⟨⟨good,finite⟩,bounded⟩,clock,ev⟩

structure Call (mc : Bitboard.Moves.X86Config) where
  board : Position
  last : {n : Nat // 1 ≤ n ∧ n ≤ 5}
  moves : Inputs mc
  raw : Position → Int
  scratch : MoveList.Initialization.Memory Move.Move
  sort : PublicationMir.SliceSort
  nodes : Nat
  fallback : Unit → PublicationMir.SearchResult

def Call.expected (q : Call mc) (fc : Cutoffs.FlipConfig) : Option Int :=
  if q.last.val = 5 then some (256 * EndgameTreeCache.discValue mc fc q.board.val) else none

private theorem weak_report (sort : PublicationMir.SliceSort) (rows : List Row) (depth lvl nodes fallback)
    (present : 0 < rows.length) (weak : lvl ≠ 5) :
    ∃ r, PublicationMir.completed sort rows depth lvl true nodes fallback = some r ∧ PublicationMir.ExactScore.scoreIfExact r = none := by
  let captured := PublicationMir.capture sort rows depth lvl true
  have nonempty : captured.rows ≠ [] := by
    intro missing
    have perm := sort.permutation PublicationMir.key rows
    change (sort.byReverseKey PublicationMir.key rows) = [] at missing
    rw [missing] at perm
    have := perm.length_eq
    simp only [List.length_nil] at this
    omega
  obtain ⟨first,tail,ordered⟩ := List.exists_cons_of_ne_nil nonempty
  have get : PublicationMir.first captured = some first := by simp [PublicationMir.first,ordered]
  have finished : PublicationMir.completed sort rows depth lvl true nodes fallback =
      some (PublicationMir.build captured.rows first depth lvl true nodes) := by
    dsimp only [captured] at get
    simp [PublicationMir.completed,PublicationMir.record,PublicationMir.intoResult,get,captured]
    rfl
  exact ⟨_,finished,by simp [PublicationMir.build,PublicationMir.ExactScore.scoreIfExact,weak]⟩

/-- Enter the selectivity ladder with the current rows, aspiration window and memory. -/
def finish (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Call mc)
    (start : SelectiveLadder.Start (Memory l)) := do
  let final ← ((OuterMir.ladder.take 4).take q.last.val).foldlM (SelectiveLadder.advance (FullLadder.trial p c q.board q.moves fc)) start
  if q.last.val ≠ 5 then
    let result ← PublicationMir.completed q.sort final.rows (Board.emptyCount q.board.val).toNat q.last.val true q.nodes q.fallback
    return (result,final.memory)
  let (_,last) ← SerialCache.run (search p c q.board (q.moves.order 5) fc) 12 (SerialStack.first final.machine final.rows) final.memory
  let result ← PublicationMir.completed q.sort (RootMoveStorage.sortRows last.answer.rows 0)
    (Board.emptyCount q.board.val).toNat 5 true q.nodes q.fallback
  return (result,last.after)

theorem finish_returns (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Call mc)
    (start : SelectiveLadder.Start (Memory l)) (addr : c.address = p.address)
    (fresh : start.valid mc q.board (ReadyMemory p mc fc)) :
    ∃ result next, finish p c fc q start = some (result,next) ∧ ReadyMemory p mc fc next ∧
      PublicationMir.ExactScore.scoreIfExact result = q.expected fc := by
  obtain ⟨final,fold,_,squares,positive,retained,pc,window⟩ := SelectiveLadder.ladder ((OuterMir.ladder.take 4).take q.last.val) start fresh
    (fun i hi => earlier p c q.board q.moves fc addr i (List.mem_of_mem_take hi))
  by_cases full : q.last.val = 5
  · obtain ⟨prior,last,r,ran,_,_,_,_,_,post,_,_,printed,score,_⟩ := publication _
      (exact_trial p c q.board (q.moves.order 5) fc addr) retained squares positive _ pc window q.sort q.nodes q.fallback
    refine ⟨r,last.after,?_,post,by simpa [Call.expected,full] using score⟩
    unfold finish
    rw [fold]
    simp [full,ran,printed]
  · obtain ⟨r,published,unflagged⟩ := weak_report q.sort final.rows (Board.emptyCount q.board.val).toNat
      q.last.val q.nodes q.fallback positive full
    refine ⟨r,final.memory,?_,retained,by simpa [Call.expected,full] using unflagged⟩
    unfold finish
    rw [fold]
    simp [full,published]

/-- A direct request advances the TT generation and computes its starting window from that table. -/
def run (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Call mc) (old : Memory l) := do
  let m := increment old
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let base := InitialEstimate.estimate m.tt.image mc fc q.raw q.board
  finish p c fc q ⟨rows,SerialEntry.firstMachine base,m⟩

theorem returns (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Call mc) (old : Memory l)
    (addr : c.address = p.address) (ready : ReadyMemory p mc fc old)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ result next, run p c fc q old = some (result,next) ∧ ReadyMemory p mc fc next ∧
      PublicationMir.ExactScore.scoreIfExact result = q.expected fc := by
  let m := increment old
  have init : ReadyMemory p mc fc m := begin_valid ready
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers born
  let base := InitialEstimate.estimate m.tt.image mc fc q.raw q.board
  have bounds := InitialEstimate.warm_bounded m.tt mc fc q.raw q.board m.supported init.2
  simpa only [run,born,bind,Option.bind] using finish_returns p c fc q
    ⟨NativeRootBirth.initial mc q.board,SerialEntry.firstMachine base,m⟩ addr
    ⟨cover,List.length_pos_iff.mpr pos,init,rfl,SerialEntry.firstMachine_window base bounds.1 bounds.2⟩

def runAll (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) :
    List (Call mc) → Memory l → Option (List PublicationMir.SearchResult × Memory l)
  | [], m => some ([],m)
  | q :: qs, m => do
    let (r,next) ← run p c fc q m
    let (rs,after) ← runAll p c fc qs next
    return (r::rs,after)

theorem all_returns (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (qs : List (Call mc)) (old : Memory l) (addr : c.address = p.address)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : ∀ q ∈ qs, Board.hasLegalMoves mc q.board.val = true) (ready : ReadyMemory p mc fc old) :
    ∃ results after, runAll p c fc qs old = some (results,after) ∧ ReadyMemory p mc fc after ∧
      results.map PublicationMir.ExactScore.scoreIfExact = qs.map (fun q => q.expected fc) := by
  induction qs generalizing old with
  | nil => exact ⟨[],old,rfl,ready,rfl⟩
  | cons q qs ih =>
    obtain ⟨r,next,done_,post,score⟩ := returns p c fc q old addr ready checked (mobile q List.mem_cons_self)
    obtain ⟨rs,after,rest,valid,scores⟩ := ih next (fun t ht => mobile t (.tail _ ht)) post
    exact ⟨r::rs,after,by simp [runAll,done_,rest],valid,by simp only [List.map_cons,score,scores]⟩

/-- Any finite sequence of direct, single-worker, single-PV endgames can reuse the tables.
    Earlier calls may stop at any requested selectivity. The initial table is created once. Every later search computes its own base from the slots
    left by its predecessors, irrespective of the seven-bit generation's wraparound. -/
def Contract : Prop :=
  ∀ (l : Layout) (p : Pool) mc fc (c : SerialScout.Config) (qs : List (Call mc)) certificate,
    c.address = p.address →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    (∀ q ∈ qs, Board.hasLegalMoves mc q.board.val = true) →
    ∃ results after, runAll p c fc qs (Memory.empty l 0 p) = some (results,after) ∧ ReadyMemory p mc fc after ∧
      results.map PublicationMir.ExactScore.scoreIfExact =
        qs.map (fun q => q.expected fc)

theorem repeated : Contract := by
  intro l p mc fc c qs cert addr checked mobile
  exact all_returns p c fc qs (Memory.empty l 0 p) addr checked mobile
    ⟨Memory.cold_valid ..,PvsParallel.SerialTable.exactScores_empty⟩

end NeuralReversi.AspirationMir.SerialTable.Repeated
