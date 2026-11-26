import NeuralReversi.EndgameSearch.SerialLeafRestore
import NeuralReversi.EndgameSearch.CacheHistory
import NeuralReversi.EndgameSearch.Thresholds

namespace NeuralReversi.EndgameSearch.SerialLeafCache

open EndgameTreeCache (Table Store entryFor soundHistory)
open CacheHistory (Pool Local)
open CacheState (Cells)
open RecursionCacheMir (Reads Steps)
open PvsParallel.SerialScout (Config)
open PvsBoardTree (Position)
open Cancellation (EmptyState)
open ScoreRefresh (Score)
open Bitboard.Moves (X86Config)

/-- Allocate each Rust table independently, then apply its worker's stores in order. -/
def materialize (p : Pool) (past : List Store) : Option Cells := do
  let deep ← CacheMir.newCache (CacheMir.budget p.size .deep)
  let shallow ← CacheMir.newCache (CacheMir.budget p.size .shallow)
  past.foldlM (CacheMir.writeCells p.width p.key) (fun | .deep => deep | .shallow => shallow)

def snapshot (p : Pool) (h : List Store) : Cells :=
  h.foldl (CacheState.write p.address) p.empty.cells

private theorem shape_after (p : Pool) (h : List Store) (cells : Cells)
    (shape : ∀ t, (cells t).size = p.sizes t) :
    ∀ t, (h.foldl (CacheState.write p.address) cells t).size = p.sizes t := by
  induction h generalizing cells with
  | nil => exact shape
  | cons w ws ih =>
    apply ih
    intro t
    by_cases same : t = w.table <;> simp [CacheState.write,same,shape]

theorem materialize_eq (p : Pool) (past : List Store) : materialize p past = some (snapshot p past) := by
  have init := CacheMir.new_represents p.width (CacheMir.budget p.size) p.key
    (CacheMir.budget_fits p.size p.width p.word)
  unfold materialize
  rw [init.1 .deep,init.1 .shallow]
  dsimp only [Option.bind_some,bind,pure]
  have both : (fun | .deep => p.empty.cells .deep | .shallow => p.empty.cells .shallow) = p.empty.cells := by
    funext t; cases t <;> rfl
  change past.foldlM (CacheMir.writeCells p.width p.key) (fun | Table.deep => p.empty.cells .deep | .shallow => p.empty.cells .shallow) = _
  rw [both]
  exact CacheMir.replay_eq _ (fun t => by simp [CacheHistory.Pool.empty,CacheState.empty])
    (CacheMir.budget_fits p.size p.width p.word)

theorem snapshot_shape (p : Pool) (h : List Store) : ∀ t, (snapshot p h t).size = p.sizes t :=
  shape_after p h p.empty.cells (fun _ => by simp [CacheHistory.Pool.empty,CacheState.empty])

theorem snapshot_arrays (p : Pool) (h : List Store) : CacheState.represents p.address h (snapshot p h) := by
  have initial := CacheMir.new_represents p.width (CacheMir.budget p.size) p.key
    (CacheMir.budget_fits p.size p.width p.word)
  have boot : CacheState.represents p.address [] p.empty.cells := initial.2
  simpa only [List.nil_append,snapshot] using CacheState.replay_represents boot h

/-- Includes both native board words and both signed bounds for any hash, even after collisions. -/
theorem physical_reads (p : Pool) (h : List Store) :
    Reads p.width p.key p.address h (snapshot p h) := by
  intro t b a
  have atSlot := snapshot_arrays p h t b
  simp only [CacheHistory.Pool.address,CacheMir.address] at atSlot
  simp [CacheMir.probe,snapshot_shape,atSlot,CacheMir.probeCell_eq,Pool.address]

/-- Enhanced cutoff scans the generated children in Rust order and stops at the first usable bound.
    The outer Option distinguishes an invalid index from a cache miss. -/
def enhanced (p : Pool) (cells : Cells) (alpha : Int) : List Board.Board → Option (Option Int)
  | [] => some none
  | b :: bs => do
    let childAlpha := -(alpha+1)
    let hit ← CacheMir.probe (cells .deep) (CacheMir.slot p.width (cells .deep).size (p.key b)) b childAlpha
    match hit with
    | some v => if v ≤ childAlpha then return some (-v) else enhanced p cells alpha bs
    | none => enhanced p cells alpha bs

theorem enhanced_eq (p : Pool) (h : List Store) (cells : Cells)
    (reads : Reads p.width p.key p.address h cells) (alpha : Int) (bs : List Board.Board) :
    enhanced p cells alpha bs = some (EndgameCache.enhancedCutoff (entryFor p.address h .deep) alpha bs) := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    rw [enhanced,reads]
    cases hit : EndgameCache.probe (entryFor p.address h .deep b) b (-(alpha+1)) with
    | none => simpa [EndgameCache.enhancedCutoff,hit] using ih
    | some v =>
      by_cases cutoff : v ≤ -(alpha+1)
      · simp [EndgameCache.enhancedCutoff,hit,cutoff]
      · simpa [EndgameCache.enhancedCutoff,hit,cutoff] using ih

def probe (p : Pool) (cells : Cells) (table : Table) (b : Board.Board) (a : Int) :=
  CacheMir.probe (cells table) (CacheMir.slot p.width (cells table).size (p.key b)) b a

/-- Only a mobile shallow node consults the table. A child being delegated to
    the four-empty solver queries its own board before that solve. -/
def shallowNode (p : Pool) (cells : Cells) (b : Position) (moves : List Move) (a : Int) (h : List Store)
    (other : Bool) (terminal : Int) (passed : Unit → Result) (stable : Option Int)
    (stages : List (List Move)) (child : Move → List Store → Result) : Option Result :=
  match moves with
  | [] => some (ShallowMir.node b [] a h other terminal passed none stable stages child)
  | _ :: _ => do
    let hit ← probe p cells .shallow b.val a
    return ShallowMir.node b moves a h other terminal passed hit stable stages child

theorem shallowNode_eq (p : Pool) (cells : Cells) (b : Position) (moves : List Move) (a : Int) (h : List Store)
    (reads : Reads p.width p.key p.address h cells) (other : Bool) (terminal : Int) (passed : Unit → Result)
    (stable : Option Int) (stages : List (List Move)) (child : Move → List Store → Result) :
    shallowNode p cells b moves a h other terminal passed stable stages child =
      some (ShallowMir.node b moves a h other terminal passed (EndgameCache.probe (entryFor p.address h .shallow b.val)
        b.val a) stable stages child) := by
  cases moves with
  | nil => rfl
  | cons => simp [shallowNode,probe,reads .shallow b.val a]

def shallowChild (p : Pool) (cells : Cells) (b : Position) (a : Int) (h : List Store) (count : Nat)
    (stable : Option Int) (solveFour : Unit → Int) (recurse : Unit → Result) : Option Result :=
  if count = 4 then do
    let hit ← probe p cells .shallow b.val a
    return ShallowMir.child b a h count hit stable solveFour recurse
  else some (ShallowMir.child b a h count none stable solveFour recurse)

theorem shallowChild_eq (p : Pool) (cells : Cells) (b : Position) (a : Int) (h : List Store)
    (reads : Reads p.width p.key p.address h cells) (count : Nat) (stable : Option Int)
    (solveFour : Unit → Int) (recurse : Unit → Result) :
    shallowChild p cells b a h count stable solveFour recurse =
      some (ShallowMir.child b a h count (EndgameCache.probe (entryFor p.address h .shallow b.val) b.val a)
        stable solveFour recurse) := by
  by_cases four : count = 4
  · simp [shallowChild,four,probe,reads .shallow b.val a]
  · simp [shallowChild,ShallowMir.child,four]

def deepEarly (p : Pool) (cells : Cells) (fc : FlipConfig) (paired : Bool) (b : Position)
    (s : Cancellation.State) (a : Int) (first : Bitboard.Square) (rest : List Bitboard.Square) :
    Option (Option Int) := do
  let hit ← probe p cells .deep b.val a
  if let some score := hit then return some score
  if let some score := EndgameStability.tryCutoff b.val s.empties.count.toNat ⟨a⟩ then return some score.value
  let wipe := EndgameCutoffs.wipesOut fc b.val
  if rest.isEmpty then return if wipe first then some 64 else none
  let generated := MoveList.SmallGenerate.generate paired wipe (first :: rest)
  if !generated.1 then return some 64
  if Thresholds.minEcEtcEmpties ≤ s.empties.count.toNat then
    enhanced p cells a (generated.2.map fun sq => Board.place b.val
      (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq)
  else return none

theorem deepEarly_eq (p : Pool) (cells : Cells) (fc : FlipConfig) (paired : Bool) (b : Position)
    (s : Cancellation.State) (reads : Reads p.width p.key p.address s.history cells)
    (a : Int) (first : Bitboard.Square) (rest : List Bitboard.Square) :
    deepEarly p cells fc paired b s a first rest =
      some (DeepRecursionMir.early fc paired b s p.address a first rest) := by
  simp only [deepEarly,DeepRecursionMir.early,probe,reads .deep b.val a,enhanced_eq p _ _ reads]
  simp only [bind,pure,Option.bind]
  cases EndgameCache.probe (entryFor p.address s.history .deep b.val) b.val a <;> try rfl
  cases EndgameStability.tryCutoff b.val s.empties.count.toNat ⟨a⟩ <;> try rfl
  cases rest.isEmpty <;> simp only [Bool.false_eq_true,ite_false,ite_true] <;> try rfl
  cases (MoveList.SmallGenerate.generate paired (EndgameCutoffs.wipesOut fc b.val) (first :: rest)).1 <;>
    simp only [Bool.not_false,Bool.not_true,Bool.false_eq_true,ite_false,ite_true] <;> try rfl
  split <;> rfl

/-- Every save and every possible read between saves has a matching operation on the actual-sized arrays. -/
private theorem saves (p : Pool) (past ws : List Store) :
    Steps p.width p.key p.address past (snapshot p past) ws (snapshot p (past ++ ws)) := by
  induction ws generalizing past with
  | nil => simpa using Steps.done (physical_reads p past)
  | cons w ws ih =>
    have stored := CacheMir.write_eq (hash := p.key) w (snapshot_shape p past) (CacheMir.budget_fits p.size p.width p.word)
    have atWrite : CacheMir.writeCells p.width p.key (snapshot p past) w = some (snapshot p (past ++ [w])) := by
      simpa [snapshot,Pool.address] using stored
    have later := ih (past ++ [w])
    exact .more (physical_reads p past) atWrite (by simpa using later)

structure Call where
  config : Config
  board : Position
  empties : EmptyState
  alpha : Score
  beta : Score
  aligned : empties.matches board
  size : empties.count.toNat ≤ 12
  null : beta.raw.toInt = alpha.raw.toInt + 1

def Call.leaf (i : Call) (p : Pool) (mc : X86Config) (fc : FlipConfig) (past : List Store) :=
  SerialLeafRestore.leaf {i.config with address := p.address} mc fc i.board i.empties i.alpha i.beta past

/-- Retain the actual arrays across successive calls, applying only the new saves. -/
def Call.execute (i : Call) (p : Pool) (mc : X86Config) (fc : FlipConfig) (prev : Local) :
    Option ((Int × EmptyState) × Local) := do
  let ((score,history),empties) ← i.leaf p mc fc prev.history
  let cells ← (history.drop prev.history.length).foldlM (CacheMir.writeCells p.width p.key) prev.cells
  return ((score,empties),⟨history,cells⟩)

def Reachable (p : Pool) (mc : X86Config) (fc : FlipConfig) (state : Local) : Prop :=
  soundHistory mc fc state.history ∧ materialize p state.history = some state.cells

theorem cold (p : Pool) (mc : X86Config) (fc : FlipConfig) : Reachable p mc fc p.empty :=
  ⟨fun _ mem => (List.not_mem_nil mem).elim,materialize_eq p []⟩

structure Returned (i : Call) (p : Pool) (mc : X86Config) (fc : FlipConfig)
    (prior : Local) (score : Int) (next : Local) : Prop where
  executed : i.execute p mc fc prior = some ((score,i.empties),next)
  returned : i.leaf p mc fc prior.history = some ((score,next.history),i.empties)
  bound : PvsParallel.boundedReplay 1 (PvsBoardTree.build mc fc i.board) false i.alpha.raw.toInt i.beta.raw.toInt score
  nextReachable : Reachable p mc fc next
  saves : ∃ ws, prior.history ++ ws = next.history ∧
    Steps p.width p.key p.address prior.history prior.cells ws next.cells

theorem leaf_step (i : Call) (p : Pool) (mc : X86Config) (fc : FlipConfig) (prior : Local)
    (reachable : Reachable p mc fc prior) : ∃ score next, Returned i p mc fc prior score next := by
  obtain ⟨v,h,ret,bound,valid,ws,grows⟩ := SerialLeafRestore.contract_proven
    {i.config with address := p.address} mc fc i.board i.empties i.alpha i.beta prior.history
    i.aligned i.size reachable.1 i.null
  change i.leaf p mc fc prior.history = some ((v,h),i.empties) at ret
  have start : prior.cells = snapshot p prior.history :=
    Option.some.inj (reachable.2.symm.trans (materialize_eq p prior.history))
  have checkedSaves := CacheMir.replay_eq (hash := p.key) ws (snapshot_shape p prior.history)
    (CacheMir.budget_fits p.size p.width p.word)
  subst h
  have writes : ws.foldlM (CacheMir.writeCells p.width p.key) prior.cells =
      some (snapshot p (prior.history ++ ws)) := by simpa [start,snapshot,Pool.address] using checkedSaves
  refine ⟨v,⟨_,snapshot p (prior.history ++ ws)⟩,?_,ret,bound,⟨valid,materialize_eq p _⟩,ws,rfl,?_⟩
  · simp [Call.execute,ret,writes]
  · exact start ▸ saves p prior.history ws

/-- At any store boundary inside this returned leaf, the concrete arrays reproduce the exact
    source probe and every concrete hit is a sound bound. The incoming cache may be warm. -/
theorem Returned.prefix_reads (out : Returned i p mc fc prev score next)
    (saved later : List Store) (atPrefix : next.history = saved ++ later) :
    ∃ cells, materialize p saved = some cells ∧ Reads p.width p.key p.address saved cells ∧
      ∀ t b a v, probe p cells t b a = some (some v) →
        EndgameCache.CacheRange v ∧ Solve2.NWS a (EndgameTreeCache.discValue mc fc b) v := by
  have valid : soundHistory mc fc saved := by
    intro e mem
    exact out.nextReachable.1 e (atPrefix ▸ List.mem_append_left _ mem)
  refine ⟨snapshot p saved,materialize_eq p saved,physical_reads p saved,?_⟩
  intro t b a v hit
  unfold probe at hit
  rw [physical_reads p saved t b a] at hit
  exact EndgameTreeCache.probe_sound valid p.address t b a v (Option.some.inj hit)

def execute (p : Pool) (mc : X86Config) (fc : FlipConfig) (calls : List Call) : Option Local :=
  calls.foldlM (fun prev i => (i.execute p mc fc prev).map Prod.snd) p.empty

theorem from_cold (p : Pool) (mc : X86Config) (fc : FlipConfig) (calls : List Call) :
    ∃ state, execute p mc fc calls = some state ∧ Reachable p mc fc state := by
  have go (prev : Local) (boot : Reachable p mc fc prev) :
      ∃ state, calls.foldlM (fun s i => (i.execute p mc fc s).map Prod.snd) prev = some state ∧
        Reachable p mc fc state := by
    induction calls generalizing prev with
    | nil => exact ⟨prev,rfl,boot⟩
    | cons i rest ih =>
      obtain ⟨score,next,ret⟩ := leaf_step i p mc fc prev boot
      simpa [List.foldlM_cons,ret.executed] using ih next ret.nextReachable
  exact go p.empty (cold p mc fc)

def Contract : Prop :=
  ∀ (p : Pool) (mc : X86Config) (fc : FlipConfig),
    (∀ h, materialize p h = some (snapshot p h) ∧ Reads p.width p.key p.address h (snapshot p h)) ∧
    (∀ i prior, Reachable p mc fc prior → ∃ score next, Returned i p mc fc prior score next) ∧
    ∀ calls, ∃ final, execute p mc fc calls = some final ∧ Reachable p mc fc final

theorem contract_proven : Contract := fun p mc fc =>
  ⟨fun h => ⟨materialize_eq p h,physical_reads p h⟩,fun i prior => leaf_step i p mc fc prior,from_cold p mc fc⟩

end NeuralReversi.EndgameSearch.SerialLeafCache
