module
public import NeuralReversi.AspirationMir.SerialStack
public import NeuralReversi.PvsParallel.SerialTermination

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCacheStack

open SerialCalls (Request Program Body Executes Tree)
open SerialExecution (Frame Inputs push)
open EndgameTreeCache (Store soundHistory)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)

abbrev History := List Store

/-- This overrides the local cache input. Every other operand is the
    corresponding load of the original activation. -/
def source (inputs : Inputs mc fc) (histories : Nat → History) : Inputs mc fc :=
  fun i ctx => {inputs i ctx with cache := histories i}

structure SharedFacts (inputs : Inputs mc fc) (limit : Nat := 64) : Prop where
  probe : ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat → ctx.request.depth.toNat ≤ limit →
    TableReads.ProbeReads (inputs i ctx).table (inputs i ctx).probe
  table : ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat → ctx.request.depth.toNat ≤ limit →
    ExactTableHistory.Publications mc fc (inputs i ctx).table
  hints : ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat → ctx.request.depth.toNat ≤ limit →
    TableReads.LegalHints (inputs i ctx).table

def capped (inputs : Inputs mc fc) (limit : Nat) : Inputs mc fc := fun i ctx =>
  if ctx.request.depth.toNat ≤ limit then inputs i ctx else
    {inputs i ctx with table := [], probe := [], etc := []}

theorem shared_reads {inputs : Inputs mc fc} (facts : SharedFacts inputs limit) (h : History) (sound : soundHistory mc fc h) :
    SerialExecution.ReadFacts (source (capped inputs limit) (fun _ => h)) ∧
    SerialTermination.Hints (source (capped inputs limit) (fun _ => h)) := by
  refine ⟨⟨fun _ _ _ _ _ => sound, ?_, ?_⟩, ?_⟩
  · intro i ctx role depth
    by_cases inside : ctx.request.depth.toNat ≤ limit
    · simpa [source,capped,inside] using facts.probe i ctx role depth inside
    · simp [source,capped,inside,TableReads.ProbeReads,TableReads.ProbeReadsFrom]
  · intro i ctx role depth
    by_cases inside : ctx.request.depth.toNat ≤ limit
    · simpa [source,capped,inside] using facts.table i ctx role depth inside
    · simpa [source,capped,inside] using (ExactTableHistory.Publications.initial : ExactTableHistory.Publications mc fc [])
  · intro i ctx role depth
    by_cases inside : ctx.request.depth.toNat ≤ limit
    · simpa [source,capped,inside] using facts.hints i ctx role depth inside
    · simp [source,capped,inside,TableReads.LegalHints]

theorem body_congr {inputs other : Inputs mc fc} (c : SerialScout.Config) (eq : inputs i ctx = other i ctx) :
    SerialExecution.body c inputs i ctx = SerialExecution.body c other i ctx := by
  unfold SerialExecution.body
  rw [eq]

theorem body_at (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame)
    (h : History) (inside : ctx.request.depth.toNat ≤ limit) :
    SerialExecution.body c (source (capped inputs limit) (fun _ => h)) i ctx =
      SerialExecution.body c (source inputs (fun _ => h)) i ctx := by
  have same : capped inputs limit i ctx = inputs i ctx := ite_eq_left inside
  apply body_congr
  unfold source
  rw [same]

abbrev Output (α : Type) := Nat × α × History
abbrev Child := Nat → Frame → History → Option (Output Int)

/-- A continuation receives both the score and the cache left by the child. -/
def interpret {mc : X86Config} {fc : FlipConfig} (descend : Child) (i : Nat) (ctx : Frame) (past : History) :
    Program Request α → Option (Output α)
  | .pure value => some (i,value,past)
  | .fail => none
  | .call q resume => do
    let child ← push mc fc ctx q
    let (k,word,after) ← descend (i+1) child past
    interpret (mc := mc) (fc := fc) descend k ctx after (resume word)

def runFuel (c : SerialScout.Config) (inputs : Inputs mc fc) : Nat → Child
  | 0, _, _, _ => none
  | fuel+1, i, ctx, past =>
    let q := ctx.request
    if q.pv = false ∧ q.depth.toNat ≤ 12 then
      (EndgameSearch.SerialLeafRestore.leaf c mc fc q.board ctx.empties
        (SerialCalls.ScoreWords.encode q.alpha) (SerialCalls.ScoreWords.encode q.beta) past).map
        fun out => (i,out.1.1,out.1.2)
    else interpret (mc := mc) (fc := fc) (runFuel c inputs fuel) i ctx past
      (SerialExecution.body c (source inputs (fun _ => past)) i ctx)

def run (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame) (past : History) :=
  runFuel c inputs (Board.playRank mc ctx.request.board.val + 1) i ctx past

/-- The root owner returns its physical rows and the same cache used in its last child. -/
def trial (c : SerialScout.Config) (b : PvsBoardTree.Position) (inputs : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : AspirationMir.SerialStack.State) (past : History) :
    Option (Output AspirationMir.SmallRoot.Answer) :=
  let a := s.regs.alpha
  let z := s.regs.beta
  interpret (mc := mc) (fc := fc) (run c inputs) s.opening (SerialRootExecution.context b a z) past
    (SerialRootExecution.attempt c inputs s.opening b root s.rows 0 a z)

end NeuralReversi.PvsParallel.SerialCacheStack
