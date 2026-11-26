import NeuralReversi.PvsParallel.NativeFrames.Invocations
import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Program

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

open FrameReturns.OwnerCalls (ChildReturn)
open Invocations (Calls Exit Entry Scope)
open Returns (argsFields forOwner)

/-- Native calls between child searches (for example a split). Their descendants
    can search and return on this same worker. -/
structure SideCall where
  entry : Entry
  exit : Invocations.Exit
  silent : ∀ v, exit ≠ .normal v
  beforeScore : Calls
  afterScore : Calls

def SideCall.count (s : SideCall) := 1+s.beforeScore.count+s.exit.score.toList.length+s.afterScore.count+1

def wrap : List SideCall → Calls → Calls
  | [], tail => tail
  | s::ss, tail => .call s.entry s.exit s.beforeScore s.afterScore (wrap ss tail)

private theorem wrap_direct (ss : List SideCall) (tail : Calls) (clock inputs) (i : Nat) :
    (wrap ss tail).direct clock inputs i = tail.direct clock inputs (i+(ss.map SideCall.count).sum) := by
  induction ss generalizing i with
  | nil => simp [wrap]
  | cons s ss ih =>
    have skip : s.exit.direct (inputs (clock i)) = none := by
      cases h : s.exit with
      | normal word => exact (s.silent word h).elim
      | discarded | leave => rfl
    simp only [wrap,Calls.direct,skip,Option.toList_none,List.nil_append,List.map_cons,List.sum_cons]
    rw [ih]
    congr 1
    dsimp [SideCall.count]
    omega

structure Site where
  beforeScore : Calls
  afterScore : Calls
  preceding : List SideCall

/-- The register widths at the recursive PVS ABI. A field record read from Rust
    always satisfies these equalities, regardless of the returned score. -/
def params (c : ChildReturn) : Args :=
  ⟨c.board,BitVec.ofNat 32 c.depth,⟨BitVec.ofInt 32 c.alpha⟩,⟨BitVec.ofInt 32 c.beta⟩,
    true,Cutoffs.exactSelectivity,c.pv,c.cut,none⟩

def value (c : ChildReturn) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 c.score⟩

def Fits (c : ChildReturn) : Prop := argsFields (params c) (value c).raw.toInt = c

theorem fits_registers (args : Args) (word : ScoreRefresh.Score) : Fits (argsFields args word.raw.toInt) := by
  simp [Fits,argsFields,params,value]

private theorem fits_window (board : Board.Board) (depth : TranspositionTable.Depth) (pv cut : Bool)
    (win : Aspiration.Window a z) (word : BitVec 32) :
    Fits ⟨board,depth.toNat,pv,cut,a,z,word.toInt⟩ := by
  obtain ⟨lo,ordered,hi⟩ := win
  have alpha : (BitVec.ofInt 32 a).toInt = a := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have beta : (BitVec.ofInt 32 z).toInt = z := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  simp [Fits,argsFields,params,value,alpha,beta]

theorem pass_fits (r : Row) (pv : Bool) (word : ScoreRefresh.Score) :
    Fits (FrameReturns.OwnerCalls.passCall r.frame r.depth pv word.raw.toInt) :=
  fits_window _ _ _ _ (PvsMoveLoop.reflectedWindow r.window) word.raw

/-- Window operands are bounded by the search itself. A native return register
    can contain any signed i32, independently at each call. -/
theorem emitted_fits (r : Row) (visits : List (RootOwnerMir.Slice.Visit (PvsBoardTree.GeneratedSquare mc r.frame.position.val)))
    (operands : ∀ v ∈ visits, Aspiration.Window v.alpha r.frame.beta ∧
      ∀ pv x, v.word pv = some x → ∃ w : BitVec 32, w.toInt = x) :
    ∀ c ∈ FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits, Fits c := by
  intro c member
  obtain ⟨v,mem,pv,x,got,rfl⟩ := FrameReturns.OwnerCalls.emitted_iff.mp member
  obtain ⟨win,words⟩ := operands v mem
  obtain ⟨w,rfl⟩ := words pv x got
  apply fits_window _ (r.depth-1) _ _ _ w
  cases pv with
  | false => exact PvsMoveLoop.scoutWindow win
  | true => exact PvsMoveLoop.reflectedWindow win

def moveSite (presearch : Bool) (index : Nat) (child : ChildReturn) : Role :=
  if presearch && index == 0 then .presearch else if child.pv then .wide else .scout

theorem forwards (role : Role) :
    EndgameSearch.CacheOwner.Edge.mk .pvs role.site .pvs .forward ∈ EndgameSearch.CacheOwner.transfers := by
  cases role <;> decide

/-- Place each returning child at its call instruction. The site's activities
    before and after it finalizes the score may contain arbitrary nested calls. -/
def issue (op : Nat → ChildReturn → Role) (receiver : EndgameSearch.CacheOwner.Resource) (sites : Nat → Site) (after : List SideCall) (k : Nat) : List ChildReturn → Calls
  | [] => wrap after .nil
  | c::cs =>
    let site := sites k
    wrap site.preceding (.call ⟨(op k c).site,receiver,none⟩ (.normal (value c)) site.beforeScore site.afterScore
      (issue op receiver sites after (k+1) cs))

/-- Extract the entry register writes at the dynamic instruction indices. -/
def entries (sites : Nat → Site) (clock : Nat → Nat) (k i : Nat) : List ChildReturn → List (Nat × Args)
  | [] => []
  | c::cs =>
    let site := sites k
    let start := i+(site.preceding.map SideCall.count).sum
    (clock start,params c)::entries sites clock (k+1) (start+1+site.beforeScore.count+1+site.afterScore.count+1) cs

theorem issue_direct (op : Nat → ChildReturn → Role) (receiver) (sites : Nat → Site) (after : List SideCall) (clock inputs) (k i : Nat) (rows : List ChildReturn)
    (regs : ∀ p ∈ entries sites clock k i rows, inputs p.1 = some p.2) (fits : ∀ c ∈ rows, Fits c) :
    (issue op receiver sites after k rows).direct clock inputs i = rows := by
  induction rows generalizing k i with
  | nil => simp [issue,wrap_direct,Calls.direct]
  | cons c cs ih =>
    let site := sites k
    let start := i+(site.preceding.map SideCall.count).sum
    let next := start+1+site.beforeScore.count+1+site.afterScore.count+1
    have loaded : inputs (clock start) = some (params c) := regs _ List.mem_cons_self
    have tail := ih (k+1) next (fun p mem => regs p (List.mem_cons_of_mem _ mem))
      (fun d mem => fits d (List.mem_cons_of_mem _ mem))
    have got : (Exit.normal (value c)).direct (inputs (clock start)) = some c := by
      rw [loaded]
      change some (argsFields (params c) (value c).raw.toInt) = some c
      exact congrArg some (fits c List.mem_cons_self)
    rw [issue,wrap_direct,Calls.direct]
    change ((Exit.normal (value c)).direct (inputs (clock start))).toList ++
      (issue op receiver sites after (k+1) cs).direct clock inputs next = c::cs
    rw [got,tail]
    rfl

private theorem indexed (sites : Nat → Site) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (rows : List ChildReturn) (k i : Nat) :
    ((entries sites clock k i rows).map Prod.fst).Nodup ∧ ∀ p ∈ entries sites clock k i rows, clock i ≤ p.1 := by
  induction rows generalizing i k with
  | nil => simp [entries]
  | cons c cs ih =>
    let site := sites k
    let start := i+(site.preceding.map SideCall.count).sum
    let next := start+1+site.beforeScore.count+1+site.afterScore.count+1
    have afterStart := time (show start < next by dsimp [next]; omega)
    have startAt : clock i ≤ clock start := by
      by_cases eq : i = start
      · exact eq ▸ Nat.le_refl _
      · exact Nat.le_of_lt (time (by dsimp [start]; omega))
    obtain ⟨distinct,later⟩ := ih (k+1) next
    change ((clock start) :: (entries sites clock (k+1) next cs).map Prod.fst).Nodup ∧ _
    refine ⟨List.nodup_cons.mpr ⟨?_,distinct⟩,?_⟩
    · intro seen
      obtain ⟨p,mem,same⟩ := List.mem_map.mp seen
      have bound := later p mem
      omega
    · intro p mem
      rcases List.mem_cons.mp mem with rfl | more
      · exact startAt
      · exact Nat.le_trans startAt (Nat.le_trans (Nat.le_of_lt afterStart) (later p more))

/-- Direct-child entry registers can be composed with all other register samples
    without overwriting any sibling, even when both children have the same board. -/
def install (sites : Nat → Site) (clock : Nat → Nat) (i : Nat) (rows : List ChildReturn)
    (rest : Nat → Option Args) (t : Nat) : Option Args :=
  ((entries sites clock 0 i rows).lookup t).orElse (fun _ => rest t)

theorem installed (sites : Nat → Site) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (i : Nat) (rows : List ChildReturn) (rest) :
    ∀ p ∈ entries sites clock 0 i rows, install sites clock i rows rest p.1 = some p.2 := by
  have unique := (indexed sites clock time rows 0 i).1
  have get (xs : List (Nat × Args)) (keys : (xs.map Prod.fst).Nodup) (p : Nat × Args) (mem : p ∈ xs) :
      xs.lookup p.1 = some p.2 := by
    induction xs with
    | nil => cases mem
    | cons q qs ih =>
      rcases List.nodup_cons.mp keys with ⟨absent,others⟩
      rcases List.mem_cons.mp mem with rfl | later
      · exact List.lookup_cons_self
      · have ne : p.1 ≠ q.1 := fun eq => absent (List.mem_map.mpr ⟨p,later,eq⟩)
        simpa [List.lookup,beq_eq_false_iff_ne.mpr ne] using ih others later
  intro p mem
  simp [install,get _ unique p mem]

/-- A native body built from the recursive call sites supplies the owner's child
    records, with no pool-wide record matching premise. -/
structure Execution (p : EndgameSearch.CacheHistory.Pool) (inputs : Nat → Nat → Option Args)
    (stream : List Record) (f : FrameReturns.Frame) (op : Nat → ChildReturn → Role) (rows : List ChildReturn) where
  scope : Scope p stream f
  sites : Nat → Site
  afterCalls : List SideCall
  instructions : scope.body = issue op scope.entry.received sites afterCalls 0 rows
  args : ∀ x ∈ entries sites scope.clock 0 (scope.opening+1) rows, inputs f.worker x.1 = some x.2
  widths : ∀ c ∈ rows, Fits c

/-- Compose the child ABI snapshots with the unmodified readings outside this
    worker's call sites. No per-child lookup successes are required as input. -/
def callInputs (worker : Nat) (sites : Nat → Site) (clock : Nat → Nat) (i : Nat) (rows : List ChildReturn)
    (others : Nat → Nat → Option Args) (w : Nat) : Nat → Option Args :=
  if w = worker then install sites clock i rows (others w) else others w

def Execution.from_instructions (ctx : Scope p stream f) (op : Nat → ChildReturn → Role) (sites : Nat → Site) (after : List SideCall)
    (rows : List ChildReturn) (others : Nat → Nat → Option Args)
    (body : ctx.body = issue op ctx.entry.received sites after 0 rows) (widths : ∀ c ∈ rows, Fits c) :
    Execution p (callInputs f.worker sites ctx.clock (ctx.opening+1) rows others) stream f op rows := by
  refine ⟨ctx,sites,after,body,?_,widths⟩
  simpa only [callInputs,ite_true] using installed sites ctx.clock ctx.time (ctx.opening+1) rows (others f.worker)

/-- PV owners and the root load i32 replies at the two MIR call sites and update
    their window between visits. These instructions suffice to construct the
    entry registers for every emitted call. -/
def Execution.from_pv {mc : Bitboard.Moves.X86Config} {r : Row}
    {square : PvsBoardTree.GeneratedSquare mc r.frame.position.val → Bitboard.Square}
    {prior pending : List (PvsBoardTree.GeneratedSquare mc r.frame.position.val)}
    {out : RootOwnerMir.Slice.Exit (PvsBoardTree.GeneratedSquare mc r.frame.position.val)}
    {visits : List (RootOwnerMir.Slice.Visit (PvsBoardTree.GeneratedSquare mc r.frame.position.val))}
    (ctx : Scope p stream r.frame)
    (src : Nat → PvsBoardTree.GeneratedSquare mc r.frame.position.val → Bool → Int → Int → ScoreRefresh.Score)
    (path : RootOwnerMir.Slice.Path square (fun i m pv a z => (src i m pv a z).raw.toInt) schedule
      r.depth.toNat limit r.frame.beta i prior pending fresh (RootReplay.start r.frame.alpha) visits out)
    (sites : Nat → Site) (after : List SideCall) (others : Nat → Nat → Option Args)
    (body : ctx.body = issue (moveSite false) ctx.entry.received sites after 0
      (FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits)) :=
  Execution.from_instructions ctx (moveSite false) sites after _ others body
    (emitted_fits r visits (path.operands r.window _ (fun _ _ _ _ _ => ⟨_,rfl⟩)))

def Execution.from_pass {r : Row} (ctx : Scope p stream r.frame) (pv : Bool) (word : ScoreRefresh.Score)
    (sites : Nat → Site) (after : List SideCall) (others : Nat → Nat → Option Args)
    (body : ctx.body = issue (fun _ _ => .passChild) ctx.entry.received sites after 0
      [FrameReturns.OwnerCalls.passCall r.frame r.depth pv word.raw.toInt]) :=
  Execution.from_instructions ctx (fun _ _ => .passChild) sites after _ others body
    (fun _ mem => List.mem_singleton.mp mem ▸ pass_fits r pv word)

theorem Execution.records (ex : Execution p inputs stream f op rows) (parsed : decode p inputs stream = some cap) :
    (Returns.poolReturns p inputs stream).filterMap (forOwner f) = rows := by
  rw [ex.scope.records parsed,ex.instructions,issue_direct _ _ _ _ _ _ _ _ _ ex.args ex.widths]

theorem Execution.observed (ex : Execution p inputs stream f op rows) (mc fc)
    (parsed : decode p inputs stream = some cap) (identify before seq) :
    (FrameReturns.OwnerCalls.observed (logOf mc fc cap.rows cap.table identify before seq) f).Perm rows := by
  rw [← ex.records parsed]
  exact Returns.owner_returns mc fc f parsed identify before seq

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites
