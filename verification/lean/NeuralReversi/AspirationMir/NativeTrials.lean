import NeuralReversi.AspirationMir.NativeCacheResult

namespace NeuralReversi.AspirationMir.NativeTrials

open PvsParallel
open NativeFrames (Capture Row Args Record Event)
open EndgameSearch.CacheOwner (Node)
open PvsBoardTree (Position)

structure Trial where
  opening : Nat
  returned : Nat
  score : Word

private structure Open where
  opening : Nat
  node : Node

private structure Scan where
  stack : List Open := []
  entered : Bool := false
  closed : Bool := false
  trials : List Trial := []

private def site (caller callee : Node) (ordinal : Nat) : Bool :=
  EndgameSearch.CacheOwner.transfers.any (fun e => e.caller == caller && e.callee == callee && e.site == ordinal)

theorem compiledSites : site .solve .endWindow 0 = true ∧ site .endWindow .pvs 0 = true := by decide

private def Scan.advance (solve opening worker : Nat) (s : Scan) : Record → Option Scan
  | .table _ => some s
  | .localCall w event =>
    if w != worker then some s else
    match event with
    | .scored .. => some s
    | .discarded _ begin _ => do
      let top :: rest := s.stack | none
      if top.opening != begin || top.node != .pvs ||
          rest.head?.any (fun p => p.opening == opening && p.node == .endWindow) then none
      else some {s with stack := rest}
    | .entry t pc node .. =>
      let opened := {s with stack := ⟨t,node⟩ :: s.stack}
      if t = opening then
        if !s.entered && node == .endWindow && s.stack.head?.any (fun p => p.node == .solve && p.opening == solve) &&
            site .solve .endWindow pc then some {opened with entered := true} else none
      else if s.stack.head?.any (fun p => p.opening == opening && p.node == .endWindow) then
        if node == .pvs && site .endWindow .pvs pc then some opened else none
      else some opened
    | .leave t begin node _ normal => do
      let top :: rest := s.stack | none
      if begin != top.opening || node != top.node then none else
      let popped := {s with stack := rest}
      if begin = opening then
        if s.entered && !s.closed && node == .endWindow then some {popped with closed := true} else none
      else if rest.head?.any (fun p => p.opening == opening && p.node == .endWindow) then
        let value ← normal
        if node == .pvs then some {popped with trials := ⟨begin,t,value.raw⟩ :: s.trials} else none
      else some popped

/-- Direct normal returns of the designated root aspiration activation, in call order. -/
def collect (worker solve opening : Nat) (stream : List Record) : Option (List Trial) := do
  let run ← stream.foldlM (Scan.advance solve opening worker) {}
  if run.closed then some run.trials.reverse else none

def stamp (ts : List Trial) (n : Nat) := ((ts[n]?).map Trial.returned).getD 0

def matchRow? (c : Capture pool) (b : Position) (worker line : Nat) (t : Trial) (a z : Word) : Option Row := do
  let row ← rootReturn? c b worker line t.returned a z
  if row.opening = t.opening ∧ row.rawScore.raw = t.score then some row else none

theorem matched (got : matchRow? c b w line t a z = some row) :
    rootReturn? c b w line t.returned a z = some row ∧ row.opening = t.opening ∧ row.rawScore.raw = t.score := by
  obtain ⟨q,found,identity⟩ := Option.bind_eq_some_iff.mp got
  split at identity <;> try contradiction
  cases identity
  exact ⟨found,by assumption⟩

def permit (c : Capture pool) (b : Position) (w line : Nat) (ts : List Trial) (m : Machine) : Bool :=
  match driverCode[m.pc]? with
  | some (.query _) =>
    ((ts[m.issued]?).bind (fun t => matchRow? c b w line t m.regs.alpha m.regs.beta)).isSome
  | _ => true

theorem checked_query (edge : driverCode[m.pc]? = some (.query next)) (allowed : permit c b w line ts m = true) :
    ∃ t row, ts[m.issued]? = some t ∧ matchRow? c b w line t m.regs.alpha m.regs.beta = some row ∧
      rootQuery c b w line (stamp ts) m.issued m.regs.alpha m.regs.beta = t.score := by
  simp only [permit,edge,Option.isSome_iff_exists] at allowed
  obtain ⟨row,bound⟩ := allowed
  obtain ⟨t,slot,ok⟩ := Option.bind_eq_some_iff.mp bound
  have evidence := matched ok
  exact ⟨t,row,slot,ok,by simp [rootQuery,stamp,slot,evidence.1,evidence.2.2]⟩

inductive Path (c : Capture pool) (b : Position) (w line : Nat) (ts : List Trial) :
    Machine → Word → Call → Prop
  | done : execute (rootQuery c b w line (stamp ts)) (fun _ => false) m = some (.ret score last) →
      m.issued = ts.length → Path c b w line ts m score last
  | next : permit c b w line ts m = true →
      execute (rootQuery c b w line (stamp ts)) (fun _ => false) m = some (.step next) →
      Path c b w line ts next score last → Path c b w line ts m score last

theorem Path.ran (p : Path c b w line ts m score last) :
    Returns (rootQuery c b w line (stamp ts)) (fun _ => false) m score last := by
  induction p with
  | done done _ => exact .done done
  | next _ step _ ih => exact .next step ih

structure Result (c : Capture pool) (b : Position) (w line : Nat) (ts : List Trial) (m : Machine) where
  score : Word
  last : Call
  row : Row
  path : Path c b w line ts m score last
  receipt : ∃ t, ts[last.index]? = some t ∧ row.opening = t.opening ∧ row.rawScore.raw = t.score
  found : rootReturn? c b w line (stamp ts last.index) last.alpha last.beta = some row

private def remaining (n : Nat) (m : Machine) : Nat :=
  (n - m.issued) * 15 + match m.pc with
  | 0 => 2 | 1 => 1 | 2 => 0 | 3 => 9 | 4 => 8 | 5 => 7 | 6 => 1
  | 7 => 6 | 8 => 5 | 9 => 4 | 10 => 1 | 11 => 2 | _ => 0

private theorem decreases (m next : Machine) (safe : permit c b w idx ts m = true)
    (edge : execute (rootQuery c b w idx (stamp ts)) (fun _ => false) m = some (.step next)) :
    remaining ts.length next < remaining ts.length m := by
  match pc : m.pc with
  | 0 | 1 | 3 | 4 | 6 | 9 | 10 | 11 =>
    simp [execute,driverCode,pc] at edge
    subst next
    simp [remaining,pc]
  | 2 =>
    obtain ⟨_,_,slot,_,_⟩ := checked_query (by simp [pc,driverCode] : driverCode[m.pc]? = some (.query 3)) safe
    have bound := (List.getElem?_eq_some_iff.mp slot).1
    simp [execute,driverCode,pc] at edge
    subst next
    simp [remaining,pc]
    omega
  | 5 | 8 =>
    simp [execute,driverCode,pc] at edge
    subst next
    cases m.flag <;> simp [remaining,pc]
  | 7 =>
    simp only [execute,driverCode,pc] at edge
    cases h : applyWiden m.regs m.last.score with
    | none => simp [h] at edge
    | some pair =>
      simp [h] at edge
      subst next
      simp [remaining,pc]
  | 12 | 13 => simp [execute,driverCode,pc] at edge
  | n+14 => simp [execute,driverCode,pc] at edge

def run (c : Capture pool) (b : Position) (w line : Nat) (ts : List Trial) (m : Machine) :
    Option (Result c b w line ts m) := do
  match edge : execute (rootQuery c b w line (stamp ts)) (fun _ => false) m with
  | none => none
  | some (.step next) =>
    if h : permit c b w line ts m then
      let out ← run c b w line ts next
      some ⟨out.score,out.last,out.row,.next h edge out.path,out.receipt,out.found⟩
    else none
  | some (.ret score last) =>
    if exactLen : m.issued = ts.length then
      match slot : ts[last.index]? with
      | none => none
      | some t =>
        match found : matchRow? c b w line t last.alpha last.beta with
        | none => none
        | some row => some ⟨score,last,row,.done edge exactLen,⟨t,slot,(matched found).2⟩,
            by simpa only [stamp,slot,Option.map_some,Option.getD_some] using (matched found).1⟩
    else none
termination_by remaining ts.length m
decreasing_by exact decreases m next h edge

def initial (a z : Word) : Machine := ⟨0,⟨a,z,0⟩,0,⟨0,0,0,0⟩,false,0⟩

structure Line (c : Capture pool) (b : Position) (w idx : Nat) where
  trials : List Trial
  alpha : Word
  beta : Word
  result : Result c b w idx trials (initial alpha beta)

/-- The first root call receives the caller's unchanged alpha and beta. Its real entry
    operands seed the VM, and every retry must match the next physical call and return. -/
def decode (c : Capture pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    (b : Position) (worker idx solve opening : Nat) : Option (Line c b worker idx) := do
  let trials ← collect worker solve opening stream
  let first ← trials.head?
  let args ← inputs worker first.opening
  let a := args.alpha.raw
  let z := args.beta.raw
  let out ← run c b worker idx trials (initial a z)
  return ⟨trials,a,z,out⟩

theorem physical_return (ok : decode c inputs stream b w idx solve opening = some line) :
    ∃ t, (collect w solve opening stream).bind (fun ts => ts[line.result.last.index]?) = some t ∧
      t.opening = line.result.row.opening ∧ t.score = line.result.score := by
  obtain ⟨ts,raw,next⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨first,_,next⟩ := Option.bind_eq_some_iff.mp next
  obtain ⟨args,_,next⟩ := Option.bind_eq_some_iff.mp next
  obtain ⟨out,_,endOf⟩ := Option.bind_eq_some_iff.mp next
  cases endOf
  obtain ⟨t,member,same,word⟩ := out.receipt
  have h := returned_inside out.path.ran rfl
  have eq := h.1.1
  change rootQuery c b w idx (stamp ts) out.last.index out.last.alpha out.last.beta = out.last.score at eq
  simp only [rootQuery,out.found,Option.map_some,Option.getD_some] at eq
  exact ⟨t,by simp [raw,member],same.symm,word.symm.trans (eq.trans h.2.symm)⟩

theorem publicScore {out : NativeFrameCache.Checked pool}
    (cache : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → NativeFrames.Branch c log endian row)
    (stores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ e ∈ c.table, FrameReturns.WriteStep log endian e)
    (b : Position) (worker solve opening : Nat) {line : Line out.physical b worker 0}
    (ok : decode out.physical inputs stream b worker 0 solve opening = some line)
    (initialRows scored : List RootMoveStorage.Row) (pv : Bitboard.Square → List Bitboard.Square)
    (covers : Root.Covers mc b (initialRows.map RootMoveStorage.Row.square)) (positive : 0 < initialRows.length)
    (root : let row := line.result.row
      FrameReturns.RootRows (NativeFrames.logOf mc fc out.physical.rows out.physical.table identify before sequenced)
        row.frame row.depth initialRows 0 row.frame.alpha row.frame.beta pv row.frame.score scored)
    (sort : PublicationMir.SliceSort) (minimum nodes : Nat) (reportable : minimum ≤ 5)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ (t : Trial) (result : PublicationMir.SearchResult), (collect worker solve opening stream).bind (fun ts => ts[line.result.last.index]?) = some t ∧
      t.opening = line.result.row.opening ∧
      OuterMir.finish minimum 5 sort scored line.result.row.depth.toNat nodes old fallback = some result ∧
      PublicationMir.score result = some t.score.toInt ∧ t.score.toInt = PvsBoardTree.minimax mc fc b := by
  obtain ⟨t,member,same,word⟩ := physical_return ok
  obtain ⟨result,returned,score,truth⟩ := publicCacheScore cache identify before sequenced irrefl trans scores stores
    b worker (stamp line.trials) line.result.path.ran rfl line.result.found
    initialRows scored pv covers positive root sort minimum nodes reportable old fallback
  exact ⟨t,result,member,same,returned,word.symm ▸ score,word.symm ▸ truth⟩

/-- Root storage before and after the accepted call. Intervening lower-selectivity
    trials can reset or reorder only the active suffix. -/
structure Stage where
  opening : Nat
  prepared : List RootMoveStorage.Row
  physical : List RootMoveStorage.Row
  pv : Bitboard.Square → List Bitboard.Square

def runStages (c : Capture pool) (inputs : Nat → Nat → Option Args) (stream : List Record) (b : Position)
    (w solve : Nat) : Nat → Nat → List RootMoveStorage.Row → List Stage → Option (List RootMoveStorage.Row)
  | _,idx,rows,[] => if idx = rows.length then some rows else none
  | prior,idx,_,stage :: rest => do
    if prior < stage.opening ∧ idx < stage.prepared.length then
      let _ ← decode c inputs stream b w idx solve stage.opening
      runStages c inputs stream b w solve stage.opening (idx+1) (RootMoveStorage.sortRows stage.physical idx) rest
    else none

def Storage (c : Capture pool) (log : FrameReturns.Log mc fc) (inputs : Nat → Nat → Option Args)
    (stream : List Record) (b : Position) (w solve : Nat) :
    Nat → List RootMoveStorage.Row → List Stage → Prop
  | _,_,[] => True
  | idx,rows,stage :: rest =>
    Prepare idx rows stage.prepared ∧
      (∀ line : Line c b w idx, decode c inputs stream b w idx solve stage.opening = some line →
        let row := line.result.row
        FrameReturns.RootRows log row.frame row.depth stage.prepared idx row.frame.alpha row.frame.beta
          stage.pv row.frame.score stage.physical) ∧
      Storage c log inputs stream b w solve (idx+1) (RootMoveStorage.sortRows stage.physical idx) rest

theorem lines (stages : List Stage)
    (done : runStages c inputs stream b w solve prior idx rows stages = some final)
    (storage : Storage c log inputs stream b w solve idx rows stages) : NativeLines c log b depth idx rows final := by
  induction stages generalizing prior idx rows with
  | nil =>
    simp only [runStages] at done
    split at done <;> try contradiction
    rename_i equal
    cases done
    exact equal ▸ .done _
  | cons stage rest ih =>
    simp only [runStages] at done
    split at done <;> try contradiction
    rename_i both
    obtain ⟨line,checked,more⟩ := Option.bind_eq_some_iff.mp done
    exact .next w (stamp line.trials) _ _ _ line.result.row stage.pv stage.prepared rfl
      line.result.path.ran line.result.found storage.1 both.2 (storage.2.1 line checked) (ih more storage.2.2)

theorem publicLines {out : NativeFrameCache.Checked pool} (cache : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → NativeFrames.Branch c log endian row)
    (stores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ e ∈ c.table, FrameReturns.WriteStep log endian e)
    (worker solve : Nat) (stages : List Stage) (done : runStages out.physical inputs stream b worker solve solve 0 rows stages = some final)
    (storage : Storage out.physical (NativeFrames.logOf mc fc out.physical.rows out.physical.table identify before sequenced)
      inputs stream b worker solve 0 rows stages)
    (covers : Root.Covers mc b (rows.map RootMoveStorage.Row.square)) (nonempty : rows ≠ [])
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final (Board.emptyCount b.val).toNat nodes old fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square :=
  publicCacheLines cache identify before sequenced irrefl trans scores stores (lines stages done storage)
    covers nonempty minimum reportable sort nodes old fallback

end NeuralReversi.AspirationMir.NativeTrials
