import NeuralReversi.PvsParallel.NativeFrames.OwnerSites

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

open FrameReturns.OwnerCalls (ChildReturn)
open EndgameSearch.CacheHistory.NativeTrace (Activation Worker Cursor)
open EndgameSearch.CacheOwner (Node)

/-- Semantic guards at an exact PVS score instruction, before packing its registers. -/
structure Admissible (c : ChildReturn) : Prop where
  board : Board.wellFormed c.board
  depth : c.depth = (Board.emptyCount c.board).toNat
  window : Aspiration.Window c.alpha c.beta
  narrow : c.pv = false → c.beta = c.alpha+1

theorem pass_admissible (r : Row) (v : Int) :
    Admissible (FrameReturns.OwnerCalls.passCall r.frame r.depth r.frame.pv v) := by
  refine ⟨(PvsBoardTree.pass r.frame.position).property,
    congrArg BitVec.toNat (ExactDepth.pass_depth _ _ r.full),PvsMoveLoop.reflectedWindow r.window,?_⟩
  intro nonPv
  have unit := r.narrow nonPv
  change -r.frame.alpha = -r.frame.beta+1
  omega

theorem emitted_admissible (r : Row)
    (visits : List (RootOwnerMir.Slice.Visit (PvsBoardTree.GeneratedSquare mc r.frame.position.val)))
    (windows : ∀ v ∈ visits, Aspiration.Window v.alpha r.frame.beta) :
    ∀ c ∈ FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits, Admissible c := by
  intro c found
  obtain ⟨v,mem,pv,x,_,rfl⟩ := FrameReturns.OwnerCalls.emitted_iff.mp found
  refine ⟨(PvsBoardTree.play mc fc r.frame.position v.move).val.property,
    congrArg BitVec.toNat (ExactDepth.move_depth mc fc _ _ v.move r.full).2.1,?_,?_⟩
  · cases pv with
    | true => exact PvsMoveLoop.reflectedWindow (windows _ mem)
    | false => exact PvsMoveLoop.scoutWindow (windows _ mem)
  · rintro rfl
    change -v.alpha = -(v.alpha+1)+1
    omega

theorem pv_admissible {mc : Bitboard.Moves.X86Config} (r : Row)
    {order : PvsBoardTree.GeneratedSquare mc r.frame.position.val → Bitboard.Square}
    {src : Nat → PvsMoveLoop.ChildSearch (PvsBoardTree.GeneratedSquare mc r.frame.position.val)}
    {visits : List (RootOwnerMir.Slice.Visit (PvsBoardTree.GeneratedSquare mc r.frame.position.val))}
    {prior pending : List (PvsBoardTree.GeneratedSquare mc r.frame.position.val)}
    {out : RootOwnerMir.Slice.Exit (PvsBoardTree.GeneratedSquare mc r.frame.position.val)}
    (path : RootOwnerMir.Slice.Path order src schedule r.depth.toNat limit r.frame.beta i prior pending fresh
      (RootReplay.start r.frame.alpha) visits out) :
    ∀ c ∈ FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits, Admissible c :=
  emitted_admissible r visits (fun v mem => (path.operands r.window (fun _ => True) (by simp) v mem).1)

/-- Build the row that the decoder will create, including its real caller and the
    not-yet-returned score. No success of the frame decoder is assumed. -/
theorem create (a : Args) (word : ScoreRefresh.Score) (ok : Admissible (Returns.argsFields a word.raw.toInt))
    (size worker opening parent stamp : Nat) :
    ∃ r, a.row size worker opening parent stamp word = some r ∧
      r.opening = opening ∧ r.frame.parent = parent ∧ r.frame.scoreStep = stamp ∧
      r.frame.returnStep = none ∧ r.rawScore = word ∧ FrameReturns.OwnerCalls.fields r.frame = Returns.argsFields a word.raw.toInt := by
  have board : a.board.player &&& a.board.opponent = 0 := ok.board
  have depth : a.depth = Board.emptyCount a.board := BitVec.eq_of_toNat_eq ok.depth
  have guards : a.depth = Board.emptyCount a.board ∧ -32767 ≤ a.alpha.raw.toInt ∧
      a.alpha.raw.toInt < a.beta.raw.toInt ∧ a.beta.raw.toInt ≤ 32767 ∧
      (a.pv = false → a.beta.raw.toInt = a.alpha.raw.toInt+1) :=
    ⟨depth,ok.window.lower,ok.window.ordered,ok.window.upper,ok.narrow⟩
  unfold Args.row
  dsimp only
  split
  · refine ⟨_,rfl,rfl,rfl,rfl,rfl,rfl,?_⟩
    simp [FrameReturns.OwnerCalls.fields,TraceCheck.FrameReturn.board,Returns.argsFields]
  · exact False.elim (by apply ‹¬ _›; exact board)

private theorem lookup_site (e : EndgameSearch.CacheOwner.Edge) (member : e ∈ EndgameSearch.CacheOwner.transfers) :
    EndgameSearch.CacheOwner.transfers.find? (fun x => x.caller == e.caller && x.site == e.site) = some e := by
  have checked : EndgameSearch.CacheOwner.transfers.all (fun op =>
    EndgameSearch.CacheOwner.transfers.find? (fun x => x.caller == op.caller && x.site == op.site) == some op) = true := by decide
  exact beq_iff_eq.mp (List.all_eq_true.mp checked e member)

theorem enter_routed {w : Worker p} (s : Cursor w) (e : EndgameSearch.CacheOwner.Edge)
    (member : e ∈ EndgameSearch.CacheOwner.transfers) (parent : s.parent.node = e.caller)
    (route : e.flow.apply p.outside s.parent.received = some receiver)
    {args : Option EndgameSearch.CacheHistory.NativeTrace.Operands}
    (metadata : args.isSome = (e.callee == .boundary) ∧ (e.callee = .boundary → s.busy = false)) :
    ∃ next : Cursor w, s.enter stamp e.site e.callee receiver args = some next ∧
      next.finished = s.finished ∧ next.boundarySites = (if e.callee = .boundary then []
        else if e.caller = .boundary then (e.site,e.callee)::s.boundarySites else s.boundarySites) := by
  unfold Cursor.enter
  dsimp only
  have found : EndgameSearch.CacheOwner.transfers.find? (fun x => x.caller == s.parent.node && x.site == e.site) =
      some e := by rw [parent]; exact lookup_site e member
  split
  · rename_i absent
    rw [found] at absent
    contradiction
  · rename_i edge got
    have eq := Option.some.inj (got.symm.trans found)
    subst edge
    by_cases boundary : e.callee = .boundary
    · have free := metadata.2 boundary
      simp [parent,route,metadata.1,boundary,free]
    · simp [parent,route,metadata.1,boundary]

/-- Enter a MIR callee with its actual receiver. Non-boundary routines use no
    cache metadata, and only PVS has a search-register snapshot. -/
theorem enter_call {w : Worker p} (s : State w) (e : EndgameSearch.CacheOwner.Edge)
    (member : e ∈ EndgameSearch.CacheOwner.transfers) (plain : e.callee ≠ .boundary)
    (parent : s.cursor.parent.node = e.caller) (route : e.flow.apply p.outside s.cursor.parent.received = some receiver)
    (ordered : s.cursor.clock ≤ stamp) (input : (inputs stamp).isSome = (e.callee == .pvs)) :
    ∃ out : State w, s.advance inputs (.entry stamp e.site e.callee receiver none) = some out ∧
      out.cursor.clock = stamp+1 ∧ out.rows = s.rows ∧ ∃ top : Activation w, out.cursor.stack = top :: s.cursor.stack ∧
        top.first = stamp ∧ top.node = e.callee ∧ top.received = receiver ∧ top.operands = none ∧
        out.cursor.finished = s.cursor.finished ∧ out.cursor.boundarySites =
          (if e.caller = .boundary then (e.site,e.callee)::s.cursor.boundarySites else s.cursor.boundarySites) := by
  obtain ⟨next,ran,done_,sites⟩ := enter_routed (args:=none) (stamp:=stamp) s.cursor e member parent route (by simp [plain])
  obtain ⟨top,pushed,_,first,node,rec,args,_⟩ :=
    EndgameSearch.CacheHistory.NativeTrace.checked_entry _ _ ran
  have tick := (EndgameSearch.CacheHistory.NativeTrace.advance_clock s.cursor
    (.enter stamp e.site e.callee receiver none)
    (by simpa [Cursor.advance,EndgameSearch.CacheHistory.NativeTrace.Raw.step,ordered] using ran)).2
  refine ⟨⟨next,s.rows⟩,?_,tick,rfl,top,pushed,first,node,rec,args,done_,by simpa only [ite_eq_right plain] using sites⟩
  simp [State.advance,input,Cursor.advance,EndgameSearch.CacheHistory.NativeTrace.Raw.step,ordered,ran,plain]

/-- A nested activation can interpose new rows after scoring. Finishing the
    current call updates just its own row beneath that prefix. -/
private theorem finish_before (recent rest : List Row) (r : Row) (opened : r.opening = opening)
    (pending : r.frame.returnStep = none) (score : r.rawScore = word)
    (later : r.frame.scoreStep < finish) (newer : ∀ a ∈ recent, opening < a.opening) :
    ∃ completed, finishRow opening finish word (recent ++ r :: rest) = some (recent ++ completed :: rest) ∧
      completed.opening = opening ∧ completed.frame.returnStep = some finish := by
  have same : r.frame.score = word.raw.toInt := score ▸ r.value
  obtain ⟨completed,done_,key,returned⟩ : ∃ q, r.finish finish word = some q ∧
      q.opening = opening ∧ q.frame.returnStep = some finish := by
    unfold Row.finish
    simp [later,pending,same,opened]
  refine ⟨completed,?_,key,returned⟩
  induction recent with
  | nil => simp [finishRow,opened,done_]
  | cons a more ih =>
    have bound := newer a List.mem_cons_self
    have apart : a.opening ≠ opening := by omega
    simp [finishRow,apart,ih (fun b mem => newer b (List.mem_cons_of_mem _ mem))]

/-- A normal child can score on its actual caller stack as soon as its input is
    admissible. The time and uniqueness follow from the surrounding call order. -/
theorem score_child {w : Worker p} (s : State w) (top : Activation w) (rest : List (Activation w))
    (stack : s.cursor.stack = top :: rest) (kind : top.node = .pvs) (key : top.first = opening)
    (a : Args) (word : ScoreRefresh.Score) (ok : Admissible (Returns.argsFields a word.raw.toInt))
    (exact_ : a.exact = true) (input : inputs opening = some a)
    (ordered : s.cursor.clock ≤ stamp) (fresh : ∀ r ∈ s.rows, r.opening ≠ opening) :
    ∃ (out : State w) (r : Row), s.advance inputs (.scored stamp word) = some out ∧
      out.cursor = {s.cursor with clock := stamp+1} ∧ out.rows = r::s.rows ∧
      r.opening = opening ∧ r.frame.scoreStep = stamp ∧ r.frame.returnStep = none ∧ r.rawScore = word ∧
      FrameReturns.OwnerCalls.fields r.frame = Returns.argsFields a word.raw.toInt := by
  let parent := match rest with
    | caller :: _ => if caller.node == Node.pvs then frameKey (EndgameSearch.CacheOwner.poolPlan p.size).length w.ordinal caller.first else 0
    | [] => 0
  obtain ⟨r,created,opened,_,atScore,pending,actual,fields⟩ :=
    create a word ok (EndgameSearch.CacheOwner.poolPlan p.size).length w.ordinal opening parent stamp
  have unique : (s.rows.any (fun a => a.opening == opening)) = false := by
    simpa using fresh
  refine ⟨⟨{s.cursor with clock := stamp+1},r::s.rows⟩,r,?_,rfl,rfl,opened,atScore,pending,actual,fields⟩
  simp [State.advance,stack,kind,key,input,Nat.not_lt_of_ge ordered,exact_,unique]
  cases rest <;> simpa [parent,stack] using congrArg (fun result => result.bind (fun q =>
    some (⟨{s.cursor with clock := stamp+1},q::s.rows⟩ : State w))) created

/-- A recursive PVS return pops the activation at its opening time. Descendants
    that completed after its score remain in their original order. -/
theorem return_child {w : Worker p} (s : State w) (top : Activation w) (rest : List (Activation w))
    (stack : s.cursor.stack = top::rest) (kind : top.node = .pvs) (key : top.first = opening)
    (noBoundary : top.operands = none) (ordered : s.cursor.clock ≤ finish)
    (exact_ : a.exact = true) (input : inputs opening = some a)
    (recent older : List Row) (r : Row) (rows : s.rows = recent ++ r::older)
    (opened : r.opening = opening) (pending : r.frame.returnStep = none)
    (score : r.rawScore = word) (later : r.frame.scoreStep < finish)
    (newer : ∀ a ∈ recent, opening < a.opening) :
    ∃ (out : State w) (completed : Row),
      s.advance inputs (.leave finish opening .pvs none (some word)) = some out ∧
      out.cursor.stack = rest ∧ out.cursor.clock = finish+1 ∧ out.rows = recent ++ completed::older ∧
      completed.opening = opening ∧ completed.frame.returnStep = some finish := by
  obtain ⟨completed,done_,same,ret⟩ := finish_before recent older r opened pending score later newer
  refine ⟨⟨{s.cursor with clock := finish+1,stack := rest},recent ++ completed::older⟩,completed,?_,rfl,rfl,rfl,same,ret⟩
  simp [State.advance,Cursor.advance,EndgameSearch.CacheHistory.NativeTrace.Raw.step,ordered,Cursor.leave,
    stack,kind,key,noBoundary,input,exact_,rows,done_]

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites
