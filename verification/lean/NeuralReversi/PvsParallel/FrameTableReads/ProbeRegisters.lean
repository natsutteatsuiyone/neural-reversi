import NeuralReversi.PvsParallel.FrameTableReads.Schedule

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution

open TranspositionTable
open SeqLock (Word)

def agrees (s t : Supply) (i : Nat) : Op → Prop
  | .slot r (.load _ _) => s.word r.address i = t.word r.address i
  | .generation a => s.generationByte a i = t.generationByte a i
  | _ => True

private theorem slot_eq (r : Site) (endian : Bool)
    (read : Runs (agrees s t) r.opening (r.emit s (snapCode endian))) :
    r.emit s (snapCode endian) = r.emit t (snapCode endian) ∧
    snapshot endian (r.words s) = snapshot endian (r.words t) := by
  obtain ⟨steps,value⟩ := (snapshot_path endian).congr (r.words s) (r.words t) (fun i loc mod found =>
    r.issued_at read found)
  exact ⟨congrArg (List.map (Op.slot r)) steps,value⟩

private theorem round_eq (s t : Supply) (allocation : Nat) (clusters key : Word)
    (endian : Bool) (q : ProbeRetry.Key) (start : Nat) :
    let r := roundAt s allocation clusters key endian q start
    let v := roundAt t allocation clusters key endian q start
    Runs (agrees s t) start (r.emit s endian q) →
      r.emit s endian q = v.emit t endian q ∧ (r.pass s endian q).exit = (v.pass t endian q).exit := by
  intro r v ran
  let a : Site := ⟨slotAt allocation clusters key 0,start⟩
  let b : Site := ⟨slotAt allocation clusters key 1,a.next s (snapCode endian)⟩
  have layout := r.emit_slots s endian q (placed ..).branch
  rw [layout] at ran
  obtain ⟨ops,_⟩ := ran.append
  obtain ⟨slots,gen⟩ := ops.append
  obtain ⟨front,back⟩ := slots.append
  have first : r.front = a := by dsimp [r,roundAt]; split <;> rfl
  rw [first] at front back gen
  obtain ⟨aOps,aVal⟩ := slot_eq a endian front
  have nextA : a.next s (snapCode endian) = a.next t (snapCode endian) :=
    congrArg (fun xs : List Op => start+xs.length) aOps
  have finish (same : r = v) (raw : r.read s endian = r.read t endian)
      (backOps : ∀ b ∈ r.back, b.emit s (snapCode endian) = b.emit t (snapCode endian)) :
      r.emit s endian q = v.emit t endian q ∧ (r.pass s endian q).exit = (v.pass t endian q).exit := by
    have passEq : r.pass s endian q = r.pass t endian q := by simp only [Round.pass,raw]
    rw [← same]
    refine ⟨?_,congrArg ReaderMir.Pass.exit passEq⟩
    rw [layout, r.emit_slots t endian q (by rw [same]; exact (placed ..).branch)]
    rw [raw,first,aOps]
    cases hb : r.back with
    | none => rfl
    | some site => simp [backOps site (by simp [hb])]
  let backAt := fun u => Site.mk (slotAt allocation clusters key 1) (a.next u (snapCode endian))
  let readsGen := fun u => ReaderMir.readsGeneration (ProbeRetry.classify q (snapshot endian (a.words u)))
    (ProbeRetry.classify q (snapshot endian ((backAt u).words u)))
  have form (u : Supply) : roundAt u allocation clusters key endian q start =
      (if ReaderMir.probeSecond q (snapshot endian (a.words u)) then
        ⟨a,some (backAt u),if readsGen u then u.generationByte allocation ((backAt u).next u (snapCode endian)) &&& 127 else 0⟩
      else ⟨a,none,0⟩) := rfl
  cases second : ReaderMir.probeSecond q (snapshot endian (a.words s)) with
  | false =>
    have rs : r = ⟨a,none,0⟩ := by rw [show r = _ from form s,second]; rfl
    have rt : v = ⟨a,none,0⟩ := by rw [show v = _ from form t,← aVal,second]; rfl
    apply finish (rs.trans rt.symm)
    · simp [rs,Round.read,aVal]
    · simp [rs]
  | true =>
    let atGen := b.next s (snapCode endian)
    let g := if readsGen s then s.generationByte allocation atGen &&& 127 else 0
    have rs : r = ⟨a,some b,g⟩ := by rw [show r = _ from form s,second]; rfl
    have joined : a.opening + (a.emit s (snapCode endian)).length = b.opening := rfl
    have bRead : Runs (agrees s t) b.opening (b.emit s (snapCode endian)) := by
      simpa [rs,← joined] using back
    obtain ⟨bOps,bVal⟩ := slot_eq b endian bRead
    have nextB : b.next s (snapCode endian) = b.next t (snapCode endian) :=
      congrArg (fun xs : List Op => b.opening+xs.length) bOps
    have generation (used : readsGen s = true) : s.generationByte allocation atGen = t.generationByte allocation atGen := by
      have taken : ReaderMir.readsGeneration (ProbeRetry.observe q (r.read s endian)).first
          (ProbeRetry.observe q (r.read s endian)).second = true := by
        simpa [rs,Round.read,ProbeRetry.observe,readsGen,backAt,b] using used
      rw [taken,rs] at gen
      have w := gen 0 (.generation allocation) (by simp [a,slotAt])
      simpa [atGen,Site.next,rs,b,a,Nat.add_assoc,agrees] using w
    have site : backAt t = b := congrArg (Site.mk _) nextA.symm
    have readEq : readsGen t = readsGen s := by
      dsimp only [readsGen]
      rw [site,← bVal,← aVal]
    have rt : v = ⟨a,some b,g⟩ := by
      rw [show v = _ from form t,← aVal,second,ite_eq_left rfl,site,readEq,← nextB]
      cases used : readsGen s with
      | false => simp [g,used]
      | true => simp [g,used,atGen,generation used]
    apply finish (rs.trans rt.symm)
    · simp [rs,Round.read,aVal,bVal]
    · simpa [rs] using bOps

private theorem sweep_eq (s t : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool)
    (q : ProbeRetry.Key) (fuel start : Nat) (last : Fin 2) :
    let x := sweep s allocation clusters key endian q fuel start last
    let y := sweep t allocation clusters key endian q fuel start last
    Runs (agrees s t) start (x.emit s endian q) →
      x.emit s endian q = y.emit t endian q ∧ x.result = y.result := by
  induction fuel generalizing start last with
  | zero => exact fun _ => ⟨rfl,rfl⟩
  | succ fuel ih =>
    let r := roundAt s allocation clusters key endian q start
    let v := roundAt t allocation clusters key endian q start
    simp only [sweep,←show r = roundAt s allocation clusters key endian q start from rfl,←show v = roundAt t allocation clusters key endian q start from rfl]
    cases branch : (r.pass s endian q).exit with
    | done result =>
      intro ran
      have one : Runs (agrees s t) start (r.emit s endian q) := by simpa [Probed.emit] using ran
      obtain ⟨ops,exitEq⟩ := round_eq s t allocation clusters key endian q start one
      rw [exitEq.symm.trans branch]
      exact ⟨by simpa [Probed.emit] using ops,rfl⟩
    | retry chosen =>
      intro ran
      let next := start+(r.emit s endian q).length
      let tail := sweep s allocation clusters key endian q fuel next (chosen.getD last)
      have whole : Runs (agrees s t) start (r.emit s endian q ++ tail.emit s endian q) := ran
      obtain ⟨one,more⟩ := whole.append
      obtain ⟨ops,exitEq⟩ := round_eq s t allocation clusters key endian q start one
      obtain ⟨tailOps,tailResult⟩ := ih next (chosen.getD last) more
      rw [exitEq.symm.trans branch]
      change r.emit s endian q = v.emit t endian q at ops
      rw [← congrArg List.length ops]
      exact ⟨(congrArg (fun xs => xs ++ tail.emit s endian q) ops).trans (congrArg (List.append _) tailOps),tailResult⟩

/-- Replaying a probe needs agreement only for the loads and generation
    reads reached by its actual control flow. -/
theorem probe_congr (s t : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (q : ProbeRetry.Key) :
    let x := probe s allocation clusters key endian q 0
    let y := probe t allocation clusters key endian q 0
    Runs (agrees s t) 0 (x.emit s endian q) → x.emit s endian q = y.emit t endian q ∧ x.result = y.result :=
  sweep_eq s t allocation clusters key endian q _ 0 0

private theorem lookup_slot (r : Site) (b : PvsBoardTree.Position)
    (covered : Runs (agrees s t) r.opening (r.emit s (lookupCode endian b))) :
    r.emit s (lookupCode endian b) = r.emit t (lookupCode endian b) ∧
      lookupValue endian b (r.words s) = lookupValue endian b (r.words t) := by
  obtain ⟨path,value⟩ := (lookup_path endian b).congr (r.words s) (r.words t) (fun i loc mode used =>
    r.issued_at covered used)
  exact ⟨congrArg (List.map (Op.slot r)) path,value⟩

theorem lookup_congr (s t : Supply) (allocation : Nat) (clusters key : Word) (endian : Bool) (b : PvsBoardTree.Position) :
    let x := lookup s allocation clusters key endian b 0
    let y := lookup t allocation clusters key endian b 0
    Runs (agrees s t) 0 (x.emit s endian b) →
      x.emit s endian b = y.emit t endian b ∧ x.pair s endian b = y.pair t endian b := by
  let first : Site := ⟨slotAt allocation clusters key 0,0⟩
  let back := fun u => Site.mk (slotAt allocation clusters key 1) (first.next u (lookupCode endian b))
  have shape (u : Supply) : lookup u allocation clusters key endian b 0 =
      (match lookupValue endian b (first.words u) with
        | some _ => ⟨first,none⟩ | none => ⟨first,some (back u)⟩) := rfl
  dsimp only
  rw [shape s,shape t]
  cases read : lookupValue endian b (first.words s) with
  | some v =>
    intro ran
    obtain ⟨ops,eq⟩ := lookup_slot first b (by simpa [Looked.emit,first] using ran)
    rw [←eq,read]
    simp [Looked.emit,Looked.pair,ops,eq]
  | none =>
    intro ran
    obtain ⟨front,remaining⟩ := ran.append
    obtain ⟨ops,eq⟩ := lookup_slot first b front
    have atBack : back t = back s := congrArg (fun xs : List Op => Site.mk (slotAt allocation clusters key 1)
        (first.opening+xs.length)) ops.symm
    have tailRun : Runs (agrees s t) (back s).opening ((back s).emit s (lookupCode endian b)) := by
      simpa [back,first,Site.next] using remaining
    obtain ⟨more,last⟩ := lookup_slot (back s) b tailRun
    rw [←eq,read,atBack]
    simp [Looked.emit,Looked.pair,ops,more,eq,last]

theorem Call.readProbe_eq (c : Call) (kind : c.method = .probe q) :
    readProbe c.allocation c.clusters c.key c.endian q c.records =
      some (probe c.supply c.allocation c.clusters c.key c.endian q 0).result := by
  have agree : Runs (agrees c.supply (Execution.supply c.records)) 0 c.ops := by
    intro i op found
    cases op with
    | slot r action =>
      cases action with
      | fence => trivial
      | load => simp [agrees,Execution.supply,c.atIndex,found,Call.atomic]
    | generation => simp [agrees,Execution.supply,c.atIndex,found,Call.atomic]
    | spin => trivial
  have ops := c.instructions
  simp only [Call.ops,kind] at agree ops
  obtain ⟨path,out⟩ := sweep_eq c.supply (Execution.supply c.records) c.allocation c.clusters c.key c.endian q _ 0 0 agree
  unfold readProbe
  dsimp only
  split
  · exact congrArg some out.symm
  · next mismatch => exact (mismatch (ops.trans (congrArg (List.map instruction) path))).elim

/-- Other readers and writers may take arbitrarily many turns between two loads. -/
theorem readProbe_dispatched (calls : Caller → Call) (turns : List Turn) (who : Caller)
    (finished : (calls who).ops.length ≤ budget who turns) (kind : (calls who).method = .probe q) :
    let c := calls who
    readProbe c.allocation c.clusters c.key c.endian q (selected who (dispatch calls (fun _ => 0) turns)) =
      some (probe c.supply c.allocation c.clusters c.key c.endian q 0).result := by
  rw [complete calls turns who finished]
  exact (calls who).readProbe_eq kind

def captureProbe (calls : Caller → Call) (turns : List Turn) (who : Caller)
    (finished : (calls who).ops.length ≤ budget who turns) (kind : (calls who).method = .probe q) :
    ProbeCapture q :=
  let c := calls who
  ⟨c.allocation,c.clusters,c.key,c.endian,selected who (dispatch calls (fun _ => 0) turns),
    (probe c.supply c.allocation c.clusters c.key c.endian q 0).result,readProbe_dispatched calls turns who finished kind⟩

/-- The same raw hit supplies both the MIR getter and the NonPV presearch operand.
    The trace alone determines whether the square is legal for the caller's board. -/
theorem presearch_operand (mc : Bitboard.Moves.X86Config) (board : PvsBoardTree.Position)
    (calls : Caller → Call) (turns : List Turn) (who : Caller)
    (finished : (calls who).ops.length ≤ budget who turns)
    (kind : (calls who).method = .probe (Cutoffs.key board)) :
    let c := calls who
    let p := probe c.supply c.allocation c.clusters c.key c.endian (Cutoffs.key board) 0
    let raw := GeneralNonPvMir.probeHint board (p.rounds.map (fun r => r.read c.supply c.endian))
    let hint := ReaderMir.bestMove (captureProbe calls turns who finished kind).result
    raw = hint ∧ ((GeneralNonPvMir.checkedHint mc board raw).isSome = raw.isSome ↔
      ∀ sq, hint = some sq → (Board.getMoves mc board.val).getLsbD sq.val = true) := by
  intro c p raw hint
  have same : raw = hint := by
    unfold raw GeneralNonPvMir.probeHint
    rw [List.map_map]
    exact congrArg ReaderMir.bestMove (probe_result c.supply c.allocation c.clusters c.key c.endian (Cutoffs.key board) 0)
  exact ⟨same, by rw [← same]; exact GeneralNonPvMir.checkedHint_present ..⟩

end NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution
