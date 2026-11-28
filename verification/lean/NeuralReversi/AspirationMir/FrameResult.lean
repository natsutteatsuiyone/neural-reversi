module
public import NeuralReversi.AspirationMir.PublicScore
public import NeuralReversi.PvsParallel.FrameWrites
public import NeuralReversi.PvsParallel.FrameRows
public import NeuralReversi.OuterMir.Correct

@[expose] public section

namespace NeuralReversi.AspirationMir

open PvsParallel
open FrameReturns (Log Frame RootStep Decoded)
open RootMoveStorage (Row)

theorem publicFrameScore {log : Log mc fc} (decoded : Decoded log endian)
    (f : Frame) (member : f ∈ log.frames)
    (ran : Returns query (fun _ => false) machine raw last) (entry : machine.pc = 0)
    (sameAlpha : f.alpha = last.alpha.toInt) (sameBeta : f.beta = last.beta.toInt)
    (sameCall : f.score = (query last.index last.alpha last.beta).toInt)
    (covers : Root.Covers mc f.position (initial.map Row.square)) (positive : 0 < initial.length)
    (root : RootStep log f depth (initial.map Row.square) 0 f.alpha f.beta f.score snapshot.scores)
    (best : snapshot.state.best = f.score) (writes : RootSuffix.Writes initial 0 f.alpha f.beta physical snapshot)
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort physical depth.toNat nodes prior fallback = some result ∧
      PublicationMir.score result = some raw.toInt ∧ raw.toInt = PvsBoardTree.minimax mc fc f.position := by
  obtain ⟨_, _, _, _, _, _, _, window, _, _⟩ := decoded.frames f member
  obtain ⟨⟨used,low,high⟩, rfl⟩ := returned_inside ran entry
  have value := sameCall.trans (congrArg BitVec.toInt used)
  have inside : f.alpha < f.score ∧ f.score < f.beta := by
    rw [sameAlpha,value,sameBeta]
    exact ⟨low,high⟩
  have enough : 0 < (initial.map Row.square).length := by simpa using positive
  have attempt := (decoded.rootAttempt root covers).sound covers enough window
  have truth := (attempt.1.exact inside.1 inside.2).trans (Root.value_zero covers enough)
  rw [← best] at inside attempt
  obtain ⟨result, published, score⟩ := publishSingle covers positive writes inside attempt sort depth.toNat nodes fallback
  exact ⟨result, by simpa [OuterMir.finish, reportable] using published,
    score.trans (congrArg some (truth.symm.trans value)), value ▸ truth⟩

theorem publicRootStep {log : Log mc fc} (decoded : Decoded log endian)
    (f : Frame) (member : f ∈ log.frames)
    (ran : Returns query (fun _ => false) machine raw last) (entry : machine.pc = 0)
    (sameAlpha : f.alpha = last.alpha.toInt) (sameBeta : f.beta = last.beta.toInt)
    (sameCall : f.score = (query last.index last.alpha last.beta).toInt)
    (covers : Root.Covers mc f.position (initial.map Row.square)) (positive : 0 < initial.length)
    (root : RootStep log f depth (initial.map Row.square) 0 f.alpha f.beta f.score scores)
    (pv : Bitboard.Square → List Bitboard.Square) (minimum : Nat) (reportable : minimum ≤ 5)
    (sort : PublicationMir.SliceSort) (nodes : Nat)
    (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ physical snapshot result, RootSuffix.Writes initial 0 f.alpha f.beta physical snapshot ∧ snapshot.scores = scores ∧
      OuterMir.finish minimum 5 sort physical depth.toNat nodes prior fallback = some result ∧
      PublicationMir.score result = some raw.toInt ∧ raw.toInt = PvsBoardTree.minimax mc fc f.position := by
  obtain ⟨physical,snapshot,writes,best,same⟩ := root.materialize covers pv
  obtain ⟨result,published,score,exactScore⟩ := publicFrameScore decoded f member ran entry sameAlpha sameBeta sameCall
    covers positive (same.symm ▸ root) best writes minimum reportable sort nodes prior fallback
  exact ⟨physical,snapshot,result,writes,same,published,score,exactScore⟩

/-- Earlier trials overwrite candidate rows. A completed selectivity also sorts the same suffix. -/
inductive Prepare (idx : Nat) : List Row → List Row → Prop
  | start : Prepare idx rows rows
  | trial : Prepare idx rows before → RootSuffix.Writes before idx a z after snapshot → Prepare idx rows after
  | sort : Prepare idx rows before → Prepare idx rows (RootMoveStorage.sortRows before idx)

theorem Prepare.preserves (prepared : Prepare idx rows before) (unique : (rows.map Row.square).Nodup) :
    (before.map Row.square).Perm (rows.map Row.square) ∧ before.take idx = rows.take idx := by
  induction prepared with
  | start => exact ⟨.refl _,rfl⟩
  | trial _ writes ih =>
    have kept := writes.scores (ih.1.nodup_iff.mpr unique)
    exact ⟨kept.1 ▸ ih.1, kept.2.1.trans ih.2⟩
  | sort _ ih =>
    exact ⟨((RootMoveStorage.sortRowsContract_proven _ _).1.map Row.square).trans ih.1,
      (RootMoveStorage.sortRowsContract_proven _ _).2.1.trans ih.2⟩

/-- The accepted call of each exact Multi-PV line, including the array passed to the next line.
    Suffix calls have their own frame ids; only the first line searches the whole root. -/
inductive FrameLines (log : Log mc fc) (b : PvsBoardTree.Position) (d : TranspositionTable.Depth) :
    Nat → List Row → List Row → Prop
  | done (rows : List Row) : FrameLines log b d rows.length rows rows
  | line (f : Frame) (q : Query) (m : Machine) (raw : Word) (last : Call)
      (pv : Bitboard.Square → List Bitboard.Square) (prepared : List Row) :
      f.position = b → f.depth = d.toNat → f.pv = true → f.cut = false →
      m.pc = 0 → Returns q (fun _ => false) m raw last →
      f.alpha = last.alpha.toInt → f.beta = last.beta.toInt → f.score = (q last.index last.alpha last.beta).toInt →
      -32767 ≤ f.alpha → f.beta ≤ 32767 →
      Prepare idx before prepared →
      idx < prepared.length →
      FrameReturns.RootRows log f d prepared idx f.alpha f.beta pv f.score physical →
      FrameLines log b d (idx+1) (RootMoveStorage.sortRows physical idx) final →
      FrameLines log b d idx before final

theorem FrameLines.checked {log : Log mc fc} (runs : FrameLines log b d idx rows final)
    (decoded : Decoded log endian) (covers : Root.Covers mc b (rows.map Row.square)) :
    RootSuffix.Lines mc fc b idx rows final := by
  induction runs with
  | done => exact .done _
  | line f q m raw last pv prepared board _ _ _ entry ran alpha beta score lower upper earlier within source _ ih =>
    obtain ⟨order,front⟩ := earlier.preserves (Root.roots_nodup covers)
    have covered := order.trans covers
    obtain ⟨snapshot,root,writes,value⟩ := source.checked (board ▸ covered)
    obtain ⟨⟨used,lo,hi⟩,_⟩ := returned_inside ran entry
    have inside : f.alpha < f.score ∧ f.score < f.beta := by
      rw [alpha,beta,score,used]
      exact ⟨lo,hi⟩
    have window : Aspiration.Window f.alpha f.beta := ⟨lower,by omega,upper⟩
    have attempt := decoded.rootAttempt root (board ▸ covered)
    rw [board] at attempt
    have sound := attempt.sound covered (by simpa using within) window
    have names := (writes.scores (Root.roots_nodup covered)).1
    apply RootSuffix.Lines.acceptedWrites covered order front within writes (value ▸ inside) (value ▸ sound)
    apply ih
    exact ((RootMoveStorage.sortRowsContract_proven _ _).1.map Row.square).trans (names ▸ covered)

theorem publicFrameLines {log : Log mc fc} (decoded : Decoded log endian) (runs : FrameLines log b d 0 rows final)
    (covers : Root.Covers mc b (rows.map Row.square)) (nonempty : rows ≠ [])
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final d.toNat nodes prior fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square := by
  have lines := runs.checked decoded covers
  obtain ⟨result,returned,best⟩ := publicMultiPV lines covers nonempty sort d.toNat nodes fallback
  exact ⟨result,by simpa [OuterMir.finish,reportable] using returned,best,(lines.exactRows covers (by simp)).2⟩

end NeuralReversi.AspirationMir
