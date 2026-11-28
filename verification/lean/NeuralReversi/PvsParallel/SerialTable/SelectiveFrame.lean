module
public import NeuralReversi.PvsParallel.SerialTable.Stack
public import NeuralReversi.PvsParallel.SerialTable.SelectiveFrame.ProbCut

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.SelectiveFrame

open Bitboard.Moves (X86Config)
open TranspositionTable (Selectivity Depth EntryData)
open TranspositionTable.ProbeRetry (RawAttempt)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare legalSquares)
open SerialCalls (Request Program)
open SerialCalls.ScoreWords (Finite)
open SerialNonPvCalls (scout prelude)
open GeneralNonPvMir (entryAction probeHint checkedHint)
open TableReads (ChildLookupFrom)
open Writes (Decision)

abbrev emptyReads (mc : X86Config) (fc : FlipConfig) (b : Position) :=
  ([] : List (ChildLookupFrom mc fc b (·.publication [])))

def preAction (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) :=
  entryAction mc fc b d a z paired checked mem [] (emptyReads mc fc b)

def direct (b : Position) (d : Depth) (z : ScoreRefresh.Score) (checked : Bool) (level : Selectivity)
    (attempts : List RawAttempt) : Option Int := do
  let .hit _ e := Writes.parentProbe b attempts | none
  let ok ← (TranspositionTable.canCut checked e ⟨z.raw⟩ d level true).toOption
  if ok then some e.score.toInt else none

def query (d : Depth) (level : Selectivity) (a : ScoreRefresh.Score) : PvsEnhancedCutoff.Query :=
  ⟨d,level,true,⟨a.raw⟩⟩

def ordinary (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (cut : Bool) (attempts : List RawAttempt) (priority : GeneratedSquare mc b.val → Int) :
    Program Request (Decision (GeneratedSquare mc b.val) a.raw.toInt) :=
  match legalSquares mc b.val with
  | [] => .fail
  | first :: rest =>
    let tt := probeHint b attempts
    Writes.mobile (scout mc fc b d cut a.raw.toInt) priority (checkedHint mc b tt) tt.isSome
      first rest a.raw.toInt z.raw.toInt

def decision (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (level : Selectivity)
    (estimates : Estimates) (attempts : List RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) : Program Request (Decision (GeneratedSquare mc b.val) a.raw.toInt) :=
  if preAction mc fc b d a z paired checked mem ≠ .rest then
    (prelude mc fc b d a z paired checked cut mem [] (emptyReads mc fc b)).map (fun v => ⟨v,.skip⟩)
  else
    match direct b d z checked level attempts with
    | some score => .pure ⟨score,.skip⟩
    | none =>
      match (if 14 ≤ d.toNat then Writes.enhanced d a reads level else none) with
      | some choice => .pure choice
      | none =>
        match probCut d level cut z estimates with
        | some score => .pure ⟨score,.skip⟩
        | none => ordinary mc fc b d a z cut attempts priority

def frame (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (level : Selectivity)
    (estimates : Estimates) (attempts : List RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) : Program Request Writes.Result :=
  (decision mc fc b d a z paired checked cut mem level estimates attempts reads priority).map
    (Decision.emit d false (Writes.parentPv false b attempts) a.raw.toInt z.raw.toInt)

abbrev Allowed (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (r : Request) :=
  ExactDepth.RecursiveCall mc fc b d r.board r.depth ∧ Aspiration.Window r.alpha r.beta ∧
    (r.pv = false → r.beta = r.alpha+1)

theorem frame_ready {mc fc b d a z paired checked cut mem level estimates attempts}
    {reads : List (ChildLookupFrom mc fc b (·.publication history))} {priority}
    (full : ExactDepth.Full b d) (deep : 12 < d.toNat) (w : Aspiration.Window a.raw.toInt z.raw.toInt)
    (null : z.raw.toInt = a.raw.toInt+1) (seen : TableReads.ProbeReads history attempts)
    (hints : TableReads.LegalHints history) :
    Program.Ready (Allowed mc fc b d)
      (frame mc fc b d a z paired checked cut mem level estimates attempts reads priority) := by
  apply Program.Ready.map
  unfold decision
  split
  next precheck =>
    apply Program.Ready.map
    have live := SerialNonPvReady.frame_ready mc fc b d a z paired checked cut mem [] (emptyReads mc fc b)
      priority full deep w null (by simp [TableReads.ProbeReads,TableReads.ProbeReadsFrom])
      (by simp [TableReads.LegalHints])
    unfold preAction at precheck
    simpa only [SerialNonPvCalls.frame,ite_eq_right precheck] using live
  next precheck =>
    split <;> try exact .pure
    split <;> try exact .pure
    split <;> try exact .pure
    unfold ordinary
    have reached : preAction mc fc b d a z paired checked mem = .rest := by simpa using precheck
    have available := SerialNonPvReady.mobile_of_rest reached
    cases moves : legalSquares mc b.val with
    | nil => simp [PvEntryMir.generated_count,moves] at available
    | cons m ms =>
      apply Stack.ready_of_map (f := Decision.score)
      rw [Writes.mobile_score]
      apply SerialNonPvReady.mobile_ready
      · intro mv
        exact ⟨.ply b mv d,PvsMoveLoop.scoutWindow w,fun _ => by dsimp [scout,SerialCalls.Sites.scoutRequest]; omega⟩
      · simpa only [Option.isSome_map] using congrArg Option.isSome (GeneralNonPvMir.observedHint mc b seen hints)

theorem frame_produced {mc fc b d a z paired checked cut mem level estimates attempts}
    {reads : List (ChildLookupFrom mc fc b source)} {priority}
    (ran : Program.Runs (frame mc fc b d a z paired checked cut mem level estimates attempts reads priority) cs out)
    (saved : out.store = some w) (null : z.raw.toInt = a.raw.toInt+1) (finite : Finite out.score) :
    Writes.Produced b d false (Writes.parentPv false b attempts) a.raw.toInt z.raw.toInt out.score w := by
  obtain ⟨result,_,rfl⟩ := Program.runs_map ran
  exact result.nonPv_produced saved null finite

private theorem negate_finite {v : Int} (h : Finite v) : Finite (-v) := by
  unfold Finite at *; omega

private theorem owner_finite (req : Move → Request) (a : Int) (moves : List Move) (best : Int)
    (bounds : -32767 ≤ best ∧ best < 32767) (started : moves = [] → Finite best)
    (ran : Program.Runs (SerialNonPvCalls.owner req a moves best) cs value)
    (children : ∀ t ∈ cs, Finite t.2) : Finite value := by
  induction moves generalizing best cs value with
  | nil => cases ran; exact started rfl
  | cons m ms ih =>
    simp only [SerialNonPvCalls.owner,SerialCalls.Sites.ownerWith,Bool.false_and,Bool.false_eq_true,ite_false] at ran
    cases ran with
    | @call _ _ _ _ raw more =>
      have val := negate_finite (children _ List.mem_cons_self)
      change Finite (-raw) at val
      by_cases improved : best < -raw
      · by_cases cut : a < -raw
        · simp only [GeneralOwnerMir.after,improved,cut,ite_true] at more
          cases more; exact val
        · simp only [GeneralOwnerMir.after,improved,ite_true,ite_eq_right cut,Bool.false_eq_true,ite_false] at more
          exact ih _ ⟨Int.le_of_lt val.1,val.2⟩ (fun _ => val) more (fun t h => children t (.tail _ h))
      · have fin : Finite best := ⟨by have := val.1; omega,bounds.2⟩
        simp only [GeneralOwnerMir.after,ite_eq_right improved,Bool.false_eq_true,ite_false] at more
        exact ih _ bounds (fun _ => fin) more (fun t h => children t (.tail _ h))

private theorem mobile_finite [DecidableEq Move] (req : Move → Request) (priority : Move → Int)
    (hint : Option Move) (raw : Bool) (first : Move) (rest : List Move) (a z : Int)
    {result : Decision Move a} (ran : Program.Runs (Writes.mobile req priority hint raw first rest a z) cs result)
    (children : ∀ t ∈ cs, Finite t.2) : Finite result.score := by
  have run := Stack.runs_map ran Decision.score
  rw [Writes.mobile_score] at run
  dsimp only [SerialNonPvCalls.mobile] at run
  split at run <;> rename_i fits
  · split at run
    · cases run with
      | @call _ _ _ _ rawWord more =>
        have word := negate_finite (children _ List.mem_cons_self)
        dsimp only at word more
        split at more
        · exact Option.some.inj (Program.runs_ofOption (x := some (-rawWord)) more).1 ▸ word
        · exact owner_finite _ _ _ _ ⟨Int.le_of_lt word.1,word.2⟩ (fun _ => word) more
            (fun t h => children t (.tail _ h))
    · rename_i noPre
      have noHint : hint = none := by
        cases hint <;> cases raw <;> simp_all
      subst hint
      have perm := GeneralNonPvMir.orderPerm priority (first::rest) none (by simp) (by simp)
      apply owner_finite _ _ _ _ (by decide) _ run children
      intro empty
      have len := perm.length_eq
      rw [empty] at len
      simp at len
  · cases run

private theorem prelude_finite {mc fc b d a z paired checked cut mem}
    (early : preAction mc fc b d a z paired checked mem ≠ .rest) (full : ExactDepth.Full b d)
    (w : Aspiration.Window a.raw.toInt z.raw.toInt) (null : z.raw.toInt = a.raw.toInt+1)
    (ran : Program.Runs (prelude mc fc b d a z paired checked cut mem [] (emptyReads mc fc b)) cs v)
    (children : ∀ t ∈ cs, Finite t.2) : Finite v := by
  by_cases pass : preAction mc fc b d a z paired checked mem = .pass
  · unfold preAction at pass
    simp only [prelude,pass,ite_true] at ran
    cases ran with
    | call more => cases more; exact negate_finite (children _ List.mem_cons_self)
  · have empty : cs = [] := by
      unfold preAction at pass
      simp only [prelude,ite_eq_right pass] at ran
      exact (Program.runs_ofOption ran).2
    have whole : Program.Runs (SerialNonPvCalls.frame mc fc b d a z paired checked cut mem [] (emptyReads mc fc b)
        (fun _ => 0)) cs v := by
      unfold preAction at early
      simpa only [SerialNonPvCalls.frame,ite_eq_right early] using ran
    have replay := SerialNonPvCalls.frame_replay mc fc b d a z paired checked cut mem [] (emptyReads mc fc b)
      (fun _ => 0) 0 full w null (by simp [TableReads.ProbeReads,TableReads.ProbeReadsFrom])
      .initial (fun _ _ => by decide) whole (by simp [empty])
    exact SerialCalls.ScoreWords.finite_of_replay w (fun _ => null) replay

private theorem direct_finite {attempts : List RawAttempt} (seen : TableReads.ProbeReads history attempts)
    (published : ∀ b e, (b,e) ∈ history → Finite e.score.toInt)
    (hit : direct b d z checked level attempts = some v) : Finite v := by
  unfold direct at hit
  cases actual : Writes.parentProbe b attempts with
  | miss => simp [actual] at hit
  | hit slot e =>
    obtain ⟨tried,reached,read,occupied⟩ := TranspositionTable.ProbeRetry.contract_proven.2 _ _ _ _ actual
    have ok := published b e (seen.published.matches slot e tried reached read occupied)
    simp only [actual] at hit
    obtain ⟨cut,_,result⟩ := Option.bind_eq_some_iff.mp hit
    cases cut <;> cases result
    exact ok

private theorem enhanced_finite {mc fc b d} {a : ScoreRefresh.Score} {level} {reads : List (TableReads.ChildLookup mc fc b history)}
    (published : ∀ b e, (b,e) ∈ history → Finite e.score.toInt)
    {result : Decision (GeneratedSquare mc b.val) a.raw.toInt} (hit : Writes.enhanced d a reads level = some result) : Finite result.score := by
  have scored := Writes.enhanced_score d a reads level
  rw [hit] at scored
  have scan : (PvsEnhancedCutoff.scan (query d level a) (reads.map ChildLookupFrom.observation)).map Prod.snd =
      some result.score := scored.symm
  obtain ⟨found,got,val⟩ := Option.map_eq_some_iff.mp scan
  obtain ⟨before,p,after,order,⟨e,sampled,_,_,_,same⟩,_⟩ :=
    ((PvsEnhancedCutoff.contract_proven _ _).2.1 found).mp got
  obtain ⟨actual,_,rfl⟩ := List.mem_map.mp (show p ∈ reads.map ChildLookupFrom.observation by rw [order]; simp)
  obtain ⟨origin⟩ := actual.source e sampled
  have word := negate_finite (published _ e origin.source)
  exact val ▸ (congrArg Prod.snd same ▸ word)

theorem frame_finite {mc fc b d a z paired checked cut mem level estimates attempts}
    {reads : List (TableReads.ChildLookup mc fc b history)} {priority}
    (full : ExactDepth.Full b d) (w : Aspiration.Window a.raw.toInt z.raw.toInt) (null : z.raw.toInt = a.raw.toInt+1)
    (seen : TableReads.ProbeReads history attempts) (published : ∀ b e, (b,e) ∈ history → Finite e.score.toInt)
    (ran : Program.Runs (frame mc fc b d a z paired checked cut mem level estimates attempts reads priority) cs out)
    (children : ∀ t ∈ cs, Finite t.2) : Finite out.score := by
  obtain ⟨result,issued,rfl⟩ := Program.runs_map ran
  change Finite result.score
  unfold decision at issued
  split at issued
  next early =>
    obtain ⟨v,got,rfl⟩ := Program.runs_map issued
    exact prelude_finite early full w null got children
  next =>
    split at issued
    next v hit => cases issued; exact direct_finite seen published hit
    next =>
      split at issued
      next value hit =>
        cases issued
        split at hit <;> try contradiction
        exact enhanced_finite published hit
      next =>
        split at issued
        next v cut => cases issued; exact (probCut_finite w cut).2
        next =>
          unfold ordinary at issued
          split at issued
          · cases issued
          · exact mobile_finite _ _ _ _ _ _ _ _ issued children

end NeuralReversi.PvsParallel.SerialTable.SelectiveFrame
