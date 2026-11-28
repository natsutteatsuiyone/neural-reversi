module
public import NeuralReversi.PvsParallel.SerialExecution
public import NeuralReversi.TranspositionTable.Access

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.Writes

open PvsBoardTree (Position GeneratedSquare legalSquares)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig key exactSelectivity)
open TranspositionTable (Depth Bound EntryData)
open TranspositionTable.ProbeRetry (RawAttempt)
open SerialCalls (Program Request)
open GeneralNonPvMir (entryAction probeHint checkedHint order)

structure Proposal where
  board : Position
  depth : Depth
  score : Int
  bound : Bound
  move : Option (Fin 64)
  ttPv : Bool

def Proposal.pack (w : Proposal) (generation : BitVec 8) : EntryData :=
  TranspositionTable.Packing.newData ⟨BitVec.ofInt 32 w.score⟩ w.move w.bound w.depth exactSelectivity generation true w.ttPv

structure Result where
  score : Int
  store : Option Proposal

def parentProbe (b : Position) (attempts : List RawAttempt) :=
  TranspositionTable.ProbeRetry.probe (attempts.map (TranspositionTable.ProbeRetry.observe (key b)))

def parentPv (pv : Bool) (b : Position) (attempts : List RawAttempt) : Bool :=
  pv || match parentProbe b attempts with
  | .miss _ => false
  | .hit _ e => TranspositionTable.Access.isPv e

inductive Site (Move : Type) (alpha score : Int) where
  | skip
  | tail (move : Option Move)
  | lower (move : Move) (above : alpha < score)

structure Decision (Move : Type) (alpha : Int) where
  score : Int
  site : Site Move alpha score

def Decision.emit {mc : X86Config} {b : Position} (d : Depth) (pv ttPv : Bool) (a z : Int)
    (x : Decision (GeneratedSquare mc b.val) a) : Result :=
  let make (bound : Bound) (move : Option (GeneratedSquare mc b.val)) : Proposal :=
    ⟨b,d,x.score,bound,move.map Subtype.val,ttPv⟩
  ⟨x.score,match x.site with
    | .skip => none
    | .tail move => some (make (TranspositionTable.classify pv ⟨BitVec.ofInt 32 x.score⟩
        ⟨BitVec.ofInt 32 a⟩ ⟨BitVec.ofInt 32 z⟩) move)
    | .lower m _ => some (make .lower (some m))⟩

private theorem map_comp (p : Program Req A) (f : A → B) (g : B → C) :
    (p.map f).map g = p.map (g ∘ f) := by
  induction p with
  | pure | fail => rfl
  | call q k ih => exact congrArg (Program.call q) (funext ih)

private theorem map_id (p : Program Req A) : p.map id = p := by
  induction p with
  | pure | fail => rfl
  | call q k ih => exact congrArg (Program.call q) (funext ih)

def owner (req : Move → Request) (a : Int) :
    List Move → Int → Option Move → Program Request (Decision Move a)
  | [],best,previous => .pure ⟨best,.tail previous⟩
  | m :: more,best,previous =>
    SerialCalls.Sites.ownerWith false false a (req m) (req m) fun _ _ score =>
      let update := GeneralOwnerMir.after a best score
      let chosen := if update.2.1 then some m else previous
      if update.2.2 then .pure ⟨score,.tail chosen⟩ else owner req a more update.1 chosen

private theorem owner_score (req : Move → Request) (a : Int) (ms : List Move) (best : Int) (prev : Option Move) :
    (owner req a ms best prev).map Decision.score = SerialNonPvCalls.owner req a ms best := by
  induction ms generalizing best prev with
  | nil => rfl
  | cons m ms ih =>
    simp only [owner,SerialNonPvCalls.owner,SerialCalls.Sites.ownerWith,Bool.false_and,Bool.false_eq_true,ite_false,Program.map]
    apply congrArg (Program.call _)
    funext w
    split <;> first | rfl | apply ih

def mobile [DecidableEq Move] (req : Move → Request) (priority : Move → Int) (hint : Option Move)
    (raw : Bool) (first : Move) (rest : List Move) (a z : Int) : Program Request (Decision Move a) :=
  let ms := first :: rest
  let ordered := order priority hint ms
  if hint.isSome = raw then
    if ms.length = 1 || raw then
      let m := hint.getD first
      SerialCalls.Sites.presearchWith (req m) fun score =>
        if cut : (a < score) && (z ≤ score) then .pure ⟨score,.lower m (by simpa using (Bool.and_eq_true_iff.mp cut).1)⟩
        else owner req a (ordered.drop 1) score (if a < score then some m else none)
    else owner req a ordered (-32767) none
  else .fail

theorem mobile_score [DecidableEq Move] (req : Move → Request) (priority : Move → Int) (hint : Option Move)
    (raw : Bool) (first : Move) (rest : List Move) (a z : Int) :
    (mobile req priority hint raw first rest a z).map Decision.score =
      SerialNonPvCalls.mobile req priority hint raw first rest a z := by
  simp only [mobile,SerialNonPvCalls.mobile]
  split <;> try rfl
  split
  · simp only [SerialCalls.Sites.presearchWith,Program.map]
    apply congrArg (Program.call _)
    funext w
    split <;> first | rfl | apply owner_score
  · apply owner_score

def enhanced {mc : X86Config} {fc : FlipConfig} {b : Position} (d : Depth) (a : ScoreRefresh.Score)
    (reads : List (TableReads.ChildLookupFrom mc fc b source)) (level := exactSelectivity) : Option (Decision (GeneratedSquare mc b.val) a.raw.toInt) :=
  let query : PvsEnhancedCutoff.Query := ⟨d,level,true,⟨a.raw⟩⟩
  match hit : PvsEnhancedCutoff.scan query (reads.map TableReads.ChildLookupFrom.observation) with
  | none => none
  | some (sq,v) =>
    have legal := TableReads.enhancedHint query reads hit sq rfl
    have witness := ((PvsEnhancedCutoff.contract_proven query (reads.map TableReads.ChildLookupFrom.observation)).2.1 _).mp hit
    have larger : a.raw.toInt < v := by
      obtain ⟨_,_,_,_,⟨_,_,_,_,above,eq⟩,_⟩ := witness
      have equal := congrArg Prod.snd eq
      change -_ = v at equal
      exact equal ▸ above
    some ⟨v,.lower ⟨sq,(Board.getMoves_exact mc b.val sq).mpr legal⟩ larger⟩

def nonPvDecision (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (TableReads.ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int) :
    Program Request (Decision (GeneratedSquare mc b.val) a.raw.toInt) :=
  let action := entryAction mc fc b d a z paired checked mem attempts reads
  if action = .rest then
    match legalSquares mc b.val with
    | [] => .fail
    | first::rest =>
      let tt := probeHint b attempts
      mobile (SerialNonPvCalls.scout mc fc b d cut a.raw.toInt) priority (checkedHint mc b tt) tt.isSome first rest a.raw.toInt z.raw.toInt
  else if action = .enhanced then .ofOption (enhanced d a reads)
  else (SerialNonPvCalls.prelude mc fc b d a z paired checked cut mem attempts reads).map (fun v => ⟨v,.skip⟩)

def nonPv (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (TableReads.ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int) : Program Request Result :=
  (nonPvDecision mc fc b d a z paired checked cut mem attempts reads priority).map
    (Decision.emit d false (parentPv false b attempts) a.raw.toInt z.raw.toInt)

def pvUpdate (beta : Int) (previous : Int × Option Move) (visit : RootOwnerMir.Slice.Visit Move) :
    Int × Option Move :=
  match visit.wide.or visit.scout with
  | none => previous
  | some word =>
    let score := -word
    let d := RootUpdateMir.decision false visit.alpha beta previous.1 score
    (if d.raisesBest then score else previous.1,if d.takesMove then some visit.move else previous.2)

/-- The visit includes the scout and wide return words at this exact occurrence.
    The final search supplies the word consumed by the Rust best-move update. -/
def pvMove (beta : Int) (visits : List (RootOwnerMir.Slice.Visit Move)) : Option Move :=
  (visits.foldl (pvUpdate beta) (-PvsMoveLoop.scoreInf,none)).2

def pvDecision (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (a z : Int) (moves : List (GeneratedSquare mc b.val)) :
    Program Request (Decision (GeneratedSquare mc b.val) a) :=
  match SerialPvCalls.action mc fc b d paired mem with
  | .continueSearch =>
    ((SerialOwnerCalls.run Subtype.val z moves (RootReplay.start a)).mapRequests
      (SerialPvCalls.moveRequest b mc fc d cut)).map fun out => ⟨out.2.state.best,.tail (pvMove z out.1)⟩
  | .pass => (SerialCalls.Sites.pass (PvsBoardTree.pass b) d true cut a z).map (fun v => ⟨v,.skip⟩)
  | _ => (Program.ofOption (PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none))).map
      (fun v => ⟨v,.skip⟩)

def pv (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (a z : Int) (moves : List (GeneratedSquare mc b.val)) :
    Program Request Result :=
  (pvDecision mc fc b d paired checked cut mem a z moves).map (Decision.emit d true true a z)

theorem pv_score (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (a z : Int) (moves : List (GeneratedSquare mc b.val)) :
    (pv mc fc b d paired checked cut mem a z moves).map Result.score =
      SerialPvCalls.frame mc fc b d paired checked cut mem a z moves := by
  unfold pv pvDecision SerialPvCalls.frame
  rw [map_comp]
  cases SerialPvCalls.action mc fc b d paired mem <;> dsimp only <;> rw [map_comp] <;> first | rfl | apply map_id

theorem enhanced_score {mc : X86Config} {fc : FlipConfig} {b : Position} (d : Depth) (a : ScoreRefresh.Score)
    (reads : List (TableReads.ChildLookupFrom mc fc b source)) (level := exactSelectivity) :
    (enhanced d a reads level).map Decision.score = (PvsEnhancedCutoff.scan
      ⟨d,level,true,⟨a.raw⟩⟩ (reads.map TableReads.ChildLookupFrom.observation)).map Prod.snd := by
  unfold enhanced
  dsimp only
  split <;> rename_i h <;> exact (congrArg (Option.map Prod.snd) h).symm

theorem nonPv_score (mc : X86Config) (fc : FlipConfig) (b : Position) (d : Depth) (a z : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (TableReads.ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int) :
    (nonPv mc fc b d a z paired checked cut mem attempts reads priority).map Result.score =
      SerialNonPvCalls.frame mc fc b d a z paired checked cut mem attempts reads priority := by
  unfold nonPv
  rw [map_comp]
  change (nonPvDecision _ _ _ _ _ _ _ _ _ _ _ _ _).map Decision.score = _
  by_cases regular : entryAction mc fc b d a z paired checked mem attempts reads = .rest
  · simp only [nonPvDecision,SerialNonPvCalls.frame,regular,ite_true]
    cases legalSquares mc b.val with
    | nil => rfl
    | cons => apply mobile_score
  by_cases action : entryAction mc fc b d a z paired checked mem attempts reads = .enhanced
  · have result := enhanced_score d a reads
    have unit : GeneralNonPvMir.runPrefix mc fc b d a z paired checked cut mem attempts reads (fun _ _ _ _ _ => none) =
        (enhanced d a reads).map Decision.score := by
      unfold GeneralNonPvMir.runPrefix
      unfold entryAction at action
      dsimp only
      rw [action,result]
      rfl
    simp only [nonPvDecision,SerialNonPvCalls.frame,action,reduceCtorEq,ite_false,ite_true,
      SerialNonPvCalls.prelude,unit]
    cases enhanced d a reads <;> rfl
  · simp only [nonPvDecision,SerialNonPvCalls.frame,ite_eq_right regular,ite_eq_right action,map_comp]
    apply map_id

theorem pvUpdate_issued (first : Bool) (m : Move) (a z : Int) (previous : Int × Option Move)
    (scout wide : Req) :
    SerialCalls.Sites.ownerWith true first a scout wide
      (fun s w _ => .pure (pvUpdate z previous ⟨m,a,s,w⟩)) =
    SerialCalls.Sites.ownerWith true first a scout wide (fun _ _ score =>
      let d := RootUpdateMir.decision first a z previous.1 score
      .pure (if d.raisesBest then score else previous.1, if d.takesMove then some m else previous.2)) := by
  cases first
  · unfold SerialCalls.Sites.ownerWith
    simp only [Bool.true_and,Bool.false_eq_true,ite_false]
    apply congrArg (Program.call _)
    funext w
    split <;> rfl
  · rfl

def word (v : Int) : TranspositionTable.Score := ⟨BitVec.ofInt 32 v⟩

structure Produced (b : Position) (d : Depth) (pv ttPv : Bool) (a z score : Int) (w : Proposal) : Prop where
  board : w.board = b
  depth : w.depth = d
  value : w.score = score
  hint : TableReads.LegalChoice b w.move
  pvTag : w.ttPv = ttPv
  bound : w.bound = TranspositionTable.classify pv (word score) (word a) (word z)

private theorem choice (m : Option (GeneratedSquare mc b.val)) : TableReads.LegalChoice b (m.map Subtype.val) := by
  intro sq eq
  obtain ⟨w,_,rfl⟩ := Option.map_eq_some_iff.mp eq
  exact (Board.getMoves_exact mc b.val w.val).mp w.property

theorem emit_tail {mc : X86Config} {b : Position} {pv tag : Bool} (m : Option (GeneratedSquare mc b.val)) {score : Int}
    (saved : ((⟨score,.tail m⟩ : Decision (GeneratedSquare mc b.val) a).emit d pv tag a z).store = some w) :
    Produced b d pv tag a z score w := by
  cases Option.some.inj saved
  exact ⟨rfl,rfl,rfl,choice _,rfl,rfl⟩

theorem Decision.nonPv_produced {mc : X86Config} {b : Position} {d : Depth} {a z : ScoreRefresh.Score}
    (result : Decision (GeneratedSquare mc b.val) a.raw.toInt)
    (saved : (result.emit d false tag a.raw.toInt z.raw.toInt).store = some w)
    (null : z.raw.toInt = a.raw.toInt+1) (finite : SerialCalls.ScoreWords.Finite result.score) :
    Produced b d false tag a.raw.toInt z.raw.toInt result.score w := by
  rcases result with ⟨v,site⟩
  cases site with
  | skip => contradiction
  | tail m => exact emit_tail m saved
  | lower m higher =>
    change -32767 < v ∧ v < 32767 at finite
    cases Option.some.inj saved
    refine ⟨rfl,rfl,rfl,choice (some m),rfl,?_⟩
    have val : (word v).raw.toInt = v :=
      BitVec.toInt_ofInt_eq_self (by decide) (by have := finite.1; omega) (by have := finite.2; omega)
    have beta : (word z.raw.toInt).raw.toInt = z.raw.toInt := by simp [word]
    change Bound.lower = TranspositionTable.classify false (word v) _ _
    simp [TranspositionTable.classify,val,beta,show z.raw.toInt ≤ v by omega]

theorem nonPv_produced {mc : X86Config} {fc : FlipConfig} {b : Position} {d : Depth} {a z : ScoreRefresh.Score}
    {mem : MoveList.Initialization.Memory Move.Move} {attempts : List RawAttempt}
    {reads : List (TableReads.ChildLookupFrom mc fc b source)} {priority : GeneratedSquare mc b.val → Int}
    (ran : Program.Runs (nonPv mc fc b d a z paired checked cut mem attempts reads priority) children out)
    (saved : out.store = some w) (null : z.raw.toInt = a.raw.toInt+1)
    (finite : SerialCalls.ScoreWords.Finite out.score) :
    Produced b d false (parentPv false b attempts) a.raw.toInt z.raw.toInt out.score w := by
  obtain ⟨result,_,rfl⟩ := Program.runs_map ran
  exact result.nonPv_produced saved null finite

theorem pv_produced {mc : X86Config} {fc : FlipConfig} {b : Position} {d : Depth}
    {mem : MoveList.Initialization.Memory Move.Move} {moves : List (GeneratedSquare mc b.val)}
    (ran : Program.Runs (pv mc fc b d paired checked cut mem a z moves) children out)
    (saved : out.store = some w) : Produced b d true true a z out.score w := by
  obtain ⟨result,run_,rfl⟩ := Program.runs_map ran
  unfold pvDecision at run_
  cases which : SerialPvCalls.action mc fc b d paired mem <;> rw [which] at run_ <;>
    obtain ⟨value,_,rfl⟩ := Program.runs_map run_ <;> first | contradiction | exact emit_tail _ saved

end NeuralReversi.PvsParallel.SerialTable.Writes
