module
public import NeuralReversi.AspirationMir.NativeExecution
public import NeuralReversi.PublicationMir.ReturnedCapture

@[expose] public section

namespace NeuralReversi.AspirationMir.NativePublication

open PvsParallel
open PublicationMir.ReturnedScores (Report Move Value Matches Maximum)

inductive Event where
  | sample (s : NativeRows.Sample)
  | saved (worker step solve object : Nat) (depth : BitVec 32) (level : Nat) (endgame : Bool) (rows : List Move)
  | returned (worker step solve : Nat) (boundary : Option Nat) (result : Option Report)

def samples (events : List Event) : List NativeRows.Sample :=
  events.filterMap fun
    | .sample s => some s
    | .saved .. => none
    | .returned w t solve boundary _ => some (.engine (.localCall w (.leave t solve .solve boundary none)))

inductive Start where
  | single (trial opening returned : Nat) (score : ScoreRefresh.Score)
  | multi (index sortedAt : Nat)

def begins (w object : Nat) (start : Start) : Event → Bool
  | .sample (.engine (.localCall who (.leave t op .pvs _ (some v)))) =>
    match start with
    | .single _ opening returned score => who == w && op == opening && t == returned && v == score
    | _ => false
  | .sample (.locked who t obj (.sort index)) =>
    match start with
    | .multi idx sortedAt => who == w && t == sortedAt && obj == object && index == idx
    | _ => false
  | _ => false

inductive Phase where
  | leaving (trial : Nat)
  | sorting
  | ready
  deriving DecidableEq

abbrev values (rows : List RootMoveStorage.Row) : List Value := rows.map fun r => (r.square,r.score)

def after (w solve object depth : Nat) (roots : List Value) :
    Phase → Option (List Move) → List Event → Option Report
  | _, _, [] => none
  | phase, last, .returned who _ opening _ result :: rest =>
    if who == w && opening == solve then do
      let out ← result
      let saved ← last
      if phase = .ready ∧ Matches roots saved depth out then some out else none
    else after w solve object depth roots phase last rest
  | phase, last, .saved who _ owner obj d sel eg rows :: rest =>
    if who == w && owner == solve then
      if phase == .ready && obj == object && d.toNat == depth && sel == 5 && eg then
        after w solve object depth roots phase (some rows) rest
      else none
    else after w solve object depth roots phase last rest
  | phase, last, .sample (.locked who _ obj mutation) :: rest =>
    if obj != object then after w solve object depth roots phase last rest else
    match phase, mutation with
    | .sorting, .sort 0 => if who == w then after w solve object depth roots .ready last rest else none
    | _, _ => none
  | phase, last, .sample (.engine (.localCall who event)) :: rest =>
    if who != w then after w solve object depth roots phase last rest else
    match event with
    | .entry _ _ .endWindow .. | .entry _ _ .pvs .. => none
    | .leave _ opening .solve .. => if opening == solve then none else after w solve object depth roots phase last rest
    | .leave _ opening .endWindow .. =>
      if phase == .leaving opening then after w solve object depth roots .sorting last rest else none
    | _ => after w solve object depth roots phase last rest
  | phase, last, .sample (.engine (.table _)) :: rest => after w solve object depth roots phase last rest

private def receipt (w solve : Nat) (out : Report) : Event → Prop
  | .returned who _ opening _ (some actual) => who = w ∧ opening = solve ∧ actual = out
  | _ => False

private theorem after_observed (events : List Event) (phase : Phase) (last : Option (List Move))
    (run : after w solve object depth roots phase last events = some out) :
    (∃ saved, Matches roots saved depth out) ∧ ∃ ev ∈ events, receipt w solve out ev := by
  induction events generalizing phase last with
  | nil => contradiction
  | cons event rest ih =>
    have tail {p l} (step : after w solve object depth roots p l rest = some out) :
        (∃ saved, Matches roots saved depth out) ∧ ∃ ev ∈ event :: rest, receipt w solve out ev := by
      obtain ⟨accepted,ev,mem,r⟩ := ih p l step
      exact ⟨accepted,ev,List.mem_cons_of_mem _ mem,r⟩
    cases event with
    | returned who tick op boundary result =>
      simp only [after] at run
      split at run
      · rename_i selected
        obtain ⟨r,ret,s,saved,ok⟩ := (by simpa only [bind,Option.bind_eq_some_iff] using run :
          ∃ r, result = some r ∧ ∃ s, last = some s ∧
            (if phase = .ready ∧ Matches roots s depth r then some r else none) = some out)
        split at ok <;> try contradiction
        rename_i condition
        cases ok
        simp only [Bool.and_eq_true,beq_iff_eq] at selected
        exact ⟨⟨s,condition.2⟩,_,List.mem_cons_self,by simp [receipt,ret,selected.1,selected.2]⟩
      · exact tail run
    | saved =>
      simp only [after] at run
      repeat' first | exact tail run | contradiction | split at run
    | sample s =>
      cases s with
      | locked => simp only [after] at run; repeat' first | exact tail run | contradiction | split at run
      | engine record =>
        cases record with
        | table => exact tail run
        | localCall => simp only [after] at run; repeat' first | exact tail run | contradiction | split at run

def decode (events : List Event) (w solve object depth : Nat) (start : Start)
    (rows : List RootMoveStorage.Row) : Option Report := do
  let _ :: rest := events.dropWhile (fun e => !begins w object start e) | none
  after w solve object depth (values rows) (match start with | .single t .. => .leaving t | .multi .. => .ready) none rest

/-- The final capture/return suffix is accepted for every nonempty vector of machine
    scores. Both the saved moves and the reported fields come from the extracted code. -/
theorem captured_suffix (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (nonempty : rows ≠ []) (depth : BitVec 32) (w solve object index tick nodes : Nat)
    (signed : ∀ r ∈ rows, (BitVec.ofInt 32 r.score).toInt = r.score) :
    ∃ it report, (PublicationMir.record sort rows depth.toNat 5 true).map Prod.fst = some (some it) ∧
      PublicationMir.intoResult (some it) nodes (fun _ => .noLegalMove) = some (.bestMove report) ∧
      let encode := PublicationMir.ReturnedScores.raw
      let words : Report := ⟨encode report.best,report.alternatives.map encode,BitVec.ofNat 32 report.stamp.depth,
        report.stamp.selectivity,report.endgame⟩
      decode [.sample (.locked w tick object (.sort index)),
          .saved w (tick+1) solve object depth 5 true (it.rows.map (encode ∘ PublicationMir.project)),
          .returned w (tick+2) solve none (some words)] w solve object depth.toNat (.multi index tick) rows = some words := by
  obtain ⟨it,report,stored,result,fields⟩ := PublicationMir.ReturnedScores.capture_returns sort rows nonempty
    depth nodes (fun _ => .noLegalMove) signed
  refine ⟨it,report,stored,result,?_⟩
  simp [decode,begins,after,values,fields]

theorem captured_single_suffix (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (nonempty : rows ≠ []) (depth : BitVec 32) (w solve object trial opening tick nodes : Nat)
    (pvsScore : ScoreRefresh.Score) (signed : ∀ r ∈ rows, (BitVec.ofInt 32 r.score).toInt = r.score) :
    let sorted := RootStorageMir.sort 0 rows
    ∃ it report, (PublicationMir.record sort sorted depth.toNat 5 true).map Prod.fst = some (some it) ∧
      PublicationMir.intoResult (some it) nodes (fun _ => .noLegalMove) = some (.bestMove report) ∧
      let encode := PublicationMir.ReturnedScores.raw
      let words : Report := ⟨encode report.best,report.alternatives.map encode,BitVec.ofNat 32 report.stamp.depth,
        report.stamp.selectivity,report.endgame⟩
      decode [.sample (.engine (.localCall w (.leave tick opening .pvs none (some pvsScore)))),
          .sample (.engine (.localCall w (.leave (tick+1) trial .endWindow none none))),
          .sample (.locked w (tick+2) object (.sort 0)),
          .saved w (tick+3) solve object depth 5 true (it.rows.map (encode ∘ PublicationMir.project)),
          .returned w (tick+4) solve none (some words)]
        w solve object depth.toNat (.single trial opening tick pvsScore) rows = some words := by
  intro sorted
  have perm : sorted.Perm rows := by
    simpa only [sorted,RootStorageMir.sort_eq] using (RootMoveStorage.sortRowsContract_proven rows 0).1
  have inhabited : sorted ≠ [] := fun nil => nonempty (List.length_eq_zero_iff.mp (by simpa [nil] using perm.length_eq.symm))
  obtain ⟨it,report,stored,result,fields⟩ := PublicationMir.ReturnedScores.capture_returns sort sorted inhabited
    depth nodes (fun _ => .noLegalMove) (fun r mem => signed r (perm.mem_iff.mp mem))
  have same := fields.1.trans (perm.map (fun r => (r.square,r.score)))
  have actual : Matches (values rows) _ _ _ := ⟨same,fields.2⟩
  refine ⟨it,report,stored,result,?_⟩
  simp [decode,begins,after,actual]

theorem observed (run : decode events w solve object depth start rows = some out) :
    Maximum (values rows) out.best.2.toInt ∧
    (out.moves.map PublicationMir.ReturnedScores.value).Perm (values rows) ∧
    ∃ t boundary, Event.returned w t solve boundary (some out) ∈ events := by
  unfold decode at run
  generalize eq : events.dropWhile (fun e => !begins w object start e) = suffix at run
  cases suffix with
  | nil => contradiction
  | cons head rest =>
    obtain ⟨⟨saved,h⟩,ev,mem,receipt⟩ := after_observed _ _ _ run
    obtain ⟨best,allRows⟩ := PublicationMir.ReturnedScores.observed h
    have present : ev ∈ events := (List.dropWhile_sublist ..).subset (eq ▸ List.mem_cons_of_mem head mem)
    cases ev <;> simp_all [NativePublication.receipt]
    case returned who time op boundary result =>
      cases result <;> simp_all
      exact ⟨time,boundary,present⟩

private def descending : PublicationMir.SliceSort where
  byReverseKey key rows := rows.mergeSort fun a b => decide (key b ≤ key a)
  permutation _ _ := List.mergeSort_perm ..
  ordered key rows := by
    let le := fun (a b : RootMoveStorage.Row) => decide (key b ≤ key a)
    have trans (a b c) (ab : le a b) (bc : le b c) : le a c := by
      simp only [le,decide_eq_true_eq] at *
      omega
    have total (a b) : le a b || le b a := by
      simp only [le,Bool.or_eq_true,decide_eq_true_eq]
      exact Int.le_total _ _
    exact (List.pairwise_mergeSort trans total rows).imp of_decide_eq_true

private theorem maximum_of_finish {rows : List RootMoveStorage.Row}
    (finished : OuterMir.finish 0 5 descending rows d 0 none (fun _ => .noLegalMove) = some result)
    (score : PublicationMir.score result = some v) : Maximum (values rows) v := by
  let ordered := descending.byReverseKey PublicationMir.key rows
  have perm : ordered.Perm rows := descending.permutation ..
  have order := descending.ordered PublicationMir.key rows
  change ordered.Pairwise _ at order
  simp only [OuterMir.finish, Nat.zero_le, ite_true, PublicationMir.completed,PublicationMir.record,
    PublicationMir.capture,PublicationMir.first,PublicationMir.snapshot,PublicationMir.intoResult] at finished
  cases shape : ordered with
  | nil =>
    have eq : descending.byReverseKey PublicationMir.key rows = [] := shape
    simp [eq] at finished
  | cons head tail =>
    have eq : descending.byReverseKey PublicationMir.key rows = head :: tail := shape
    simp [eq] at finished
    subst result
    obtain rfl := Option.some.inj score
    rw [shape] at perm order
    refine ⟨⟨(head.square,head.score), List.mem_map.mpr ⟨head,perm.mem_iff.mp (by simp),rfl⟩,rfl⟩,?_⟩
    intro r mem
    obtain ⟨row,member,rfl⟩ := List.mem_map.mp mem
    rcases List.mem_cons.mp (perm.mem_iff.mpr member) with rfl | later
    · exact Int.le_refl _
    · exact (List.pairwise_cons.mp order).1 row later

variable {out : NativeFrameCache.Checked pool}

/-- Execute the extracted capture and return for any order of equal keys.
    Only the winning score needs to fit in the public score word. -/
private theorem return_maximum (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (winner : Maximum (values rows) v) (fits : (BitVec.ofInt 32 v).toInt = v)
    (depth nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ report, PublicationMir.completed sort rows depth 5 true nodes fallback = some (.bestMove report) ∧
      (PublicationMir.ReturnedScores.raw report.best).2.toInt = v ∧
      report.alternatives.Perm (rows.map PublicationMir.project) := by
  obtain ⟨⟨_, member, value⟩, upper⟩ := winner
  obtain ⟨best, included, rfl⟩ := List.mem_map.mp member
  have perm := sort.permutation PublicationMir.key rows
  have order := sort.ordered PublicationMir.key rows
  cases sorted : sort.byReverseKey PublicationMir.key rows with
  | nil => rw [sorted] at perm; simp [perm.symm.eq_nil] at included
  | cons head tail =>
    rw [sorted] at perm order
    have bound := upper (head.square,head.score) (List.mem_map_of_mem (perm.mem_iff.mp (by simp)))
    have other : best.score ≤ head.score := by
      rcases List.mem_cons.mp (perm.mem_iff.mpr included) with rfl | later
      · exact Int.le_refl _
      · exact (List.pairwise_cons.mp order).1 best later
    have score : head.score = v := by change best.score = v at value; change head.score ≤ v at bound; omega
    simp [PublicationMir.completed,PublicationMir.record,PublicationMir.first,PublicationMir.capture,
      PublicationMir.snapshot,PublicationMir.intoResult,PublicationMir.build,PublicationMir.project,
      PublicationMir.ReturnedScores.raw,sorted,score]
    exact ⟨by simpa using fits, by simpa [PublicationMir.project,score] using perm.map PublicationMir.project⟩

private theorem minimax_fits (b : PvsBoardTree.Position) :
    (BitVec.ofInt 32 (PvsBoardTree.minimax mc fc b)).toInt = PvsBoardTree.minimax mc fc b := by
  have range := PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc b)
  change -16384 ≤ PvsBoardTree.minimax mc fc b ∧ PvsBoardTree.minimax mc fc b ≤ 16384 at range
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

private theorem move_fits (b : PvsBoardTree.Position) (sq : Bitboard.Square) :
    (BitVec.ofInt 32 (Root.squareValue mc fc b sq)).toInt = Root.squareValue mc fc b sq := by
  unfold Root.squareValue
  split
  · rename_i legal
    unfold Root.childValue PvsBoardTree.minimax
    have range := PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b ⟨sq,legal⟩).val)
    unfold PvsMoveLoop.scoreMax at range
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega
  · decide

/-- Construct the signed result from the checked search prefix. This includes the
    single-PV suffix sort that precedes the final snapshot in solve_root. -/
theorem returnSingle {samples : List NativeRows.Sample} (e : NativeExecution.Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs vals (NativeRows.engine samples) = some out)
    (checked : e.check inputs samples = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (w solve opening : Nat) {line : NativeTrials.Line out.physical b w 0}
    (ran : NativeTrials.decode out.physical inputs (NativeRows.engine samples) b w 0 solve opening = some line)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ stored report, NativeExecution.finalRows e samples line.result.row = some stored ∧
      PublicationMir.completed sort (RootStorageMir.sort 0 stored.rows) line.result.row.depth.toNat 5 true nodes fallback =
        some (.bestMove report) ∧
      (PublicationMir.ReturnedScores.raw report.best).2.toInt = PvsBoardTree.minimax mc fc b := by
  obtain ⟨_,physical,_,_,_,found,finished,score,truth⟩ :=
    NativeExecution.publicScore e cache checked irrefl b w solve opening ran descending 0 0 (by decide) none
      (fun _ => .noLegalMove)
  have best := maximum_of_finish finished score
  rw [truth] at best
  have reorder := (RootMoveStorage.sortRowsContract_proven physical.rows 0).1.map
    (fun r => (r.square,r.score))
  obtain ⟨report,published,word,_⟩ := return_maximum sort (RootStorageMir.sort 0 physical.rows)
    (v := PvsBoardTree.minimax mc fc b) (by
      rw [RootStorageMir.sort_eq]
      obtain ⟨⟨r,mem,eq⟩,bound⟩ := best
      exact ⟨⟨r,reorder.mem_iff.mpr mem,eq⟩,fun x hx => bound x (reorder.mem_iff.mp hx)⟩)
    (minimax_fits b) _ nodes fallback
  exact ⟨physical,report,found,published,word⟩

theorem returnMulti {samples : List NativeRows.Sample} (e : NativeExecution.Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs vals (NativeRows.engine samples) = some out)
    (checked : e.check inputs samples = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (w solve object : Nat) (stages : List NativeRowLines.Stage)
    (ran : NativeRootBirth.run out.physical _ inputs samples b w solve object e.lines paired mem stages = some rows)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ report, PublicationMir.completed sort rows (Board.emptyCount b.val).toNat 5 true nodes fallback =
      some (.bestMove report) ∧
      (PublicationMir.ReturnedScores.raw report.best).2.toInt = PvsBoardTree.minimax mc fc b ∧
      ∀ move ∈ report.alternatives, (PublicationMir.ReturnedScores.raw move).2.toInt =
        Root.squareValue mc fc b move.square := by
  obtain ⟨_,finished,score,truth⟩ := NativeExecution.publicLines e cache checked irrefl b w solve object stages
    ran 0 (by decide) descending 0 none (fun _ => .noLegalMove)
  obtain ⟨report,published,best,moves⟩ :=
    return_maximum sort rows (maximum_of_finish finished score) (minimax_fits b) _ nodes fallback
  refine ⟨report,published,best,?_⟩
  intro move member
  obtain ⟨row,seen,rfl⟩ := List.mem_map.mp (moves.mem_iff.mp member)
  change (BitVec.ofInt 32 row.score).toInt = Root.squareValue mc fc b row.square
  rw [truth row seen,move_fits]

theorem single (e : NativeExecution.Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs vals (NativeRows.engine (samples events)) = some out)
    (checked : e.check inputs (samples events) = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (w solve opening : Nat) {line : NativeTrials.Line out.physical b w 0}
    (ran : NativeTrials.decode out.physical inputs (NativeRows.engine (samples events)) b w 0 solve opening = some line)
    (published : (do
      let row := line.result.row
      let physical ← NativeExecution.finalRows e (samples events) row
      let t ← row.frame.returnStep
      decode events w solve (e.receiver row) row.depth.toNat (.single opening row.opening t row.rawScore)
        physical.rows) = some report) :
    report.best.2.toInt = PvsBoardTree.minimax mc fc b ∧
    ∃ tick boundary, Event.returned w tick solve boundary (some report) ∈ events := by
  obtain ⟨_,physical,result,_,_,found,finished,score,truth⟩ :=
    NativeExecution.publicScore e cache checked irrefl b w solve opening ran descending 0 0 (by decide) none
      (fun _ => .noLegalMove)
  obtain ⟨store,storeEq,rest⟩ := Option.bind_eq_some_iff.mp published
  rw [found] at storeEq
  cases storeEq
  obtain ⟨t,_,got⟩ := Option.bind_eq_some_iff.mp rest
  obtain ⟨actual,_,returned⟩ := observed got
  exact ⟨(PublicationMir.ReturnedScores.maximum_unique actual (maximum_of_finish finished score)).trans truth,returned⟩

theorem multi (e : NativeExecution.Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs vals (NativeRows.engine (samples events)) = some out)
    (checked : e.check inputs (samples events) = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (w solve object : Nat) (stages : List NativeRowLines.Stage)
    (ran : NativeRootBirth.run out.physical _ inputs (samples events) b w solve object e.lines paired mem stages = some rows)
    (published : (do
      let last ← stages.getLast?
      decode events w solve object (Board.emptyCount b.val).toNat (.multi (stages.length - 1) last.sortedAt) rows) = some report) :
    report.best.2.toInt = PvsBoardTree.minimax mc fc b ∧
    (∀ move ∈ report.moves, move.2.toInt = Root.squareValue mc fc b move.1) ∧
    ∃ tick boundary, Event.returned w tick solve boundary (some report) ∈ events := by
  obtain ⟨result,finished,score,truth⟩ := NativeExecution.publicLines e cache checked irrefl b w solve object stages
    ran 0 (by decide) descending 0 none (fun _ => .noLegalMove)
  obtain ⟨_,_,got⟩ := Option.bind_eq_some_iff.mp published
  obtain ⟨actual,moves,returned⟩ := observed got
  refine ⟨PublicationMir.ReturnedScores.maximum_unique actual (maximum_of_finish finished score),?_,returned⟩
  intro move member
  obtain ⟨row,present,eq⟩ := List.mem_map.mp (moves.mem_iff.mp (List.mem_map_of_mem member))
  exact (congrArg Prod.snd eq).symm.trans ((truth row present).trans (congrArg (Root.squareValue mc fc b) (congrArg Prod.fst eq)))

end NeuralReversi.AspirationMir.NativePublication
