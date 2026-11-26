import NeuralReversi.PvsParallel.Interleaving.ScoreReuse

open Lean
open NeuralReversi
open NeuralReversi.Threading.AtomicWord
open NeuralReversi.PvsParallel.Interleaving.ScoreReuse
open NeuralReversi.PvsParallel.SplitSchedulerMir

structure Origin where
  generation : Nat
  helper : Nat
  deriving FromJson, BEq, Repr

structure Report where
  origin : Origin
  score : Int
  deriving FromJson, BEq

/-- A state from TLC's JSON trace. Helper IDs are the bit indices in the mask. -/
structure Snapshot where
  phase : String
  generation : Nat
  words : List Int
  reports : List Report
  origins : List Origin
  publisher : Origin
  first : Int
  last : Int
  source : Int
  lower : Int
  active : List Nat
  unreported : List Nat
  deriving FromJson

private def scoreWord (score : Int) : Write :=
  ⟨⟨0, 0⟩, (BitVec.ofInt 32 score).zeroExtend 64, false, false⟩

private def require (condition : Bool) (reason : String) : Except String Unit :=
  unless condition do throw reason

private def releaseSequence (mask : List Write) (i zero : Fin mask.length) : Bool :=
  @decide (InReleaseSequence mask i zero) (by unfold InReleaseSequence; infer_instance)

/-- Invoke the compiled worker path on one scout return with the lock held.
    This null-window call stays below alpha, so neither a re-search nor a cutoff
    intervenes. commit.best records the set_best_score branch. -/
private def bestStore (h : Nat) (score best : Int) : Except String (Option Int) := do
  let i : PvsParallel.SplitWorkerMir.Input :=
    ⟨h,13,false,false,false,1025,1024,1024,BitVec.ofInt 32 (-score),0⟩
  let some searched := PvsParallel.SplitWorkerMir.search i | throw "worker search failed"
  require (searched.chosen.toInt == score) "incorrect child score word"
  return (PvsParallel.SplitWorkerMir.commit i ⟨false,1024,BitVec.ofInt 32 best⟩ searched.chosen).best.map BitVec.toInt

private def checkPublishers (trace : List Snapshot) : Except String Unit := do
  let some initial := trace.head? | throw "missing slot initialization"
  require (initial.words == [-512] && initial.origins == [⟨1,0⟩] && initial.reports.isEmpty)
    "invalid first slot initializer"
  for (prev, cur) in trace.zip trace.tail do
    let mut words := prev.words
    let mut origins := prev.origins
    let mut reports := prev.reports
    if prev.phase == "working" && cur.phase == "working" && prev.unreported != cur.unreported then
      let [h] := prev.unreported.filter (!cur.unreported.contains ·) | throw "missing reporting worker"
      let some r := cur.reports[reports.length]? | throw "missing child report"
      let some best := words.getLast? | throw "missing score"
      require (r.origin == ⟨prev.generation,h⟩ && (r.score == 0 || r.score == 512))
        "wrong worker or unsupported child score"
      reports := reports ++ [r]
      if let some value ← bestStore h r.score best then
        words := words ++ [value]
        origins := origins ++ [r.origin]
    else if prev.phase == "zero" && (cur.phase == "working" || cur.phase == "choose") then
      let some seed := cur.words.getLast? | throw "missing new initializer"
      words := words ++ [seed]
      origins := origins ++ [⟨prev.generation+1,0⟩]
    require (cur.words == words && cur.origins == origins && cur.reports == reports)
      "publisher or store disagrees with protected worker update"

/-- Rebuild the field's modification order from the recorded reporters and seeds.
    At each reset, the split lock orders the preceding score stores before the RMW. -/
private def replay (mode : String) (trace : List Snapshot) : Except String String := do
  let current := trace.filter fun s => s.generation == 2
  let some birth := current.head? | throw "missing reused slot"
  let working := current.takeWhile (fun s => s.phase == "working")
  let some finished := working.getLast? | throw "missing active helpers"
  let [wait, choose, doneState] := current.drop working.length | throw "missing ordered zero wait, choice or read"
  require (wait.phase == "zero" && choose.phase == "choose" && doneState.phase == "done")
    "missing ordered zero wait, choice or read"
  require (finished.active.isEmpty && wait.active.isEmpty && wait.words == finished.words)
    "returning wait did not observe the current empty mask"
  checkPublishers trace
  let some seed := birth.words.getLast? | throw "missing initializer"
  let some nextSeed := doneState.words.getLast? | throw "missing later initializer"
  require (birth.active.mergeSort == [1, 2] && birth.unreported.mergeSort == [1, 2])
    "missing booked helpers"
  let clock : Clock := ⟨0, id, fun _ _ h => h⟩
  let previous := birth.words.dropLast.map scoreWord
  let mut best := seed
  let mut improvements : Array Write := #[]
  let mut resets : Array (Write × Nat) := #[]
  for (before, after) in working.zip working.tail do
    let reports := before.unreported.filter (fun h => !after.unreported.contains h)
    let finishes := before.active.filter (fun h => !after.active.contains h)
    match reports, finishes with
    | [h], [] =>
      require (before.active == after.active && before.unreported.erase h == after.unreported)
        "report changed another helper"
      if after.words.length > before.words.length then
        let some value := after.words.getLast? | throw "missing protected best store"
        best := value
        improvements := improvements.push (scoreWord value)
    | [], [h] =>
      require (!before.unreported.contains h && before.unreported == after.unreported &&
        before.active.erase h == after.active && before.words == after.words) "invalid helper reset"
      let bits := after.active.foldl (fun n i => n ||| (1 <<< i)) (0 : Nat)
      let rmw : Write := ⟨⟨h, 1⟩, BitVec.ofNat 64 bits, mode != "relaxedReset", true⟩
      resets := resets.push (rmw, previous.length + improvements.size)
    | _, _ => throw "expected one helper report or reset"
  let later : OwnerFields.Initialization :=
    ⟨0, ⟨initPhases.length, by omega⟩, BitVec.ofInt 32 nextSeed, 0, 0, [], [], []⟩
  let score : Scores := ⟨previous, BitVec.ofInt 32 seed, improvements.toList, [later]⟩
  let words := score.words clock 0 1
  let last := (score.last clock 0 1).val
  require (doneState.words == (List.finRange words.length).map (signed32At words)) "wrong store sequence"
  require (choose.words == wait.words ++ [nextSeed] && doneState.words == choose.words)
    "wrong initializer after zero wait"
  require (doneState.first == previous.length && doneState.last == last) "wrong generation indices"
  let mask := resets.toList.map Prod.fst
  require (mask.length == 2) "missing helper reset"
  let some zero := (List.finRange mask.length).getLast? | throw "missing wait source"
  require ((mask.get zero).value == 0) "wait did not see zero"
  let mut lower := previous.length
  if mode != "relaxedWait" then
    for i in List.finRange mask.length do
      if releaseSequence mask i zero then
        let some (_, published) := resets[i.val]? | throw "missing release"
        lower := max lower published
  require (wait.lower == lower && wait.last == last && wait.first == previous.length &&
    doneState.lower == wait.lower) "zero-wait bounds disagree"
  let src := doneState.source.toNat
  require (doneState.source == src && lower ≤ src && src < words.length) "incoherent score source"
  require (mode == "unorderedReuse" || src ≤ last) "read crossed the next initializer"
  let read := doneState.words[src]?
  let some physical := doneState.origins[src]? | throw "unidentified physical store"
  let publisher := doneState.publisher
  if mode != "lastEqual" && mode != "firstAcross" then
    require (publisher == physical) "reported helper did not publish this score"
  match mode with
  | "tie1" | "tie2" | "lastEqual" | "firstAcross" =>
    let [first, second] := doneState.reports.filter (·.origin.generation == 2) | throw "missing current reports"
    require (first.score == second.score && first.origin.helper != second.origin.helper &&
      src == last && physical == first.origin && seed < first.score) "missing tie after first improvement"
    if mode == "tie1" || mode == "tie2" then
      require (first.origin.helper == (if mode == "tie1" then 1 else 2)) "wrong tie order"
    else if mode == "lastEqual" then
      require (publisher == second.origin) "missing last-equal misattribution"
    else
      let some old := doneState.reports.find? (·.score == best) | throw "missing earlier report"
      require (publisher == old.origin && publisher.helper == physical.helper && publisher.generation == 1 &&
        (doneState.origins.zip doneState.words).contains (publisher,best)) "missing previous-generation publisher"
  | "step" => require (src == last && previous.length < last && best == nextSeed) "no equal-score reuse"
  | "seedOnly" => require (src == last && last == previous.length) "initializer was improved"
  | "relaxedWait" | "relaxedReset" => require (src < last && read != some best) "no stale result"
  | "unorderedReuse" => require (src == last+1 && read == some best) "no equal-score substitution"
  | _ => throw "unknown scheduling mode"
  return s!"{mode}: read store {src} ({read.getD 0}), current last store {last} ({best}); physical {repr physical}, reported {repr publisher}"

def main (args : List String) : IO Unit := do
  let [mode, path] := args | throw (IO.userError "expected mode and snapshot file")
  let input ← IO.FS.readFile path
  let json ← IO.ofExcept (Json.parse input)
  let states ← IO.ofExcept (fromJson? json : Except String (List Snapshot))
  IO.println (← IO.ofExcept (replay mode states))
