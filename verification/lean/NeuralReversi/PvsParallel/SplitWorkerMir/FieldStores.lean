module
public import NeuralReversi.PvsParallel.SplitWorkerMir.Loop

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

open Threading
open AtomicWord (Write)
open SplitResult (Report)
open SplitWindow (AlphaStores BestStores MoveStores raisesAlpha choosesMove scan)

structure Fields where
  alpha : Word
  best : Word
  picked : Option (Fin 64)

def Fields.state (s : Fields) : SplitWindow.State := ⟨s.alpha.toInt,s.best.toInt⟩

structure Writes where
  alpha : Option Write
  best : Option Write
  move : Option Write

def low32 (write : Write) := write.value.setWidth 32
def low8 (write : Write) := write.value.setWidth 8

structure Record where
  input : Input
  read : Locked
  body : Searched
  returned : search input = some body
  writes : Writes

def Record.update (r : Record) := commit r.input r.read r.body.chosen

def Record.report (r : Record) : Option Report :=
  if r.read.stopped then none else some (selected r.input r.body)

def Fields.after (s : Fields) (r : Record) (atIndex : Nat → Option (Fin 64)) : Fields :=
  ⟨r.update.alpha.getD s.alpha,r.update.best.getD s.best,
    if r.update.moveSet then atIndex r.input.index else s.picked⟩

structure Observed (s : Fields) (r : Record) (pv : Bool) (beta : Int)
    (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) : Prop where
  tag : r.input.pv = pv
  upper : r.input.beta.toInt = beta
  alphaRead : r.read.alpha = s.alpha
  bestRead : r.read.best = s.best
  alphaWrite : r.writes.alpha.map low32 = r.update.alpha
  bestWrite : r.writes.best.map low32 = r.update.best
  moveWrite : r.writes.move.map low8 =
    if r.update.moveSet then some (encode (atIndex r.input.index)) else none

private theorem commit_fields (r : Record) (normal : r.read.stopped = false) :
    let s : SplitWindow.State := ⟨r.read.alpha.toInt,r.read.best.toInt⟩
    let score := r.body.chosen.toInt
    r.update.alpha = (if raisesAlpha r.input.pv r.input.beta.toInt s score then some r.body.chosen else none) ∧
    r.update.best = (if s.best < score then some r.body.chosen else none) ∧
    r.update.moveSet = decide (choosesMove s score) := by
  unfold Record.update
  rw [commit_eq]
  cases pv : r.input.pv <;> by_cases a : r.read.alpha.toInt < r.body.chosen.toInt <;>
    by_cases b : r.read.best.toInt < r.body.chosen.toInt <;> by_cases z : r.body.chosen.toInt < r.input.beta.toInt <;>
    simp [normal,a,b,z,raisesAlpha,choosesMove]

structure Log where
  fields : Fields
  reports : List Report
  alpha : List Write
  best : List Write
  moves : List Write

def Log.append (l : Log) (r : Record) (atIndex : Nat → Option (Fin 64)) : Log :=
  ⟨l.fields.after r atIndex,l.reports ++ r.report.toList,l.alpha ++ r.writes.alpha.toList,
    l.best ++ r.writes.best.toList,l.moves ++ r.writes.move.toList⟩

structure Invariant (pv : Bool) (beta : Int) (initial : Fields)
    (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) (l : Log) : Prop where
  state : l.fields.state = scan pv beta initial.state (l.reports.map Report.score)
  best : l.fields.best.toInt = (l.reports.map Report.score).foldl max initial.best.toInt
  alphaWrites : AlphaStores pv beta initial.state (l.reports.map Report.score) l.alpha
  bestWrites : BestStores initial.best.toInt (l.reports.map Report.score) l.best
  moveWrites : MoveStores encode atIndex pv beta initial.state initial.picked l.reports l.fields.picked l.moves

def initialLog (initial : Fields) (a b m : Write) : Log := ⟨initial,[],[a],[b],[m]⟩

theorem seed (initial : Fields) (a b m : Write) (alpha : low32 a = initial.alpha)
    (best : low32 b = initial.best) (move : low8 m = encode initial.picked) :
    Invariant pv beta initial encode atIndex (initialLog initial a b m) := by
  exact ⟨rfl,rfl,.initialized a (by simpa [Fields.state,low32] using alpha),
    .initialized b (congrArg BitVec.toInt best),.initialized m move⟩

private theorem written {f : α → β} {x : Option α} {v : β} (ok : x.map f = some v) :
    ∃ raw, x = some raw ∧ f raw = v := Option.map_eq_some_iff.mp ok

theorem append (h : Invariant pv beta initial encode atIndex l)
    (o : Observed l.fields r pv beta encode atIndex) :
    Invariant pv beta initial encode atIndex (l.append r atIndex) := by
  have ha := o.alphaWrite
  have hb := o.bestWrite
  have hm := o.moveWrite
  cases stopped : r.read.stopped with
  | true =>
    have no : r.update = ⟨.stopped,none,none,false,none,false⟩ := by
      simp [Record.update,commit_eq,stopped]
    simp [no] at ha hb hm
    simpa [Log.append,Record.report,stopped,Fields.after,no,ha,hb,hm] using h
  | false =>
    let report := selected r.input r.body
    let scores := l.reports.map Report.score
    let prior := scan pv beta initial.state scores
    have locked : (⟨r.read.alpha.toInt,r.read.best.toInt⟩ : SplitWindow.State) = prior := by
      simpa [Fields.state,o.alphaRead,o.bestRead,prior,scores] using h.state
    obtain ⟨alpha,best,pick⟩ := commit_fields r stopped
    rw [locked,o.tag,o.upper] at alpha
    rw [locked] at best pick
    rw [alpha] at ha
    rw [best] at hb
    rw [pick] at hm
    have reportEq : r.report = some report := by simp [Record.report,stopped,report]
    have state : (l.fields.after r atIndex).state = SplitWindow.publish pv beta prior report.score := by
      have step := publish_eq r.input r.read r.body.chosen stopped
      rw [locked,o.tag,o.upper] at step
      simpa [Fields.after,Fields.state,Record.update,lockedState,o.alphaRead,o.bestRead,report,selected] using step
    have priorBest : prior.best = scores.foldl max initial.best.toInt :=
      (congrArg SplitWindow.State.best h.state).symm.trans h.best
    constructor
    · simpa [Log.append,reportEq,SplitWindow.scan,prior,scores] using state
    · have eq := congrArg SplitWindow.State.best state
      simpa [Log.append,reportEq,SplitWindow.publish,Fields.state,priorBest,scores] using eq
    · change AlphaStores _ _ _ ((l.reports ++ r.report.toList).map Report.score) (l.alpha ++ r.writes.alpha.toList)
      rw [reportEq]
      by_cases raises : raisesAlpha pv beta prior r.body.chosen.toInt
      · rw [ite_eq_left raises] at ha
        obtain ⟨raw,store,value⟩ := written ha
        simpa [store,report,selected] using
          AlphaStores.raised r.body.chosen.toInt raw h.alphaWrites raises (by simpa [low32] using value)
      · rw [ite_eq_right raises] at ha
        have same : r.writes.alpha = none := by simpa using ha
        simpa [same,report,selected] using AlphaStores.unchanged _ h.alphaWrites raises
    · change BestStores _ ((l.reports ++ r.report.toList).map Report.score) (l.best ++ r.writes.best.toList)
      rw [reportEq]
      by_cases gains : prior.best < r.body.chosen.toInt
      · rw [ite_eq_left gains] at hb
        obtain ⟨raw,store,value⟩ := written hb
        simpa [store,report,selected] using
          BestStores.improved _ raw h.bestWrites (priorBest ▸ gains) (congrArg BitVec.toInt value)
      · rw [ite_eq_right gains] at hb
        have same : r.writes.best = none := by simpa using hb
        simpa [same,report,selected] using BestStores.unchanged _ h.bestWrites (Int.le_of_not_gt (priorBest ▸ gains))
    · change MoveStores _ _ _ _ _ _ (l.reports ++ r.report.toList) (l.fields.after r atIndex).picked
        (l.moves ++ r.writes.move.toList)
      rw [reportEq]
      by_cases chooses : choosesMove prior r.body.chosen.toInt
      · simp [chooses] at pick hm
        obtain ⟨raw,store,value⟩ := hm
        simpa [store,Fields.after,pick,report,selected] using MoveStores.improved report raw h.moveWrites chooses value
      · simp [chooses] at pick hm
        simpa [hm,Fields.after,pick,report,selected] using MoveStores.unchanged report h.moveWrites chooses

end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores
