module
public import NeuralReversi.PvsParallel.SerialTable.MidgameChild

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MidgameValue

open AspirationMir.SerialTable.Repeated (ReadyMemory)
open SerialCalls (Request)
open SerialCalls.ScoreWords (Finite)
open EndgameSearch.CacheHistory (Pool)
open MidgameChild (Plan eval eval_ready)
open TranspositionTable
open PvsBoardTree (Position)
open PvsMoveLoop (scoreMax)

abbrev Value := {v : Int // Finite v}

def Value.neg (x : Value) : Value := ⟨-x.val,MidgameScoreRange.finite_neg x.property⟩
def Value.max (x y : Value) : Value := ⟨Max.max x.val y.val,MidgameScoreRange.finite_max x.property y.property⟩
def ofRange {v : Int} (range : -scoreMax ≤ v ∧ v ≤ scoreMax) : Value :=
  ⟨v,by unfold Finite; unfold scoreMax at range; omega⟩

def known {q : Request} (scout : q.pv = false) (v : Value) : Plan mc fc q := .nonPvWord scout v.val

theorem eval_known (p : Pool) (q : Request) (scout : q.pv = false) (v : Value) (m : Memory l) :
    eval (mc := mc) (fc := fc) p (known scout v) m = (v.val,m) :=
  MidgameChild.eval_nonPvWord p q m scout v.val v.property

def call (r : Request) (w : Aspiration.Window r.alpha r.beta) (sub : Plan mc fc r)
    (next : Value → Plan mc fc q) : Plan mc fc q :=
  .before r w sub fun v =>
    if h : -32767 < v ∧ v < 32767 then next ⟨v,h⟩ else .stopped

theorem eval_call (p : Pool) (r : Request) (w : Aspiration.Window r.alpha r.beta) (sub : Plan mc fc r)
    (next : Value → Plan mc fc q) (m : Memory l) (ready : ReadyMemory p mc fc m) :
    eval p (call r w sub next) m =
      eval p (next ⟨(eval p sub m).1,(eval_ready p sub m w ready).2.1⟩) (eval p sub m).2 := by
  have finite := (eval_ready p sub m w ready).2.1
  change (let pair := eval p sub m
          eval p (if h : -32767 < pair.1 ∧ pair.1 < 32767 then next ⟨pair.1,h⟩ else .stopped) pair.2) = _
  dsimp only
  split <;> first | rfl | contradiction

abbrev Sample := {e : EntryData // Finite e.score.toInt}

def Sample.word (e : Sample) : Value := ⟨e.val.score.toInt,e.property⟩

def checked (entry : Option EntryData) : Option Sample := do
  let e ← entry
  if h : -32767 < e.score.toInt ∧ e.score.toInt < 32767 then some ⟨e,h⟩ else none

def probe (b : Position) (next : Fin 2 → Option Sample → Plan mc fc q) : Plan mc fc q :=
  .observed b fun i e => next i (checked e)

def lookup (b : Position) (next : Option Sample → Plan mc fc q) : Plan mc fc q :=
  .lookupAt b fun e => next (checked e)

theorem probe_hit (m : Memory l) (ready : ReadyMemory p mc fc m) (b : Position)
    (read : m.tt.image.probe b = .hit i e) :
    ∃ seen : Sample, checked (some e) = some seen ∧ seen.val = e := by
  have reads := m.tt.image.probe_reads m.supported b
  obtain ⟨attempt,tried,observed,occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ read
  have h := ready.1.2 b e (reads.published.matches i e attempt tried observed occupied)
  exact ⟨⟨e,h⟩,by dsimp only [checked,bind,Option.bind]; split <;> first | rfl | contradiction,rfl⟩

theorem lookup_hit (m : Memory l) (ready : ReadyMemory p mc fc m) (b : Position)
    (read : m.tt.image.lookup b = some e) :
    ∃ seen : Sample, checked (some e) = some seen ∧ seen.val = e := by
  obtain ⟨origin⟩ := m.tt.image.lookup_source m.supported b read
  have h := ready.1.2 b e origin.source
  exact ⟨⟨e,h⟩,by dsimp only [checked,bind,Option.bind]; split <;> first | rfl | contradiction,rfl⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameValue
