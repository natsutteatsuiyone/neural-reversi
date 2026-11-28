module
public import NeuralReversi.AspirationMir.SerialTable.Repeated

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.OnePlySeed

open PvsParallel
open SerialCalls (Program Request)
open EndgameSearch.CacheHistory (Pool)
open PvsParallel.SerialTable (Memory Layout)
open PvsParallel.SerialTable.Writes (Proposal)
open PvsBoardTree (Position)
open PvsMoveLoop (scoreMax)
open RootMoveStorage (Row)
open TranspositionTable (Selectivity)
open Repeated (ReadyMemory)

/-- A fixed midgame iteration at depth one evaluates both narrow and wide children at depth zero.
    Its root may write an exact-bound score without the endgame flag. -/
structure Inputs where
  raw : Request → Int
  order : SerialRecursion.RootInput
  scratch : MoveList.Initialization.Memory Move.Move
  level : Selectivity

def save (m : Memory l) (slot : l.Index) (level : Selectivity) : Option Proposal → Memory l
  | none => m
  | some w =>
    ⟨m.tt.store slot w.board (SerialTable.Writes.word w.score) w.bound w.depth w.move level false w.ttPv,
      m.tt.store_supported m.supported ..,m.cache⟩

def reply (ops : Inputs) (q : Request) : Int :=
  (InitialEstimate.static (fun _ => ops.raw q) q.board).val

def run (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (paired : Bool)
    (ops : Inputs) (before : Memory l) := do
  let m := Repeated.increment before
  let rows ← NativeRootBirth.started mc fc b paired ops.scratch
  let code := SerialTable.Root.frame mc fc b 1 paired ops.scratch rows 0 ops.order (-32767) 32767
  let out ← code.eval (fun q => some (reply ops q))
  return (out.answer.value,save m (m.tt.image.slot b) ops.level out.proposal)

private theorem eval_some {code : Program Req α} (ready : Program.Ready allowed code) (f : Req → Int) :
    ∃ v, code.eval (fun q => some (f q)) = some v := by
  induction ready with
  | pure => exact ⟨_,rfl⟩
  | @call q _ _ _ ih => exact ih (f q)

private theorem save_ready (m : Memory l) (ok : ReadyMemory p mc fc m) (slot : l.Index) (level : Selectivity)
    {proposal : Option Proposal} (witness : ∀ w, proposal = some w → TableReads.LegalChoice w.board w.move ∧
      -scoreMax ≤ w.score ∧ w.score ≤ scoreMax) : ReadyMemory p mc fc (save m slot level proposal) := by
  cases proposal with
  | none => exact ok
  | some w =>
    obtain ⟨legal,range⟩ := witness w rfl
    have signed : (SerialTable.Writes.word w.score).raw.toInt = w.score := BitVec.toInt_ofInt_eq_self (by decide)
      (by unfold scoreMax at range; omega) (by unfold scoreMax at range; omega)
    exact ⟨⟨⟨m.tt.store_other ok.1.1.1 _ _ _ _ _ _ _ false _ (Or.inl rfl) legal,ok.1.1.2⟩,
      m.tt.store_finite m.supported ok.1.2 _ _ _ _ _ _ _ _ _⟩,m.tt.store_exactScores m.supported ok.2 _ _ _ _ _ _ _ _ _
        (fun _ => by rw [signed]; exact range)⟩

theorem prepares (p : Pool) (mc fc) (b : Position) (paired : Bool) (ops : Inputs) (before : Memory l)
    (ready : ReadyMemory p mc fc before) (checked : Std.Tactic.BVDecide.LRAT.check cert (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc b.val = true) :
    ∃ score after, run mc fc b paired ops before = some (score,after) ∧ ReadyMemory p mc fc after ∧
      -scoreMax ≤ score ∧ score ≤ scoreMax := by
  let m := Repeated.increment before
  have initial := started_of_certificate mc fc b paired ops.scratch checked mobile
  obtain ⟨cover,nonempty⟩ := NativeRootBirth.covers initial
  have positive := List.length_pos_iff.mpr nonempty
  let rows := NativeRootBirth.initial mc b
  let code := SerialTable.Root.frame mc fc b 1 paired ops.scratch rows 0 ops.order (-32767) 32767
  have window : Aspiration.Window (-32767#32).toInt (32767#32).toInt := ⟨by decide,by decide,by decide⟩
  have live := SerialRootCalls.frame_ready_at_depth (fc := fc) (key := ops.order.key) (sort := ops.order.sort)
    (pv := ops.order.pvLine) (depth := 1) (paired := paired) (mem := ops.scratch) cover positive window
  rw [← SerialTable.Root.erase mc fc b 1 paired ops.scratch rows 0 ops.order] at live
  obtain ⟨out,returned⟩ := eval_some (code := code) (SerialTable.Stack.ready_of_map live) (reply ops)
  obtain ⟨cs,issued,words⟩ := Program.runs_of_eval code _ returned
  have ranges (t) (belongs : t ∈ cs) : -scoreMax ≤ t.2 ∧ t.2 ≤ scoreMax := by
    rw [← Option.some.inj (words t belongs)]
    exact (InitialEstimate.static _ _).property
  have finite := (SerialTable.WeakRoot.result issued cover positive (fun t ht => by
    have r := ranges t ht; unfold scoreMax at r; unfold SerialCalls.ScoreWords.Finite; omega)).1
  have bounded := SerialTable.WeakTrial.root_range issued window (fun t ht _ _ _ => ranges t ht)
  have range := bounded finite.1 finite.2
  refine ⟨out.answer.value,save m (m.tt.image.slot b) ops.level out.proposal,?_,?_,range⟩
  · simp only [run,initial,bind,Option.bind]; rw [returned]; rfl
  · apply save_ready m (Repeated.begin_valid ready)
    intro w found
    obtain ⟨_,board,_,score,_,_,legal⟩ := SerialTable.Root.writes issued found
    rw [board,score]
    exact ⟨legal,range⟩

def Contract : Prop :=
  ∀ (l : Layout) (p : Pool) mc fc (c : SerialScout.Config) (first : Position) (ops : Inputs)
    (qs : List (Repeated.Call mc)) certificate, c.address = p.address →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc first.val = true → (∀ q ∈ qs, Board.hasLegalMoves mc q.board.val = true) →
    ∃ score warm results next, run mc fc first c.paired ops (Memory.empty l 0 p) = some (score,warm) ∧
      Repeated.runAll p c fc qs warm = some (results,next) ∧
      results.map PublicationMir.ExactScore.scoreIfExact = qs.map (fun q => q.expected fc)

theorem after_midgame : Contract := by
  intro l p mc fc c first ops qs cert addr checked initial mobile
  obtain ⟨v,warm,mid,ready,_⟩ := prepares p mc fc first c.paired ops (Memory.empty l 0 p)
    ⟨Memory.cold_valid ..,PvsParallel.SerialTable.exactScores_empty⟩ checked initial
  obtain ⟨results,post,ends,_,scores⟩ := Repeated.all_returns p c fc qs warm addr checked mobile ready
  exact ⟨v,warm,results,post,mid,ends,scores⟩

end NeuralReversi.AspirationMir.SerialTable.OnePlySeed
