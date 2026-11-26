import NeuralReversi.PvsParallel.FrameReturns
import NeuralReversi.PvsParallel.FrameMove
import NeuralReversi.TranspositionTable.WriterMir.Events
import NeuralReversi.EndgameSearch.CacheHistory

namespace NeuralReversi.PvsParallel.FrameReturns

open TranspositionTable
open WriterMir
open CausalReplay (SearchSave PrefixWriteFrom)
open PvsBoardTree (Position build)
open ConcurrentReplay (Action UsedChild)

private abbrev priorReads (log : Log mc fc) (endian : Bool) (call : WriterTrace.Call Nat) :=
  ConcurrentReplay.ReadAt log.before log.events call.finish endian

structure SavedScore (log : Log mc fc) (call : WriterTrace.Call Nat) (f : Frame) (save : SearchSave) : Prop where
  present : f ∈ log.frames
  earlier : log.before (log.identify f.use) call.finish
  alpha : f.alpha = save.alpha.raw.toInt
  beta : f.beta = save.beta.raw.toInt
  score : f.score = save.score.raw.toInt

theorem SavedScore.determined {log : Log mc fc} (saved : SavedScore log call f save) :
    ConcurrentReplay.Determined (some f.id) (ConcurrentReplay.predecessors log.before log.events call.finish)
      (build mc fc f.position) f.pv save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt := by
  refine ⟨f.id, rfl, _, ConcurrentReplay.mem_predecessors.mpr ⟨log.scored f saved.present, saved.earlier⟩, ?_⟩
  simp only [Frame.result, Frame.position, saved.alpha, saved.beta, saved.score]

def boundarySave (i : EndgameSearch.CacheHistory.Input) (score : Score) (generation : BitVec 8)
    (pv : Bool) (observed : Save.Observation) : SearchSave :=
  ⟨⟨i.alpha.raw⟩,⟨i.beta.raw⟩,score,none,i.depth,generation,pv,observed,0⟩

inductive WriteStep (log : Log mc fc) (endian : Bool) : ConcurrentReplay.Event → Prop
  | clear (event : Nat → Nat) (current seq metadata last : SeqLock.Word) (more : Bool)
      (req : Request) (call : WriterTrace.Call Nat) :
      publishRound event endian current (clearStep endian more (decide (current = seq)) seq metadata last) =
        some (req,call) → WriteStep log endian ⟨call.finish, .cleared call⟩
  | penalty (b : Position) (frame : Option Nat) (event : Nat → Nat) (current : SeqLock.Word)
      (expected : EntryData) (snap : Option SeqLock.Snapshot) (req : Request) (call : WriterTrace.Call Nat) :
      TableReads.SlotLookupFrom (priorReads log endian call) b (some expected) →
      publishRound event endian current (penaltyStep snap.isSome b.val.player b.val.opponent expected
        (snap.getD dummy) (Packing.unpack endian (snap.getD dummy).metadata)) = some (req,call) →
      WriteStep log endian ⟨call.finish, .stored frame call b req.entry⟩
  | other (b : Position) (frame : Option Nat) (event : Nat → Nat) (incoming : EntryData)
      (move : Option Bitboard.Square) (obs : Save.Observation) (req : Request) (call : WriterTrace.Call Nat) :
      ¬Cutoffs.ExactEndgame incoming → MoveOrigin mc fc b (priorReads log endian call) move →
      incoming.bestMove = Packing.encodeBestMove move →
      observations (priorReads log endian call) endian obs (saveRound endian b.val.player b.val.opponent incoming obs) →
      publishRound event endian obs.current (saveRound endian b.val.player b.val.opponent incoming obs) = some (req,call) →
      WriteStep log endian ⟨call.finish, .stored frame call b req.entry⟩
  | searched (f : Frame) (save : SearchSave) (event : Nat → Nat) (slot : Nat) (req : Request) (call : WriterTrace.Call Nat) :
      SavedScore log call f save → Aspiration.Window f.alpha f.beta → (f.pv = false → f.beta = f.alpha+1) →
      MoveOrigin mc fc f.position (priorReads log endian call) save.move →
      observations (priorReads log endian call) endian save.observed (saveRound endian f.position.val.player
        f.position.val.opponent (save.incoming (classify f.pv save.score save.alpha save.beta)) save.observed) →
      storeRound event endian slot f.position save (classify f.pv save.score save.alpha save.beta) = some (req,call) →
      WriteStep log endian ⟨call.finish, .stored (some f.id) call f.position req.entry⟩
  | boundary {p : EndgameSearch.CacheHistory.Pool} (trace : EndgameSearch.CacheHistory.Timeline p)
      (i : EndgameSearch.CacheHistory.Input) (worker opening frame : Nat) (value : Score) (gen : BitVec 8)
      (pv : Bool) (obs : Save.Observation) (event : Nat → Nat) (slot : Nat) (req : Request) (call : WriterTrace.Call Nat) :
      i.depth.toNat = 12 → i.depth = i.empties.count → i.parallel = true → i.aborted = false →
      i.arm = .deep i.active → Aspiration.Window i.alpha.raw.toInt i.beta.raw.toInt →
      i.beta.raw.toInt = i.alpha.raw.toInt+1 →
      (i.run mc fc p.address (trace.readAt mc fc worker opening).history).score = some value.raw.toInt →
      observations (priorReads log endian call) endian obs (saveRound endian i.board.val.player i.board.val.opponent
        ((boundarySave i value gen pv obs).incoming (classify false value ⟨i.alpha.raw⟩ ⟨i.beta.raw⟩)) obs) →
      storeRound event endian slot i.board (boundarySave i value gen pv obs) (classify false value ⟨i.alpha.raw⟩ ⟨i.beta.raw⟩) = some (req,call) →
      WriteStep log endian ⟨call.finish,.stored (some frame) call i.board req.entry⟩
  | enhanced (b : Position) (frame : Option Nat) (event : Nat → Nat) (slot : Nat)
      (save : SearchSave) (query : PvsEnhancedCutoff.Query) (hit : PvsEnhancedCutoff.Hit)
      (req : Request) (call : WriterTrace.Call Nat)
      (reads : List (TableReads.ChildLookupFrom mc fc b (priorReads log endian call))) :
      query.selectivity = Cutoffs.exactSelectivity → query.endgame = true →
      PvsEnhancedCutoff.scan query (reads.map TableReads.ChildLookupFrom.observation) = some hit →
      Aspiration.Window query.alpha.raw.toInt save.beta.raw.toInt → save.beta.raw.toInt = query.alpha.raw.toInt+1 →
      save.score.raw.toInt = hit.2 → save.depth = query.depth → save.move = some hit.1 →
      storeRound event endian slot b save .lower = some (req,call) →
      WriteStep log endian ⟨call.finish, .stored frame call b req.entry⟩

theorem WriteStep.physical {log : Log mc fc} {e : ConcurrentReplay.Event} (step : WriteStep log endian e) :
    e.action.Words endian ∧ (∀ call, e.call = some call → e.id = call.finish) := by
  have marked (call : WriterTrace.Call Nat) : ∀ c, some call = some c → call.finish = c.finish := by
    intro c same; cases same; rfl
  cases step with
  | clear _ _ _ _ _ _ _ call hit => exact ⟨clear_event hit, marked call⟩
  | penalty b frame event current expected snap req call previous hit =>
    have pair := penalty_event (mc := mc) (fc := fc)
      (computed := ConcurrentReplay.Determined frame (ConcurrentReplay.predecessors log.before log.events call.finish))
      frame previous hit
    exact ⟨pair.2, marked call⟩
  | other _ frame _ _ _ _ _ call _ _ _ _ hit => exact ⟨(save_publication frame hit).2, marked call⟩
  | searched f _ _ _ _ call _ _ _ _ _ hit => exact ⟨(store_publication (some f.id) hit).2, marked call⟩
  | boundary _ _ _ _ frame _ _ _ _ _ _ _ call _ _ _ _ _ _ _ _ _ hit =>
    exact ⟨(store_publication (some frame) hit).2,marked call⟩
  | enhanced _ frame _ _ _ _ _ _ call _ _ _ _ _ _ _ _ _ hit => exact ⟨(store_publication frame hit).2, marked call⟩

theorem WriteStep.checked {log : Log mc fc} {e : ConcurrentReplay.Event} (step : WriteStep log endian e)
    (prior : TableReads.LegalHints (ConcurrentReplay.history (ConcurrentReplay.predecessors log.before log.events e.id)))
    (seen : ∀ raw, ConcurrentReplay.ReadAt log.before log.events e.id endian raw →
      raw.publication (ConcurrentReplay.history (ConcurrentReplay.predecessors log.before log.events e.id))) :
    e.action.Valid mc fc endian (UsedChild log.identify log.before log.events e.id)
      (ConcurrentReplay.ReadAt log.before log.events e.id endian) (ConcurrentReplay.predecessors log.before log.events e.id) := by
  cases step with
  | clear => trivial
  | penalty b frame event current expected snap req call previous hit =>
    exact (penalty_event frame previous hit).1
  | other b frame event incoming move obs req call outside hint encoded sampled hit =>
    have pair := regular_event
      (computed := ConcurrentReplay.Determined frame (ConcurrentReplay.predecessors log.before log.events call.finish))
      frame outside (hint.otherHint prior seen) encoded sampled rfl hit
    exact pair.1
  | searched f save event slot req call saved window null hint sampled result =>
    rw [saved.alpha, saved.beta] at window null
    exact (exact_prefix (some f.id) save f.pv window null saved.determined (hint.hint prior seen) sampled result).1
  | boundary trace i worker opening frame v gen pv obs event slot req call _ _ _ _ branch window narrow scored observed hit =>
    let p := (trace.readAt mc fc worker opening).history
    have aligned := i.aligned
    have cache := (trace.reads mc fc worker opening).1
    unfold EndgameSearch.CacheHistory.Input.run at scored
    rw [branch] at scored
    have bound := DeepLeafMir.local_bound i.target i.bmi2 i.paired mc fc _ i.board i.alpha i.beta (i.state p)
      i.active i.checked i.aborted aligned cache narrow scored
    have good : v.raw.toInt.natAbs < 32767 := by
      unfold storeRound at hit
      rw [store_input] at hit
      split at hit
      · assumption
      · contradiction
    let save := {boundarySave i v gen pv obs with sequence := SeqLock.publish call.expected}
    exact .localBoundary save window rfl (PvsMoveLoop.Bound.toSoundResult window bound (by change -32767 < v.raw.toInt; omega)
      (by change v.raw.toInt < 32767; omega)) (preserves_observed_hint observed) (store_publication (some frame) hit).1
  | enhanced b frame event slot save query hit req call reads exactTag endgame scan window narrow score depth move result =>
    have pair := store_publication frame result
    let s := {save with sequence := SeqLock.publish call.expected}
    refine PrefixWriteFrom.enhanced query reads hit s exactTag endgame scan window narrow score ?_
    simpa only [SearchSave.incoming, depth, move] using pair.1

theorem decodedOfSteps (log : Log mc fc) (irrefl : ∀ i, ¬log.before i i)
    (trans : ∀ {i j k}, log.before i j → log.before j k → log.before i k)
    (frames : ∀ f ∈ log.frames, ∃ depth a z, ExactDepth.Full f.position depth ∧
      f.depth = depth.toNat ∧ f.alpha = a.raw.toInt ∧ f.beta = z.raw.toInt ∧
      Aspiration.Window f.alpha f.beta ∧ (f.pv = false → f.beta = f.alpha + 1) ∧ ScoreStep log endian f depth a z)
    (writes : ∀ e ∈ log.writes, WriteStep log endian e) : Decoded log endian :=
  ⟨irrefl, trans, frames, fun e mem =>
    let step := writes e mem
    ⟨step.physical.1, step.physical.2, step.checked⟩⟩

end NeuralReversi.PvsParallel.FrameReturns
