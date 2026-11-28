module
public import NeuralReversi.TranspositionTable.WriterMir.Origins
public import NeuralReversi.TranspositionTable.WriterMir
public import NeuralReversi.TranspositionTable.Penalty

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word)
open PvsParallel

def requested : List SourceOp → Option Request
  | [] => none
  | .claim request :: _ => some request
  | _ :: tail => requested tail

theorem forward_exact {entry : EntryData} (bound : entry.bound = Packing.boundByte .exact)
    (more found cas p o saved old) :
    saveStep more found cas p o entry saved old = ⟨[.exact p o entry], false⟩ := by
  simp [saveStep, bound]

theorem exact_request (more cas : Bool) (seq p o : Word) (entry : EntryData) :
    requested (exactStep more cas seq p o entry).ops =
      if more ∧ seq &&& 1 = 0 then some ⟨seq, p, o, entry⟩ else none := by
  cases more <;> cases cas <;> by_cases stable : seq &&& 1#64 = 0#64 <;>
    simp [exactStep, requested, stable]

theorem nonexact_request {entry : EntryData} (bound : entry.bound ≠ Packing.boundByte .exact)
    (cas p o) (saved : SeqLock.Snapshot) (old : EntryData) :
    requested (saveStep true true cas p o entry saved old).ops =
      (Save.proposal (p,o) (saved.player,saved.opponent) entry old).map
        (fun d => ⟨saved.sequence, p, o, d⟩) := by
  unfold saveStep Save.proposal
  dsimp only
  cases cas <;> cases same : sameBoardBits p o saved.player saved.opponent <;>
    by_cases empty : old.bound = 0#8 <;> simp_all [requested]
  all_goals
    by_cases a : old.depth.toNat ≤ (entry.depth.setWidth 32 + 2).toNat <;>
      by_cases b : selectivityRank old.selectivity < entry.selectivity.toNat <;>
      by_cases c : old.depth.toNat ≤ (entry.depth.setWidth 32 + 4).toNat <;>
      by_cases g : entry.generation = old.generation <;>
      by_cases m : entry.bestMove = Packing.noMove <;> simp_all [requested]
  all_goals (repeat' split) <;> first | rfl | (exfalso; omega)


def dummy : SeqLock.Snapshot := ⟨0,0,0,0⟩

def saveRound (endian : Bool) (p o : Word) (d : EntryData) (obs : Save.Observation) : SourceStep :=
  let snap := Snapshot.read endian obs.first obs.last obs.sample
  let s := snap.getD dummy
  let step := saveStep true snap.isSome (decide (obs.current = s.sequence)) p o d s
      (Packing.unpack endian s.metadata)
  match step.ops with
  | [.exact player opponent entry] =>
      exactStep true (decide (obs.current = obs.first)) obs.first player opponent entry
  | _ => step

def outcome (current : Word) (step : SourceStep) : Save.Outcome :=
  match requested step.ops with
  | some req => Save.claimWrite current req.expected req.entry
  | none => if step.retry then .retry else .rejected

theorem ordinary_stop {entry : EntryData} (bound : entry.bound ≠ Packing.boundByte .exact)
    (cas p o saved old) :
    ¬(∃ x y d, (saveStep true true cas p o entry saved old).ops = [.exact x y d]) ∧
    (requested (saveStep true true cas p o entry saved old).ops = none →
      (saveStep true true cas p o entry saved old).retry = false) := by
  simp only [saveStep, bound, ↓reduceIte]
  repeat' first | split | constructor
  all_goals simp [requested]

theorem expand_eq {s : SourceStep}
    (ordinary : ¬∃ p o d, s.ops = [.exact p o d]) :
    (match s.ops with
      | [.exact p o d] => exactStep true cas seq p o d
      | _ => s) = s := by
  split
  · exact False.elim (ordinary ⟨_,_,_,by assumption⟩)
  · rfl

theorem round_agrees (endian p o entry obs) :
    outcome obs.current (saveRound endian p o entry obs) = Save.attempt endian (p,o) entry obs := by
  by_cases bound : entry.bound = Packing.boundByte .exact
  · simp only [saveRound, forward_exact bound, Save.attempt, bound, ↓reduceIte]
    by_cases stable : obs.first &&& 1#64 = 0#64
    · simp [outcome, exact_request, stable]
    · simp [outcome, exactStep, requested, stable]
  · cases sample : Snapshot.read endian obs.first obs.last obs.sample with
    | none => simp [saveRound, Save.attempt, sample, bound, saveStep, outcome, requested]
    | some s =>
      let old := Packing.unpack endian s.metadata
      let cas := decide (obs.current = s.sequence)
      have normal := ordinary_stop bound cas p o s old
      simp only [saveRound, sample, Option.getD_some, Option.isSome_some, Save.attempt, bound, ↓reduceIte]
      rw [expand_eq normal.1]
      unfold outcome
      rw [nonexact_request bound]
      cases prop : Save.proposal (p,o) (s.player,s.opponent) entry old with
      | none =>
        have gone := nonexact_request bound cas p o s old
        rw [prop] at gone
        simp only [Option.map_none] at gone
        change (if (saveStep true true cas p o entry s old).retry then _ else _) = _
        rw [normal.2 gone]
        simp [old, prop] at *
      | some write => change _ = _; simp [old, prop] at *

theorem request_key {endian p o entry obs req}
    (issued : requested (saveRound endian p o entry obs).ops = some req) :
    req.player = p ∧ req.opponent = o := by
  by_cases bound : entry.bound = Packing.boundByte .exact
  · simp only [saveRound, forward_exact bound, exact_request, true_and] at issued
    split at issued <;> cases issued
    exact ⟨rfl,rfl⟩
  · cases sample : Snapshot.read endian obs.first obs.last obs.sample with
    | none => simp [saveRound, sample, bound, saveStep, requested] at issued
    | some s =>
      simp only [saveRound, sample, Option.getD_some, Option.isSome_some] at issued
      rw [expand_eq (ordinary_stop bound ..).1, nonexact_request bound] at issued
      obtain ⟨d, _, rfl⟩ := Option.map_eq_some_iff.mp issued
      exact ⟨rfl,rfl⟩

def publishRound (event : Nat → Nat) (endian : Bool) (current : Word) (s : SourceStep) :
    Option (Request × WriterTrace.Call Nat) := do
  let req ← requested s.ops
  if even : req.expected &&& 1 = 0 then
    let trace := writeStable endian ⟨_, even⟩ req.player req.opponent req.entry
      (decide (current = req.expected))
    return (req, ← publication event trace)
  else none

theorem save_publication {board : PvsBoardTree.Position} (frame : Option Nat)
    {event endian entry obs req call}
    (hit : publishRound event endian obs.current
      (saveRound endian board.val.player board.val.opponent entry obs) = some (req, call)) :
    Save.attempt endian (Cutoffs.key board) entry obs =
        .written (SeqLock.publish call.expected) req.entry ∧
      ConcurrentReplay.Action.Words endian (.stored frame call board req.entry) := by
  generalize issued : requested (saveRound endian board.val.player board.val.opponent entry obs).ops =
    candidate
  unfold publishRound at hit
  rw [issued] at hit
  cases candidate with
  | none => contradiction
  | some candidate =>
    have key := request_key issued
    change (if even : candidate.expected &&& 1 = 0 then (publication event (writeStable endian ⟨_,even⟩ candidate.player candidate.opponent candidate.entry (decide (obs.current = candidate.expected)))).bind (fun c => some (candidate,c)) else none) = some (req,call) at hit
    split at hit <;> try contradiction
    rename_i even
    by_cases fresh : obs.current = candidate.expected
    · simp [fresh, published] at hit
      obtain ⟨rfl,rfl⟩ := hit
      constructor
      · change Save.attempt endian (board.val.player,board.val.opponent) entry obs = _
        rw [← round_agrees, outcome, issued]
        exact (Save.claimWrite_iff ..).2 ⟨fresh,rfl,rfl⟩
      · simp [ConcurrentReplay.Action.Words, SlotJournal.packed, key]
    · simp [fresh, writeStable, publication] at hit

theorem clear_request (endian more cas : Bool) (seq metadata last : Word) :
    requested (clearStep endian more cas seq metadata last).ops =
      if more ∧ seq &&& 1 = 0 ∧ metadata ≠ 0 then some ⟨seq, 0, 0, Packing.unpack endian 0⟩ else none := by
  cases more <;> cases cas <;> by_cases stable : seq &&& 1#64 = 0#64 <;>
    by_cases empty : metadata = 0#64 <;> by_cases equal : last = seq <;>
    simp [clearStep, requested, stable, empty, equal]

theorem penalty_agrees (endian : Bool) (p o current : Word) (expected : EntryData)
    (observation : Option SeqLock.Snapshot) :
    (do let req ← requested (penaltyStep observation.isSome p o expected (observation.getD dummy)
            (Packing.unpack endian (observation.getD dummy).metadata)).ops
        if current = req.expected then some req.entry else none) =
      Penalty.tryPenalize endian p o expected observation current := by
  cases observation with
  | none => by_cases empty : expected.depth = 0#8 <;>
      simp [penaltyStep, requested, Penalty.tryPenalize, empty]
  | some s =>
    by_cases empty : expected.depth = 0#8 <;>
      by_cases equal : Packing.unpack endian s.metadata = expected <;>
      by_cases board : p = s.player ∧ o = s.opponent <;>
      simp [penaltyStep, requested, Penalty.tryPenalize, empty, equal, sameBoardBits_exact,
        SeqLock.claim_success_iff, board, eq_comm (a := s.player), eq_comm (a := s.opponent)]

end NeuralReversi.TranspositionTable.WriterMir




