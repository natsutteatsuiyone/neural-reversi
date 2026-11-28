module
public import NeuralReversi.PvsParallel.Interleaving.AlphaOrigin
public import NeuralReversi.PvsParallel.Interleaving.Drive

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.SchedulerReplay

open Threading SplitWorkerMir
open AtomicWord (Event)
open SplitCompletion (Action)
open MoveList.ConcurrentTrace (Write)
open Tickets (State)

/-- A merge of the physical mask RMWs, iterator operations and acquired lock
    sections for one slot generation. Exhaustion keeps the load and failed-CAS
    sources that produced the observed counter word. -/
inductive Instruction (w : Nat) where
  | mask (at_ : Event)
  | claim (cas : Write w)
  | exhausted (worker load : Nat) (failures : List Nat)
  | lock (acquire : Event)

def maskOf {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) :=
  SlotFrameMir.RawProjection.actions b.generation

abbrev window {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) :=
  b.owner.exit.clock.atStep (3+b.owner.exit.log.length)

abbrev slotOf {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) :=
  b.flagAddress (b.generation.owner,b.bankIndex)

/-- Only a section whose physical stop test returned true needs the corresponding
    stop-source evidence. The worker's release is found in the mask modification log. -/
abbrev Stops {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t) :=
  (rank : Nat) → (posted : List Tickets.Post) → (poll : Event) →
    Option (StopSchedule.Evidence b.memory b.owner.check (window b) (slotOf b) posted poll)

def release (complete : List Action) (poll : Event) : Option (StopSchedule.Release complete poll) :=
  (List.finRange complete.length).findSome? fun i =>
    match h : complete.get i with
    | .book .. => none
    | .finish w e =>
      if before : w.val = poll.worker ∧ poll.worker = e.worker ∧ poll.step < e.step then
        some ⟨i,w,e,h,before.1,before.2⟩
      else none

abbrev seq {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t)
    (root cut : Bool) (depth : Nat) (ms : List Action) (locks : List FieldStores.Raw) (s : State t.width) :=
  ScoreOrigin.RunWith t b.memory b.owner.check (window b) (slotOf b) (AlphaOrigin.words b.registers.reports)
    (ScoreOrigin.task t root cut depth) (fun _ _ _ _ _ => True) b.generation.owner (maskOf b)
    (AlphaOrigin.source b.registers.reports) ms locks s

/-- Lists remaining at the end must all be empty. Consequently an accepted tape includes
    every effective booking/reset, successful iterator CAS and protected report or discard. -/
def walk {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t)
    (root cut : Bool) (depth : Nat) (stop : Stops b) (s : State t.width)
    (ms : List Action) (locks : List FieldStores.Raw)
    (bits : List Action) (issued : List (Write t.width)) (packets : List FieldStores.Raw)
    (old : seq b root cut depth ms locks s) : (instructions : List (Instruction t.width)) →
    Option {state : State t.width // seq b root cut depth (ms++bits) (locks++packets) state}
  | [] => by
    cases bits with
    | cons => exact none
    | nil =>
      cases issued with
      | cons => exact none
      | nil =>
        cases packets with
        | cons => exact none
        | nil => exact some ⟨s,by simpa using old⟩
  | inst :: rest => by
    cases inst with
    | mask stamp =>
      cases bits with
      | nil => exact none
      | cons head tail =>
        if stamp ≠ head.event then exact none else
        cases head with
        | book w e =>
          exact do
            match h : Drive.step rc.length s (.book w.val) with
            | none => none
            | some n =>
              let ans ← walk b root cut depth stop n _ locks tail issued packets (.book w e old (Drive.step_sound h)) rest
              return ⟨ans.val,by simpa using ans.property⟩
        | finish w e =>
          exact do
            match h : Drive.step rc.length s (.leave w.val) with
            | none => none
            | some n =>
              let ans ← walk b root cut depth stop n _ locks tail issued packets (.depart w e old (Drive.step_sound h)) rest
              return ⟨ans.val,by simpa using ans.property⟩
    | claim cas =>
      cases issued with
      | nil => exact none
      | cons head tail =>
        if head ≠ cas then exact none else
        if actor : ScoreOrigin.poll t s (.take cas.worker) = true then
          exact match h : Drive.step rc.length s (.take cas.worker cas.old.toNat) with
            | none => none
            | some n => walk b root cut depth stop n ms locks bits tail packets (.poll old (Drive.step_sound h) actor) rest
        else exact none
    | exhausted worker load failures =>
      exact do
        let seen ← MoveList.ConcurrentTrace.observed rc.length
          (MoveList.ConcurrentTrace.words t.start t.state.writes) load failures
        guard (seen = s.current)
        match h : Drive.step rc.length s (.empty worker) with
        | none => none
        | some n => walk b root cut depth stop n ms locks bits issued packets (.poll old (Drive.step_sound h) rfl) rest
    | lock acq =>
      cases packets with
      | nil => exact none
      | cons p tail =>
        exact do
          let rank := locks.length
          let obs ← t.journal[rank]?
          guard (obs.acquired = acq ∧ acq.worker = obs.worker ∧ obs.polled.worker = obs.worker)
          if stopped : p.stopped = true then
            if actor : ScoreOrigin.actor t p.input.index = some obs.polled.worker then
              let samples ← stop rank s.reports obs.polled
              let reset ← release (maskOf b) obs.polled
              let ⟨(l,r),split⟩ ← Drive.remove (obs.polled.worker,p.input.index) s.pending
              let newer := StopSchedule.discard s l r obs.polled.worker p.input.index
              let ans ← walk b root cut depth stop newer ms _ bits issued tail
                (.stop p obs.polled old actor stopped split samples reset) rest
              return ⟨ans.val,by simpa using ans.property⟩
            else none
          else
            if found : (AlphaOrigin.next t.reports s.reports).isSome then
              let src := AlphaOrigin.source (root := root) (cut := cut) (depth := depth) b.registers.reports
              let row : src.Observation s.reports := ⟨found⟩
              let commit := src.commit s.reports row
              if actor : ScoreOrigin.actor t commit.index = some acq.worker then
                if args : ScoreOrigin.operands p.input = ScoreOrigin.operands commit.input then
                  match h : Drive.step rc.length s (.report acq.worker commit.post) with
                  | none => none
                  | some n =>
                    let ans ← walk b root cut depth stop n ms _ bits issued tail
                      (.report p row old actor args (Bool.eq_false_iff.mpr stopped)
                        ⟨True.intro,fun _ => True.intro⟩ (Drive.step_sound h)) rest
                    return ⟨ans.val,by simpa using ans.property⟩
                else none
              else none
            else none
termination_by structural instructions => instructions

/-- The decoder supplies the scored schedule and the final scheduler state together.
    Child-return obligations for live reports can subsequently be attached by the
    ordinary root/interior return-register checker. -/
def capture {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.Operands t)
    (root cut : Bool) (depth : Nat) (stop : Stops b) (instructions : List (Instruction t.width)) :
    Option (AlphaOrigin.Capture t root cut depth (fun _ _ _ _ _ => True)) := do
  let state ← walk b root cut depth stop (Tickets.initial t.start b.generation.owner.val) [] []
    (maskOf b) t.state.writes t.calls .initial instructions
  return { toOperands := b, state := state.val, run := state.property }

end NeuralReversi.PvsParallel.Interleaving.SchedulerReplay
