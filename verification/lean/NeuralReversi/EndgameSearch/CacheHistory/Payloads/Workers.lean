module
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open NativeTrace (Event Raw workerEvents)

theorem requests_append (xs ys : List (Event S)) (worker first last : Nat) :
    requests (xs ++ ys) worker first last = requests xs worker first last ++ requests ys worker first last := by
  simp [requests,List.filterMap_append]

def localRequests (raw : List Raw) (worker first last : Nat) :=
  requests (S:=Unit) (raw.map (Event.worker worker)) worker first last

private theorem one_foreign (raw : Raw) (different : n ≠ worker) :
    requests (S:=S) [.worker n raw] worker first last = [] := by
  cases raw with
  | enter => rfl
  | leave _ _ node _ =>
    cases found : target node <;> simp [requests,site?,found,Interval.contains,Ne.symm different]

theorem requests_local (stream : List (Event S)) (worker first last : Nat) :
    requests stream worker first last = localRequests (workerEvents worker stream) worker first last := by
  induction stream with
  | nil => rfl
  | cons ev rest ih =>
    change requests ([ev] ++ rest) _ _ _ = _
    rw [requests_append,ih]
    cases ev with
    | shared => exact List.nil_append _
    | worker n raw =>
      by_cases same : n = worker
      · subst n
        rw [show workerEvents worker (.worker worker raw :: rest) = raw :: workerEvents worker rest by simp [workerEvents]]
        change _ = requests (S:=Unit) ([Event.worker worker raw] ++ _) worker first last
        rw [requests_append]
        cases raw <;> rfl
      · simp [workerEvents,same,one_foreign raw same,localRequests]

theorem collect_interleaved (values : Values) (left : List (Event S)) (right : List (Event T)) (worker first last : Nat)
    (same : workerEvents worker left = workerEvents worker right) :
    collect values left worker first last = collect values right worker first last := by
  simp only [collect,requests_local,same]

theorem completeVisit_interleaved (p : Pool) (timeline : Timeline p) (values : Values)
    (left : List (Event S)) (right : List (Event T)) (v : Visit p)
    (same : workerEvents v.worker left = workerEvents v.worker right) :
    completeVisit p timeline mc fc values left v = completeVisit p timeline mc fc values right v := by
  unfold completeVisit
  rw [collect_interleaved values left right _ _ _ same]

theorem pendingWorker_interleaved (p : Pool) (timeline : Timeline p) (values : Values)
    (left : List (Event S)) (right : List (Event T)) (w : (owner : NativeTrace.Worker p) × NativeTrace.Cursor owner)
    (same : workerEvents w.1.ordinal left = workerEvents w.1.ordinal right) :
    pendingWorker p timeline mc fc values left w = pendingWorker p timeline mc fc values right w := by
  unfold pendingWorker
  cases w.2.stack.find? (fun frame => frame.node == .boundary) with
  | none => rfl
  | some frame => simp only [collect_interleaved values left right _ frame.first w.2.clock same]

end NeuralReversi.EndgameSearch.CacheHistory.Payloads
