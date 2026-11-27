--------------------------- MODULE EndgameResult ---------------------------
EXTENDS Integers

(***************************************************************************
 A bounded result model for pvs.rs::search_split_point and the owner's
 search() return path.  A child search is an oracle satisfying the fail-soft
 contract, except that a stopped child (global/ancestor abort or local
 split-point cutoff) may unwind early and return the dummy score zero.

 Claim, alpha loads, scout completion, retry decision, wide-window alpha load,
 wide completion, locked publication, owner readback, and owner stop check are
 separate actions.  In particular another worker can raise alpha between a
 scout and its retry decision.  The retry decision uses the alpha *of that
 scout*, as the Rust implementation does.

 This is a sequentially consistent abstraction.  Iterator CAS correctness,
 slot lifetime, memory-order visibility, and child search correctness are
 independent obligations covered by the other TLA+/Lean models.
 ***************************************************************************)

CONSTANTS N, NumMoves, Mutant
Workers == 1..N
Scores == -1..3
ScoreMax == 3
Sentinel == -2

Max(a, b) == IF a >= b THEN a ELSE b
Bound(v, a, b, r) ==
    /\ (r <= a => v <= r)
    /\ (r >= b => r <= v)
    /\ (a < r /\ r < b => r = v)

VARIABLES originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next,
          phase, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop,
          owner, observed, accepted

vars == <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next,
          phase, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop,
          owner, observed, accepted>>

AllWorkersDone == \A w \in Workers : phase[w] = "done"

Init ==
    /\ originalAlpha \in {-1, 0}
    /\ pv \in BOOLEAN
    /\ beta \in IF pv THEN {2, 4} ELSE {originalAlpha + 1}
    /\ \E pvVal \in Scores \cup {Sentinel}, pvScore \in Scores \cup {Sentinel} :
           /\ (pvVal = Sentinel) = (pvScore = Sentinel)
           /\ pv => pvVal # Sentinel
           /\ pvScore < beta /\ pvScore < ScoreMax
           /\ Bound(pvVal, originalAlpha, beta, pvScore)
           /\ nodeValue = pvVal
           /\ alpha = IF pv THEN Max(originalAlpha, pvScore) ELSE originalAlpha
           /\ best = pvScore
    /\ unvalued = NumMoves
    /\ cutoff = FALSE
    /\ aborted = FALSE
    /\ next = 1
    /\ phase = [w \in Workers |-> "claim"]
    /\ moveVal = [w \in Workers |-> Sentinel]
    /\ scoutAlpha = [w \in Workers |-> 0]
    /\ wideAlpha = [w \in Workers |-> 0]
    /\ score = [w \in Workers |-> 0]
    /\ lockOwner = 0
    /\ stopSeen = [w \in Workers |-> FALSE]
    /\ reportedCount = 0
    /\ publishedAfterObservedStop = FALSE
    /\ owner = "wait"
    /\ observed = Sentinel
    /\ accepted = FALSE

\* Abstracts the already-verified unique ConcurrentMoveIterator claim.
\* On stop a worker may retire directly: no further score may be published.
\* Workers are symmetric prior to their first claim, so WLOG worker w only
\* claims when w <= next (preserves liveness without TLC SYMMETRY).
Claim(w) ==
    /\ phase[w] = "claim"
    /\ IF cutoff \/ aborted \/ next > NumMoves THEN
           /\ phase' = [phase EXCEPT ![w] = "done"]
           /\ UNCHANGED next
       ELSE
           /\ w <= next
           /\ next' = next + 1
           /\ phase' = [phase EXCEPT ![w] = "scout_read"]
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

ScoutRead(w) ==
    /\ phase[w] = "scout_read"
    /\ scoutAlpha' = [scoutAlpha EXCEPT ![w] = alpha]
    /\ phase' = [phase EXCEPT ![w] = "scout"]
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, moveVal, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

\* Child search true values are materialized lazily on first Bound evaluation
\* (exact bisimulation with choosing values \in [Moves -> Scores] in Init).
\* When aborted or local cutoff holds, thread.should_stop() may unwind early
\* and return dummy 0, or the child may still return a sound Bound value.
Scout(w) ==
    /\ phase[w] = "scout"
    /\ \/ /\ aborted \/ cutoff
          /\ score' = [score EXCEPT ![w] = 0]
          /\ UNCHANGED <<nodeValue, unvalued, moveVal>>
       \/ \E v \in Scores, r \in Scores :
              /\ Bound(v, scoutAlpha[w], scoutAlpha[w] + 1, r)
              /\ score' = [score EXCEPT ![w] = r]
              /\ nodeValue' = Max(nodeValue, v)
              /\ unvalued' = unvalued - 1
              /\ moveVal' = [moveVal EXCEPT ![w] = IF pv /\ r > scoutAlpha[w] THEN v ELSE Sentinel]
    /\ phase' = [phase EXCEPT ![w] = "decide"]
    /\ UNCHANGED <<originalAlpha, pv, beta, alpha, best, cutoff, aborted, next,
          scoutAlpha, wideAlpha, lockOwner, stopSeen, reportedCount,
          publishedAfterObservedStop, owner, observed, accepted>>

Decide(w) ==
    /\ phase[w] = "decide"
    /\ LET threshold == IF Mutant = "retry_against_new_alpha"
                           THEN alpha ELSE scoutAlpha[w]
           retry == pv /\ score[w] > threshold /\ Mutant # "skip_pv_retry"
       IN /\ phase' = [phase EXCEPT ![w] = IF retry THEN "wide_read" ELSE "publish"]
          /\ score' = IF retry THEN [score EXCEPT ![w] = 0] ELSE score
          /\ moveVal' = IF retry THEN moveVal ELSE [moveVal EXCEPT ![w] = Sentinel]
    /\ scoutAlpha' = [scoutAlpha EXCEPT ![w] = 0]
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, wideAlpha,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

WideRead(w) ==
    /\ phase[w] = "wide_read"
    /\ wideAlpha' = [wideAlpha EXCEPT ![w] = alpha]
    /\ phase' = [phase EXCEPT ![w] = "wide"]
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, moveVal, scoutAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

Wide(w) ==
    /\ phase[w] = "wide"
    /\ \/ /\ aborted \/ cutoff
          /\ score' = [score EXCEPT ![w] = 0]
          /\ moveVal' = [moveVal EXCEPT ![w] = Sentinel]
          /\ UNCHANGED <<nodeValue, unvalued>>
       \/ IF moveVal[w] # Sentinel THEN
              \E r \in Scores :
                  /\ Bound(moveVal[w], wideAlpha[w], beta, r)
                  /\ score' = [score EXCEPT ![w] = r]
                  /\ moveVal' = [moveVal EXCEPT ![w] = Sentinel]
                  /\ UNCHANGED <<nodeValue, unvalued>>
          ELSE
              \E v \in Scores, r \in Scores :
                  /\ Bound(v, wideAlpha[w], beta, r)
                  /\ score' = [score EXCEPT ![w] = r]
                  /\ nodeValue' = Max(nodeValue, v)
                  /\ unvalued' = unvalued - 1
                  /\ UNCHANGED moveVal
    /\ wideAlpha' = [wideAlpha EXCEPT ![w] = 0]
    /\ phase' = [phase EXCEPT ![w] = "publish"]
    /\ UNCHANGED <<originalAlpha, pv, beta, alpha, best, cutoff, aborted, next,
          scoutAlpha, lockOwner, stopSeen, reportedCount,
          publishedAfterObservedStop, owner, observed, accepted>>

\* The split lock serializes publishers, but NOT the external stop writer.
\* Stop can therefore occur after this check and before Commit's stores.
LockCheck(w) ==
    /\ phase[w] = "publish" /\ lockOwner = 0
    /\ LET rejects == cutoff \/ (aborted /\ Mutant # "publish_after_abort")
       IN /\ phase' = [phase EXCEPT ![w] = IF rejects THEN "done" ELSE "commit"]
          /\ lockOwner' = IF rejects THEN 0 ELSE w
          /\ stopSeen' = [stopSeen EXCEPT ![w] = IF rejects THEN FALSE ELSE aborted]
          /\ score' = IF rejects THEN [score EXCEPT ![w] = 0] ELSE score
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, moveVal, scoutAlpha, wideAlpha,
          reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

Commit(w) ==
    /\ phase[w] = "commit" /\ lockOwner = w
    /\ LET improves == score[w] > best /\ score[w] > alpha
           raises == improves /\ pv /\ score[w] < beta
           newCutoff == cutoff \/
                (improves /\ (~pv \/ score[w] >= beta \/ score[w] >= ScoreMax))
       IN /\ best' = Max(best, score[w])
          /\ alpha' = IF raises THEN score[w] ELSE alpha
          /\ cutoff' = newCutoff
          /\ phase' = [phase EXCEPT ![w] = IF newCutoff THEN "done" ELSE "claim"]
    /\ reportedCount' = reportedCount + 1
    /\ publishedAfterObservedStop' = (publishedAfterObservedStop \/ stopSeen[w])
    /\ lockOwner' = 0
    /\ score' = [score EXCEPT ![w] = 0]
    /\ stopSeen' = [stopSeen EXCEPT ![w] = FALSE]
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          aborted, next, moveVal, scoutAlpha, wideAlpha, owner, observed, accepted>>

\* An external/global/ancestor stop may occur during a child or after readback.
Stop ==
    /\ ~aborted /\ owner # "done"
    /\ aborted' = TRUE
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, next, phase, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, owner, observed, accepted>>

OwnerRead ==
    /\ owner = "wait"
    /\ IF Mutant = "read_before_workers_drain"
           THEN (AllWorkersDone \/ next > NumMoves) ELSE AllWorkersDone
    /\ observed' = best
    /\ owner' = "check"
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, phase, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, accepted>>

\* search() checks should_stop_after_split() before accepting the tuple.
OwnerCheck ==
    /\ owner = "check"
    /\ accepted' = (~aborted \/ Mutant = "accept_after_abort")
    /\ owner' = "done"
    /\ UNCHANGED <<originalAlpha, pv, beta, nodeValue, unvalued,
          alpha, best, cutoff, aborted, next, phase, moveVal, scoutAlpha, wideAlpha, score,
          lockOwner, stopSeen, reportedCount, publishedAfterObservedStop, observed>>

Worker(w) == Claim(w) \/ ScoutRead(w) \/ Scout(w) \/ Decide(w)
             \/ WideRead(w) \/ Wide(w) \/ LockCheck(w) \/ Commit(w)
Done == owner = "done" /\ AllWorkersDone /\ UNCHANGED vars
Next == (\E w \in Workers : Worker(w)) \/ Stop \/ OwnerRead \/ OwnerCheck \/ Done
Spec == Init /\ [][Next]_vars
        /\ (\A w \in Workers : WF_vars(Worker(w)))
        /\ WF_vars(OwnerRead) /\ WF_vars(OwnerCheck)

TypeOK ==
    /\ originalAlpha \in {-1, 0}
    /\ pv \in BOOLEAN /\ beta \in {0, 1, 2, 4}
    /\ nodeValue \in Scores \cup {Sentinel}
    /\ unvalued \in 0..NumMoves
    /\ alpha \in originalAlpha..ScoreMax
    /\ best \in Scores \cup {Sentinel}
    /\ cutoff \in BOOLEAN /\ aborted \in BOOLEAN
    /\ next \in 1..(NumMoves + 1)
    /\ phase \in [Workers -> {"claim", "scout_read", "scout", "decide", "wide_read", "wide", "publish", "commit", "done"}]
    /\ moveVal \in [Workers -> Scores \cup {Sentinel}]
    /\ scoutAlpha \in [Workers -> originalAlpha..ScoreMax]
    /\ wideAlpha \in [Workers -> originalAlpha..ScoreMax]
    /\ score \in [Workers -> Scores]
    /\ lockOwner \in Workers \cup {0}
    /\ stopSeen \in [Workers -> BOOLEAN]
    /\ reportedCount \in 0..NumMoves /\ publishedAfterObservedStop \in BOOLEAN
    /\ owner \in {"wait", "check", "done"}
    /\ observed \in Scores \cup {Sentinel} /\ accepted \in BOOLEAN

LockOwnership == \A w \in Workers : (phase[w] = "commit") = (lockOwner = w)
SearchWindow == alpha < beta
OwnerReadAfterDrain == owner # "wait" => AllWorkersDone
NoInterruptedPublication == ~publishedAfterObservedStop
NoInterruptedAcceptance == accepted => ~aborted
ReturnedBound ==
    accepted =>
        IF unvalued = 0
        THEN Bound(nodeValue, originalAlpha, beta, observed)
        ELSE \A u \in Scores : Bound(Max(nodeValue, u), originalAlpha, beta, observed)
ReturnedCoverage == accepted => (reportedCount = NumMoves \/ observed >= beta \/ observed = ScoreMax)
CutoffWitness == cutoff => (best >= beta \/ best = ScoreMax)
EventuallyCompletes == <> (owner = "done" /\ AllWorkersDone)
=============================================================================
