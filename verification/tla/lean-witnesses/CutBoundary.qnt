// Concurrent TT cut-boundary and multi-session handoff model:
// 1) Models the 7-event key-mismatch litmus (worker 0 seqlock write vs worker 1 fast lookup)
//    comparing HB-only cuts (which drop incomparable RF write 6) against RF+HB-closed cuts.
// 2) Models an adopted TT hit (sequence acquire syncs-with finish), verifying HB closure
//    already retains the committed epoch's writes.
// 3) Models multi-session handoff (drained writers vs undrained odd sequence lock).
module CutBoundary {
  type State = {
    phase: str,
    // Events emitted in 7-event litmus: W0={0:begin,1:cas,2:fence,6:player}, W1={3:seq,4:player(rf=6),5:opp}
    w0Steps: int,        // 0..4 (4 = wrote player at event 6; 5 = finished full epoch)
    w1Steps: int,        // 0..3 (3 = completed fast lookup)
    cutK: int,           // chosen prefix cut k in 0..7
    hbClosed: bool,
    rfClosed: bool,
    session1Drained: bool,
    session2ReadInherited: bool,
  }
  var s: State

  action init = s' = {
    phase: "litmus",
    w0Steps: 0,
    w1Steps: 0,
    cutK: -1,
    hbClosed: true,
    rfClosed: true,
    session1Drained: true,
    session2ReadInherited: false,
  }

  // Worker 0 emits events 0, 1 (cas), 2 (fence), and later 6 (store player).
  action w0Advance = all {
    s.phase == "litmus",
    s.w0Steps < 4,
    s' = { ...s, w0Steps: s.w0Steps + 1 },
  }

  // Worker 1 executes fast key-mismatch lookup (events 3:seq, 4:player reading 6, 5:opp).
  action w1Mismatch = all {
    s.phase == "litmus",
    s.w0Steps == 4,
    s.w1Steps < 3,
    s' = { ...s, w1Steps: s.w1Steps + 1 },
  }

  // Choose a cut boundary k for the 7-event trace.
  // Normal mode closes under BOTH HB and RF (so k=6 expands to k=7).
  // Mutant hbOnly mode allows cutting at k=6 (HB-closed, NOT RF-closed because 6 -> 4).
  action cutTrace(allowHbOnly: bool): bool = all {
    s.phase == "litmus",
    s.w0Steps == 4,
    s.w1Steps == 3,
    any {
      all {
        s' = { ...s, phase: "handoff", cutK: 7, hbClosed: true, rfClosed: true },
      },
      all {
        allowHbOnly,
        s' = { ...s, phase: "handoff", cutK: 6, hbClosed: true, rfClosed: false },
      },
    },
  }

  // Session 1 completes writer finish (drained) or stops mid-write (undrained),
  // then Session 2 worker adopts the inherited TT entry.
  action handoffSession(allowUndrained: bool): bool = all {
    s.phase == "handoff",
    any {
      all {
        s' = { ...s,
          phase: "done",
          session1Drained: true,
          session2ReadInherited: true,
        },
      },
      all {
        allowUndrained,
        s' = { ...s,
          phase: "done",
          session1Drained: false,
          session2ReadInherited: true,
        },
      },
    },
  }

  action advance(allowHbOnly: bool, allowUndrained: bool): bool = any {
    w0Advance,
    w1Mismatch,
    cutTrace(allowHbOnly),
    handoffSession(allowUndrained),
    all { s.phase == "done", s' = s },
  }

  action step = advance(false, false)
  action hbOnlyCut = advance(true, false)
  action undrainedHandoff = advance(false, true)

  val cutPreservesRF = (s.cutK == -1) or (s.hbClosed and s.rfClosed)
  val handoffDrained = not(s.session2ReadInherited) or s.session1Drained
  val inv = cutPreservesRF and handoffDrained

  // Witness selectors
  val noNormalDone = not(s.phase == "done" and s.cutK == 7 and s.rfClosed and s.session1Drained and s.session2ReadInherited)
  val noHbOnlyDrop = cutPreservesRF
  val noUndrainedHandoff = handoffDrained
}
