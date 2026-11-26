// Unified split-point lifecycle model:
// - Split init -> helper booking & ActivePointer startup
// - Worker 1 scout alpha load -> concurrent Worker 2 alpha/best update -> Worker 1 wide re-search alpha load
// - cutoff_epoch miss + cache hit (distinguishing physical write index from reused 64-bit value)
// - 32-bit AbortState generation wrap (0xFFFFFFFF -> 0) and external abort propagation to owner
// - Participation mask release -> owner acquire zero -> owner best_score/best_move return -> immediate slot reuse
module SplitLifecycle {
  type State = {
    phase: str,
    // Alpha modification order indices: 0=init(0), 1=worker2(256)
    alphaUpdated: bool,
    scoutSrc: int,
    wideSrc: int,
    // cutoff_epoch writes: [0, 1, 2, 1] (index 1 and index 3 share value 1)
    epochReused: bool,
    cachedEpochSrc: int,
    hitEpochSrc: int,
    // AbortState 32-bit wrap: prevGen=max, begin wraps gen=0 (requested=max), request sets requested=0
    abortRequested: bool,
    workerAbortSrc: int,   // 0=begin(gen 0, not stopped), 1=request(gen 0, stopped)
    ownerAbortSrc: int,    // 0=begin, 1=request
    // Mask & owner score return + immediate reuse writing identical score 256 (index 1 vs index 2)
    maskReleased: bool,
    slotReused: bool,
    ownerScoreSrc: int,    // 0=init(0), 1=worker2(256), 2=reuse(256)
  }
  var s: State

  action init = s' = {
    phase: "init",
    alphaUpdated: false,
    scoutSrc: -1,
    wideSrc: -1,
    epochReused: false,
    cachedEpochSrc: -1,
    hitEpochSrc: -1,
    abortRequested: false,
    workerAbortSrc: -1,
    ownerAbortSrc: -1,
    maskReleased: false,
    slotReused: false,
    ownerScoreSrc: -1,
  }

  // Worker 1 executes scout alpha load and first cutoff_epoch miss (populates cache from epoch write 1).
  action workerScout = all {
    s.phase == "init",
    s' = { ...s, phase: "scouted", scoutSrc: 0, cachedEpochSrc: 1 },
  }

  // Worker 2 updates alpha and best_score to +256 under split lock.
  action remoteUpdate = all {
    s.phase == "scouted" or s.phase == "epochBumped",
    not(s.alphaUpdated),
    s' = { ...s, alphaUpdated: true },
  }

  // Background thread bumps cutoff_epoch to 2 and then wraps/reuses value 1 at write index 3.
  action bumpReuseEpoch = all {
    s.phase == "scouted",
    not(s.epochReused),
    s' = { ...s, phase: "epochBumped", epochReused: true },
  }

  // Worker 1 synchronizes on split lock and executes wide re-search alpha load.
  action workerWide(backward: bool): bool = all {
    s.alphaUpdated,
    s.wideSrc == -1,
    s' = { ...s, wideSrc: if (backward) 0 else 1 },
  }

  // External thread requests abort in wrapped generation 0.
  action requestAbort = all {
    s.wideSrc != -1,
    not(s.abortRequested),
    s' = { ...s, abortRequested: true },
  }

  // Worker 1 executes cutoff_epoch hit + non-flag AbortState load, then releases participation mask bit.
  action workerPollAndRelease(reusedEpochHit: bool): bool = all {
    s.abortRequested,
    s.epochReused,
    not(s.maskReleased),
    s' = { ...s,
      phase: "released",
      hitEpochSrc: if (reusedEpochHit) 3 else 1,
      workerAbortSrc: 1,
      maskReleased: true,
    },
  }

  // Owner acquires zero mask, reads post-wait AbortState and best_score, then immediately reuses slot.
  action ownerReturnAndReuse(missAbort: bool, readReuseScore: bool): bool = all {
    s.phase == "released",
    s.maskReleased,
    s' = { ...s,
      phase: "done",
      ownerAbortSrc: if (missAbort) 0 else 1,
      ownerScoreSrc: if (readReuseScore) 2 else 1,
      slotReused: true,
    },
  }

  action advance(backward: bool, reusedEpochHit: bool, missAbort: bool, readReuseScore: bool): bool = any {
    workerScout,
    remoteUpdate,
    bumpReuseEpoch,
    workerWide(backward),
    requestAbort,
    workerPollAndRelease(reusedEpochHit),
    ownerReturnAndReuse(missAbort, readReuseScore),
    all { s.phase == "done", s' = s },
  }

  action step = advance(false, false, false, false)
  action backwardAlpha = advance(true, false, false, false)
  action reusedEpoch = advance(false, true, false, false)
  action missedWrappedAbort = advance(false, false, true, false)
  action reuseScoreRead = advance(false, false, false, true)

  val alphaCoherent = (s.wideSrc == -1) or (s.scoutSrc <= s.wideSrc and s.wideSrc == 1)
  val epochHitExact = (s.hitEpochSrc == -1) or (s.hitEpochSrc == s.cachedEpochSrc)
  val abortPropagates = (s.ownerAbortSrc == -1) or (s.workerAbortSrc <= s.ownerAbortSrc)
  val scoreBeforeReuse = (s.ownerScoreSrc == -1) or (s.ownerScoreSrc == 1)

  val inv = alphaCoherent and epochHitExact and abortPropagates and scoreBeforeReuse

  // Witness selectors
  val noNormalDone = not(s.phase == "done" and s.alphaUpdated and s.epochReused and s.slotReused)
  val noBackwardAlpha = alphaCoherent
  val noReusedEpochHit = epochHitExact
  val noMissedWrappedAbort = abortPropagates
  val noReuseScoreRead = scoreBeforeReuse
}
