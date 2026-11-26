// Synchronization between ownerActive (split-point lock only) and foreign
// book_into (worker state lock + child split lock) on worker 0's active cell.
// Locks: 0 = worker 0 state lock, 1 = slot 0 lock, 2 = slot 1 lock.
// Each actor and lock carries the set of write events it has observed via
// program order and release->acquire of the same lock.
module OwnerActiveSync {
  type State = {
    phase: str,
    searching: bool,
    detached: bool,
    booked: bool,
    cloned1: bool,
    cloned2: bool,
    bookLock: int,
    interveners: List[int],
    // Bitmask of observed events: 1=ownerActive, 2=detach0, 4=foreignBook
    actorVis: List[int],
    lockVis: List[int],
    ownerActiveKnownAtBook: bool,
    bookKnownAtOwnerActive: bool,
    bookKnownAtClone2: bool,
    earlyDispatch: bool,
  }
  var s: State

  pure def hasBit(mask: int, bit: int): bool = (mask / bit) % 2 == 1
  pure def joinBits(a: int, b: int): int =
    (if (hasBit(a, 1) or hasBit(b, 1)) 1 else 0) +
    (if (hasBit(a, 2) or hasBit(b, 2)) 2 else 0) +
    (if (hasBit(a, 4) or hasBit(b, 4)) 4 else 0)

  action init = s' = {
    phase: "ownerInit",
    searching: true,
    detached: false,
    booked: false,
    cloned1: false,
    cloned2: false,
    bookLock: -1,
    interveners: [],
    actorVis: [0, 0, 0],
    lockVis: [0, 0, 0],
    ownerActiveKnownAtBook: false,
    bookKnownAtOwnerActive: false,
    bookKnownAtClone2: false,
    earlyDispatch: false,
  }

  // Worker 0 executes ownerActive under slot 0 lock (lock 1), NOT worker 0 lock.
  action ownerActive = {
    val vis = joinBits(joinBits(s.actorVis.nth(0), s.lockVis.nth(1)), 1)
    all {
      s.phase == "ownerInit",
      s' = { ...s,
        phase: "ownerRun",
        actorVis: s.actorVis.replaceAt(0, vis),
        lockVis: s.lockVis.replaceAt(1, vis),
        bookKnownAtOwnerActive: hasBit(vis, 4),
      },
    }
  }

  // Worker 0 clones its active pointer under worker 0 lock (lock 0) and dispatches.
  action ownerClone(early: bool): bool = {
    val vis = joinBits(s.actorVis.nth(0), s.lockVis.nth(0))
    all {
      s.phase == "ownerRun",
      not(s.cloned1),
      s' = { ...s,
        cloned1: true,
        earlyDispatch: early,
        actorVis: s.actorVis.replaceAt(0, vis),
        lockVis: if (early) s.lockVis else s.lockVis.replaceAt(0, vis),
      },
    }
  }

  // Worker 0 detaches under worker 0 lock (lock 0) and sets searching = false.
  action ownerDetach = {
    val vis = joinBits(joinBits(s.actorVis.nth(0), s.lockVis.nth(0)), 2)
    all {
      s.phase == "ownerRun",
      s.cloned1,
      not(s.detached),
      not(s.earlyDispatch),
      s' = { ...s,
        detached: true,
        searching: false,
        actorVis: s.actorVis.replaceAt(0, vis),
        lockVis: s.lockVis.replaceAt(0, vis),
      },
    }
  }

  // Worker 2 can acquire and release worker 0 lock while idle without changing active.
  action intervene = {
    val vis = joinBits(s.actorVis.nth(2), s.lockVis.nth(0))
    all {
      s.phase == "ownerRun",
      s.detached,
      not(s.booked),
      s.interveners.length() == 0,
      s' = { ...s,
        interveners: [2],
        actorVis: s.actorVis.replaceAt(2, vis),
        lockVis: s.lockVis.replaceAt(0, vis),
      },
    }
  }

  // Worker 1 books worker 0 into child split (slot 1) under lock `lock` (0 normal, 2 wrong).
  action foreignBook(ignoreSearching: bool, wrongLock: bool): bool = {
    val lock = if (wrongLock) 2 else 0
    val acquired = joinBits(joinBits(s.actorVis.nth(1), s.lockVis.nth(2)), s.lockVis.nth(lock))
    val vis = joinBits(acquired, 4)
    all {
      s.phase == "ownerRun",
      not(s.booked),
      ignoreSearching or not(s.searching),
      s' = { ...s,
        booked: true,
        searching: true,
        bookLock: lock,
        ownerActiveKnownAtBook: hasBit(acquired, 1),
        actorVis: s.actorVis.replaceAt(1, vis),
        lockVis: s.lockVis.replaceAt(2, vis).replaceAt(lock, vis),
      },
    }
  }

  // Worker 0 acquires lock 0, clones the foreign booking, releases lock 0, dispatches.
  action helperClone = {
    val vis = joinBits(s.actorVis.nth(0), s.lockVis.nth(0))
    all {
      s.phase == "ownerRun",
      s.booked,
      s.detached,
      not(s.cloned2),
      s' = { ...s,
        phase: "done",
        cloned2: true,
        bookKnownAtClone2: hasBit(vis, 4),
        actorVis: s.actorVis.replaceAt(0, vis),
        lockVis: s.lockVis.replaceAt(0, vis),
      },
    }
  }

  action advance(ignoreSearching: bool, wrongLock: bool, early: bool): bool = any {
    ownerActive,
    ownerClone(early),
    ownerDetach,
    intervene,
    foreignBook(ignoreSearching, wrongLock),
    helperClone,
    all { s.phase == "done", s' = s },
    all { s.earlyDispatch, s' = s },
  }

  action step = advance(false, false, false)
  action raceBook = advance(true, false, false)
  action wrongLock = advance(false, true, false)
  action earlyRelease = advance(false, false, true)

  // Every reachable state is DRF between ownerActive and foreignBook, and the
  // second clone observes the foreign booking after lock release.
  val safe =
    (s.booked implies (s.ownerActiveKnownAtBook or s.bookKnownAtOwnerActive)) and
    (s.phase == "done" implies (s.ownerActiveKnownAtBook and s.bookKnownAtClone2)) and
    not(s.earlyDispatch)

  val noNormalWithIntervener = not(s.phase == "done" and s.interveners.length() == 1)
  val noDataRace = not(s.booked and not(s.ownerActiveKnownAtBook) and not(s.bookKnownAtOwnerActive))
  val noWrongLock = not(s.phase == "done" and s.bookLock == 2 and not(s.bookKnownAtClone2))
  val noEarlyDispatch = not(s.earlyDispatch)
}
