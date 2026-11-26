// One serially visible execution family. Owner 0 retains parent P while idle,
// then helps nested C owned by helper 1. Other participants may set either
// flag. Helper 1 restores P after C. Participation spans both polls, and both
// P and C close before their physical slots are initialized a second time.
// Epoch starts at 1 so reset(epoch) = epoch-1 needs no integer wrap here.
module NestedCutoff {
  type Write = { generation: int, value: bool }
  type Booking = { generation: int, parentSeed: int, childSeed: int }
  type Sample = { slot: int, source: int }
  type State = {
    phase: str, generation: int, parent: List[Write], child: List[Write], bookings: List[Booking],
    parentMask: Set[int], helperActive: int, restored: int, assigned: int, usedAssignment: int,
    seen: int, epoch: int, cached: bool, lastSlot: int, lastSource: int, previousParentSource: int,
    nestedStop: bool, expectedNested: bool, afterStop: bool, parentSource: int, pollKind: str,
  }
  var s: State

  action init = s' = {
    phase: "start", generation: 0, parent: [], child: [], bookings: [],
    parentMask: Set(), helperActive: -1, restored: -1, assigned: 0, usedAssignment: 0,
    seen: 0, epoch: 1, cached: false, lastSlot: -1, lastSource: -1, previousParentSource: -1,
    nestedStop: false, expectedNested: false, afterStop: false, parentSource: -1, pollKind: "",
  }

  action openParent = all {
    (s.phase == "start" or s.phase == "closed") and s.generation < 2, s.parentMask == Set(),
    val g = s.generation + 1
    // Initialize P (physical slot 0), book helpers 1 and 2. Worker 2 can
    // publish P's cutoff while the owner helps C.
    s' = { ...s, phase: "parent", generation: g, helperActive: 0, restored: -1, parentMask: Set(0,1,2),
           parent: s.parent.append({generation: g, value: false}), nestedStop: false,
           previousParentSource: if (s.nestedStop and s.lastSlot == 0) s.lastSource else -1,
           afterStop: false, parentSource: -1, assigned: 0, usedAssignment: 0 },
  }
  action openChild = all {
    s.phase == "parent", s.parentMask.contains(1),
    // Helper 1 saves the assigned parent in C's task and initializes slot 1.
    // Another child participant can publish a cutoff after booking.
    s' = { ...s, phase: "child", child: s.child.append({generation: s.generation, value: false}) },
  }
  action detach = all {
    s.phase == "child",
    // search_active_split_point detaches owner 0 and clears its bit.
    // Parent P stays live through helper 1 and the other parent participant.
    s' = { ...s, phase: "idle", parentMask: s.parentMask.exclude(Set(0)), helperActive: -1,
           seen: s.epoch - 1, cached: false, lastSlot: -1, lastSource: -1, pollKind: "" },
  }
  action help = all {
    s.phase == "idle", s.helperActive == -1, s.parentMask.contains(1),
    // can_join: deepest owned P still has child owner 1; book_into installs C.
    val job = {generation: s.generation, parentSeed: s.parent.length()-1, childSeed: s.child.length()-1}
    s' = { ...s, phase: "first", helperActive: 1, assigned: job.generation, bookings: s.bookings.append(job) },
  }

  action mark(parent: bool): bool = all {
    s.phase == "first" or s.phase == "second" or (s.phase == "after" and parent),
    val tape = if (parent) s.parent else s.child
    val marked = tape.append({generation: s.generation, value: true})
    all {not(tape.nth(tape.length()-1).value), s' = { ...s, epoch: s.epoch+1,
           parent: if (parent) marked else s.parent, child: if (parent) s.child else marked }},
  }

  // Reconstruct the sources from the install and this lifetime's flag writes.
  // Source ranks are local to their physical slot's modification order.
  def latest(tape: List[Write], seed: int): int =
    tape.indices().filter(i => i >= seed and tape.nth(i).generation == tape.nth(seed).generation)
        .fold(seed, (highest, i) => if (highest > i) highest else i)
  def walk(job: Booking): Sample = {
    val c = latest(s.child, job.childSeed)
    val p = latest(s.parent, job.parentSeed)
    if (s.child.nth(c).value) {slot: 1, source: c}
    else if (s.parent.nth(p).value) {slot: 0, source: p}
    else {slot: -1, source: -1}
  }

  action poll(stale: bool): bool = all {
    s.phase == "first" or s.phase == "second",
    // Diagnostic substitutes the old helpful booking for the new one. The
    // physical slots, tape and actual assignments of both workers stay intact.
    val job = if (stale) s.bookings.head() else s.bookings.nth(s.bookings.length()-1)
    val rescan = not(s.cached) and s.epoch != s.seen
    val found = if (rescan) walk(job) else {slot: s.lastSlot, source: s.lastSource}
    val stop = if (rescan) found.slot >= 0 else s.cached
    s' = { ...s, phase: if (s.phase == "first") "second" else "retire",
           seen: if (rescan) s.epoch else s.seen, cached: stop, nestedStop: stop,
           expectedNested: s.parent.nth(s.parent.length()-1).value or s.child.nth(s.child.length()-1).value,
           usedAssignment: job.generation, lastSlot: found.slot, lastSource: found.source,
           pollKind: if (rescan) "walk" else if (s.cached) "latch" else "equalEpoch" },
  }
  action retire = all {
    s.phase == "retire",
    // Helper 0 detaches from C. Worker 1's finalize_split_point restores C's
    // saved parent and should_stop_after_split walks P directly on worker 1.
    // Keep worker 0's final poll registers for the witness.
    s' = { ...s, phase: "after", helperActive: -1, restored: 0 },
  }
  action after = all {
    s.phase == "after", s.restored == 0,
    val i = s.parent.length()-1
    s' = { ...s, phase: "done", parentSource: i, afterStop: s.parent.nth(i).value },
  }
  action closeParent = all {
    s.phase == "done", s.generation == 1,
    // The remaining parent participants detach and the owner observes a zero mask.
    s' = { ...s, phase: "closed", parentMask: Set(), restored: -1 },
  }
  action advance(stale: bool): bool = any {
    openParent, openChild, detach, help, mark(true), mark(false), poll(stale), retire, after, closeParent,
    all {s.phase == "done", s.generation == 2, s' = s},
  }
  action step = advance(false)
  action staleAssignment = advance(true)

  val checked = s.phase == "second" or s.phase == "retire" or s.phase == "after" or s.phase == "done"
  val witness = s.phase == "done" and s.generation == 2
  val sourceCurrent = if (not(s.nestedStop)) true
    else if (s.lastSlot == 0) s.parent.nth(s.lastSource).generation == s.generation
    else s.lastSlot == 1 and s.child.nth(s.lastSource).generation == s.generation
  val safe = (not(checked) or (s.usedAssignment == s.assigned and s.nestedStop == s.expectedNested and sourceCurrent)) and
    (s.phase != "done" or (s.parent.nth(s.parentSource).generation == s.generation and
                          s.afterStop == s.parent.nth(s.parentSource).value))
  // First-generation parent used the same positive word. Require a cached
  // second poll to retain the original flag write for the current assignment.
  val bothPositive = witness and s.nestedStop and s.lastSlot == 0 and s.afterStop and
    s.pollKind == "latch" and not(s.child.nth(s.child.length()-1).value) and
    s.previousParentSource >= 0
  val noReuse = not(bothPositive and sourceCurrent)
  val noStaleAlias = not(bothPositive and s.previousParentSource == s.lastSource and not(sourceCurrent))
  val noStaleStop = not(witness and s.nestedStop and s.lastSlot == 0 and not(s.afterStop) and s.previousParentSource == s.lastSource and
                          not(s.child.nth(s.child.length()-1).value) and not(sourceCurrent))
  val noLocalOnly = not(witness and s.lastSlot == 1 and s.nestedStop and not(s.afterStop))
  val noRestoredMark = not(witness and not(s.nestedStop) and s.pollKind == "equalEpoch" and s.afterStop)
}
