// One physical helpers_mask with two completed generations and the owner's next clear and mark.
// Owner 0 books workers 1 and 2 while the split lock is held. Each retires once
// during normal dispatch. "raw" also admits a subsequent ineffective Release RMW,
// as RawMask does. That extra operation is not a claimed production path.
module ParentMaskZero {
  type State = {
    phase: str, pending: Set[int], finishes: List[int], words: List[int],
    ownerFinish: int, effective: int, duplicate: int, source: int,
  }
  var s: State
  pure val workers = Set(0, 1, 2)
  pure def word(active: Set[int]): int =
    active.fold(0, (bits, w) => bits + (if (w == 0) 1 else if (w == 1) 2 else 4))

  // Previous clear, mark, finish; current clear, mark and the two booked bits.
  action init = s' = {
    phase: "retiring", pending: workers, finishes: [], words: [0,1,0,0,1,3,7],
    ownerFinish: -1, effective: -1, duplicate: -1, source: -1,
  }

  action finish(w: int): bool = {
    val rest = s.pending.exclude(Set(w))
    val index = s.words.length()
    all {
      s.phase == "retiring", s.pending.contains(w),
      s' = { ...s, phase: if (rest.size() == 0) "empty" else "retiring",
        pending: rest, finishes: s.finishes.append(w), words: s.words.append(word(rest)),
        ownerFinish: if (w == 0) index else s.ownerFinish,
        effective: if (rest.size() == 0) index else s.effective },
    }
  }

  // Build the complete physical MO before selecting a source; the next clear is
  // program-ordered AFTER the read. An extra reset by owner 0 is also before the read.
  action close(extra: int): bool = {
    val words = if (extra < 0) s.words else s.words.append(0)
    all { s.phase == "empty",
      s' = { ...s, phase: "choose", duplicate: extra, words: words.append(0).append(1),
        ownerFinish: if (extra == 0) s.words.length() else s.ownerFinish },
    }
  }

  // With the common split lock, all mask modifications through the owner's own
  // retirement precede its wait. Later helper releases precede this acquire
  // only up to its chosen release-sequence source, unless another edge orders them.
  action read(i: int, checkOrder: bool): bool = all {
    s.phase == "choose", s.words.nth(i) == 0,
    not(checkOrder) or (s.ownerFinish <= i and i < s.words.length()-2),
    s' = { ...s, phase: "done", source: i },
  }

  action advance(extraActors: Set[int], checkOrder: bool): bool = any {
    { nondet w = workers.oneOf(); finish(w) },
    close(-1),
    { nondet w = extraActors.oneOf(); close(w) },
    { nondet i = s.words.indices().oneOf(); read(i, checkOrder) },
    all { s.phase == "done", s' = s },
  }
  action step = advance(Set(), true)
  action raw = advance(Set(1,2), true)
  action ownerRaw = advance(Set(0), true)
  action valueOnly = advance(Set(), false)

  val safe = s.phase != "done" or (s.effective <= s.source and s.source < s.words.length()-2)
  val unique = s.phase != "done" or s.source == s.effective
  val latestPhysical = s.phase != "done" or s.source == s.words.length()-3
  val noPriorFinish = s.phase != "done" or s.source != 2
  val noNextClear = s.phase != "done" or s.source != s.words.length()-2
  val ownerOrdered = s.phase != "done" or s.duplicate != 0 or s.source == s.effective+1
  val noOwnerOrdered = s.phase != "done" or s.duplicate != 0 or s.source != s.effective+1
  val noCurrentFinish = s.phase != "done" or s.source != s.effective
}
