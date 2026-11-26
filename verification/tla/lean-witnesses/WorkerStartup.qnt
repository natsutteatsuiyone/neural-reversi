// mutex_for_state publication in book_into -> search_active_split_point.
// Each actor/lock retains the latest of three program points it has observed:
// 1 = first lifetime, 2 = new slot initialization, 3 = new booking. Carry
// these facts only over program order and unlock/acquire of the same object.
module WorkerStartup {
  type State = {
    phase: str, late: bool, writerLock: int, turns: List[int], known: List[int],
    lockWords: List[int], active: int, oldClone: int, selected: int,
    address: int, selectedAddress: int, atBook: int,
  }
  // Mutex indices 0,1,2 represent the target worker, another worker, and the slot.
  var s: State
  pure def newest(a: int, b: int): int = if (a > b) a else b
  pure def arrived(actor: int, lock: int, minds: List[int], locks: List[int]): int =
    newest(minds.nth(actor), locks.nth(lock))

  // Previous dispatch cloned bank 7, detached and was cleared. Both owner 0
  // and helper 1 saw that generation. The previous clone remains in the log.
  action init = s' = {
    phase: "reuse", late: false, writerLock: -1, turns: [], known: [1,1,0,0],
    lockWords: [1,0,1], active: 0, oldClone: 1, selected: 0,
    address: 7, selectedAddress: -1, atBook: 0,
  }

  action reuse(late: bool): bool = all {
    s.phase == "reuse",
    // The owner's initialize_split_point writes the new task under the slot lock.
    // An initial booking stays inside it. Late booking reacquires it after init.
    s' = { ...s, phase: "book", late: late, known: s.known.replaceAt(0,2),
           lockWords: if (late) s.lockWords.replaceAt(2,2) else s.lockWords },
  }

  action book(wrong: bool): bool = {
    val who = if (s.late) 1 else 0
    val lock = if (wrong) 1 else 0
    val entered = if (s.late) arrived(who,2,s.known,s.lockWords) else s.known.nth(who)
    val acquired = newest(entered,s.lockWords.nth(lock))
    all { s.phase == "book",
      s' = { ...s, phase: "between", active: 2, writerLock: lock, atBook: acquired,
        known: s.known.replaceAt(who,3), lockWords: s.lockWords.replaceAt(lock,3).replaceAt(2,3) },
    }
  }

  // Other callers can acquire the worker lock, observe searching and release
  // without changing the assignment. The helper only acquires after these guards.
  action intervene(w: int): bool = {
    val carry = arrived(w,0,s.known,s.lockWords)
    all { s.phase == "between", not(s.turns.indices().exists(i => s.turns.nth(i) == w)),
      s' = { ...s, turns: s.turns.append(w), known: s.known.replaceAt(w,carry),
        lockWords: s.lockWords.replaceAt(0,carry) },
    }
  }

  action clone(stale: bool): bool = {
    val vis = arrived(1,0,s.known,s.lockWords)
    all { s.phase == "between",
      s' = { ...s, phase: "done", selected: if (stale) s.oldClone else s.active,
        selectedAddress: s.address, known: s.known.replaceAt(1,vis), lockWords: s.lockWords.replaceAt(0,vis) },
    }
  }

  action advance(wrong: bool, stale: bool): bool = any {
    { nondet late = Bool.oneOf(); reuse(late) },
    book(wrong),
    { nondet w = Set(2,3).oneOf(); intervene(w) },
    clone(stale),
    all { s.phase == "done", s' = s },
  }
  action step = advance(false,false)
  // Change only the booker's lock target. This probes the worker-lock transfer;
  // the independent Release/Acquire on searching is outside this litmus.
  action wrongWorker = advance(true,false)
  // Substitute the earlier clone of the same allocation in the decoded result.
  action oldClone = advance(false,true)
  val safe = s.phase != "done" or (s.known.nth(1) == 3 and s.atBook == 2 and s.selected == 2)
  val noInitialChain = not(s.phase == "done" and not(s.late) and s.turns.length() == 2)
  val noLateChain = not(s.phase == "done" and s.late and s.turns.length() == 2)
  val noMissingLock = not(s.phase == "done" and not(s.late) and s.turns.length() == 2 and
    s.selected == 2 and s.known.nth(1) == 1)
  val noOldAlias = not(s.phase == "done" and s.turns.length() == 2 and s.selected == 1 and
    s.selectedAddress == s.address and s.known.nth(1) == 3)
  val noSelfHandoff = not(s.phase == "done" and s.late and s.writerLock == 1 and
    s.turns.length() == 2 and s.known.nth(1) == 3)
}
