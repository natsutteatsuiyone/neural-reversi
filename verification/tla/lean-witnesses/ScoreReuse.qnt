// Owner's relaxed best_score load at the second use of one split-point slot.
// Each generation starts just after the owner finished its own participation,
// with two booked helpers. Each helper reports once and resets its bit once.
// The owner takes no further split-point lock before its returning score load.
module ScoreReuse {
  type Origin = { generation: int, helper: int }
  type Report = { origin: Origin, score: int }
  type State = {
    phase: str,
    generation: int,
    words: List[int],
    origins: List[Origin],
    reports: List[Report],
    publisher: Origin,
    first: int,
    last: int,
    published: int,
    lower: int,
    source: int,
    unreported: Set[int],
    active: Set[int],
  }

  var s: State
  pure val helpers = Set(1, 2)
  pure val initialScores = Set(-512, 0, 512)

  pure def origin(g: int, h: int): Origin = { generation: g, helper: h }

  action init = s' = {
    phase: "working", generation: 1, words: [-512], first: 0, last: -1,
    origins: [origin(1, 0)], reports: [], publisher: origin(0, 0),
    published: 0, lower: 0, source: -1, unreported: helpers, active: helpers,
  }

  // The split lock serializes strict improvements and helper resets.
  action report(h: int, score: int): bool = {
    val who = origin(s.generation, h)
    val improves = score > s.words.nth(s.words.length() - 1)
    all {
      s.phase == "working",
      s.unreported.contains(h),
      s' = { ...s,
        words: if (improves) s.words.append(score) else s.words,
        origins: if (improves) s.origins.append(who) else s.origins,
        reports: s.reports.append({ origin: who, score: score }),
        unreported: s.unreported.exclude(Set(h)),
      },
    }
  }

  action reset(h: int, release: bool): bool = all {
    s.phase == "working",
    s.active.contains(h),
    not(s.unreported.contains(h)),
    s' = { ...s,
      active: s.active.exclude(Set(h)),
      published: if (release) s.words.length() - 1 else s.published,
    },
  }

  // All remaining mask changes are RMWs. An Acquire of the final zero
  // synchronizes with every preceding Release reset in this release sequence.
  // The owner's own initializer is ordered even for a Relaxed zero load.
  action waitZero(acquire: bool): bool = all {
    s.phase == "working",
    s.active == Set(),
    s' = { ...s, phase: "zero", last: s.words.length() - 1,
           lower: if (acquire) s.published else s.first },
  }

  action reuse(seed: int): bool = all {
    s.phase == "zero",
    s.generation == 1,
    s' = { ...s, phase: "working", generation: 2, words: s.words.append(seed),
           origins: s.origins.append(origin(2, 0)),
           first: s.words.length(), published: s.words.length(), last: -1,
           unreported: helpers, active: helpers },
  }

  // Include the following initializer in modification order before choosing
  // the second load's reads-from edge. The owner executes that initializer
  // AFTER the load. Enumeration order here is not Rust program order.
  action nextInitializer(seed: int): bool = all {
    s.phase == "zero",
    s.generation == 2,
    s' = { ...s, phase: "choose", words: s.words.append(seed), origins: s.origins.append(origin(3, 0)) },
  }

  action chooseSource(i: int, orderedReuse: bool, policy: str): bool = {
    val word = s.words.nth(i)
    val matches = s.reports.select(r => r.score == word)
    val current = matches.select(r => r.origin.generation == s.generation)
    val publisher =
      if (policy == "lastEqual" and current.length() > 0) current.nth(current.length()-1).origin
      else if (policy == "firstAcross" and matches.length() > 0) matches.head().origin
      else s.origins.nth(i)
    all {
      s.phase == "choose",
      // Write-read coherence against scores before the returning load.
      s.lower <= i,
      // Read-write coherence against the following initializer.
      not(orderedReuse) or i < s.words.length() - 1,
      s' = { ...s, phase: "done", source: i, publisher: publisher },
    }
  }

  action advance(acquire: bool, release: bool, orderedReuse: bool, policy: str): bool = any {
    { nondet h = helpers.oneOf(); nondet score = Set(0, 512).oneOf(); report(h, score) },
    { nondet h = helpers.oneOf(); reset(h, release) },
    waitZero(acquire),
    { nondet seed = initialScores.oneOf(); reuse(seed) },
    { nondet seed = initialScores.oneOf(); nextInitializer(seed) },
    { nondet i = 0.to(s.words.length() - 1).oneOf(); chooseSource(i, orderedReuse, policy) },
    all { s.phase == "done", s' = s },
  }

  action step = advance(true, true, true, "physical")
  action relaxedWait = advance(false, true, true, "physical")
  action relaxedReset = advance(true, false, true, "physical")

  // Dropping the owner program-order edge is a fault in the reuse protocol.
  action unorderedReuse = advance(true, true, false, "physical")

  // Diagnostic reconstructions keep the score and its physical read source intact.
  action lastEqual = advance(true, true, true, "lastEqual")
  action firstAcross = advance(true, true, true, "firstAcross")

  val currentReports = s.reports.select(r => r.origin.generation == s.generation)
  val safe = if (s.phase == "done") {
    val word = s.words.nth(s.last)
    val expected = if (word == s.words.nth(s.first)) origin(s.generation, 0)
      else currentReports.select(r => r.score == word).head().origin
    s.source == s.last and s.publisher == expected
  } else true

  val tie = if (s.phase == "done" and s.last > s.first and currentReports.length() == 2)
    currentReports.nth(0).score == currentReports.nth(1).score and s.source == s.last
    else false
  val firstHelperWon = if (tie) currentReports.head().origin.helper == 1 else false
  val secondHelperWon = if (tie) currentReports.head().origin.helper == 2 else false
  val noFirstHelperWon = not(firstHelperWon)
  val noSecondHelperWon = not(secondHelperWon)
  val tieOwner = if (tie) s.publisher == currentReports.head().origin else true
  // Equal helper number and payload across two activations of the same slot.
  val noPriorPublisher = if (tie)
    not(s.publisher.generation == 1 and s.publisher.helper == currentReports.head().origin.helper and
        s.origins.indices().exists(i => s.origins.nth(i) == s.publisher and s.words.nth(i) == s.words.nth(s.source)))
    else true

  // The returned score and the next initializer have different store identities.
  val samePayload = if (s.phase == "done") {
    s.source == s.last and s.first < s.last and s.words.nth(s.last) == s.words.nth(s.last + 1)
  } else false

  val seedOnly = s.phase == "done" and s.source == s.first and s.first == s.last

  val aliasWrong = if (s.phase == "done") {
    s.source == s.last + 1 and s.first < s.last and s.words.nth(s.last) == s.words.nth(s.source)
  } else false

  val noEqualPayload = not(samePayload)
  val noSeedOnly = not(seedOnly)
  val noEqualSubstitution = not(aliasWrong)
}
