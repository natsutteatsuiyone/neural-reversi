// One relaxed load of one TT payload field, three successful writers and an
// initial word. The recorded writes are in modification order. The read's
// position on the recorded tape orders it only against its own worker's writes.
module TTReadback {
  type State = {
    workers: List[int], words: List[int], cut: int, source: int, candidates: List[int],
  }
  var s: State

  action init = s' = { workers: [], words: [0], cut: -1, source: -1, candidates: [] }

  action write(w: int, word: int): bool = all {
    s.workers.length() < 3,
    s' = { ...s, workers: s.workers.append(w), words: s.words.append(word) },
  }
  action issue = all {
    s.cut == -1,
    s' = { ...s, cut: s.workers.length() },
  }

  // Place the read after write i in modification order. Its projection onto
  // worker 0 must have the same program order as that worker's recorded tape.
  // There are no cross-worker synchronization edges to this read in the litmus.
  def ownCount(k: int): int =
    range(0, k).select(j => s.workers.nth(j) == 0).length()
  def legal(i: int): bool = ownCount(i) == ownCount(s.cut)

  // Decoder sees the loaded word. Positive indices identify physical stores;
  // index zero identifies initialization. Tape numbering alone gives no HB.
  def decode(seen: int, policy: str): List[int] = {
    val own = range(1, s.words.length()).select(k => s.workers.nth(k-1) == 0)
    val before = own.select(k => k <= s.cut)
    val after = own.select(k => k > s.cut)
    range(0, s.words.length()).select(i =>
      s.words.nth(i) == seen and
      before.indices().forall(j => before.nth(j) <= i) and
      (policy == "omitUpper" or after.indices().forall(j => i < after.nth(j))) and
      (policy != "latestOnly" or i == s.cut))
  }

  action readback(i: int, policy: str): bool = all {
    s.cut >= 0,
    s.workers.length() == 3,
    s.source == -1,
    legal(i),
    s' = { ...s, source: i, candidates: decode(s.words.nth(i), policy) },
  }
  action advance(policy: str): bool = any {
    { nondet w = Set(0, 1).oneOf(); nondet v = Set(0, 1).oneOf(); write(w, v) },
    issue,
    { nondet i = 0.to(s.words.length()-1).oneOf(); readback(i, policy) },
    all { s.source >= 0, s' = s },
  }
  action step = advance("both")
  action latestOnly = advance("latestOnly")
  action omitUpper = advance("omitUpper")

  val done = s.source >= 0
  val exact = if (done) {
    val seen = s.words.nth(s.source)
    val allowed = range(0, s.words.length()).select(i => legal(i) and s.words.nth(i) == seen)
    s.candidates == allowed
  } else true

  val stale = if (done and s.source > 0 and s.source < s.cut)
    s.workers.nth(s.cut-1) != 0 and s.words.nth(s.source) == 1 and s.words.nth(s.cut) == 0
    else false
  val ambiguous = if (done and s.source > 0)
    s.words.nth(s.source) == 1 and s.candidates.select(i => i > 0).length() >= 2
    else false
  // Require identical words on both sides of the nearest same-worker bracket.
  val bracket = if (done and s.cut > 0 and s.cut < s.workers.length())
    s.source == s.cut and s.workers.nth(s.cut-1) == 0 and s.workers.nth(s.cut) == 0 and
    s.workers.indices().exists(j => s.workers.nth(j) == 1) and
    s.words.nth(s.cut) == 1 and s.words.nth(s.cut+1) == 1
    else false

  val noStale = not(stale)
  val noAmbiguity = not(ambiguous)
  val noBracket = not(bracket)
  val keepsStale = stale implies s.candidates.indices().exists(j => s.candidates.nth(j) == s.source)
  val respectsBracket = bracket implies s.candidates == [s.source]
}
