// Two worker stacks each have an active parent making two synchronous PVS calls.
// All four calls use the same board, depth, node kind, cut flag and [-1, 0] window.
// Each normal return independently chooses one of two possible fail-high bounds.
module CallReturns {
  type Receipt = { id: int, score: int }
  type State = {
    // Array position allocates the child ID at entry. Parents 0 and 1 are already active.
    begins: List[int],
    returns: List[Receipt],
    consumed: List[List[int]],
  }

  var s: State
  pure val workers = Set(0, 1)
  action init = s' = { begins: [], returns: [], consumed: [[], []] }

  def children(p: int): List[int] =
    range(0, s.begins.length()).select(i => s.begins.nth(i) == p)
  def receipts(p: int): List[Receipt] =
    s.returns.select(r => s.begins.nth(r.id) == p)
  def tape(p: int): List[int] =
    children(p).select(i => s.returns.indices().exists(j => s.returns.nth(j).id == i))
  def consumed(p: int): List[int] = s.consumed.nth(p)

  action beginCall(p: int): bool = all {
    children(p).length() < 2,
    children(p).length() == consumed(p).length(),
    s' = { ...s, begins: s.begins.append(p) },
  }

  action leaveCall(p: int, score: int): bool = all {
    children(p).length() > receipts(p).length(),
    s' = { ...s, returns: s.returns.append({
      id: children(p).nth(receipts(p).length()), score: score,
    }) },
  }

  // Each observation consumes a request whose actual direct child has returned.
  // Faults either drop its ordinal or take the first operand match across all parents.
  action consume(p: int, policy: str): bool = {
    val n = consumed(p).length()
    all {
      n < receipts(p).length(),
      s' = { ...s, consumed: s.consumed.replaceAt(p, consumed(p).append(
        if (policy == "ordinal") tape(p).nth(n)
        else if (policy == "firstInParent") tape(p).head()
        else s.returns.head().id)) },
    }
  }

  action advance(policy: str): bool = any {
    { nondet p = workers.oneOf(); beginCall(p) },
    { nondet p = workers.oneOf(); nondet score = Set(512, 1024).oneOf(); leaveCall(p, score) },
    { nondet p = workers.oneOf(); consume(p, policy) },
    all { workers.forall(p => consumed(p).length() == 2), s' = s },
  }
  action step = advance("ordinal")
  action firstInParent = advance("firstInParent")
  action firstAcrossParents = advance("firstAcrossParents")

  val correctID = workers.forall(p => consumed(p).indices().forall(n =>
    consumed(p).nth(n) == receipts(p).nth(n).id))
  val correctScore = workers.forall(p => consumed(p).indices().forall(n =>
    s.returns.indices().exists(j => {
      val used = s.returns.nth(j)
      used.id == consumed(p).nth(n) and used.score == receipts(p).nth(n).score
    })))
  val safe = correctID and correctScore
  val identityOnEqualScores = correctScore implies correctID

  // An actual overlap with the first two starts on different workers, second
  // starter returning first, and distinct words on both of worker 0's calls.
  val overlap = if (workers.forall(p => consumed(p).length() == 2))
    s.begins.nth(0) == 0 and s.begins.nth(1) == 1 and s.returns.nth(0).id == 1
      and receipts(0).nth(0).score != receipts(0).nth(1).score
    else false
  val noOverlap = not(overlap)
}
