"""Emit direct boundary call ordinals alongside the concrete null-window poller type."""

from cache_owner_mir import Body, THREAD
from mir_common import checked
from extract_cache_owner import GET, STRATEGY, routing


def render(dump, repo):
    stem = STRATEGY + "{impl#1}-shallow_search"
    body = Body(dump, stem)
    calls = {block: function for block, function, _, _ in body.calls}
    selectors = {
        GET: "access",
        "shallow_search::<5>": "shallow5",
        "shallow_search::<6>": "shallow6",
        "null_window_search::<" + THREAD + ">": "poll",
        "null_window_search::<IgnoreSplitPointCutoff>": "ignore",
    }
    sites = {}
    for ordinal, (block, target, _) in enumerate(routing(dump, stem)):
        function = calls[block]
        label = selectors.get(function)
        checked(label is not None and label not in sites, "unexpected boundary callee: " + function)
        sites[label] = f"({ordinal},.{target})"
    checked(sites.keys() == set(selectors.values()), "missing boundary call")
    first = sites["access"]
    return "\n".join([
        "module", "public import NeuralReversi.EndgameSearch.CacheHistory.Boundary",
        "public import NeuralReversi.EndgameSearch.CacheOwner.Syntax",
        "",
        "@[expose] public section", "", "namespace NeuralReversi.EndgameSearch.CacheHistory.NativeTrace",
        "",
        "def expectedSites : Arm → Option (List (Nat × CacheOwner.Node))",
        "  | .failed => none",
        "  | .table _ => some []",
        f"  | .fixed _ => some [{first}]",
        f"  | .shallow six => some [{first},if six then {sites['shallow6']} else {sites['shallow5']}]",
        f"  | .deep poll => some [{first},if poll then {sites['poll']} else {sites['ignore']}]",
        "",
        "end NeuralReversi.EndgameSearch.CacheHistory.NativeTrace",
        "",
    ])
