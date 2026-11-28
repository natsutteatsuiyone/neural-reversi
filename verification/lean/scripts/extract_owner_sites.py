"""Bind the exact PVS call roles to the indexed call edges of the compiled Rust function."""

from pathlib import Path
import tempfile

from cache_owner_mir import Body, PASS
from mir_common import checked
from extract_cache_owner import routing
from extract_header import initial

PVS = "search-pvs-search"
SCOUT = "search::<NonPV, SS>"
WIDE = "search::<PV, SS>"
PASS_CHILD = "search::<NT, SS>"


def role_blocks(dump):
    body = Body(dump, PVS)
    _, header = initial(body.source, body.cfg)
    calls = {pc: name for pc, name, _, _ in body.calls if name in (SCOUT, WIDE, PASS_CHILD)}
    before = body.reachable(0, stop_at={header})
    first = body.reachable(header, stop_at=calls)

    def only(name, pcs, role):
        matches = [pc for pc, callee in calls.items() if callee == name and pc in pcs]
        checked(len(matches) == 1, f"expected one {role} call, got {matches}")
        return matches[0]

    passing = only(PASS_CHILD, before, "pass")
    pre = only(SCOUT, before, "presearch")
    scout = only(SCOUT, first, "owner scout")
    wide = only(WIDE, first, "owner wide")
    retry = only(SCOUT, body.reachable(header) - first, "reduced scout retry")
    roles = dict(passChild=passing, presearch=pre, scout=scout, reducedRetry=retry, wide=wide)
    checked(set(roles.values()) == set(calls), "unclassified recursive PVS call")
    return body, roles


def render(dump):
    _, roles = role_blocks(dump)
    edges = {pc: ordinal for ordinal, (pc, target, flow) in enumerate(routing(dump, PVS))
             if target == "pvs" and flow == "forward"}
    checked(set(edges) == set(roles.values()), "recursive call roles disagree with worker receiver transfers")
    return "\n".join([
        "module", "public import NeuralReversi.EndgameSearch.CacheOwner.Program", "",
        "@[expose] public section", "", "namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites", "",
        "inductive Role where", *[f"  | {role}" for role in roles], "  deriving DecidableEq", "",
        "def Role.site : Role → Nat", *[f"  | .{role} => {edges[pc]}" for role, pc in roles.items()], "",
        "def Role.block : Role → Nat", *[f"  | .{role} => {pc}" for role, pc in roles.items()], "",
        "end NeuralReversi.PvsParallel.NativeFrames.OwnerSites", "",
    ])


def check_mutants(dump):
    body, roles = role_blocks(dump)
    original = render(dump)
    mutations = {"passChild": SCOUT, "presearch": WIDE, "scout": WIDE, "wide": SCOUT, "reducedRetry": PASS_CHILD}
    for role, changed in mutations.items():
        pc = roles[role]
        name = next(fn for n, fn, _, _ in body.calls if n == pc)
        term = body.cfg[pc][-1]
        line = term.replace(name, changed)
        checked(line != term and body.source.count(term) == 1, f"failed to mutate {role} call")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / f"reversi_core.{PVS}{PASS}").write_text(body.source.replace(term, line), encoding="utf-8")
            try:
                result = render(target)
            except ValueError:
                continue
            checked(result != original, f"accepted incorrect {role} target")
    return len(mutations)
