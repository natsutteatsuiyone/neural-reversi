"""Check the executable suspended calls in SerialCalls/Sites against pinned PVS MIR."""

import argparse
from itertools import product
from pathlib import Path
import re
import tempfile

from cache_owner_mir import PASS
from extract_header import initial, lmr_returns_zero, transition
from extract_nonpv_leaf_mir import impl_constant
from extract_owner_sites import PVS, role_blocks
from extract_pvs_context_mir import Graph, bracket
from extract_root_wipeout import FILENAME, constants
from extract_single import SIGNATURE, check_score_operators
from mir_common import CALL, checked


REPO = Path(__file__).resolve().parents[3]
SITES = Path("verification/lean/NeuralReversi/PvsParallel/SerialCalls/Sites.lean")
REDUCTION = f"reversi_core.search-pvs-compute_lmr_reduction{PASS}"
PV = "const <NT as search::node_type::NodeType>::PV_NODE"
ROOT = PV.replace("PV_NODE", "ROOT_NODE")
NEG = "<Score as Neg>::neg"


class Trace:
    """The existing header interpreter emits an aggregate result. Keep every visited PC."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = []

    def __getitem__(self, pc):
        self.path.append(pc)
        return self.cfg[pc]


def specializations(dump, repo):
    checked(impl_constant(repo, dump, "strategy", "impl SearchStrategy for EndGameStrategy {",
                          "IS_ENDGAME", "bool") == "true", "unexpected EndGameStrategy")
    traits = constants(repo)
    result = {}
    for name, node in (("Root", "rootID"), ("PV", "pvID"), ("NonPV", "nonPvID")):
        values = traits.copy()
        for trait, key in (("PV_NODE", PV), ("ROOT_NODE", ROOT)):
            value = impl_constant(repo, dump, "node_type", f"impl NodeType for {name} {{", trait, "bool")
            checked(value in ("false", "true"), f"unknown {name} {trait}")
            values[key] = value == "true"
        checked(values[PV] == (name != "NonPV") and values[ROOT] == (name == "Root"), "unexpected node type")
        result[node] = values
    return result


def boolean(path, operand, cut_reg, cut):
    """Read the actual reaching definition, including a join specialized by the trace."""
    if operand == cut_reg:
        return cut
    if operand in ("const true", "const false"):
        return operand == "const true"
    if m := re.fullmatch(r"(?:copy|move) (.+)", operand):
        return boolean(path, m[1], cut_reg, cut)
    if m := re.fullmatch(r"Not\((.+)\)", operand):
        return not boolean(path, m[1], cut_reg, cut)
    checked(re.fullmatch(r"_\d+", operand), f"unsupported cut operand {operand}")
    for i in reversed(range(len(path))):
        if m := re.fullmatch(operand + r" = (.+);", path[i]):
            return boolean(path[:i], m[1], cut_reg, cut)
    raise ValueError(f"no reaching definition for {operand}")


def cut_formula(cases):
    for lean, fn in (("(!pv && !cut)", lambda pv, cut: not pv and not cut),
                     ("!cut", lambda pv, cut: not cut), ("false", lambda pv, cut: False)):
        if all(v == fn(pv, cut) for (pv, cut), v in cases.items()):
            return lean
    raise ValueError(f"unsupported child cut formula {cases}")


def request(expr):
    minus = ("field", ("SubWithOverflow", "arg3", "const 1_u32"), 0)
    exact = ("field", ("SubWithOverflow", minus, ("compute_lmr_reduction::<NT, SS>",)), 0)
    names = {"arg3": "depth", minus: "depth - 1", exact: "depth - 1",
             (NEG, "arg5"): "-beta", (NEG, "arg4"): "-alpha",
             (NEG, ("<Score as Add<i32>>::add", "arg4", "const 1_i32")): "-(alpha + 1)"}
    checked(expr in names, f"unsupported native request field {expr}")
    return names[expr]


def inspect(dump, traits):
    body, roles = role_blocks(dump)
    checked(SIGNATURE in body.source.splitlines(), "unexpected PVS function type")
    lmr_returns_zero((dump / REDUCTION).read_text(encoding="utf-8"))
    g = Graph(dump, PVS, 7)
    kinds = bracket(dump, PVS)
    calls = {pc: (name, args, ret) for pc, name, args, ret in body.calls if name.startswith("search::<")}
    checked(set(calls) == set(roles.values()) == set(kinds), "unaccounted recursive call")
    regs, head = initial(body.source, body.cfg)
    # Recover the two incoming values of all_node at the join immediately after entry.
    entry = re.fullmatch(r"(_\d+) = (.+);", g.cfg[0][0])
    checked(entry, "unexpected PVS entry instruction")
    pv_reg, arm = entry.groups()
    branch = re.fullmatch(rf"switchInt\(move {pv_reg}\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", g.cfg[0][-1])
    checked(arm == PV and branch and len(g.cfg[0]) == 2, "wrong pass node specialization")
    arms = [int(n) for n in branch.groups()]
    checked(g.cfg[arms[0]][-1] == g.cfg[arms[1]][-1] and
            re.fullmatch(r"goto -> bb\d+;", g.cfg[arms[0]][-1]), "pass arms do not join")
    checked({pc for pc, ins in body.ops if ins.startswith(regs["all_node"] + " = ")} == set(arms)
            and not g.defs.get(regs["cut_node"]), "pass-cut reaching definitions")

    by_pc = {pc: role for role, pc in roles.items()}
    cuts = {role: {} for role in roles}
    for node, t in traits.items():
        pv = t[PV]
        # In the single-worker pool can_split is false and claim is Untracked. Every
        # other independent header predicate is free, including the two depth gates.
        for first, cut, improved, deep, many, fresh, speculative, tt_pv in product((False, True), repeat=8):
            trace = Trace(body.cfg)
            flags = (True, first, deep, many, False, False, fresh, improved)
            out = transition(trace, regs, head, flags, t, 2, node, cut, tt_pv, speculative)
            issued = [by_pc[n] for n in trace.path if n in by_pc]
            expected = ["wide"] if pv and first else ["scout"] + (["wide"] if pv and improved else [])
            checked(issued == expected, f"{node}/{first}/{improved}: returned through {issued}")
            flags = ("true" if "scout" in issued else "false") + " " + ("true" if "wide" in issued else "false")
            word = "-pv" if issued[-1] == "wide" else "-scout"
            checked(out == f".child (idx + 1) {flags} ({word})", "wrong selected return register")
            for role, prefix in (
                ("passChild", [arms[int(pv)], roles["passChild"]]), ("presearch", [roles["presearch"]]),
                *((role, trace.path[:trace.path.index(roles[role]) + 1]) for role in issued)):
                pc = roles[role]
                path = [ins for n in prefix for ins in g.cfg[n]]
                v = boolean(path, calls[pc][1][6], regs["cut_node"], cut)
                checked(cuts[role].setdefault((pv, cut), v) == v, f"{role} cut depends on previous call")
    # A wide child is only reached from PV. Require its cut flag to be literal false.
    checked(calls[roles["wide"]][1][-1] == "const false", "wide cut is not false")

    descriptors = {}
    for role, pc in roles.items():
        _, args, dest = calls[pc]
        ret = re.fullmatch(CALL, g.cfg[pc][-1])
        after = g.cfg[int(ret[4])]
        neg = re.fullmatch(CALL, after[-1])
        checked(len(after) == 1 and neg and neg[2] == NEG and neg[3] == "move " + dest,
                f"{role}: resume did not negate this call's return word")
        if role == "passChild":
            undo = re.fullmatch(CALL, g.cfg[int(neg[4])][-1])
            checked(undo and undo[2] == "SearchContext::<'_>::undo_pass" and
                    g.cfg[int(undo[4])][0] == f"_0 = copy {neg[1]};", "pass returned a different word")
        if role == "reducedRetry":
            continue
        checked(len(args) == 7 and g.val(args[0]) == "arg1" and g.val(args[5]) == "arg6",
                f"{role}: changed receiver or thread")
        # bracket() verified next is exactly the switched/moved board paired with the live context.
        pv = {".pass": "pv", "(.move true)": "true", "(.move false)": "false"}[kinds[pc]]
        cut = "false" if role == "wide" else cut_formula(cuts[role])
        depth, alpha, beta = (request(g.val(args[index])) for index in (2, 3, 4))
        descriptors[role] = f"⟨next, {depth}, {pv}, {cut}, {alpha}, {beta}⟩"
    return descriptors


def render(dump, traits):
    return """import NeuralReversi.PvsParallel.SerialCalls

namespace NeuralReversi.PvsParallel.SerialCalls.Sites

/-- The native loop suspends at each recursive call. A wide re-search consumes
    the scout's signed return before deciding whether to issue a second call. -/
def ownerWith (pv first : Bool) (alpha : Int) (scout wide : Req)
    (finish : Option Int → Option Int → Int → Program Req A) : Program Req A :=
  if pv && first then
    .call wide fun w => finish none (some w) (-w)
  else
    .call scout fun s =>
      if pv && decide (alpha < -s) then
        .call wide fun w => finish (some s) (some w) (-w)
      else finish (some s) none (-s)

def owner (pv first : Bool) (alpha : Int) (scout wide : Req) : Program Req Int :=
  ownerWith pv first alpha scout wide fun _ _ v => .pure v

def scoutRequest (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (cut : Bool) (alpha : Int) : Request :=
  {scout}

def wideRequest (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (alpha beta : Int) : Request :=
  {wide}

def ownerMove (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (pv cut first : Bool) (alpha beta : Int) : Program Request Int :=
  owner pv first alpha (scoutRequest next depth cut alpha) (wideRequest next depth alpha beta)

def presearchWith (request : Req) (finish : Int → Program Req A) : Program Req A :=
  .call request fun w => finish (-w)

def presearch (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (cut : Bool) (alpha : Int) : Program Request Int :=
  presearchWith {presearch} .pure

def pass (next : PvsBoardTree.Position) (depth : ExactDepth.Depth)
    (pv cut : Bool) (alpha beta : Int) : Program Request Int :=
  .call {passChild} fun w => .pure (-w)

end NeuralReversi.PvsParallel.SerialCalls.Sites
""".format(**inspect(dump, traits))


def check(dump, repo=REPO):
    traits = specializations(dump, repo)
    check_score_operators(repo)
    expected = (repo / SITES).read_text(encoding="utf-8")
    checked(render(dump, traits) == expected, "SerialCalls/Sites.lean differs from native calls")
    body, roles = role_blocks(dump)
    calls = {pc: (name, args, dest) for pc, name, args, dest in body.calls}

    def replace_in(pc, old, new):
        pattern = re.compile(rf"(^    bb{pc}: \{{\n)(.*?)(^    \}})", re.M | re.S)
        match = pattern.search(body.source)
        checked(match and match[2].count(old) == 1, f"no mutation anchor {old} at bb{pc}")
        return body.source[:match.start(2)] + match[2].replace(old, new) + body.source[match.end(2):]

    scout, pre, wide, passing = (roles[r] for r in ("scout", "presearch", "wide", "passChild"))
    names, header = initial(body.source, body.cfg)
    traced = Trace(body.cfg)
    transition(traced, names, header, (True, False, False, True, False, False, False, True),
               traits["pvID"], 2, "pvID")
    between = traced.path[traced.path.index(scout) + 1:traced.path.index(wide)]
    high, = [pc for pc in between if pc in calls and calls[pc][0] == "<Score as PartialOrd>::gt"]
    ret = int(re.fullmatch(CALL, body.cfg[wide][-1])[4])
    # Only one conditional between the scout's signed comparison and the wide arm.
    conditional = int(re.fullmatch(CALL, body.cfg[high][-1])[4])
    test = re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", body.cfg[conditional][-1])
    checked(test, "expected the PV re-search branch")
    retry_cmp, = [pc for pc in between if any("Gt(" in op and "0_u32)" in op for op in body.cfg[pc])]
    guard = re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", body.cfg[retry_cmp][-1])
    checked(guard, "expected the LMR retry branch")
    pass_arm, = [pc for pc, ins in body.ops if ins == f"{names['all_node']} = const false;"]
    # Run native changes against the same executable Lean artifact. An altered stage must either
    # violate a checked data-flow constraint or produce a different Lean request/continuation.
    changes = {
        "presearch uses parent cut": replace_in(pre, calls[pre][1][6], "copy _7"),
        "wide cut": replace_in(wide, "const false", "const true"),
        "presearch uses stale window": replace_in(pre, calls[pre][1][3], calls[pre][1][4]),
        "pass counts a move": replace_in(passing, calls[passing][1][2], "const 0_u32"),
        "scout returned instead of wide": replace_in(ret, "move " + calls[wide][2], "copy " + calls[scout][2]),
        "research is non-strict": replace_in(high, "PartialOrd>::gt", "PartialOrd>::ge"),
        "wrong continuation target": replace_in(wide, f"return: bb{ret}", f"return: bb{ret+1}"),
        "wide only after failed scout": replace_in(conditional, test[0], test[0].replace(
            f"0: bb{test[2]}, otherwise: bb{test[3]}", f"0: bb{test[3]}, otherwise: bb{test[2]}")),
        "endgame reduced retry": replace_in(retry_cmp, "0: bb" + guard[2], "0: bb" + guard[3]),
        "PV pass marked cut": replace_in(pass_arm, "const false", "const true"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        test = Path(tmp)
        (test / REDUCTION).symlink_to((dump / REDUCTION).resolve())
        for label, source in changes.items():
            (test / FILENAME).write_text(source, encoding="utf-8")
            try:
                different = render(test, traits) != expected
            except (ValueError, KeyError):
                different = True
            checked(different, f"accepted mutant: {label}")
    return len(changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    print(f"Checked compiled child requests, continuations and {check(parser.parse_args().dump)} mutants.")
