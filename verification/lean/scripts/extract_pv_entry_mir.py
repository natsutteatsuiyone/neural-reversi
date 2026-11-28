"""Internal PV entry, empty move list, pass, and cached wipeout in the compiled PVS."""

from itertools import product
import re

from mir_common import checked, normal_target, split_call
from extract_nonpv_leaf_mir import final_evaluation, follow_pvs, impl_constant, parameters, pvs_body
from extract_root_wipeout import constants


def positive(body, traits, cut, mobile, opponent, wipes, to_owner=None):
    env = dict(_1="ctx", _2="board", _3="depth", _4="alpha", _5="beta", _6="thread", _7=cut)
    calls = []
    moves = ("new", "board")
    passed = ("switch", "board")
    candidate = ("wipe", moves)
    square = ("square", candidate)
    bb, visited = 0, set()

    def operand(s):
        if s in traits:
            return traits[s]
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(_\d+)", s):
            return env[m[1] or m[2] or m[3]]
        if s == "&(*((*_1).5: &transposition_table::TranspositionTable))":
            return "tt"
        if s == "copy ((*_1).10: u32)":
            return "empties"
        if m := re.fullmatch(r"Eq\(move (_\d+), const 0_(usize|u32)\)", s):
            if m[2] == "usize":
                checked(env[m[1]] == ("count", moves), "unexpected move count")
                return not mobile
            checked(env[m[1]] == "depth", "unexpected depth test")
            return False
        if m := re.fullmatch(r"discriminant\((_\d+)\)", s):
            checked(env[m[1]] == candidate, "unexpected discriminator")
            return int(wipes)
        if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: square::Square\)", s):
            checked(env[m[1]] == candidate and wipes, "wrong wipeout move")
            return square
        raise ValueError(f"unexpected PV entry operand {s}")

    before = [("prepare", "ctx"), ("hash", "board"), ("prefetch", "tt", ("hash", "board")), moves, ("count", moves)]
    while bb not in visited:
        visited.add(bb)
        block = body[bb]
        for op in block.statements:
            if m := re.fullmatch(r"PlaceMention\((_\d+)\);", op):
                checked(m[1] in env, "uninitialized mention")
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", op)
            checked(m, f"unexpected PV instruction bb{bb}: {op}")
            env[m[1]] = operand(m[2])
        end = block.end
        if end == "return;":
            if not mobile:
                expected = [*before, passed, ("mobile", passed)]
                if opponent:
                    negB, negA = ("neg", "beta"), ("neg", "alpha")
                    child = ("child", "ctx", passed, "depth", negB, negA, "thread", False)
                    result = ("neg", child)
                    expected += [("beginPass", "ctx"), negB, negA, child, result, ("endPass", "ctx")]
                    name = "pass"
                else:
                    result = ("terminal", "board", "empties")
                    expected += [result]
                    name = "terminal"
            else:
                checked(wipes, "ordinary move unexpectedly returned")
                result = 16384
                expected = [*before, candidate, ("updatePv", "ctx", square)]
                name = "wipeout"
            checked(env["_0"] == result and calls == expected, f"wrong PV {name} return or effects: {env['_0']}, {calls}")
            return name
        if c := split_call(end):
            if c[2] == "TranspositionTable::probe":
                args = [operand(arg) for arg in c[3].split(", ")]
                checked(mobile and not wipes and calls == [*before, candidate] and
                        args == ["tt", "board", ("hash", "board")], "wrong ordinary path to TT probe")
                if to_owner is not None:
                    return to_owner(bb, env)
                return "continue"
            names = {"SearchContext::<'_>::prepare_pv": "prepare", "Board::hash": "hash",
                     "TranspositionTable::prefetch": "prefetch", "MoveList::new": "new", "MoveList::count": "count",
                     "Board::switch_players": "switch", "Board::has_legal_moves": "mobile",
                     "SearchContext::<'_>::update_pass": "beginPass", "SearchContext::<'_>::undo_pass": "endPass",
                     "<Score as Neg>::neg": "neg", "search::<NT, SS>": "child", "MoveList::wipeout_move": "wipe",
                     "Board::solve": "terminal", "SearchContext::<'_>::update_pv": "updatePv"}
            checked(c[2] in names, f"unexpected PV call {c[2]}")
            call = (names[c[2]], *(operand(arg) for arg in c[3].split(", ")))
            calls.append(call)
            env[c[1]] = opponent if call == ("mobile", passed) else call
            bb = normal_target(body, c[4], c[5])
        elif m := re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];", end):
            bb = normal_target(body, m[1] or m[2])
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];", end):
            value = operand(m[1])
            checked(isinstance(value, int), "unresolved branch")
            bb = normal_target(body, m[3] if value == int(m[2]) else m[4])
        else:
            raise ValueError(f"unexpected internal PV terminator bb{bb}: {end}")
    raise ValueError("PV entry loop")


def verify(dump, repo):
    body = pvs_body(dump)
    zero = parameters(repo, dump, True)
    final_evaluation(repo, dump)
    root = impl_constant(repo, dump, "node_type", "impl NodeType for PV {", "ROOT_NODE", "bool")
    checked(root == "false", "PV is marked root")
    traits = dict(constants(repo), **zero)
    traits["const <NT as search::node_type::NodeType>::ROOT_NODE"] = False
    checked(traits["const score::Score::MAX"] == 16384, "unexpected max score")
    for cut in (False, True):
        follow_pvs(body, zero, 0, cut, True)
        for mobile, opponent, wipes in product((False, True), repeat=3):
            actual = positive(body, traits, cut, mobile, opponent, wipes)
            expected = ("wipeout" if wipes else "continue") if mobile else "pass" if opponent else "terminal"
            checked(actual == expected, f"wrong PV branch {actual}")
    return """module

@[expose] public section

namespace NeuralReversi.PvsParallel.PvEntryMir

inductive Action where
  | zero | terminal | pass | wipeout | continueSearch
  deriving DecidableEq

def dispatch (depth : Nat) (mobile opponent wipes : Bool) : Action :=
  if depth = 0 then .zero else if mobile then
    if wipes then .wipeout else .continueSearch
  else if opponent then .pass else .terminal

end NeuralReversi.PvsParallel.PvEntryMir
"""
