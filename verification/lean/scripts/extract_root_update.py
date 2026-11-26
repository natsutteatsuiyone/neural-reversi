"""Extract the root owner's numeric updates between undo and the next loop/TT decision."""

import itertools
import re

from mir_common import checked, split_call
from extract_nonpv_leaf_mir import pvs_body
from extract_root_wipeout import FILENAME, constants


def decode(dump):
    cfg = pvs_body(dump)
    source = (dump / FILENAME).read_text(encoding="utf-8")
    names = ("ctx", "thread", "alpha", "beta", "org_alpha", "best_move", "best_score",
             "mv", "move_count", "score", "claim", "idx", "n_moves", "fresh_end")
    places = {}
    for name in names:
        # The recursive loop's shadowing declarations follow the TT presearch scope.
        found = re.findall(r"^\s*debug " + name + r" => (_\d+);$", source, re.M)
        checked(found, f"missing PVS binding {name}")
        places[name] = found[-1]
    start, = [pc for pc, block in cfg.items() if ">::undo(" in block.end and
              any(f"copy ({places['mv']}.2: square::Square)" in s for s in block.statements)]
    return cfg, places, start


def render(dump, repo):
    cfg, places, start = decode(dump)
    traits = constants(repo)
    checked(traits["const score::Score::MAX"] == 16384, "unsupported score maximum")
    return render_table({bits: post_step(cfg, places, traits, start, bits)
                         for bits in itertools.product((False, True), repeat=5)})


def post_step(cfg, places, traits, start, bits, internal=False):
    first, higher_alpha, higher_best, below_beta, at_max = bits
    env = {reg: name for name, reg in places.items()}
    effects = []
    seen = set()
    pc = start

    def operand(expr):
        if expr in traits:
            return traits[expr]
        if m := re.fullmatch(r"(?:copy|move) \((_\d+)\.2: square::Square\)", expr):
            checked(env[m[1]] == "mv", "square from a different move")
            return "sq"
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(_\d+)", expr):
            return env[m[1] or m[2] or m[3]]
        if m := re.fullmatch(r"Eq\((.+), const 1_usize\)", expr):
            checked(operand(m[1]) == "move_count", "eldest compared using a different index")
            return first
        if m := re.fullmatch(r"Option::<Square>::Some\((.+)\)", expr):
            checked(operand(m[1]) == "sq", "wrong best square")
            return "selected"
        raise ValueError(f"unrecognized root update expression {expr}")

    def finish(cut):
        if internal:
            checked(effects in (["undo", "poll", "drop"], ["undo", "poll", "pv", "drop"]), f"invalid effects {effects}")
        else:
            checked(effects == ["undo", "poll", "row", "drop"], f"invalid effects {effects}")
        alpha, best, pick = [env[places[name]] for name in ("alpha", "best_score", "best_move")]
        checked(alpha in ("alpha", "score") and best in ("best_score", "score")
                and pick in ("best_move", "selected"), "invalid numeric update")
        return ("pv" in effects if internal else visible), alpha == "score", best == "score", pick == "selected", cut

    while pc not in seen:
        seen.add(pc)
        block = cfg[pc]
        checked(not block.cleanup, "unexpected unwind")
        for s in block.statements:
            m = re.fullmatch(r"(_\d+) = (.+);", s)
            checked(m, f"unexpected root update statement {s}")
            dest, expr = m.groups()
            if dest in places.values():
                checked(dest in (places["alpha"], places["best_score"], places["best_move"]),
                        f"unexpected overwrite of {dest}")
            env[dest] = operand(expr)
        term = block.end
        if c := split_call(term):
            args = [operand(arg) for arg in c[3].split(", ")]
            fn = c[2]
            comparisons = {
                ("gt", "score", "alpha"): higher_alpha, ("gt", "score", "best_score"): higher_best,
                ("lt", "score", "beta"): below_beta, ("ge", "score", 16384): at_max,
            }
            if m := re.fullmatch(r"<Score as PartialOrd>::(gt|lt|ge)", fn):
                result = comparisons[(m[1], *args)]
            elif fn == "SearchContext::<'_>::update_root_move":
                checked(args[:3] == ["ctx", "sq", "score"] and len(args) == 4 and args[3] in (0, 1),
                        f"wrong update_root_move arguments {args}")
                effects.append("row")
                visible = bool(args[3])
                result = "unit"
            elif fn == "transposition_table::Bound::classify::<NT>":
                checked(args == [env[places["best_score"]], "org_alpha", "beta"], "wrong TT window or score")
                return finish(True)
            else:
                expected, result, effect = {
                    "SearchContext::<'_>::undo": (["ctx", "sq"], "unit", "undo"),
                    "SearchContext::<'_>::update_pv": (["ctx", "sq"], "unit", "pv"),
                    "<Arc<threading::thread::Thread> as Deref>::deref": (["thread"], "worker", None),
                    "threading::thread::Thread::should_stop": (["worker"], False, "poll"),
                }[fn]
                checked(args == expected, f"{fn} arguments {args}")
                if effect:
                    effects.append(effect)
            env[c[1]] = result
            pc = int(c[4])
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", term):
            flag = operand(m[1])
            checked(flag in (0, 1), f"non-Boolean branch {term}")
            pc = int(m[3] if flag else m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", term):
            checked(env[m[1]] == "claim", "unexpected destructor")
            effects.append("drop")
            pc = int(m[2])
        elif m := re.fullmatch(r"goto -> bb(\d+);", term):
            pc = int(m[1])
        elif m := re.fullmatch(r"falseUnwind -> \[real: bb(\d+), unwind: bb\d+\];", term):
            test = cfg[int(m[1])]
            operands = [re.fullmatch(r"(_\d+) = copy (_\d+);", s) for s in test.statements[:2]]
            checked(len(test.statements) == 3 and all(operands), "unexpected next loop header")
            left, right = operands
            comparison = re.fullmatch(r"(_\d+) = Lt\(move (_\d+), move (_\d+)\);", test.statements[2])
            checked(comparison and list(comparison.groups()[1:]) == [left[1], right[1]]
                    and [env[left[2]], env[right[2]]] == ["idx", "n_moves"]
                    and test.end.startswith(f"switchInt(move {comparison[1]})"), "wrong next loop guard")
            return finish(False)
        else:
            raise ValueError(f"unsupported root update terminator {term}")
    raise ValueError("cycle before root loop/TT decision")


def render_table(table):
    inputs = ["first", "higherAlpha", "higherBest", "belowBeta", "atMax"]

    def expr(rows, i):
        if all(v == rows[0][1] for _, v in rows):
            return str(rows[0][1]).lower()
        low, high = [expr([(k, v) for k, v in rows if k[i] == b], i + 1) for b in (False, True)]
        if low == high:
            return low
        x = inputs[i]
        if (low, high) == ("false", "true"):
            return x
        return f"(if {x} then {high} else {low})"

    fields = [expr([(k, v[i]) for k, v in table.items()], 0) for i in range(5)]
    return "\n".join([
        "namespace NeuralReversi.PvsParallel.RootUpdateMir", "",
        "structure Changes where", "  visible : Bool", "  raisesAlpha : Bool", "  raisesBest : Bool",
        "  takesMove : Bool", "  cuts : Bool", "",
        "def changes (" + " ".join(inputs) + " : Bool) : Changes :=",
        "  ⟨" + ",\n   ".join(fields) + "⟩", "", "end NeuralReversi.PvsParallel.RootUpdateMir", "",
    ])
