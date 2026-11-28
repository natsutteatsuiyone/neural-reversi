"""Extract the two allocations used by EndGameCaches::for_thread_count."""

from math import prod
import re

from extract_cache_allocation import cfg
from mir_common import checked, CALL


def render(dump, repo):
    text = (repo / "crates/reversi-core/src/search/endgame.rs").read_text(encoding="utf-8")
    all_constants = re.findall(r"^const (\w+_CACHE_BYTES): usize = ([\d\s*]+);$", text, re.M)
    constants = {name: prod(int(x.strip()) for x in expr.split("*")) for name, expr in all_constants}
    checked(all(v < 2**32 for v in constants.values()), "native cache budget overflow")
    small = [name for name, value in constants.items() if value < 18]
    checked(not small, f"{', '.join(small)} below one 18-byte entry, but budget_fits "
            "(EndgameSearch/CacheMir/BudgetProof.lean) needs at least one entry per thread")
    top = cfg(dump, "reversi_core.search-endgame-{impl#0}-for_thread_count",
              ">::for_thread_count(_1: usize) -> EndGameCaches {")
    env = {"_1": "n"}
    fields = {}
    def expr(s):
        if m := re.fullmatch(r"(?:copy|move) (_\d+)",s):
            return env[m[1]]
        if m := re.fullmatch(r"const search::endgame::(\w+_CACHE_BYTES)",s):
            return str(constants[m[1]])
        if m := re.fullmatch(r"const (\d+)_usize",s):
            return m[1]
        if m := re.fullmatch(r"(Div|Eq)\((.+), (.+)\)",s):
            a, b = expr(m[2]), expr(m[3])
            if m[1]=="Div":
                checked(b == "(max n 1)", "divisor could be zero")
                return f"({a} / {b})"
            checked((a,b) == ("(max n 1)","0"), "division check operand")
            return False
        if m := re.fullmatch(r"EndGameCaches \{ ec: (move _\d+), shallow: (move _\d+) \}",s):
            both = [expr(m[1]),expr(m[2])]
            checked([t for t,_ in both] == ["Deep","Shallow"],"cache allocation mixup")
            fields.update(both)
            return "returned"
        raise ValueError(f"unknown allocation term {s}")
    pc = 0
    seen = set()
    while pc not in seen:
        seen.add(pc)
        *stmts,end = top[pc]
        for statement in stmts:
            m = re.fullmatch(r"(_\d+) = (.+);",statement)
            checked(m,f"unknown statement {statement}")
            env[m[1]] = expr(m[2])
        if m := re.fullmatch(CALL,end):
            dest,fn,operands,nxt = m.groups()
            args = [expr(a) for a in operands.split(", ")]
            if fn=="<usize as Ord>::max":
                a,b = args
                result = f"(max {a} {b})"
            elif fn=="<usize as Ord>::clamp":
                a,lo,hi = args
                checked(int(lo) <= int(hi),"invalid clamp")
                result = f"(min (max {a} {lo}) {hi})"
            elif m := re.fullmatch(r"EndGameCache::<(Deep|Shallow)>::new",fn):
                (budget,) = args
                result = (m[1],budget)
            else:
                raise ValueError(f"unknown allocation call {fn}")
            env[dest]=result
            pc=int(nxt)
        elif m := re.fullmatch(r'assert\(!move (_\d+), "attempt to divide `\{\}` by zero", .+\) -> \[success: bb(\d+), unwind: bb\d+\];',end):
            checked(env[m[1]] is False,"nonzero divisor")
            pc=int(m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",end):
            checked(env["_0"]=="returned" and env[m[1]][0] in fields,"drop of unmoved field")
            pc=int(m[2])
        else:
            checked(end=="return;" and env["_0"]=="returned","unexpected allocation exit")
            return "\n".join(["module", "public import NeuralReversi.EndgameSearch.CacheMir","",
                "@[expose] public section", "", "namespace NeuralReversi.EndgameSearch.CacheMir","",
                "def budget (n : Nat) : EndgameTreeCache.Table → Nat",
                "  | .deep => "+fields["Deep"],"  | .shallow => "+fields["Shallow"],"",
                "end NeuralReversi.EndgameSearch.CacheMir",""])
    raise ValueError("unexpected cache allocation loop")
