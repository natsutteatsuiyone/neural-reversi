"""Recover the receiver of a call from all reaching typed MIR definitions."""

from collections import defaultdict
import re

from mir_common import CALL, blocks, checked


PASS = ".SimplifyCfg-initial.after.mir"
THREAD = "threading::thread::Thread"
POOL = "search-threading-pool-{impl#0}-"
WORKER = "search-threading-thread-{impl#1}-"
CLOSURE = r"\{closure@[^}]+\}"


class Body:
    def __init__(self, dump, stem):
        self.stem = stem
        self.source = (dump / ("reversi_core." + stem + PASS)).read_text(encoding="utf-8")
        signatures = re.findall(r"^fn (.+) \{$", self.source, re.M)
        checked(len(signatures) == 1, f"{stem}: signature")
        self.signature, = signatures
        self.cfg = blocks(self.source)
        self.ops = [(n, s) for n, lines in self.cfg.items() for s in lines]
        self.calls = []
        self.defs = defaultdict(list)
        for n,s in self.ops:
            if m := re.fullmatch(CALL,s):
                dest,fn,args,nxt = m.groups()
                operands = args.split(", ") if args else []
                self.calls.append((n,fn,operands,dest))
                self.defs[dest].append(("call",fn,operands))
            elif m := re.fullmatch(r"(_\d+) = (.+);",s):
                self.defs[m[1]].append(m[2])

    def named(self, name):
        sites = [c for c in self.calls if c[1] == name]
        checked(len(sites) == 1, f"{self.stem}: expected one {name}, got {len(sites)}")
        return sites[0]

    def arg(self, value):
        debug = re.findall(r"^\s*debug "+value+r" => (_\d+);",self.source,re.M)
        checked(len(debug)==1 and re.search(re.escape(debug[0])+r":",self.signature),f"unknown {value} input")
        return debug[0]

    def resolve(self, operand, seeds):
        """Accept a register only if every assignment gives it the same origin."""
        def origin(expr, chain):
            if isinstance(expr,tuple):
                _,fn,args = expr
                if fn in (f"<Arc<{THREAD}> as Deref>::deref",f"<Arc<{THREAD}> as Clone>::clone",
                          "<Arc<ThreadPool> as Deref>::deref"):
                    (arg,) = args
                    return origin(arg,chain)
                raise ValueError(f"{self.stem}: unexpected producer {fn}")
            if expr in seeds:
                checked(not re.fullmatch(r"_\d+",expr) or not self.defs.get(expr)
                        or (expr in seeds and isinstance(seeds[expr],tuple)),
                        f"{self.stem}: reassigned input {expr}")
                value=seeds[expr]
                if isinstance(value,tuple):
                    kind,expected,result = value
                    checked(kind=="checked" and self.defs[expr] == [expected],f"conflicting definition of {expr}")
                    return result
                return value
            if m := re.fullmatch(r"(_\d+)",expr):
                reg=m[1]
                checked(reg not in chain, f"{self.stem} recursive alias {reg}")
                options = [origin(rhs,chain|{reg}) for rhs in self.defs[reg]]
                checked(options and all(x==options[0] for x in options),f"{self.stem} conflicting {reg}: {options}")
                return options[0]
            if m := re.fullmatch(r"(?:move|copy) (.+)|&(?:mut )?(.+)|\(\*(.+)\)",expr):
                return origin(next(x for x in m.groups() if x),chain)
            if m := re.fullmatch(r"\((move _\d+),\)",expr):
                return ("one",origin(m[1],chain))
            if m := re.fullmatch(r"\(\(\*(_\d+)\)\.([01]): search::endgame::cache::EndGameCache<search::endgame::cache::(Deep|Shallow)>\)",expr):
                parent=origin(m[1],chain)
                kind={"0":"Deep","1":"Shallow"}[m[2]]
                checked(parent=="pair" and kind==m[3],f"cache field {expr}")
                return kind.lower()
            if expr.startswith("const "):
                return expr.removeprefix("const ")
            raise ValueError(f"{self.stem}: unknown origin {expr}")
        return origin(operand,set())

    def produced(self, fn, result, seeds):
        _,_,args,dest = self.named(fn)
        return {**seeds,dest:("checked",("call",fn,args),result)}

    def reachable(self, start, *, stop_at=()):
        work=[start]
        found=set()
        while work:
            pc=work.pop()
            if pc in found: continue
            found.add(pc)
            if pc in stop_at:
                continue
            end=self.cfg[pc][-1]
            if end.startswith("falseEdge") or end.startswith("falseUnwind"):
                targets=re.findall(r"real: bb(\d+)",end)
            else:
                targets=re.findall(r"(?:return: |success: |otherwise: |\d+: |goto -> )bb(\d+)",end)
            work.extend(int(x) for x in targets if int(x) in self.cfg)
        return found
