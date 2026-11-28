"""Translate the normal per-attempt traces of the four TT slot mutation methods."""

from dataclasses import dataclass
import re

from mir_common import CALL, GOTO, checked
from extract_tt_writer import read_cfg
from extract_tt_reader_mir import layout


class Branch(Exception):
    pass


@dataclass(frozen=True)
class Choice:
    guard: str
    yes: object
    no: object


def guards(run, taken):
    try:
        return run(taken)
    except Branch as branch:
        guard, = branch.args
        yes = guards(run, {**taken, guard: True})
        no = guards(run, {**taken, guard: False})
        return yes if yes == no else Choice(guard, yes, no)


FIELDS = [("score", "i16"), ("bestMove", "u8"), ("bound", "u8"), ("depth", "u8"),
          ("selectivity", "u8"), ("generation", "u8"), ("flags", "u8")]


def source_checks(repo, dump):
    src = repo / "crates/reversi-core/src"
    def tokens(path):
        return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", (src / path).read_text(encoding="utf-8")))

    tt = tokens("transposition_table.rs")
    for fragment in ["constSAVE_RETRIES:u32=4;", "constNO_MOVE:u8=u8::MAX;",
                     "pubenumBound{None=0,Lower=1,Upper=2,Exact=3,}",
                     "fnis_occupied(&self)->bool{self.bound!=0}",
                     "fndepth(&self)->Depth{self.depthasDepth}",
                     "fnselectivity(&self)->Selectivity{Selectivity::from_u8(self.selectivity)}",
                     "fngeneration(&self)->u8{self.generation}",
                     "#[derive(Clone,Copy,Debug,PartialEq)]pubstructTTEntryData{",
                     "fnsame_board_bits(lhs_player:u64,lhs_opponent:u64,rhs_player:u64,rhs_opponent:u64)->bool{"
                     "((lhs_player^rhs_player)|(lhs_opponent^rhs_opponent))==0}"]:
        checked(tt.count(fragment) == 1, f"unrecognized TT helper: {fragment}")
    board = tokens("board.rs")
    for name in ("player", "opponent"):
        checked(board.count(f"fn{name}(&self)->Bitboard{{self.{name}}}") == 1, f"Board::{name}")
    checked("fnbits(self)->u64{self.0}" in tokens("bitboard.rs"), "bitboard bits")
    selectivity = tokens("probcut.rs")
    checked("pubenumSelectivity{Mid=0,#[default]Level1=1,Level2=2,Level3=3,Level4=4,None=5,}" in selectivity,
            "selectivity discriminants")
    checked("fnas_u8(self)->u8{selfasu8}" in selectivity, "selectivity cast")
    checked("fnfrom_u8(value:u8)->Self{matchvalue{0=>Selectivity::Mid,1=>Selectivity::Level1,"
            "2=>Selectivity::Level2,3=>Selectivity::Level3,4=>Selectivity::Level4,_=>Selectivity::None,}}" in selectivity,
            "selectivity decoding")
    copy = read_cfg(dump, "{impl#3}", "with_best_move_of",
                    "(_1: TTEntryData, _2: TTEntryData) -> TTEntryData")
    names = ["score","best_move","bound","depth","selectivity","generation","flags"]
    row = ", ".join(f"{name}: " + ("move _3" if i == 1 else f"copy (_1.{i}: {FIELDS[i][1]})")
                    for i, name in enumerate(names))
    checked(copy == {0: ["_3 = copy (_2.1: u8);", "_0 = TTEntryData { " + row + " };", "return;"]},
            "with_best_move_of metadata construction")
    zero = read_cfg(dump, "{impl#3}", "from_u64", "(_1: u64) -> TTEntryData")
    checked(zero == {0: ["_2 = copy _1;",
                         "_0 = std::intrinsics::transmute::<u64, TTEntryData>(move _2) -> [return: bb1, unwind: bb2];"],
                     1: ["return;"]}, "from_u64")


def attempt(cfg, method, locs, taken):
    values = {"_1": "slot", "_2": "board" if method == "save" else "p",
              "_3": "entry" if method == "save" else "o", "_4": "entry"}
    events = []
    reads = 0
    visited = set()
    stable = "seq &&& 1 ≠ 0"

    def branch(cond):
        if isinstance(cond, bool):
            return cond
        if cond not in taken:
            raise Branch(cond)
        return taken[cond]

    def val(expr):
        if expr == "const ()":
            return "()"
        if expr in ("const true","const false"):
            return expr == "const true"
        if m := re.fullmatch(r"const (\d+)_u(?:8|32|64)",expr):
            return m[1]
        if expr == "const transposition_table::NO_MOVE":
            return "Packing.noMove"
        if expr == "const transposition_table::Bound::Exact::{constant#0}":
            return "Packing.boundByte .exact"
        if expr == "std::ops::Range::<u32> { start: const 0_u32, end: const transposition_table::TTEntry::SAVE_RETRIES }":
            return "fourRetries"
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(?:mut )?(_\d+)",expr):
            return values[next(s for s in m.groups() if s)]
        if m := re.fullmatch(r"std::sync::atomic::Ordering::(Acquire|Relaxed|Release|AcqRel)",expr):
            return "." + m[1][0].lower() + m[1][1:]
        if m := re.fullmatch(r"&\(\(\*(_\d+)\)\.(\d+): std::sync::atomic::Atomic<u64>\)",expr):
            checked(values[m[1]] == "slot", "atomic slot")
            return locs[int(m[2])]
        if m := re.fullmatch(r"copy \(\(\((_\d+) as Some\)\.0: \(u64, u64, u64, transposition_table::TTEntryData\)\)\.([0-3]): (.+)\)",expr):
            checked(values[m[1]] == "snapshot" and branch("found"), "read invalid snapshot")
            return ["saved.player", "saved.opponent", "saved.sequence", "old"][int(m[2])]
        if m := re.fullmatch(r"(?:copy|move) \((_\d+)\.(\d+): (u8|u32|bool|i16)\)",expr):
            obj = values[m[1]]
            i = int(m[2])
            if isinstance(obj, tuple):
                return obj[i]
            name, ty = FIELDS[i]
            checked(obj in ("entry","old") and m[3] == ty, "metadata field")
            return f"{obj}.{name}"
        if m := re.fullmatch(r"(move _\d+) as (u32|u8) \(IntToInt\)",expr):
            x = val(m[1])
            if m[2] == "u8":
                checked(x == "Packing.boundByte .exact", "bound cast")
                return x
            checked(x == "entry.depth", "u8->u32 cast")
            return "entry.depth.setWidth 32"
        if m := re.fullmatch(r"(Add|Sub)(WithOverflow)?\((.+), (.+)\)",expr):
            left, right = val(m[3]), val(m[4])
            if m[1] == "Add" and left == "Packing.boundByte .exact" and right == "0":
                result = left
            elif m[1] == "Add" and left == "entry.depth.setWidth 32" and right in ("2","4"):
                result = f"(entry.depth.setWidth 32 + {right}).toNat"
            elif m[1] == "Sub" and (left,right) == ("entry.depth","1"):
                checked(branch("entry.depth = 0") is False, "penalty underflows")
                result = "(entry.depth - 1)"
            else:
                raise ValueError(f"unmodeled arithmetic {left} {right}")
            return (result,False) if m[2] else result
        if m := re.fullmatch(r"discriminant\((_\d+)\)",expr):
            return {"iteration":"more","snapshot":"found"}[values[m[1]]]
        if m := re.fullmatch(r"(Eq|Ne|BitAnd|Ge|Gt)\((.*), (.*)\)",expr):
            left, right = val(m[2]), val(m[3])
            return {
                ("Eq","entry.bound","Packing.boundByte .exact"): "entry.bound = Packing.boundByte .exact",
                ("Ne","entry.bestMove","Packing.noMove"): "entry.bestMove ≠ Packing.noMove",
                ("BitAnd","seq","1"): "(seq &&& 1)",
                ("Ne","(seq &&& 1)","0"): stable,
                ("Ge","(entry.depth.setWidth 32 + 2).toNat","old.depth.toNat"):
                    "old.depth.toNat ≤ (entry.depth.setWidth 32 + 2).toNat",
                ("Ge","(entry.depth.setWidth 32 + 4).toNat","old.depth.toNat"):
                    "old.depth.toNat ≤ (entry.depth.setWidth 32 + 4).toNat",
                ("Gt","entry.selectivity","rank"): "selectivityRank old.selectivity < entry.selectivity.toNat",
                ("Ne","entry.generation","old.generation"): "entry.generation ≠ old.generation",
                ("Eq","entry.depth","0"): "entry.depth = 0",
                ("Eq","metadata","0"): "metadata = 0", ("Eq","last","seq"): "last = seq",
            }[(m[1],left,right)]
        if expr.startswith("TTEntryData { ") and expr.endswith(" }"):
            names = ["score","best_move","bound","depth","selectivity","generation","flags"]
            fields = [item.split(": ",1) for item in expr[14:-2].split(", ")]
            checked([key for key,_ in fields] == names, "penalty record field order")
            origins = [val(item) for _,item in fields]
            expected = ["entry." + name for name,ty in FIELDS]
            expected[3] = "(entry.depth - 1)"
            checked(origins == expected, f"unexpected penalty origins {origins}")
            return "(Packing.penalized entry)"
        raise ValueError(f"{method}: unexpected expression {expr}")

    def call(name, args):
        nonlocal reads
        if name == "Atomic::<u64>::load":
            loc, order = args
            events.append(f".load {loc} {order}")
            result = ["seq","metadata","last"][reads]
            reads += 1
            return result
        if name == "fence":
            order, = args
            events.append(f".fence {order}")
            return "()"
        if name == "spin_loop":
            checked(not args, "spin arguments")
            events.append(".spin")
            return "()"
        if name == "TTEntry::try_load_snapshot":
            checked(args == ["slot"], "snapshot slot")
            events.append(".snapshot")
            return "snapshot"
        if name == "TTEntry::save_exact":
            slot,p,o,entry = args
            checked(slot == "slot", "wrong exact slot")
            events.append(f".exact {p} {o} {entry}")
            return "()"
        if name == "TTEntry::seqlock_write":
            slot,seq,p,o,entry = args
            checked(slot == "slot", "wrong claim slot")
            events.append(f".claim ⟨{seq}, {p}, {o}, {entry}⟩")
            return "cas"
        known = {
            "Board::player": (["board"],"playerBits"), "Board::opponent": (["board"],"opponentBits"),
            "<std::ops::Range<u32> as IntoIterator>::into_iter": (["fourRetries"],"iterator"),
            "<std::ops::Range<u32> as Iterator>::next": (["iterator"],"iteration"),
            "TTEntryData::is_occupied": (["old"],"old.bound ≠ 0"),
            "same_board_bits": (["p","o","saved.player","saved.opponent"],
                                 "sameBoardBits p o saved.player saved.opponent"),
            "TTEntryData::depth": (["old"],"old.depth.toNat"),
            "TTEntryData::selectivity": (["old"],"storedSelectivity"),
            "Selectivity::as_u8": (["storedSelectivity"],"rank"),
            "TTEntryData::generation": (["old"],"old.generation"),
            "TTEntryData::with_best_move_of": (["entry","old"],"(Packing.withBestMoveOf entry old)"),
            "TTEntryData::from_u64": (["0"],"(Packing.unpack endian 0)"),
            "<TTEntryData as PartialEq>::ne": (["old","entry"],"old ≠ entry"),
        }
        if name == "Bitboard::bits":
            return {("playerBits",): "p", ("opponentBits",): "o"}[tuple(args)]
        expected,result = known[name]
        checked(args == expected, f"{name}: expected {expected}, got {args}")
        return result

    pc = 0
    while pc not in visited:
        visited.add(pc)
        *statements,end = cfg[pc]
        for statement in statements:
            m = re.fullmatch(r"(_\d+) = (.+);",statement)
            checked(m, f"{method}: unexpected statement {statement}")
            values[m[1]] = val(m[2])
        if m := re.fullmatch(CALL,end):
            dest,name,operands,next_pc = m.groups()
            args = [val(arg) for arg in operands.split(", ")] if operands else []
            values[dest] = call(name,args)
            pc = int(next_pc)
        elif m := re.fullmatch(GOTO,end):
            pc = int(m[1] or m[2])
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[(.+)\];",end):
            arms = dict((k,int(v)) for k,v in re.findall(r"(\d+|otherwise): bb(\d+)",m[2]))
            key = "1" if branch(values[m[1]]) else "0"
            pc = arms[key] if key in arms else arms["otherwise"]
        elif m := re.fullmatch(r"assert\(!move \((_\d+)\.1: bool\), .+\) -> \[success: bb(\d+), unwind: bb\d+\];",end):
            checked(values[m[1]][1] is False,"possibly overflowing arithmetic")
            pc = int(m[2])
        else:
            checked(end == "return;",f"{method}: unexpected exit {end}")
            return (tuple(events),False)
    checked(cfg[pc][-1].startswith("falseUnwind -> "), f"unexpected cycle bb{pc}")
    return (tuple(events),True)


def lean_tree(tree):
    nodes = {}
    lines = []
    def visit(item):
        if item in nodes:
            return nodes[item]
        if isinstance(item, Choice):
            yes = visit(item.yes)
            no = visit(item.no)
            body = f"if {item.guard} then {yes} else {no}"
        else:
            events,retry = item
            body = "⟨[" + ", ".join(events) + "], " + str(retry).lower() + "⟩"
        name = f"step{len(nodes)}"
        lines.append(f"  let {name} : SourceStep := {body}")
        nodes[item] = name
        return name
    last = visit(tree)
    return [*lines,"  " + last]


def render(dump, repo):
    locs = layout(repo)
    source_checks(repo,dump)
    lines = ["module", "public import NeuralReversi.TranspositionTable.WriterMir.SourceSyntax", "",
             "@[expose] public section", "", "namespace NeuralReversi.TranspositionTable.WriterMir", "", "open SeqLock (Word)", ""]
    signatures = {
        "save": ("(_1: &TTEntry, _2: &Board, _3: TTEntryData) -> ()",
                 "def saveStep (more found cas : Bool) (p o : Word) (entry : EntryData)"
                 " (saved : SeqLock.Snapshot) (old : EntryData)"),
        "save_exact": ("(_1: &TTEntry, _2: u64, _3: u64, _4: TTEntryData) -> ()",
                       "def exactStep (more cas : Bool) (seq p o : Word) (entry : EntryData)"),
        "clear_shared": ("(_1: &TTEntry) -> ()",
                         "def clearStep (endian more cas : Bool) (seq metadata last : Word)"),
        "penalize": ("(_1: &TTEntry, _2: u64, _3: u64, _4: TTEntryData) -> ()",
                     "def penaltyStep (found : Bool) (p o : Word) (entry : EntryData)"
                     " (saved : SeqLock.Snapshot) (old : EntryData)"),
    }
    for method,(signature,params) in signatures.items():
        body = read_cfg(dump,"{impl#6}",method,signature)
        tree = guards(lambda taken: attempt(body,method,locs,taken), {})
        lines.extend([params + " : SourceStep :=", *lean_tree(tree), ""])
    return "\n".join([*lines,"end NeuralReversi.TranspositionTable.WriterMir",""])
