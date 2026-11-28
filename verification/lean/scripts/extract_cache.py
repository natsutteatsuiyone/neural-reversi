"""Project checked element accesses and normal cell computations of EndGameCache."""

import re

from mir_common import CALL, GOTO, checked


from extract_cache_allocation import allocation, cfg, PREFIX, CONSTANT


def tokens(path):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8")))


def sentinels(dump, repo):
    src = repo / "crates/reversi-core/src"
    cache = tokens(src / "search/endgame/cache.rs")
    checked("#[repr(C,packed)]structRawEntry{player:u64,opponent:u64,lower:i8,upper:i8,}" in cache,
            "RawEntry packed byte layout")
    size = 2 * 8 + 2
    score = tokens(src / "score.rs")
    for s in ["#[derive(Clone,Copy,Debug,Default,PartialEq,Eq,PartialOrd,Ord,Hash)]"
              "#[repr(transparent)]pubstructDiscScore(i32);",
              "fnnew(value:i32)->Self{Self(value)}", "fnvalue(self)->i32{self.0}"]:
        checked(s in score,"DiscScore signed comparison and constructors")
    board = tokens(src / "board.rs")
    for f in ("player","opponent"):
        checked(f"fn{f}(&self)->Bitboard{{self.{f}}}" in board,f"Board::{f}")
    checked("fnbits(self)->u64{self.0}" in tokens(src / "bitboard.rs"),"Bitboard::bits")
    bounds = {}
    for name,op,base in [("LOWER_UNSET","Sub","MIN"),("UPPER_UNSET","Add","MAX")]:
        constant = re.search(r"pubconstDISC_SCORE_" + base + r":DiscScore=DiscScore::new\((-?\d+)\);", score)
        checked(constant, f"DiscScore {base}")
        body = cfg(dump,PREFIX+"{impl#0}-"+name,f">::{name}: i8 = {{")
        load = f"_2 = DiscScore::value(const score::DISC_SCORE_{base}) -> [return: bb1, unwind: bb3];"
        checked(body[0] == [load], "sentinel input")
        checked(body[1][0] == f"_3 = {op}WithOverflow(copy _2, const 1_i32);" and
                re.fullmatch(r'assert\(!move \(_3\.1: bool\), "attempt to compute [^"]+which would overflow", '
                             r'move _2, const 1_i32\) -> \[success: bb2, unwind: bb3\];',body[1][1]) and
                len(body[1]) == 2 and body[2] == ["_1 = move (_3.0: i32);","_0 = move _1 as i8 (IntToInt);","return;"]
                and len(body) == 3, "sentinel arithmetic")
        result = int(constant[1]) + (1 if op == "Add" else -1)
        checked(-128 <= result <= 127,"sentinel overflow")
        bounds[name] = result
    empty = cfg(dump,PREFIX+"{impl#0}-EMPTY",">::EMPTY: RawEntry = {")
    checked(empty == {0: ["_0 = RawEntry { player: const 0_u64, opponent: const 0_u64, "
                          f"lower: const {CONSTANT}LOWER_UNSET, upper: const {CONSTANT}UPPER_UNSET }};",
                          "return;"]},"empty cell")
    return size, bounds


class NeedBranch(Exception):
    pass


def tree(run, facts, indent):
    try:
        return [indent + run(facts)]
    except NeedBranch as missing:
        condition, = missing.args
        return ([indent + "if " + condition + " then"]
                + tree(run, {**facts,condition:True},indent+"  ")
                + [indent+"else"] + tree(run,{**facts,condition:False},indent+"  "))


def execute(body, method, facts):
    probe = method == "probe"
    array = probe or method == "store"
    mutable = method in ("overwrite","store_nws_result","store")
    entries = {"_1":"table" if array else "cell"}
    if array:
        entries.update({"_2":"slot","_3":"board","_4":"alpha","_5":"score"})
    elif method == "overwrite":
        entries.update({"_2":"board","_3":"alpha","_4":"score"})
    else:
        entries.update({"_2":"alpha","_3":"score"})
    values = ["e.board.player","e.board.opponent","e.lower","e.upper"]
    updated = None
    checked_index = False
    def branch(c):
        if isinstance(c,bool):
            return c
        if c not in facts:
            raise NeedBranch(c)
        return facts[c]
    def record():
        return "⟨⟨" + ", ".join(values[:2]) + "⟩, " + ", ".join(values[2:]) + "⟩"
    def field(place):
        m = re.fullmatch(r"\((?:\(\*(_\d+)\)|(_\d+))\.([0-3]): (u64|i8)\)",place)
        if not m:
            return None
        i = int(m[3])
        checked(entries[m[1] or m[2]] == "cell" and m[4] == ("u64" if i<2 else "i8"),"raw cell projection")
        return i
    def val(expr):
        table = r"\(\*\(\(\*(_\d+)\)\.0: std::boxed::Box<\[search::endgame::cache::RawEntry\]>\)\)"
        if m := re.fullmatch(r"&(?P<mut>mut )?"+table,expr):
            checked(entries[m[2]] == "table","box slice reference")
            return "mut-slice" if m["mut"] else "slice"
        if m := re.fullmatch(r"copy \((_\d+)\.0: usize\)",expr):
            checked(entries[m[1]] == "slot","slot index")
            return "index"
        if m := re.fullmatch(r"copy (.+)",expr):
            i = field(m[1])
            if i is not None:
                return values[i]
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(_\d+)|copy \(\*(_\d+)\)",expr):
            return entries[next(v for v in m.groups() if v)]
        if m := re.fullmatch(r"move (_\d+) as (i32|i8) \(IntToInt\)",expr):
            x = entries[m[1]]
            return f"{x}.toInt" if m[2] == "i32" else f"(Int8.ofInt {x})"
        if m := re.fullmatch(r"(Eq|Ne|Lt)\((.+), (.+)\)",expr):
            a,b = val(m[2]),val(m[3])
            if m[1]=="Lt":
                checked((a,b)==("index","size"),"array bound guard")
                return "inRange"
            return f"{a} " + ("=" if m[1]=="Eq" else "≠") + f" {b}"
        if m := re.fullmatch(r"Option::<DiscScore>::Some\((.*)\)",expr):
            return "some " + val(m[1])
        return {"const true":True,"const false":False,"const ()":"()", "Option::<DiscScore>::None":"none",
                "const "+CONSTANT+"LOWER_UNSET":"lowerUnset","const "+CONSTANT+"UPPER_UNSET":"upperUnset"}[expr]
    def call(fn,args):
        nonlocal updated, checked_index
        if fn == "core::slice::<impl [RawEntry]>::len":
            checked(args == ["slice"], "slice length receiver")
            return "size"
        if fn == "core::slice::<impl [RawEntry]>::get_unchecked::<usize>":
            checked(not mutable and args == ["slice", "index"] and checked_index, "unchecked probe element access")
            return "cell"
        if fn == "core::slice::<impl [RawEntry]>::get_unchecked_mut::<usize>":
            checked(mutable and args == ["mut-slice", "index"] and checked_index, "unchecked store element access")
            return "cell"
        if fn in ("Board::player","Board::opponent"):
            checked(args==["board"],"board receiver")
            return ("bitboard",fn.split("::")[1])
        if fn == "Bitboard::bits":
            (v,) = args
            checked(v[0]=="bitboard","bitboard operand")
            return "board."+v[1]
        if fn in ("DiscScore::new","DiscScore::value"):
            (x,) = args
            return x
        if fn in ("<DiscScore as PartialOrd>::gt","<DiscScore as PartialOrd>::le"):
            a,b = args
            return f"{a} " + (">" if fn.endswith("gt") else "≤") + f" {b}"
        if fn in ("<i8 as Ord>::max","<i8 as Ord>::min"):
            return "(" + fn.split("::")[-1] + " " + " ".join(args) + ")"
        if fn=="RawEntry::score_for_alpha":
            cell,alpha=args
            checked(cell=="cell" and updated is None,"probe receiver")
            return f"scoreForAlpha e {alpha}"
        if fn=="RawEntry::store_nws_result":
            cell,alpha,score=args
            checked(cell=="cell" and updated is None,"store receiver")
            updated=f"merge {record()} {alpha} {score}"
            return "()"
        if fn=="RawEntry::overwrite":
            cell,board,alpha,score=args
            checked(cell=="cell" and updated is None,"overwrite receiver")
            updated=f"overwrite e {board} {alpha} {score}"
            return "()"
        raise ValueError(f"unexpected {method} call {fn}")

    pc = 0
    visited = set()
    while pc not in visited:
        visited.add(pc)
        *statements,end = body[pc]
        for s in statements:
            l,r = s[:-1].split(" = ",1)
            if re.fullmatch(r"_\d+",l):
                entries[l] = val(r)
            else:
                i = field(l)
                checked(i is not None and mutable and updated is None, f"unexpected place write {s}")
                values[i] = val(r)
        if m := re.fullmatch(CALL,end):
            dst,fn,args,nxt = m.groups()
            entries[dst] = call(fn,[val(a) for a in args.split(", ")])
            pc = int(nxt)
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];",end):
            cond = entries[m[1]]
            if cond == "inRange":
                checked(not checked_index,"multiple element checks")
                fail_bb = int(m[2])
                checked(body[fail_bb] == [re.sub(r"_\d+", lambda x: x[0], body[fail_bb][0])]
                        and 'panic(const "assertion failed: slot.0 < self.table.len()")' in body[fail_bb][-1],
                        "illegal debug bound assertion")
                checked_index = True
                pc = int(m[3])
            else:
                pc = int(m[3] if branch(cond) else m[2])
        elif m := re.fullmatch(GOTO,end):
            pc = int(m[1] or m[2])
        else:
            checked(end=="return;" and (checked_index or not array),f"unexpected {method} exit {end}")
            if mutable:
                checked(entries["_0"]=="()", "mutation result")
                return updated or record()
            return entries["_0"]
    raise ValueError(f"unexpected cycle in {method}")


SIGS = {
    "score_for_alpha": ("(_1: RawEntry, _2: DiscScore) -> Option<DiscScore>",
                        "def scoreForAlpha (e : Entry) (alpha : Int) : Option Int"),
    "store_nws_result": ("(_1: &mut RawEntry, _2: DiscScore, _3: DiscScore) -> ()",
                         "def merge (e : Entry) (alpha score : Int) : Entry"),
    "overwrite": ("(_1: &mut RawEntry, _2: &Board, _3: DiscScore, _4: DiscScore) -> ()",
                  "def overwrite (e : Entry) (board : Board) (alpha score : Int) : Entry"),
    "probe": ("(_1: &EndGameCache<K>, _2: CacheSlot<K>, _3: &Board, _4: DiscScore) -> Option<DiscScore>",
              "def probeCell (e : Entry) (board : Board) (alpha : Int) : Option Int"),
    "store": ("(_1: &mut EndGameCache<K>, _2: CacheSlot<K>, _3: &Board, _4: DiscScore, _5: DiscScore) -> ()",
              "def storeCell (e : Entry) (board : Board) (alpha score : Int) : Entry"),
}


def render(dump,repo):
    size, bounds = sentinels(dump, repo)
    lines = ["module", "public import NeuralReversi.EndgameCache","public import NeuralReversi.MulHi","",
             "@[expose] public section", "", "namespace NeuralReversi.EndgameSearch.CacheMir","","open EndgameCache (Entry)","open Board (Board)","",
             f"def lowerUnset : Int8 := {bounds['LOWER_UNSET']}",f"def upperUnset : Int8 := {bounds['UPPER_UNSET']}",
             "def initialCell : Entry := ⟨⟨0, 0⟩, lowerUnset, upperUnset⟩","",*allocation(dump,size)]
    for method,(source,signature) in SIGS.items():
        owner = "{impl#1}-" if method in ("probe","store") else "{impl#0}-"
        body = cfg(dump,PREFIX+owner+method,">::"+method+source+" {")
        result = tree(lambda flags: execute(body,method,flags),{}, "  ")
        if method == "overwrite" and not any(re.search(r"\be\b",line) for line in result):
            signature = signature.replace("(e : Entry)", "(_prior : Entry)")
        lines.extend([signature+" :=",*result, ""])
    lines.extend([
        "def probe (cells : Array Entry) (idx : Nat) (board : Board) (alpha : Int) : Option (Option Int) :=",
        "  (cells[idx]?).map fun e => probeCell e board alpha","",
        "def store (cells : Array Entry) (idx : Nat) (board : Board) (alpha score : Int) : Option (Array Entry) :=",
        "  if h : idx < cells.size then some (cells.set idx (storeCell cells[idx] board alpha score)) else none","",
        "end NeuralReversi.EndgameSearch.CacheMir",""])
    return "\n".join(lines)
