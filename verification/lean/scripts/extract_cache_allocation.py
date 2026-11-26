"""Compile cache construction, table length and both integer-cast configurations."""

import re

from mir_common import blocks, CALL, checked


PASS = ".SimplifyCfg-initial.after.mir"
PREFIX = "reversi_core.search-endgame-cache-"
RAW = "search::endgame::cache::RawEntry"
CONSTANT = RAW + "::"
STORAGE = re.compile(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);")


def cfg(dump, name, signature=None):
    path = dump / (name + PASS)
    source = path.read_text(encoding="utf-8")
    if signature:
        checked(signature in source, f"{path.name}: unexpected signature")
    return {n: [s for s in lines if not STORAGE.fullmatch(s)] for n, lines in blocks(source).items()}


def evaluate(body, registers, size, positive):
    def val(expr):
        if m := re.fullmatch(r"(?:copy|move) (_\d+)",expr):
            return registers[m[1]]
        if m := re.fullmatch(r"const (\d+)_(?:i32|u32|usize)",expr):
            return m[1]
        if expr == "const 64_i32 as u32 (IntToInt)":
            return "64"
        if m := re.fullmatch(r"move (_\d+) as (usize|u64|u128) \(IntToInt\)",expr):
            x = registers[m[1]]
            if m[2] == "usize":
                return f"({x}.setWidth width).toNat"
            if m[2] == "u64" and x == "length":
                return "(BitVec.ofNat 64 length)"
            return f"({x}.setWidth " + {"u128":"128","u64":"64"}[m[2]] + ")"
        if m := re.fullmatch(r"(Eq|Gt|Lt|Div|Mul|MulWithOverflow|Shr)\((.+), (.+)\)",expr):
            left,right=val(m[2]),val(m[3])
            if m[1] == "Eq":
                checked((left,right) == (str(size),"0"),"cache entry division")
                return False
            if m[1] == "Gt":
                checked((left,right) == (f"(bytes / {size})","0"),"positive allocation")
                return positive
            if m[1] == "Lt":
                checked((left,right) == ("64","128"),"shift range")
                return True
            if m[1] == "Shr":
                return f"({left} >>> ({right} : Nat))"
            if m[1] == "Div":
                return f"({left} / {right})"
            result = f"({left} * {right})"
            if m[1] == "MulWithOverflow":
                checked((left,right) == ("(a.setWidth 128)","(b.setWidth 128)"),"u64×u64 fits u128")
                return (result,False)
            return result
        if m := re.fullmatch(r"move \((_\d+)\.0: u128\)",expr):
            return registers[m[1]][0]
        if m := re.fullmatch(r"&\(\*\(\(\*(_\d+)\)\.0: std::boxed::Box<\[search::endgame::cache::RawEntry\]>\)\)",expr):
            checked(registers[m[1]] == "table","cache length source")
            return "slice"
        if expr == "const "+CONSTANT+"EMPTY":
            return "initialCell"
        if expr in ("PhantomData::<K>","const ()"):
            return "()"
        if m := re.fullmatch(r"EndGameCache::<K> \{ table: move (_\d+), kind: move (_\d+) \}|CacheSlot::<K>\(move (_\d+), move (_\d+)\)",expr):
            checked(registers[m[2] or m[4]]=="()","PhantomData")
            return registers[m[1] or m[3]]
        raise ValueError(f"unknown allocation/index expression {expr}")
    pc=0
    visited=set()
    while pc not in visited:
        visited.add(pc)
        *stmts,end=body[pc]
        if stmts and stmts[0]=="_13 = &_1;":
            checked(stmts==["_13 = &_1;","_12 = (move _13,);","_16 = &(*(_12.0: &usize));"]
                    and end=="_15 = core::fmt::rt::Argument::<'_>::new_display::<usize>(move _16) -> [return: bb5, unwind: bb11];",
                    "invalid error formatting")
            checked(len(body[5]) == 6 and body[5][0]=="_14 = [move _15];" and
                    body[5][1].startswith('_18 = const b"') and body[5][2:5] ==
                    ["_17 = &(*_18);","_20 = &_14;","_19 = &(*_20);"] and
                    re.fullmatch(r"_11 = Arguments::<'_>::new::<\d+, 1>\(move _17, move _19\) -> \[return: bb6, unwind: bb11\];",body[5][5]) and
                    body[6] == ["_10 = panic_fmt(move _11) -> bb11;"],"too-small allocation must panic")
            return "none"
        for stmt in stmts:
            m=re.fullmatch(r"(_\d+) = (.+);",stmt)
            checked(m,f"unknown allocation statement {stmt}")
            registers[m[1]]=val(m[2])
        if m:=re.fullmatch(CALL,end):
            dest,name,args,nxt=m.groups()
            args=[val(a) for a in args.split(", ")] if args else []
            if name=="std::mem::size_of::<RawEntry>" and not args:
                result=str(size)
            elif name=="std::vec::from_elem_global::<RawEntry>":
                value,count=args
                result=f"(Array.replicate {count} {value})"
            elif name=="Vec::<RawEntry>::into_boxed_slice":
                (result,)=args
            elif name=="core::slice::<impl [RawEntry]>::len" and args==["slice"]:
                result="length"
            elif name=="mul_hi64":
                result="(highWord "+" ".join(args)+")"
            else:
                raise ValueError(f"unknown allocation/index call {name} {args}")
            registers[dest]=result
            pc=int(nxt)
        elif m:=re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];",end):
            checked(isinstance(registers[m[1]],bool),"condition type")
            pc=int(m[3] if registers[m[1]] else m[2])
        elif m:=re.fullmatch(r'assert\((!?)(?:move|copy) (?:(_\d+)|\((_\d+)\.1: bool\)), [^\n]+ -> \[success: bb(\d+), unwind: bb\d+\];',end):
            condition=registers[m[2]] if m[2] else registers[m[3]][1]
            checked(condition == (not m[1]), "unjustified arithmetic assertion")
            pc=int(m[4])
        elif m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",end):
            checked(registers[m[1]] == registers["_0"] and registers["_0"].startswith("(Array.replicate "),
                    "drop after moving allocated box into return position")
            pc=int(m[2])
        else:
            checked(end=="return;",f"unexpected allocation/index exit {end}")
            return registers["_0"]
    raise ValueError("unexpected loop")


def allocation(dump,size):
    make=cfg(dump,PREFIX+"{impl#1}-new",">::new(_1: usize) -> EndGameCache<K> {")
    multiply=cfg(dump,"reversi_core.util-mul_hi64", "fn mul_hi64(_1: u64, _2: u64) -> u64 {")
    slot=cfg(dump,PREFIX+"{impl#1}-slot",">::slot(_1: &EndGameCache<K>, _2: u64) -> CacheSlot<K> {")
    yes=evaluate(make,{"_1":"bytes"},size,True)
    no=evaluate(make,{"_1":"bytes"},size,False)
    checked(no=="none" and yes!="none", "invalid cache construction")
    high=evaluate(multiply,{"_1":"a","_2":"b"},size,None)
    index=evaluate(slot,{"_1":"table","_2":"key"},size,None)
    return ["def newCache (bytes : Nat) : Option (Array Entry) :=",
            f"  if 0 < bytes / {size} then some {yes} else {no}","",
            "def highWord (a b : BitVec 64) : BitVec 64 :=", "  "+high,"",
            "def slot (width length : Nat) (key : BitVec 64) : Nat :=", "  "+index,""]
