"""Ground the public store arguments, flag packing and score narrowing in MIR."""

import re

from mir_common import blocks, CALL, GOTO, checked
from extract_tt_save import source_checks
from extract_tt_writer import read_cfg, SKIP, SUFFIX


NEW_SIG = ("(_1: Score, _2: Option<Square>, _3: transposition_table::Bound, _4: u32, _5: Selectivity, "
           "_6: u8, _7: bool, _8: bool) -> TTEntryData")
STORE_SIG = ("(_1: &TranspositionTable, _2: TTSlot<'_>, _3: &Board, _4: Score, "
             "_5: transposition_table::Bound, _6: u32, _7: Option<Square>, _8: Selectivity, "
             "_9: bool, _10: bool) -> ()")


def dependencies(repo, dump):
    source_checks(repo,dump)
    src = repo / "crates/reversi-core/src"
    def source(path):
        return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", (src/path).read_text(encoding="utf-8")))
    sq = source("square.rs")
    squares = ",".join(f"{file}{rank}" for rank in range(1,9) for file in "ABCDEFGH")
    checked("pubenumSquare{" + squares + ",}" in sq, "square discriminants")
    score = source("score.rs").split("implScore{")[1]
    for text in ["pubconstINF:Self=Self(i16::MAXasi32);",
                 "pubconstfnvalue(self)->i32{self.0}",
                 "pub(crate)constfnis_finite(self)->bool{self.0.unsigned_abs()<Self::INF.0asu32}"]:
        checked(text in score, f"unknown Score primitive {text}")
    tt = source("transposition_table.rs")
    for text in ["constGENERATION_MASK:u8=0x7F;", "constFLAG_PV:u8=2;", "constFLAG_ENDGAME:u8=1;",
                 "pubstructTranspositionTable{entries:AlignedBuffer<TTEntry>,cluster_count:u64,generation:AtomicU8,}"]:
        checked(text in tt, f"unknown table representation {text}")
    generation = read_cfg(dump,"{impl#7}","generation","(_1: &TranspositionTable) -> u8")
    checked(generation == {
        0: ["_3 = &((*_1).2: std::sync::atomic::Atomic<u8>);", "_4 = std::sync::atomic::Ordering::Relaxed;",
            "_2 = Atomic::<u8>::load(move _3, move _4) -> [return: bb1, unwind: bb2];"],
        1: ["_0 = BitAnd(move _2, const transposition_table::TTEntry::GENERATION_MASK);","return;"]
    }, "generation load")
    closure = (dump / ("reversi_core.transposition_table-{impl#3}-new-{closure#0}" + SUFFIX)).read_text(encoding="utf-8")
    signature = re.search(r"^fn transposition_table::<impl at [^>]+>::new::\{closure#0\}\(_1: (\{closure@[^}]+\}), _2: Square\) -> u8 \{$",closure,re.M)
    cfg = {n:[s for s in stmts if not SKIP.fullmatch(s)] for n,stmts in blocks(closure).items()}
    checked(signature and cfg == {0: ["_3 = copy _2;", "_4 = discriminant(_3);",
                                     "_0 = move _4 as u8 (IntToInt);","return;"]}, "square cast closure")
    return signature[1]


def run(cfg, values, closure, truth):
    result = None
    pc = 0
    visited = set()
    def val(expr):
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&\(\*(_\d+)\)",expr):
            return values[m[1] or m[2]]
        if expr == "const ()":
            return "()"
        if expr == closure:
            return expr
        if m := re.fullmatch(r"(move _\d+) as (u8|i16) \(IntToInt\)",expr):
            item = val(m[1])
            return {("depth","u8"):"(depth.setWidth 8)", ("boundDiscr","u8"):"(Packing.boundByte bound)",
                    ("rawScore","i16"):"(score.raw.setWidth 16)"}[(item,m[2])]
        if m := re.fullmatch(r"discriminant\((_\d+)\)",expr):
            checked(values[m[1]] == "bound","bound discriminant")
            return "boundDiscr"
        if expr == "&(*(_2.0: &transposition_table::TTEntry))":
            checked(values["_2"] == "slot","TTSlot receiver")
            return "slot"
        if m := re.fullmatch(r"(BitAnd|BitOr)\((.+), (.+)\)",expr):
            left,right = val(m[2]), val(m[3])
            op = {"BitAnd":"&&&","BitOr":"|||"}[m[1]]
            return f"({left} {op} {right})"
        if expr.startswith("TTEntryData { ") and expr.endswith(" }"):
            fields = [field.split(": ",1) for field in expr[14:-2].split(", ")]
            checked([f for f,_ in fields] == ["score","best_move","bound","depth","selectivity","generation","flags"],
                    "invalid record initialization")
            return "⟨" + ", ".join(val(x) for _,x in fields) + "⟩"
        return {"const 0_u8":"0", "const transposition_table::NO_MOVE":"Packing.noMove",
                "const transposition_table::TTEntry::GENERATION_MASK":"127",
                "const transposition_table::FLAG_ENDGAME":"1",
                "const transposition_table::FLAG_PV":"2"}[expr]
    while pc not in visited:
        visited.add(pc)
        *stmts,end = cfg[pc]
        for s in stmts:
            m = re.fullmatch(r"(_\d+) = (.+);",s)
            checked(m,f"unrecognized statement {s}")
            values[m[1]] = val(m[2])
        if m := re.fullmatch(CALL,end):
            dest,name,args,nxt = m.groups()
            args = [val(arg) for arg in args.split(", ")]
            if name == "Score::is_finite" and args == ["score"]:
                call = "finite"
            elif name == "Score::value" and args == ["score"]:
                call = "rawScore"
            elif name == "Selectivity::as_u8" and args == ["selectivity"]:
                call = "(BitVec.ofNat 8 selectivity.val)"
            elif name == "TranspositionTable::generation" and args == ["table"]:
                call = "(generation &&& 127)"
            elif name == "TTEntryData::new":
                call = "(makeData " + " ".join(args) + ")"
            elif name == "TTEntry::save":
                checked(result is None,"multiple TT writes in store")
                result = "some (" + ", ".join(args) + ")"
                call = "()"
            elif name == f"Option::<Square>::map_or::<u8, {closure}>" and args == ["move","Packing.noMove",closure]:
                call = "(Packing.encodeBestMove move)"
            else:
                raise ValueError(f"unknown {name} {args}")
            values[dest] = call
            pc = int(nxt)
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];",end):
            pc = int(m[3] if truth[values[m[1]]] else m[2])
        elif m := re.fullmatch(GOTO,end):
            pc = int(m[1] or m[2])
        else:
            checked(end == "return;",f"unexpected exit {end}")
            return (result or "none") if values["_0"] == "()" else values["_0"]
    raise ValueError("unexpected constructor/store cycle")


def render(dump,repo):
    closure = dependencies(repo,dump)
    constructor = read_cfg(dump,"{impl#3}","new",NEW_SIG)
    store = read_cfg(dump,"{impl#7}","store",STORE_SIG)
    fields = ["score","move","bound","depth","selectivity","generation","endgame","pv"]
    registers = lambda names: {f"_{i}":name for i,name in enumerate(names,1)}
    def data(endgame,pv):
        return run(constructor,registers(fields),closure,{"endgame":endgame,"pv":pv})
    def save(finite):
        return run(store,registers(["table","slot","board","score","bound","depth","move",
                                    "selectivity","endgame","pv"]),closure,{"finite":finite})
    return "\n".join(["import NeuralReversi.TranspositionTable.Packing", "",
        "namespace NeuralReversi.TranspositionTable.WriterMir", "",
        "def makeData (score : Score) (move : Option (Fin 64)) (bound : Bound) (depth : Depth)",
        "    (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool) : EntryData :=",
        "  if endgame then", "    if pv then " + data(True,True), "    else " + data(True,False),
        "  else", "    if pv then " + data(False,True), "    else " + data(False,False), "",
        "def storeInput (slot : σ) (board : β) (score : Score) (bound : Bound) (depth : Depth)",
        "    (move : Option (Fin 64)) (selectivity : Selectivity) (generation : BitVec 8)",
        "    (endgame pv : Bool) : Option (σ × β × EntryData) :=",
        "  if score.raw.toInt.natAbs < 32767 then " + save(True), "  else " + save(False), "",
        "end NeuralReversi.TranspositionTable.WriterMir",""])
