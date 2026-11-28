"""Project the normal locked root-vector mutations from pinned rustc MIR."""
import re

from extract_root_wipeout import constants
from mir_common import run_extractor_cli, blocks, GOTO, checked, split_call


METHODS = {
    "reset": "root_move-{impl#1}-reset_scores_from_pv_idx",
    "index": "root_move-{impl#1}-pv_idx",
    "update": "root_move-{impl#1}-update",
    "sort": "root_move-{impl#1}-sort_from_pv_idx",
    "key": "root_move-{impl#1}-sort_from_pv_idx-{closure#0}",
    "find": "root_move-{impl#1}-update-{closure#0}",
    "map": "root_move-{impl#1}-update-{closure#1}",
    "context": "context-{impl#0}-update_root_move",
}
FIELDS = ["square", "score", "average", "meanSquared", "pv"]
LOCK = "std::sync::Mutex::<Vec<RootMove>>::lock"
UNWRAP = "Result::<std::sync::MutexGuard<'_, Vec<RootMove>>, PoisonError<std::sync::MutexGuard<'_, Vec<RootMove>>>>::unwrap"
DEREF = "<std::sync::MutexGuard<'_, Vec<RootMove>> as DerefMut>::deref_mut"
ITER = "core::slice::<impl [RootMove]>::iter_mut"
PV_ITER = "std::slice::Iter<'_, Option<Square>>"
SKIP = "Skip<std::slice::IterMut<'_, RootMove>>"


def program(dump, name):
    path = dump / f"reversi_core.search-{METHODS[name]}.SimplifyCfg-initial.after.mir"
    source = path.read_text(encoding="utf-8")
    method = METHODS[name].split("}-", 1)[1].replace("-{closure#","::{closure#")
    checked(source.splitlines()[0].endswith(f"::{method}" + chr(96) + " after SimplifyCfg-initial"), f"{name}: wrong MIR owner")
    normal = re.sub(r"\{closure@[^}]+\}", "CLOSURE", source)
    sig = next(line for line in normal.splitlines() if line.startswith("fn "))
    signatures = {
        "index": "(_1: &RootMoves) -> usize",
        "reset": "(_1: &RootMoves) -> ()",
        "sort": "(_1: &RootMoves) -> ()",
        "key": "(_1: &mut CLOSURE, _2: &RootMove) -> Reverse<Score>",
        "update": "(_1: &RootMoves, _2: Square, _3: Score, _4: bool, _5: &[Option<Square>; WIDTH]) -> ()",
        "find": "(_1: &mut CLOSURE, _2: &&mut RootMove) -> bool",
        "map": "(_1: &mut CLOSURE, _2: &Option<Square>) -> Option<Square>",
        "context": "(_1: &mut SearchContext<'_>, _2: Square, _3: Score, _4: bool) -> ()",
    }
    expected = re.escape("::" + method + signatures[name] + " {").replace("WIDTH", r"\d+")
    checked(re.search(expected + "$",sig) is not None, f"{name}: unexpected signature {sig}")
    return blocks(normal), source


def execute(dump, name, seeded=False, visible=False):
    cfg, src = program(dump, name)
    inputs = {
        "update": ["roots", "sq", "value", "visible", "pvSlots"],
        "reset": ["roots"], "index": ["roots"], "context": ["ctx", "sq", "value", "visible"],
        "find": ["closure", "candidate"], "map": ["closure", "slot"],
        "sort": ["roots"], "key": ["closure", "keyRow"],
    }[name]
    env = {f"_{n}": value for n, value in enumerate(inputs, 1)}
    fields = {f: "old." + f for f in FIELDS}
    effects, checks = [], []
    iterated = False
    loop_header = None
    next_call = None
    width = re.search(r"_5: &\[Option<Square>; (\d+)\]", src) if name == "update" else None

    def field(base, index):
        if base == "row":
            return FIELDS[int(index)]
        if base == "keyRow":
            return "key." + FIELDS[int(index)]
        table = {("roots", "0"): "mutex", ("roots", "1"): "indexCell",
                 ("ctx", "1"): "roots", ("candidate", "0"): "rowSquare", ("closure", "0"): "sq"}
        return table[base, index]

    def expr(raw):
        s = re.sub(r"^(?:copy |move |&mut |&)", "", raw)
        if s in env:
            return env[s]
        if m := re.fullmatch(r"\(\*(_\d+)\)", s):
            return env[m[1]]
        if m := re.fullmatch(r"\(\*\(\(\*(_\d+)\)\.(\d+): .+\)\)", s):
            return field(env[m[1]], m[2])
        if m := re.fullmatch(r"\(\(\*\(\*(_\d+)\)\)\.(\d+): .+\)", s):
            return field(env[m[1]], m[2])
        if m := re.fullmatch(r"\(\(\*(_\d+)\)\.(\d+): .+\)", s):
            key = field(env[m[1]], m[2])
            return fields[key] if env[m[1]] == "row" and not raw.startswith("&mut ") else key
        if m := re.fullmatch(r"\((_\d+)\.0: i64\)", s):
            return env[m[1]]
        if m := re.fullmatch(r"std::ops::RangeFrom::<usize> \{ start: (.+) \}",s):
            checked(expr(m[1]) == "idx", "sort starts at wrong offset")
            return "range"
        if m := re.fullmatch(r"Reverse::<Score>\((.+)\)",s):
            return "descending " + expr(m[1])
        if m := re.fullmatch(r"Lt\((.+), (.+)\)",s):
            checked((expr(m[1]),expr(m[2])) == ("idx","length"), "sort bounds")
            return "inside"
        if m := re.fullmatch(r"\(\((_\d+) as Some\)\.0: &mut search::root_move::RootMove\)", s):
            checked(env[m[1]] == "optionRow" and iterated, "invalid iterator item")
            return "row"
        if m := re.fullmatch(r"discriminant\((_\d+)\)", s):
            checked(env[m[1]] == "optionRow","invalid reset discriminant")
            return "hasItem"
        if m := re.fullmatch(r"CLOSURE \{ sq: (.+) \}", s):
            checked(expr(m[1]) == "sq","wrong captured selector square")
            return "matcher"
        if m := re.fullmatch(r"(_\d+) as &\[std::option::Option<square::Square>\] \(PointerCoercion\(Unsize, Implicit\)\)", s):
            checked(env[m[1]] == "pvSlots","wrong PV buffer")
            return "pvSlots"
        if s in ("CLOSURE","const ()","std::sync::atomic::Ordering::Relaxed","const score::Score::INF",
                 "const 2_i32","const 2_i64","const 0_i64","const -1_i64","const i64::MIN"):
            return s
        if m := re.fullmatch(r"(MulWithOverflow|Mul|AddWithOverflow|Add|Eq|BitAnd|Div)\((.+), (.+)\)", s):
            a,b = expr(m[2]),expr(m[3])
            if m[1] in ("MulWithOverflow","Mul"):
                checked([a,b] == ["value","absolute"], "signed square operands")
                return "squared"
            if m[1] in ("AddWithOverflow","Add"):
                checked([a,b] == ["old.meanSquared","squared"], "history sum operands")
                return "squareSum"
            if m[1] == "Div":
                checked(a == "squareSum" and b == "const 2_i64", "squared history division")
                return "squareSum.tdiv 2"
            if m[1] == "Eq":
                checked((a,b) in (("const 2_i64","const 0_i64"),("const 2_i64","const -1_i64"),("squareSum","const i64::MIN")),"unexpected division guard")
                return "divGuard"
            checked(a == b == "divGuard", "incorrect division checks")
            return "divGuard"
        raise ValueError(f"{name}: unsupported operand {raw}")

    def call(fn, vals):
        nonlocal next_call
        ops = {
            LOCK: (["mutex"], "locked"), UNWRAP: (["locked"], "guard"), DEREF: (["guard"], "vec"),
            "<std::sync::MutexGuard<\'_, Vec<RootMove>> as Deref>::deref": (["guard"],"vec"),
            "Vec::<RootMove>::len": (["vec"],"length"),
            "<Vec<RootMove> as IndexMut<std::ops::RangeFrom<usize>>>::index_mut": (["vec","range"],"tail"),
            "std::slice::<impl [RootMove]>::sort_by_key::<Reverse<Score>, CLOSURE>": (["tail","CLOSURE"],"sort"),
            "<Vec<RootMove> as DerefMut>::deref_mut": (["vec"],"slice"), ITER: (["slice"],"iter"),
            "<std::slice::IterMut<'_, RootMove> as Iterator>::find::<CLOSURE>": (["iter","matcher"],"found"),
            "Option::<&mut RootMove>::unwrap": (["found"],"row"), "Score::value": (["value"],"value"),
            "core::num::<impl i32>::abs": (["value"],"absolute"),
            "<Score as Neg>::neg": (["const score::Score::INF"],"negInf"),
            "<Score as PartialEq>::eq": (["old.average","negInf"],"isSeed"),
            "<Score as Add>::add": (["old.average","value"],"averageSum"),
            "<Score as Div<i32>>::div": (["averageSum","const 2_i32"],"averageSum.tdiv 2"),
            "RootMoves::pv_idx": (["roots"],"idx"),
            "<std::slice::IterMut<'_, RootMove> as Iterator>::skip": (["iter","idx"],"suffix"),
            f"<{SKIP} as IntoIterator>::into_iter": (["suffix"],"suffix"),
            f"<{SKIP} as Iterator>::next": (["suffix"], "optionRow"),
            "Atomic::<usize>::load": (["indexCell","std::sync::atomic::Ordering::Relaxed"],"index"),
            "<Square as PartialEq>::eq": (["rowSquare","sq"],"matches"),
            "SearchContext::<'_>::update_pv": (["ctx","sq"],"updatePv"),
            "SearchContext::<'_>::pv": (["ctx"],"currentPv"),
            "RootMoves::update": (["roots","sq","value","visible","currentPv"],"write"),
            "Vec::<Square>::clear": (["pv"],"clear"),
            "core::slice::<impl [Option<Square>]>::iter": (["pvSlots"],"pvIter"),
            f"<{PV_ITER} as Iterator>::map_while::<Square, CLOSURE>": (["pvIter","CLOSURE"],"pvPrefix"),
            f"<Vec<Square> as Extend<Square>>::extend::<MapWhile<{PV_ITER}, CLOSURE>>": (["pv","pvPrefix"],"extend"),
        }
        if fn == "<i64 as From<i32>>::from":
            checked(vals in (["value"],["absolute"]), "signed extension")
            return vals[0]
        if fn == "slice::<impl [RootMove]>::sort_by_key::<Reverse<Score>, CLOSURE>":
            fn = "std::" + fn
        expected,value = ops[fn]
        checked(vals == expected, f"{name}: {fn} got {vals}, expected {expected}")
        if value == "optionRow":
            checked(next_call in (None,pc),"reset must return to the same iterator next call")
            next_call = pc
        if value in ("locked","row","suffix","idx","optionRow","updatePv","currentPv","write","clear","extend","sort"):
            effects.append(value)
        if value == "clear":
            fields["pv"] = "[]"
        if value == "extend":
            checked(fields["pv"] == "[]", "destination was not cleared")
            fields["pv"] = "pvPrefix slots.toList"
        return value

    pc = 0
    for _ in range(70):
        bb = [s for s in cfg[pc] if not s.startswith(("StorageLive(","StorageDead(","FakeRead(","PlaceMention("))]
        for s in bb[:-1]:
            if m := re.fullmatch(r"(_\d+) = (.*);", s):
                env[m[1]] = expr(m[2])
            elif m := re.fullmatch(r"\(\(\*(_\d+)\)\.(\d+): [^()]+\) = (.+);", s):
                checked(env[m[1]] == "row","store to wrong object")
                f = field("row",m[2])
                fields[f] = expr(m[3])
                effects.append(f)
            else:
                raise ValueError(f"{name} bb{pc}: {s}")
        end = bb[-1]
        if end == "return;":
            return env.get("_0"), fields, effects, checks, int(width[1]) if width else None
        if m := split_call(end):
            args = [expr(v) for v in m[3].split(", ")] if m[3] else []
            env[m[1]] = call(m[2],args)
            pc = int(m[4])
        elif m := re.fullmatch(GOTO,end):
            pc = int(m[1] or m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",end):
            checked(env[m[1]] == "guard","wrong lock release")
            effects.append("unlock")
            pc = int(m[2])
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[(.+)\];",end):
            tag = expr(m[1])
            choices = dict(v.split(": bb") for v in m[2].split(", "))
            if tag == "hasItem":
                checked(loop_header in (None,pc) and next_call is not None, "incorrect reset loop backedge")
                loop_header = pc
                truth = not iterated
                iterated = True
            else:
                truth = {"isSeed":seeded,"visible":visible,"inside":visible}[tag]
            pc = int(choices.get(str(int(truth)),choices["otherwise"]))
        elif m := re.fullmatch(r'assert\(!(.+?), "[^"]+", .+\) -> \[success: bb(\d+), unwind: bb\d+\];',end):
            pred = m[1]
            if val := re.fullmatch(r'move \((_\d+)\.1: bool\)',pred):
                checks.append(env[val[1]])
            else:
                checked(expr(pred) == "divGuard","unrecognized overflow check")
            pc = int(m[2])
        else:
            raise ValueError(f"{name} bb{pc}: {end}")
    raise ValueError(f"{name}: invalid loop")


def render(dump, repo):
    inf = constants(repo.resolve())["const score::Score::INF"]
    for name, result in (("index", "index"), ("find", "matches"), ("map", "slot"), ("key","descending key.score")):
        value, _, effects, _, _ = execute(dump, name)
        checked(value == result and not effects, f"{name}: return value")
    for inside in (False,True):
        _, _, effects, _, _ = execute(dump, "sort", visible=inside)
        checked(effects == ["idx","locked"] + (["sort"] if inside else []) + ["unlock"], "incorrect suffix sort")
    (_, reset, actions, _, _) = execute(dump, "reset")
    checked(actions == ["idx","locked","suffix","suffix","optionRow","score","optionRow","unlock"],
            f"reset: incorrect suffix traversal {actions}")
    checked(reset["score"] == "negInf" and all(reset[k] == f"old.{k}" for k in FIELDS if k != "score"),
            "reset writes fields other than score")
    paths = {}
    for shown in (False, True):
        _, _, ctx, _, _ = execute(dump, "context", visible=shown)
        checked(ctx == (["updatePv"] if shown else []) + ["currentPv","write"], "PV data not read at the update")
        for seed in (False, True):
            _, path, effects, guards, length = execute(dump, "update", seed, shown)
            checked(length is not None and length > 0, "missing PV array width")
            checked(effects == ["locked","row","average","meanSquared","score"] +
                    (["clear","extend"] if shown else []) + ["unlock"], "unmatched store or lock scope")
            checked(guards in ([], ["squared"] if seed else ["squared","squareSum"]), "invalid arithmetic assert")
            paths[shown,seed] = path

    def term(key):
        def choose(test, yes, no):
            return yes if yes == no else f"(if {test} then {yes} else {no})"
        return choose("shown", *(choose("seed", paths[v,True][key], paths[v,False][key]) for v in (True,False)))

    lean = [
        "module",
        "public import NeuralReversi.RootMoveStorage",
        "public import NeuralReversi.MidgameAspiration.Arithmetic", "",
        "@[expose] public section", "",
        "namespace NeuralReversi.PvsParallel.RootStorageMir", "",
        "open Bitboard (Square)", "open RootMoveStorage (Row)", "",
        f"def capacity : Nat := {length}", "abbrev Slots := Vector (Option Square) capacity",
        f"def negInf : Int := -{inf}", "",
        "def pvPrefix : List (Option Square) → List Square",
        "  | some sq :: tail => sq :: pvPrefix tail", "  | _ => []", "",
        "def reset (idx : Nat) (rows : List Row) : List Row :=",
        "  rows.take idx ++ (rows.drop idx).map (fun old => {old with score := " + reset["score"] + "})", "",
        "def sort (idx : Nat) (rows : List Row) : List Row :=",
        "  if idx < rows.length then",
        "    rows.take idx ++ (rows.drop idx).mergeSort (fun a b => decide (b.score ≤ a.score)) else rows", "",
        "def changed (old : Row) (value : Int) (shown : Bool) (slots : Slots) : Row :=",
        "  let seed := old.average = negInf", "  let squared := value * (value.natAbs : Int)",
        "  let averageSum := old.average + value", "  let squareSum := old.meanSquared + squared",
        "  ⟨" + ", ".join(term(k) for k in FIELDS) + "⟩", "",
        "instance (bits : Nat) (x : Int) : Decidable (MidgameAspiration.FitsSigned bits x) :=",
        "  inferInstanceAs (Decidable (_ ≤ x ∧ x < _))", "",
        "def numeric (old : Row) (value : Int) : Bool :=",
        "  let squared := value * (value.natAbs : Int)",
        "  decide (MidgameAspiration.FitsSigned 32 (value.natAbs : Int) ∧ MidgameAspiration.FitsSigned 64 squared ∧",
        "    (old.average ≠ negInf → MidgameAspiration.FitsSigned 32 (old.average + value) ∧",
        "      MidgameAspiration.FitsSigned 64 (old.meanSquared + squared)))", "",
        "def update (square : Square) (value : Int) (shown : Bool) (slots : Slots) : List Row → Option (List Row)",
        "  | [] => none", "  | old :: tail => if old.square = square then",
        "      if numeric old value then some (changed old value shown slots :: tail) else none",
        "    else (update square value shown slots tail).map (old :: ·)", "",
        "end NeuralReversi.PvsParallel.RootStorageMir", "",
    ]
    return "\n".join(lean)


def main():
    run_extractor_cli(
        __doc__,
        render,
        check_message=lambda p: f"generated storage MIR differs from {p}",
    )


if __name__ == "__main__":
    main()

