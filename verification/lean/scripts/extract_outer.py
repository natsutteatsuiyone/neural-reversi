"""Translate the non-aborted endgame ladder, including normal and time-limit returns."""

import itertools
import re

from mir_common import blocks, CALL, GOTO, checked
from extract_header import binding
from check_outer_state import check as check_state


ROOT = "reversi_core.search-endgame-"
PASS = ".SimplifyCfg-initial.after.mir"
FIELDS = {
    ("task", "0", "board::Board"): "board",
    ("task", "8", "std::option::Option<std::sync::Arc<search::time_control::TimeManager>>"): "optionalTime",
    ("ctx", "1", "&search::root_move::RootMoves"): "roots",
    ("ctx", "2", "u64"): "nodes",
    ("ctx", "7", "probcut::Selectivity"): "sel",
    ("ctx", "9", "eval::EvalMode"): "mode",
    ("ctx", "10", "empty_list::EmptyList"): "empty",
}
ITER_PV = "<std::ops::Range<usize> as Iterator>::next"
ITER_SEL = "<Enumerate<std::array::IntoIter<Selectivity, 5>> as Iterator>::next"
QUERY = "search::endgame::aspiration_search"
SORT = "RootMoves::sort_from_pv_idx"
MARK = "CompletedState::line_completed"
CAPTURE = "CompletedState::iteration_completed"
FINISH = "CompletedState::into_result"


def execute(cfg, names, start, stops, options):
    timer, cb, have_row, have_pv, have_level, above, reportable, stop, timeout = options
    env = {"_1": ("ref", "ctx"), "_2": ("ref", "task"), "_3": ("ref", "arc"), "_4": "base",
           "_5": "completed", "_6": "minimum", "_7": "last"}
    env.update({v: v for v in ("ctx", "task", "board", "arc", "optionalTime", "roots", "nodes",
                               "mode", "empty", "timer", "thread", "callback", "progress")})
    env["sel"] = "level"
    before = {"board": ("ref", "board"), "n_empties": "depth", "time_manager": "time" if timer else "none",
              "endgame_clock": "clock" if timer else "none", "pv_count": "count", "alpha": "alpha", "beta": "beta"}
    if start:
        env.update({names[n]: v for n, v in before.items()})
        env[names["lines"]] = "range"
        env[names["ladder"]] = "enumerated"
    events = []
    pc = start

    def ref(value):
        checked(isinstance(value, tuple) and value[0] == "ref", f"not a reference {value}")
        return value[1]

    def place(text):
        if re.fullmatch(r"_\d+", text):
            return text
        if m := re.fullmatch(r"\(\*(_\d+)\)", text):
            return ref(env[m[1]])
        if m := re.fullmatch(r"\(\(\*(_\d+)\)\.(\d+): (.+)\)", text):
            parent = ref(env[m[1]])
            if parent == "task" and m[2] == "7" and "Option<std::sync::Arc<dyn std::ops::Fn(" in m[3]:
                return "optionalCallback"
            return FIELDS[(parent, m[2], m[3])]
        if m := re.fullmatch(r"\(\*\(\(\*(_\d+)\)\.1: &search::root_move::RootMoves\)\)", text):
            checked(ref(env[m[1]]) == "ctx", "invalid root receiver")
            return "roots"
        raise ValueError(f"unknown place {text}")

    env["optionalCallback"] = "callbackPresent" if cb else "none"

    def expr(text):
        if text == "const ()":
            return "unit"
        if text in ("const true", "const false"):
            return int(text == "const true")
        if m := re.fullmatch(r"const (\d+)_usize", text):
            return int(m[1])
        if text in ("const search::endgame::INTER_SELECTIVITY_DELTA", "const score::Score::INF",
                    "const probcut::Selectivity::ENDGAME_LADDER", "Selectivity::None", "EvalMode::Small",
                    "TimeManager::endgame_clock"):
            return text
        if m := re.fullmatch(r"std::ops::Range::<usize> \{ start: (.*), end: (.*) \}", text):
            checked([expr(m[1]),expr(m[2])] == [0,"count"], "unexpected range")
            return "range"
        if m := re.fullmatch(r"discriminant\((.*)\)", text):
            value = env[place(m[1])]
            return {"none":0, "nextLine":1, "nextLevel":1, "time":1, "rowOption":1, "callbackPresent":1}[value]
        if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: (usize|&search::time_control::TimeManager)\)", text):
            return {("nextLine","usize"):"idx", ("time","&search::time_control::TimeManager"):("ref","timer")}[(env[m[1]],m[2])]
        if m := re.fullmatch(r"copy \(\(\((_\d+) as Some\)\.0: \(usize, probcut::Selectivity\)\)\.([01]): (?:usize|probcut::Selectivity)\)", text):
            checked(env[m[1]] == "nextLevel", "invalid ladder item")
            return ("step","level")[int(m[2])]
        if m := re.fullmatch(r"copy \((_\d+)\.([01]): score::Score\)", text):
            checked(env[m[1]] == "seed", "wrong seed")
            return ("alpha","beta")[int(m[2])]
        if m := re.fullmatch(r"move \(\((_\d+) as Some\)\.0: search::root_move::RootMove\)", text):
            checked(env[m[1]] == "rowOption", "missing current PV")
            return "row"
        if m := re.fullmatch(r"&\(\(\(\(\*(_\d+)\)\.7: .+\) as Some\)\.0: std::sync::Arc<dyn std::ops::Fn\(.+\)", text):
            checked(cb and ref(env[m[1]]) == "task", "missing callback")
            return ("ref","callback")
        if m := re.fullmatch(r"(\{closure@[^}]+\})(?: \{ ([\w_]+): (move _\d+) \})?", text):
            return ("closure",m[1],(m[2],ref(expr(m[3]))) if m[2] else None)
        if m := re.fullmatch(r"\(move (_\d+),\)", text):
            return ("tuple",env[m[1]])
        if m := re.fullmatch(r"(?:copy|move) (.+)", text):
            return env[place(m[1])]
        if m := re.fullmatch(r"&(?:mut )?(.+)", text):
            return ("ref",place(m[1]))
        raise ValueError(f"unsupported expression {text}")

    def call(fn, raw):
        args = [expr(a) for a in raw.split(", ") if a]
        vals = [env[v[1]] if isinstance(v,tuple) and v[0]=="ref" else v for v in args]
        externals = {
            "Option::<Arc<TimeManager>>::as_deref": (["optionalTime"],"time" if timer else "none"),
            "TimeManager::set_endgame_mode": (["timer",1],"unit"),
            "EmptyList::count": (["empty"],"depth"),
            "CompletedState::begin_phase": (["completed"],"unit"),
            "RootMoves::count": (["roots"],"rootCount"),
            "SearchTask::pv_count": (["task","rootCount"],"count"),
            "<std::ops::Range<usize> as IntoIterator>::into_iter": (["range"],"range"),
            ITER_PV: (["range"],"nextLine" if have_pv else "none"),
            "RootMoves::set_pv_idx": (["roots","idx"],"unit"),
            "core::num::<impl usize>::checked_sub": (["idx",1],"previousIndex"),
            "initial_aspiration_window": (["idx","base","previousScore"],"seed"),
            "<[Selectivity; 5] as IntoIterator>::into_iter": (["const probcut::Selectivity::ENDGAME_LADDER"],"arrayIter"),
            "<std::array::IntoIter<Selectivity, 5> as Iterator>::enumerate": (["arrayIter"],"enumerated"),
            "<Enumerate<std::array::IntoIter<Selectivity, 5>> as IntoIterator>::into_iter": (["enumerated"],"enumerated"),
            ITER_SEL: (["enumerated"],"nextLevel" if have_level else "none"),
            "<Selectivity as PartialOrd>::gt": (["level","last"],above),
            "<Selectivity as PartialOrd>::ge": (["level","minimum"],reportable),
            QUERY: (["ctx","board","alpha","beta","arc"],"reply"),
            "<Score as Sub>::sub": (["reply","const search::endgame::INTER_SELECTIVITY_DELTA"],"low"),
            "<Score as Neg>::neg": (["const score::Score::INF"],"negInf"),
            "<Score as Ord>::max": (["low","negInf"],"nextAlpha"),
            "<Score as Add>::add": (["reply","const search::endgame::INTER_SELECTIVITY_DELTA"],"high"),
            "<Score as Ord>::min": (["high","const score::Score::INF"],"nextBeta"),
            "<Arc<threading::thread::Thread> as Deref>::deref": (["arc"],("ref","thread")),
            "threading::thread::Thread::is_search_aborted": (["thread"],0),
            SORT: (["roots"],"unit"), MARK: (["completed","ctx","depth",1],"maybeCaptured"),
            "RootMoves::current_pv": (["roots"],"rowOption" if have_row else "none"),
            "SearchProgress::from_iteration": (["ctx","row","depth","depth","reply","nodes",1],"progress"),
            "<Arc<dyn Fn(SearchProgress) + Send + Sync> as Deref>::deref": (["callback"],("ref","callback")),
            "<dyn Fn(SearchProgress) + Send + Sync as Fn<(SearchProgress,)>>::call": (["callback",("tuple","progress")],"unit"),
            "Option::<EndgameClock<'_>>::as_ref": (["clock" if timer else "none"],"clockRef" if timer else "none"),
            CAPTURE: (["completed","ctx","depth",1],"captured"),
            FINISH: (["completed","ctx","board"],"reported"),
        }
        if fn in externals:
            expected,result=externals[fn]
            checked(vals==expected,f"{fn} expected {expected}, got {vals}")
            if fn in (QUERY,MARK,CAPTURE):
                checked(env["sel"]=="level", "reported selectivity not the current level")
            if fn in (QUERY,SORT,MARK,CAPTURE,FINISH,"RootMoves::set_pv_idx","CompletedState::begin_phase"):
                events.append(fn)
            return result
        if fn.startswith("Option::<&TimeManager>::map::<EndgameClock"):
            checked(vals==["time" if timer else "none","TimeManager::endgame_clock"],"wrong clock")
            return "clock" if timer else "none"
        if m := re.fullmatch(r"Option::<usize>::and_then::<RootMove, (\{closure@[^}]+\})>",fn):
            checked(vals==["previousIndex",("closure",m[1],("ctx__root_moves","roots"))],"wrong previous root")
            return "previousRow"
        if m := re.fullmatch(r"Option::<RootMove>::map::<Score, (\{closure@[^}]+\})>",fn):
            checked(vals==["previousRow",("closure",m[1],None)],"wrong previous score")
            return "previousScore"
        if m := re.fullmatch(r"Option::<&EndgameClock<'_>>::is_some_and::<(\{closure@[^}]+\})>",fn):
            checked(vals==["clockRef" if timer else "none",("closure",m[1],("step",names["step"]))],
                    "clock observed wrong level")
            return int(timer and stop)
        if m := re.fullmatch(r"Option::<&TimeManager>::is_some_and::<(\{closure@[^}]+\})>",fn):
            checked(vals==["time" if timer else "none",("closure",m[1],None)],"time poll")
            return int(timer and timeout)
        raise ValueError(f"bb{pc} unknown {fn} {vals}")

    for ticks in range(250):
        if ticks and pc in stops:
            return stops[pc],events,env
        *stmts,end=cfg[pc]
        for s in stmts:
            if re.fullmatch(r"(?:Storage(?:Live|Dead)|PlaceMention)\(.*\);|FakeRead\((?:ForLet|ForMatchedPlace)\(None\), .+\);|AscribeUserType\(.+\);",s):
                continue
            m=re.fullmatch(r"(.+) = (.+);",s)
            checked(m,f"bb{pc} {s}")
            where=place(m[1])
            env[where]=expr(m[2])
            if where == "sel":
                events.append(("set",env[where]))
        if m:=re.fullmatch(CALL,end):
            env[m[1]]=call(m[2],m[3])
            pc=int(m[4])
        elif m:=re.fullmatch(r"switchInt\(move (_\d+)\) -> \[((?:\d+: bb\d+, )*otherwise: bb\d+)\];",end):
            cases=dict(re.findall(r"(\w+): bb(\d+)",m[2]))
            condition=env[m[1]]
            checked(isinstance(condition,int),f"unknown branch {condition}")
            pc=int(cases.get(str(int(condition)),cases["otherwise"]))
        elif m:=re.fullmatch(GOTO,end):
            pc=int(m[1] or m[2])
        elif m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",end):
            checked(m[1] in names["drops"], f"unknown destructor {m[1]}")
            pc=int(m[2])
        elif end=="return;":
            checked(env["_0"]=="reported","wrong public return")
            return "return",events,env
        else:
            raise ValueError(f"bb{pc} unsupported {end}")
    raise ValueError("loop did not reach a cutpoint")


def locations(cfg,source):
    names={s:binding(source,s) for s in ("board","n_empties","time_manager","endgame_clock","pv_count",
                                          "alpha","beta","step","selectivity")}
    iter_names=re.findall(r"^\s*debug iter => (_\d+);$",source,re.M)
    checked(len(iter_names)==2,"wrong iterator nesting")
    names["lines"],names["ladder"]=iter_names
    drop_types = {"std::iter::Enumerate<std::array::IntoIter<probcut::Selectivity, 5>>",
                  "std::option::Option<search::root_move::RootMove>", "search::root_move::RootMove",
                  "search::progress::SearchProgress", "search::result::CompletedState"}
    names["drops"] = {"_5"} | {reg for reg, ty in re.findall(r"^\s*let (?:mut )?(_\d+): (.*);$",source,re.M)
                               if ty in drop_types}
    heads={}
    for fn in (ITER_PV,ITER_SEL):
        body,=[n for n,b in cfg.items() if f" = {fn}(" in b[-1]]
        head,=[n for n,b in cfg.items() if re.fullmatch(rf"falseUnwind -> \[real: bb{body}, unwind: bb\d+\];",b[-1])]
        heads[fn]=head
    drops=[(n,int(m[1])) for n,b in cfg.items()
           if (m:=re.fullmatch(r"drop\("+names["ladder"]+r"\) -> \[return: bb(\d+), unwind: bb\d+\];",b[-1]))]
    (_,follow),=drops
    next_drop=re.fullmatch(r"drop\(_\d+\) -> \[return: bb(\d+), unwind: bb\d+\];",cfg[follow][-1])
    checked(next_drop,"missing full level iterator drop")
    return names,heads[ITER_PV],heads[ITER_SEL],int(next_drop[1])


def check_ladder(dump):
    source=(dump/(ROOT+"solve_root"+PASS)).read_text(encoding="utf-8")
    checked(source.startswith("// MIR for `search::endgame::solve_root` after SimplifyCfg-initial\n"), "wrong solve_root")
    cfg=blocks(source)
    names,lines,levels,after=locations(cfg,source)
    stops={lines:"line",levels:"level",after:"after"}
    defaults=(False,)*9
    for timed in (False,True):
        opts=(timed,*defaults[1:])
        out,seen,env=execute(cfg,names,0,stops,opts)
        checked(out=="line" and seen==["CompletedState::begin_phase",("set","Selectivity::None")]
                and env["mode"]=="EvalMode::Small" and env[names["lines"]]=="range","wrong ladder entry")
    for have_line,publish in itertools.product((False,True),repeat=2):
        opts=(False,False,False,have_line,False,False,publish,False,False)
        out,seen,env=execute(cfg,names,lines,stops,opts)
        if have_line:
            checked(out=="level" and seen==["RootMoves::set_pv_idx"] and env[names["ladder"]]=="enumerated"
                    and [env[names[n]] for n in ("alpha","beta")]==["alpha","beta"],"bad line entry")
        else:
            checked(out=="return" and seen==([CAPTURE] if publish else [])+[FINISH],"bad ladder output")
    for timed,cb,row,item,above,publish,stop in itertools.product((False,True),repeat=7):
        opts=(timed,cb,row,True,item,above,publish,stop,False)
        out,seen,env=execute(cfg,names,levels,stops,opts)
        if not item or above:
            checked(out=="after" and not seen,"ineligible level searched")
        else:
            checked(out==("after" if timed and stop else "level"),"incorrect clock transition")
            expected=[("set","level"),QUERY,SORT]+([MARK] if publish else [])
            checked(seen==expected,"unpublished or reordered root query")
            checked([env[names[n]] for n in ("alpha","beta")]==["nextAlpha","nextBeta"],"invalid next window")
    for timed,timeout in itertools.product((False,True),repeat=2):
        opts=(timed,False,False,False,False,False,False,False,timeout)
        out,seen,_=execute(cfg,names,after,stops,opts)
        checked((out,seen)==(("return",[FINISH]) if timed and timeout else ("line",[])),"wrong end of line")


def ladder_constants(dump, repo):
    path=dump/("reversi_core.probcut-{impl#0}-ENDGAME_LADDER"+PASS)
    source=blocks(path.read_text(encoding="utf-8"))
    checked(set(source)=={0} and source[0][-1]=="return;","unexpected ladder")
    levels={}
    for statement in source[0][:-1]:
        if re.fullmatch(r"Storage(Live|Dead)\(_\d+\);",statement):
            continue
        if m:=re.fullmatch(r"(_\d+) = Selectivity::(\w+);",statement):
            levels[m[1]]=m[2]
        elif m:=re.fullmatch(r"_0 = \[(move _\d+(?:, move _\d+)*)\];",statement):
            result=[levels[a.removeprefix("move ")] for a in m[1].split(", ")]
        else:
            raise ValueError(f"unsupported ladder statement {statement}")
    src=(repo/"crates/reversi-core/src/probcut.rs").read_text(encoding="utf-8")
    body,=re.findall(r"#\[repr\(u8\)\]\s*pub enum Selectivity \{(.*?)\n\}",src,re.S)
    ordinals=dict(re.findall(r"^\s*(\w+) = (\d+),$",body,re.M))
    checked([ordinals[x] for x in result] == ["1","2","3","4","5"] and result[-1]=="None","invalid ladder")
    return [int(ordinals[x]) for x in result]


def render(dump, repo):
    check_ladder(dump)
    check_state(dump)
    levels=ladder_constants(dump,repo)
    return f"""import NeuralReversi.PublicationMir.Correct

namespace NeuralReversi.OuterMir

def ladder : List Nat := {levels}

def directEntry (count : Nat) (multi : Bool) : Option Nat :=
  if count = 0 then none else some (if multi then count else 1)

def lineStamp (lines : List (Option (Nat × Nat))) (idx depth level : Nat) :
    List (Option (Nat × Nat)) × Bool :=
  let mark := some (depth, level)
  let next := lines.set idx mark
  (next, next.all (· == mark))

inductive Step where
  | endLine
  | accepted (level : Nat) (publish mayContinue : Bool)
deriving DecidableEq, Repr

def step (next : Option Nat) (last minimum : Nat) (stop : Bool) : Step :=
  match next with
  | none => .endLine
  | some level =>
    if last < level then .endLine else .accepted level (decide (minimum ≤ level)) (!stop)

def finish (minimum level : Nat) (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (depth nodes : Nat) (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    Option PublicationMir.SearchResult :=
  if minimum ≤ level then PublicationMir.completed sort rows depth level true nodes fallback
  else PublicationMir.intoResult prior nodes fallback

end NeuralReversi.OuterMir
"""

