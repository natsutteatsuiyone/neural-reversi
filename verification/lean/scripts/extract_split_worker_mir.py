"""Extract a claimed search_split_point move, including the locked score publication."""

from copy import deepcopy
import re

import check_split_fields
from extract_single import check_score_operators
from extract_root_wipeout import constants
from mir_common import PASS, run_extractor_cli, ASSERT, CALL, GOTO, blocks, checked
from mir_mutants import assert_mutants_rejected


MIR = f"reversi_core.search-pvs-search_split_point.{PASS}.after.mir"
SIGNATURE = ("fn search_split_point(_1: &mut SearchContext<'_>, _2: &SplitPointTask, "
             "_3: &ConcurrentMoveIterator, _4: &Arc<threading::thread::Thread>, _5: &Arc<SplitPoint>) -> () {")


def render(dump, repo, *, fields=False):
    text = (dump/MIR).read_text(encoding="utf-8")
    checked(text.startswith(f"// MIR for `search::pvs::search_split_point` after {PASS}\n")
            and SIGNATURE in text.splitlines(), "unexpected worker MIR")
    check_score_operators(repo)
    maximum = constants(repo)["const score::Score::MAX"]
    score = re.findall(r"^\s*debug score => (_\d+);", text, re.M)
    checked(len(score) == 1, "missing live score at publication")
    src = repo / "crates/reversi-core/src"
    def field_names(path, name):
        rust = re.sub(r"//[^\n]*", "", (src/path).read_text(encoding="utf-8"))
        body = re.search(r"pub struct "+name+r" \{([^}]+)\}", rust)[1]
        return re.findall(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:",body,re.M)
    task = field_names("search/threading/split_point.rs", "SplitPointTask")
    checked((task[0],task[7],task[8]) == ("board","depth","beta"), "task field map")
    move = field_names("move_list.rs","Move")
    checked((move[0],move[2]) == ("flipped","sq"), "move field map")
    cfg = blocks(text)
    roots = {"_1":"ctx", "_2":"task", "_3":"iter", "_4":"arcThread", "_5":"arcSplit"}
    bodies = {}
    locked_programs = set()
    locked_events = set()
    def branch(cond, yes, no):
        if yes == no:
            return yes
        ident = (cond, yes, no)
        if ident not in bodies:
            bodies[ident] = f"branch{len(bodies)}"
        return bodies[ident]

    def walk(pc, state, phase):
        env, steps, effects, seen = deepcopy(state)
        for _ in range(160):
            def val(s):
                if m := re.fullmatch(r"(?:copy |move |&mut |&)?(_\d+)", s):
                    return env[m[1]]
                if m := re.fullmatch(r"(?:copy |move |&mut |&)?\(\*(_\d+)\)", s):
                    return env[m[1]]
                if m := re.fullmatch(r"(?:move |copy )\(\(\*(_\d+)\)\.(\d+): [^()]+\)", s):
                    return {("task","0"):"board", ("task","7"):"i.depth", ("task","8"):"i.beta",
                            ("mv","0"):"flips", ("mv","2"):"sq"}[(env[m[1]],m[2])]
                if m := re.fullmatch(r"(?:move |copy )\((_\d+)\.(\d+): [^()]+\)", s):
                    return env[m[1]][int(m[2])]
                if m := re.fullmatch(r"copy \(\(\((_\d+) as Some\)\.0: \(&move_list::Move, usize\)\)\.(\d+): [^()]+\)", s):
                    checked(env[m[1]] == "issued","move/index from wrong source")
                    return ["mv","(i.index + 1)"][int(m[2])]
                if m := re.fullmatch(r"discriminant\((_\d+)\)",s):
                    checked(env[m[1]] in ("issued","noTicket"),"unknown discriminant")
                    return "true" if env[m[1]]=="issued" else "false"
                if m := re.fullmatch(r"\((move _\d+), (move _\d+), (move _\d+)\)",s):
                    return tuple(map(val,m.groups()))
                if m := re.fullmatch(r"SubWithOverflow\((copy _\d+), const 1_u32\)",s):
                    checked(val(m[1]) == "i.depth", "wrong child depth")
                    return ("(i.depth - 1)", "i.depth = 0")
                if m := re.fullmatch(r"Gt\((move _\d+), const 1_usize\)",s):
                    checked(val(m[1]) == "(i.index + 1)", "wrong claim ordinal")
                    return "1 < i.index + 1"
                if m := re.fullmatch(r"Not\((move _\d+)\)",s):
                    return "(!"+val(m[1])+")"
                if m := re.fullmatch(r"Option::<Square>::Some\((move _\d+)\)",s):
                    checked(val(m[1]) == "sq","wrong best move")
                    return "someSq"
                return {"const ()":"unit","const true":"true","const false":"false","const 1_i32":"1",
                        "const score::Score::MAX":"maximum",
                        "const <NT as search::node_type::NodeType>::PV_NODE":"i.pv",
                        "const <NT as search::node_type::NodeType>::ROOT_NODE":"i.root"}[s]

            def call(name, args):
                def exact(expected):
                    checked(args == expected, f"wrong arguments for {name}: {args}")
                if phase=="commit" and name not in ("threading::thread::Thread::should_stop",
                        "<Arc<threading::thread::Thread> as Deref>::deref"):
                    checked("poll" in seen and "unlock" not in seen,
                            f"{name} outside the checked critical section")
                simple = {
                    "<Arc<SplitPoint> as Deref>::deref": (["arcSplit"],"split"),
                    "<Arc<threading::thread::Thread> as Deref>::deref": (["arcThread"],"thread"),
                    "SplitPoint::state": (["split"],"sp"),
                    "SplitPointState::cut_node": (["sp"],"i.cut"),
                    "Board::make_move_with_flipped": (["board","flips","sq"],"child"),
                    "claim_child::<EndGameStrategy>": (["thread","child","i.depth"],"claim"),
                    "SearchContext::<'_>::update": (["ctx","sq","flips"],"updated"),
                    "SearchContext::<'_>::undo": (["ctx"],"undone"),
                    "SplitPointState::best_score": (["sp"],"s.best"),
                    "SearchContext::<'_>::update_pv": (["ctx","sq"],"unit"),
                    "SearchContext::<'_>::pv": (["ctx"],"pvline"),
                    "SplitPoint::copy_pv": (["split","pvline"],"unit"),
                }
                if name in simple:
                    expected, result = simple[name]
                    exact(expected)
                    if name == "SplitPointState::best_score":
                        checked(phase == "commit" and "poll" in seen, "best read outside locked update")
                        result = effects.get("bestRead", "s.best")
                        steps.append(".readBest " + result)
                    if result in ("claim", "updated", "undone"):
                        seen.append(result)
                    return result
                if name == "SplitPointState::alpha":
                    exact(["sp"])
                    if phase == "commit":
                        checked("poll" in seen,"locked read before polling")
                        observed = effects.get("alphaRead", "s.alpha")
                        steps.append(".readAlpha " + observed)
                        return observed
                    again = any(a.startswith(".load") for a in steps)
                    steps.append(".load "+str(again).lower())
                    return "i.fresh" if again else "i.alpha"
                if name == "<Score as Add<i32>>::add":
                    return f"({args[0]} + {args[1]})"
                if name == "<Score as Neg>::neg":
                    return "(-"+args[0]+")"
                if m := re.fullmatch(r"<Score as PartialOrd>::(gt|lt|ge)",name):
                    a,b=args
                    op={"gt":">","lt":"<","ge":"≥"}[m[1]]
                    return f"{a}.toInt {op} {b}.toInt"
                if m := re.fullmatch(r"search::<(NonPV|PV), EndGameStrategy>",name):
                    ctx, board, depth, lo, hi, thread, cut=args
                    checked((ctx,board,thread)==("ctx","child","arcThread") and seen==["claim","updated"],
                            "search did not use the claimed updated child")
                    pv = m[1]=="PV"
                    result="i.wide" if pv else "i.scout"
                    steps.append(f".returned ⟨i.index, {depth}, {str(pv).lower()}, {lo}, {hi}, {cut}, {result}⟩")
                    return result
                if name == "threading::thread::Thread::should_stop":
                    exact(["thread"]); checked(phase=="commit" and not seen,"unexpected stop poll")
                    seen.append("poll")
                    return "s.stopped"
                if name == "SplitPointState::set_all_helpers_searching_locked":
                    exact(["sp","false"]); seen.append("stopped")
                elif name == "SearchContext::<'_>::update_root_move":
                    checked(args[:2]==["ctx","sq"],"wrong reported root move")
                    effects["rootRow"]=f"(some ({args[2]}, decide ({args[3]})))"
                elif name == "SplitPointState::set_best_score":
                    checked(args[0]=="sp","wrong best-score target")
                    checked("best" not in effects, "multiple locked best stores")
                    steps.append(".writeBest " + args[1])
                    effects["bestRead"] = args[1]
                    effects["best"]=f"(some {args[1]})"
                elif name == "SplitPointState::set_best_move":
                    checked("moveSet" not in effects, "multiple locked move stores")
                    steps.append(".writeMove i.index")
                    exact(["sp","someSq"]); effects["moveSet"]="true"
                elif name == "SplitPointState::set_alpha":
                    checked(args[0]=="sp","wrong alpha target")
                    checked("alpha" not in effects, "multiple locked alpha stores")
                    steps.append(".writeAlpha " + args[1])
                    effects["alpha"]=f"(some {args[1]})"; effects["alphaRead"]=args[1]
                elif name == "threading::thread::Thread::mark_split_point_cutoff":
                    exact(["thread","sp"]); effects["cut"]="true"
                else:
                    raise ValueError(f"unsupported call {name}")
                return "unit"

            def end(exit):
                checked(phase=="commit" and "poll" in seen and "unlock" in seen and "release" in seen,
                        f"incomplete commit on {exit}")
                locked_events.add(tuple("."+a for a in seen if a in ("poll","unlock")))
                if fields:
                    return "["+", ".join(steps)+"]"
                cols=[effects.get(k,default) for k,default in [
                    ("rootRow","none"),("best","none"),("moveSet","false"),("alpha","none"),("cut","false")]]
                return "⟨."+exit+", "+", ".join(cols)+"⟩"

            def arm(dest):
                return walk(int(dest),(env,steps,effects,seen),phase)

            *assignments, term=cfg[pc]
            for line in assignments:
                if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);|PlaceMention\(_\d+\);",line):
                    continue
                m=re.fullmatch(r"(_\d+) = (.+);",line)
                checked(m,f"unsupported statement {line}")
                env[m[1]]=val(m[2])
            if m:=re.fullmatch(CALL,term):
                dst,name,ops,target=m.groups()
                args=[val(a) for a in ops.split(", ")]
                if name=="ConcurrentMoveIterator::next":
                    checked(args==["iter"],"wrong move iterator")
                    if phase=="commit":
                        return end("again")
                    checked("issued" not in env.values(),"unexpected loop")
                    env[dst]="noTicket" if phase=="empty" else "issued"
                elif name=="SplitPoint::lock":
                    checked(args==["split"] and phase=="search" and seen==["claim","updated","undone"],"lock placement")
                    env[dst]="guard"
                    chosen=env[score[0]]
                    env[score[0]]="chosen"
                    locked_programs.add(walk(int(target),(env,[],{},[]),"commit"))
                    reads=[a for a in steps if a.startswith(".load")]
                    checked(1<=len(reads)<=2,"unexpected number of alpha loads")
                    used="i.alpha" if len(reads)==1 else "i.fresh"
                    return "some ⟨["+", ".join(steps+[".lock"])+"], "+used+", "+chosen+"⟩"
                else:
                    env[dst]=call(name,args)
                pc=int(target)
            elif m:=re.fullmatch(r"switchInt\(move (_\d+)\) -> \[(0|1): bb(\d+), otherwise: bb(\d+)\];",term):
                cond=env[m[1]]
                t,f=(m[4],m[3]) if m[2]=="0" else (m[3],m[4])
                if cond in ("true","false"):
                    pc=int(t if cond=="true" else f)
                else:
                    return branch(cond,arm(t),arm(f))
            elif m:=re.fullmatch(ASSERT,term):
                cond=val(m[1])
                checked(phase=="search","unexpected commit assertion")
                return branch(cond,"none",arm(m[2]))
            elif m:=re.fullmatch(GOTO,term):
                pc=int(m[1] or m[2])
            elif m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",term):
                seen.append({"guard":"unlock","claim":"release"}[env[m[1]]])
                pc=int(m[2])
            elif term=="return;":
                if phase=="empty":
                    checked(not steps and not seen and not effects and "noTicket" in env.values(),
                            "empty iterator has an effect")
                    return ".exhausted"
                return end("stopped" if "stopped" in seen else "cutoff")
            elif re.fullmatch(r'_\d+ = (?:core::panicking::)?panic\(const ".+"\) -> bb\d+;',term) and phase=="search":
                return "none"
            else:
                raise ValueError(f"bb{pc}: {term}")
        raise ValueError("unexpected loop in claimed move")

    pre=walk(0,(roots,[],{},[]),"search")
    exhausted=walk(0,(roots,[],{},[]),"empty")
    checked(exhausted==".exhausted","unexpected empty-ticket path")
    checked(len(locked_programs)==1,"commit depends on non-carried search temporaries")
    checked(len(locked_events)==1,"protected region's poll/unlock events vary by exit")
    def function(header, typ, root):
        reachable={root}
        for (guard,yes,no),name in reversed(bodies.items()):
            if name in reachable:
                reachable.update([yes,no])
        lets=[f"  let {n} : {typ} := if {guard} then {yes} else {no}"
              for (guard,yes,no),n in bodies.items() if n in reachable]
        return [header, *lets, "  "+root, ""]
    if fields:
        check_split_fields.check(dump, repo)
        lines = function("def accesses (i : Input) (s : Locked) (chosen : Word) :=", "List Access", locked_programs.pop())
        limit = [f"private def maximum : Word := {maximum}", ""] if any("maximum" in line for line in lines) else []
        return "\n".join(["import NeuralReversi.PvsParallel.SplitWorkerMir.FieldAccess","",
            "namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores","",
            *limit, *lines, "end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores",""])
    return "\n".join(["import NeuralReversi.PvsParallel.SplitWorkerMir.Syntax","",
        "namespace NeuralReversi.PvsParallel.SplitWorkerMir","",f"def maximum : Word := {maximum}","",
        f"def emptyExit : Threading.SplitReturn.Departure := {exhausted}","",
        "def lockedSteps : List Step := ["+", ".join(locked_events.pop())+"]","",
        *function("def search (i : Input) : Option Searched :=","Option Searched",pre),
        *function("def commit (i : Input) (s : Locked) (chosen : Word) : Commit :=",
                  "Commit", locked_programs.pop()), "end NeuralReversi.PvsParallel.SplitWorkerMir",""])


def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="search_split_point",
        check_message="generated worker differs",
    )


def check_child_mutants(dump, repo):
    """Change each source of a split child's operands or its selected return."""
    cases = [
        ("scout board", "_57 = &_25;", "_57 = &_6;"),
        ("wide board", "_84 = &_25;", "_84 = &_6;"),
        ("child depth", "move _56, move _58,", "move _56, copy _7,"),
        ("scout lower bound", "move _58, move _61,", "move _58, copy _64,"),
        ("scout cut flag", "_67 = Not(move _68);", "_67 = copy _68;"),
        ("retry threshold", "_73 = &_40;", "_73 = &_8;"),
        ("fresh alpha", "_91 = copy _74;", "_91 = copy _40;"),
        ("wide lower bound", "_89 = copy _8;", "_89 = copy _40;"),
        ("wide cut flag", "move _90, move _92, const false)", "move _90, move _92, const true)"),
        ("chosen return", "_53 = move _80;", "_53 = copy _54;"),
    ]
    return check_mutations(dump, repo, cases)


def check_guard_mutants(dump, repo):
    cases = [
        ("stopped guard", "drop(_95) -> [return: bb102", "drop(_29) -> [return: bb102"),
        ("cutoff guard", "drop(_95) -> [return: bb99", "drop(_29) -> [return: bb99"),
        ("next ticket guard", "drop(_95) -> [return: bb96", "drop(_29) -> [return: bb96"),
        ("early unlock", "_133 = SplitPointState::set_best_score(move _134, move _135) -> [return: bb64",
         "drop(_95) -> [return: bb64"),
    ]
    return check_mutations(dump, repo, cases)


def check_mutations(dump, repo, cases):
    expected = render(dump, repo)
    return assert_mutants_rejected(
        dump,
        [(MIR, old, new, label) for label, old, new in cases],
        lambda d: render(d, repo),
        baseline=expected,
    )


if __name__=="__main__":
    main()
