"""Decode ordinary exact NonPV entry, stability, pass/terminal, parent TT and enhanced TT cuts."""

from dataclasses import dataclass
from itertools import product
import re

from mir_common import PASS, checked, normal_target, split_call, blocks
from extract_nonpv_leaf_mir import impl_constant, parameters, pvs_body, forwarding_body
from extract_fixed_leaf_mir import constant
from extract_root_wipeout import constants
from extract_header import ordinary
import extract_general_presearch


@dataclass(frozen=True)
class Flags:
    cut: bool
    deep: bool
    stability: bool
    move: bool
    other: bool
    wipe: bool
    hit: bool
    cut_tt: bool
    enhanced: bool


def plus_threshold(dump, repo, name):
    src = (repo / "crates/reversi-core/src/search/strategy.rs").read_text(encoding="utf-8").splitlines()
    line, = [i for i,x in enumerate(src,1) if x == "impl SearchStrategy for EndGameStrategy {"]
    mask = f"reversi_core.search-strategy-*-{name}.built.after.mir"
    found = []
    for p in dump.glob(mask):
        t = p.read_text(encoding="utf-8").replace("\\","/")
        path = f"strategy::<impl at crates/reversi-core/src/search/strategy.rs:{line}:1: {line}:40>::{name}"
        if t.splitlines()[0] != f"// MIR for `search::{path}` after built":
            continue
        checked(f"const {path}: u32 = {{" in t.splitlines(), "wrong threshold constant")
        cfg = blocks(t)
        m = re.fullmatch(r"_1 = AddWithOverflow\(const search::endgame::DEPTH_TO_NWS, const (\d+)_u32\);",cfg[0][0])
        checked(m,"unsupported threshold expression")
        delta = int(m[1])
        checked(cfg == {0:[m[0],f'assert(!move (_1.1: bool), "attempt to compute `{{}} + {{}}`, which would overflow", const search::endgame::DEPTH_TO_NWS, const {delta}_u32) -> [success: bb1, unwind: bb2];'],
                        1:["_0 = move (_1.0: u32);","return;"]}, "invalid threshold evaluation")
        found.append(constant(dump,"DEPTH_TO_NWS")+delta)
    checked(len(found)==1, f"missing compiled {name}")
    return found[0]


def prefetch_loop(cfg):
    # The iterator may contain any number of moves. Its back edge changes only the
    # iterator and temporaries, and each normal iteration prefetches one child hash.
    code = {
        83: ["_161 = &_92;", "_160 = MoveList::iter(move _161) -> [return: bb84, unwind: bb450];"],
        84: ["_159 = <std::slice::Iter<'_, move_list::Move> as IntoIterator>::into_iter(move _160) -> [return: bb85, unwind: bb450];"],
        85: ["PlaceMention(_159);","_162 = move _159;","goto -> bb86;"],
        86: ["falseUnwind -> [real: bb87, unwind: bb450];"],
        87: ["_167 = &mut _162;","_166 = &mut (*_167);","_165 = <std::slice::Iter<'_, move_list::Move> as Iterator>::next(move _166) -> [return: bb88, unwind: bb450];"],
        88: ["PlaceMention(_165);","_168 = discriminant(_165);","switchInt(move _168) -> [0: bb90, 1: bb91, otherwise: bb89];"],
        89: ["FakeRead(ForMatchedPlace(None), _165);","unreachable;"],
        90: ["falseEdge -> [real: bb92, imaginary: bb91];"],
        91: ["_170 = copy ((_165 as Some).0: &move_list::Move);","_172 = &(*_2);",
             "_173 = copy ((*_170).0: bitboard::Bitboard);","_174 = copy ((*_170).2: square::Square);",
             "_171 = Board::make_move_with_flipped(move _172, move _173, move _174) -> [return: bb93, unwind: bb450];"],
        92: ["goto -> bb97;"],
        93: ["_176 = &(*((*_1).5: &transposition_table::TranspositionTable));","_178 = &_171;","_177 = Board::hash(move _178) -> [return: bb94, unwind: bb450];"],
        94: ["_175 = TranspositionTable::prefetch(move _176, move _177) -> [return: bb95, unwind: bb450];"],
        95: ["goto -> bb86;"],
    }
    for pc, expected in code.items():
        b = cfg[pc]
        checked(not b.cleanup and [*b.statements,b.end] == expected, f"unknown child prefetch block bb{pc}")


def auxiliary(dump,repo):
    typed = forwarding_body(repo,dump,"search::strategy","impl SearchStrategy for EndGameStrategy {","try_probcut",
        "(_1: &mut SearchContext<'_>, _2: &Board, _3: u32, _4: Score, _5: bool, _6: Option<Score>, _7: &Arc<threading::thread::Thread>) -> Option<Score> {")
    def at(cfg,num): return [*cfg[num].statements,cfg[num].end]
    checked(at(typed,0)==["_8 = &mut (*_1);","_9 = &(*_2);","_10 = copy _3;","_11 = copy _4;","_12 = copy _5;",
         "_0 = search::endgame::try_probcut(move _8, move _9, move _10, move _11, move _12) -> [return: bb1, unwind: bb2];"]
         and at(typed,1)==["return;"] and at(typed,2)==["resume;"],"wrong endgame ProbCut delegation")
    exactfile=dump / f"reversi_core.search-endgame-try_probcut.{PASS}.after.mir"
    source=exactfile.read_text(encoding="utf-8")
    checked(source.startswith(f"// MIR for `search::endgame::try_probcut` after {PASS}\n") and
        "fn search::endgame::try_probcut(_1: &mut SearchContext<'_>, _2: &Board, _3: u32, _4: Score, _5: bool) -> Option<Score> {" in source.splitlines(),
        "wrong ProbCut implementation")
    ast=blocks(source)
    expected={0:["_8 = copy ((*_1).7: probcut::Selectivity);","_7 = Selectivity::is_enabled(move _8) -> [return: bb1, unwind: bb6];"],
              1:["switchInt(move _7) -> [0: bb3, otherwise: bb2];"],
              3:["_0 = Option::<Score>::None;","goto -> bb5;"],5:["return;"]}
    checked(all(ordinary(ast[n])==ops for n,ops in expected.items()),"exact ProbCut is live")
    enable=forwarding_body(repo,dump,"probcut","impl Selectivity {","is_enabled","(_1: Selectivity) -> bool {")
    checked(at(enable,0)==["_2 = &_1;","_4 = Selectivity::None;","_3 = &_4;",
          "_0 = <Selectivity as PartialEq>::ne(move _2, move _3) -> [return: bb1, unwind: bb2];"] and
          at(enable,1)==["return;"] and at(enable,2)==["resume;"],"changed exact selectivity predicate")
    closurefile=dump / f"reversi_core.search-pvs-search-{{closure#0}}.{PASS}.after.mir"
    closure=closurefile.read_text(encoding="utf-8")
    header=re.search(r"fn search::\{closure#0\}\(_1: (\{closure@[^}]+\}), _2: TTEntryData\) -> bool \{",closure)
    checked(header and {k:ordinary(x) for k,x in blocks(closure).items()}==
        {0:["_3 = &_2;","_0 = TTEntryData::is_pv(move _3) -> [return: bb1, unwind: bb2];"],1:["return;"]},
        "unexpected TT PV closure")
    return header[1]


def walk(cfg, traits, flags, resume=None):
    env = dict(_1="ctx",_2="board",_3="depth",_4="alpha",_5="beta",_6="thread",_7=flags.cut)
    gen, swap, probe = ("new","board"), ("switch","board"), ("probe","tt","board",("hash","board"))
    threshold = ("disc","alpha")
    args_stable = ("stability","board","empties",threshold)
    data, hint = ("data",probe), ("hint",probe)
    entry = ("some",data)
    child_reads = ("enhanced","ctx","board",gen,"depth","alpha",("slot",probe),"ttPv")
    events = []

    def arg(s):
        if s in traits: return traits[s]
        if m := re.fullmatch(r"copy (_\d+)|move (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(_\d+)",s):
            return env[next(x for x in m.groups() if x is not None)]
        fields = {"&((*_1).10: empty_list::EmptyList)":"list", "&mut ((*_1).12: search::counters::SearchStats)":"stats",
                  "&(*((*_1).5: &transposition_table::TranspositionTable))":"tt", "copy ((*_1).7: probcut::Selectivity)":"exact"}
        if s in fields: return fields[s]
        if m := re.fullmatch(r"Not\((.+)\)",s):
            v=arg(m[1]);checked(isinstance(v,bool),"unexpected negation");return not v
        if m := re.fullmatch(r"(Le|Ge)\(move (_\d+), const <SS as search::strategy::SearchStrategy>::(DEPTH_TO_SHALLOW|MIN_ETC_DEPTH)\)",s):
            checked(env[m[2]]=="depth","threshold uses wrong depth")
            return {("Le","DEPTH_TO_SHALLOW"):False,("Ge","MIN_ETC_DEPTH"):flags.deep}[(m[1],m[3])]
        if m := re.fullmatch(r"Eq\(move (_\d+), const 0_usize\)",s):
            checked(env[m[1]]==("count",gen),"wrong move count");return not flags.move
        if m := re.fullmatch(r"discriminant\((_\d+)\)",s):
            return int({args_stable:flags.stability,("wipe",gen):flags.wipe,data:flags.hit,child_reads:flags.enhanced}[env[m[1]]])
        if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: (score::DiscScore|square::Square|transposition_table::TTEntryData|score::Score)\)", s):
            val=env[m[1]]
            checked(val=={"score::DiscScore":args_stable,"square::Square":("wipe",gen),"transposition_table::TTEntryData":data,"score::Score":child_reads}[m[2]],"wrong optional payload")
            return ("some",val)
        if m := re.fullmatch(r"\{closure@crates[\\/]reversi-core[\\/]src[\\/]search[\\/]pvs.rs:[^}]+\}",s):
            return m[0]
        raise ValueError(f"unsupported NonPV operand {s}")

    pc,visited=0,set()
    while pc not in visited:
        visited.add(pc)
        if pc==83:
            checked((env["_92"],env["_2"],env["_1"])==(gen,"board","ctx"),"prefetch uses changed state")
            events.append(("prefetchChildren",gen))
            pc=97
        if pc==146:
            outcome="owner"
            break
        b=cfg[pc]
        for op in b.statements:
            if m:=re.fullmatch(r"PlaceMention\((_\d+)\);",op):
                checked(m[1] in env,"uninitialized match");continue
            m=re.fullmatch(r"(_\d+) = (.+);",op);checked(m,f"unexpected statement {op}")
            env[m[1]]=arg(m[2])
        term=b.end
        if term=="return;":
            outcome=env["_0"];break
        if c:=split_call(term):
            args=tuple(arg(a) for a in c[3].split(", "))
            names={"EmptyList::count":"empties","Board::final_disc_diff":"terminal","Score::to_disc_diff":"disc",
                   "stability_cutoff":"stability","SearchStats::increment_stability_cut":"stableStat","Score::from_disc_diff":"scaled",
                   "Board::hash":"hash","TranspositionTable::prefetch":"prefetch","MoveList::new":"new","MoveList::count":"count",
                   "Board::switch_players":"switch","Board::has_legal_moves":"mobile","SearchContext::<'_>::update_pass":"beginPass",
                   "SearchContext::<'_>::undo_pass":"endPass","<Score as Neg>::neg":"neg","search::<NT, SS>":"recurse",
                   "Board::solve":"terminal","MoveList::wipeout_move":"wipe","TranspositionTable::probe":"probe",
                   "SearchStats::increment_tt_probe":"probeStat","TTProbeResult::<'_>::best_move":"hint","TTProbeResult::<'_>::data":"data",
                   "TTEntryData::can_cut":"canCut","SearchStats::increment_tt_hit":"hitStat","TTEntryData::score":"score",
                   "SearchStats::increment_etc_attempt":"etcStat","TTProbeResult::<'_>::slot":"slot",
                   "enhanced_transposition_cutoff::<SS>":"enhanced","SearchStats::increment_etc_cut":"etcCut"}
            if m:=re.fullmatch(r"Option::<TTEntryData>::is_some_and::<(\{closure@[^}]+\})>",c[2]):
                checked(args==(data,m[1]) and m[1] == traits["checkedClosure"],"wrong tt_pv predicate"); call=("isPv",data);v="ttPv"
            else:
                checked(c[2] in names,f"unexpected NonPV call {c[2]}")
                call=(names[c[2]],*args)
                if call==("empties","list"): v="empties"
                elif call==("mobile",swap): v=flags.other
                elif call==("canCut",entry,"beta","depth","exact",True): v=flags.cut_tt
                else: v=call
            events.append(call);env[c[1]]=v
            pc=normal_target(cfg,c[4],c[5])
        elif m:=re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];",term):
            pc=normal_target(cfg,m[1] or m[2])
        elif m:=re.fullmatch(r"switchInt\((.+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];",term):
            x=arg(m[1]);checked(isinstance(x,int),f"unresolved branch {x}")
            pc=normal_target(cfg,m[3] if int(x)==int(m[2]) else m[4])
        else: raise ValueError(f"unknown NonPV terminator bb{pc}: {term}")
    else: raise ValueError("unexpected ordinary NonPV cycle")
    expected=[("empties","list"),threshold,args_stable]
    if flags.stability:
        v=("scaled",("some",args_stable));expected += [("stableStat","stats"),v];arm="stability"
    else:
        expected += [("hash","board"),("prefetch","tt",("hash","board")),gen,("count",gen)]
        if not flags.move:
            expected += [swap,("mobile",swap)]
            if flags.other:
                nb,na=("neg","beta"),("neg","alpha")
                child=("recurse","ctx",swap,"depth",nb,na,"thread",not flags.cut)
                v=("neg",child);expected += [("beginPass","ctx"),nb,na,child,v,("endPass","ctx")];arm="pass"
            else:
                v=("terminal","board","empties");expected += [("empties","list"),v];arm="terminal"
        else:
            expected += [("wipe",gen)]
            if flags.wipe: v=16384;arm="wipeout"
            else:
                if flags.deep: expected += [("prefetchChildren",gen)]
                expected += [probe,("probeStat","stats"),hint,data,("isPv",data),data]
                if flags.hit: expected += [("canCut",entry,"beta","depth","exact",True)]
                if flags.hit and flags.cut_tt: v=("score",entry);expected += [("hitStat","stats"),v];arm="tt"
                else:
                    v="owner";arm="owner"
                    if flags.deep:
                        expected += [("etcStat","stats"),("slot",probe),child_reads]
                        if flags.enhanced: v=("some",child_reads); expected += [("etcCut","stats")];arm="enhanced"
    checked(outcome==v and events==expected,f"incorrect {arm} result or effects: {outcome}, {events}")
    if arm=="owner" and resume is not None:
        return resume(pc,env)
    return arm


def verify(dump, repo):
    cfg=pvs_body(dump)
    traits=dict(constants(repo),**parameters(repo,dump,False))
    root=impl_constant(repo,dump,"node_type","impl NodeType for NonPV {","ROOT_NODE","bool")
    checked(root=="false","NonPV claimed root")
    traits["const <NT as search::node_type::NodeType>::ROOT_NODE"]=False
    cutoff=constant(dump,"DEPTH_TO_NWS")
    etc=plus_threshold(dump,repo,"MIN_ETC_DEPTH")
    prob=plus_threshold(dump,repo,"MIN_PROBCUT_DEPTH")
    checked(etc==cutoff+2 and prob==cutoff+1,"unexpected general NonPV levels")
    traits["checkedClosure"]=auxiliary(dump,repo)
    prefetch_loop(cfg)
    for flags in product((False,True),repeat=9):
        walk(cfg,traits,Flags(*flags))
    def reach(cut, deep, hit, cb):
        return walk(cfg,traits,Flags(cut,deep,False,True,False,False,hit,False,False),cb)
    extract_general_presearch.verify(cfg,dump,traits,reach)
    return f"""namespace NeuralReversi.PvsParallel.GeneralNonPvMir

inductive Action where
  | shallow | stability | terminal | pass | wipeout | probe | enhanced | rest
  deriving DecidableEq

def dispatch (depth : Nat) (stable mobile other wipes parentCut childCut : Bool) : Action :=
  if depth ≤ {cutoff} then .shallow else if stable then .stability else if !mobile then
    if other then .pass else .terminal
  else if wipes then .wipeout else if parentCut then .probe
  else if {etc} ≤ depth && childCut then .enhanced else .rest

inductive Presearch (Move : Type) where
  | owner (idx : Nat) (best : Int)
  | storeLower (move : Move) (score : Int)

def presearch (count : Nat) (hint : Option Move) (first : Move) (alpha beta : Int)
    (scout : Move → Int → Int → Int) : Presearch Move :=
  if count = 1 || hint.isSome then
    let move := hint.getD first
    let score := -(scout move (-(alpha + 1)) (-alpha))
    if alpha < score && beta ≤ score then .storeLower move score else .owner 1 score
  else .owner 0 (-32767)

end NeuralReversi.PvsParallel.GeneralNonPvMir
"""
