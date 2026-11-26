"""Project general exact NonPV owner scheduling and post-child publication from its MIR."""

from itertools import product

from mir_common import checked, const_value, split_call, blocks
from extract_fixed_leaf_mir import constant
from extract_header import binding, initial, lmr_returns_zero, specialization, transition
from extract_nonpv_leaf_mir import impl_constant
from extract_pv_owner_flow import walk
from extract_root_update import decode, post_step
from extract_root_wipeout import FILENAME, constants
from extract_single import check_score_operators


def verify(dump, repo):
    source = (dump / FILENAME).read_text(encoding="utf-8")
    ir = blocks(source)
    regs, header = initial(source, ir)
    t = constants(repo)
    t["const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"] = True
    checked(impl_constant(repo,dump,"strategy","impl SearchStrategy for EndGameStrategy {","IS_ENDGAME","bool")=="true",
            "unverified exact strategy")
    for name in ("PV_NODE", "ROOT_NODE"):
        checked(impl_constant(repo,dump,"node_type","impl NodeType for NonPV {",name,"bool")=="false",
                "incorrect ordinary NonPV node specialization")
        t[f"const <NT as search::node_type::NodeType>::{name}"]=False
    source_rows=(repo/"crates/reversi-core/src/search/node_type.rs").read_text(encoding="utf-8").splitlines()
    line,=[i for i,s in enumerate(source_rows,1) if s=="impl NodeType for NonPV {"]
    identity=f"node_type::<impl at crates/reversi-core/src/search/node_type.rs:{line}:1: {line}:24>::ID"
    candidates=[]
    for p in dump.glob("reversi_core.search-node_type-*-ID.built.after.mir"):
        content=p.read_text(encoding="utf-8").replace("\\","/")
        if content.startswith(f"// MIR for `search::{identity}` after built\n"):
            candidates.append(content)
    expected=[f"// MIR for `search::{identity}` after built",f"const {identity}: NodeTypeId = {{",
              "let mut _0: search::node_type::NodeTypeId;","bb0: {","_0 = NodeTypeId::NonPv;","return;","}","}"]
    checked(len(candidates)==1 and [s.strip() for s in candidates[0].splitlines() if s.strip()]==expected,
            "unverified split request node type")
    min_depth=const_value(dump,"search-endgame-MIN_SPLIT_DEPTH");max_depth=constant(dump,"SPECULATIVE_SPLIT_MAX_DEPTH")
    check_score_operators(repo)
    specialization(repo)
    lmr_returns_zero((dump/"reversi_core.search-pvs-compute_lmr_reduction.SimplifyCfg-initial.after.mir").read_text(encoding="utf-8"))
    # All nodes scout once. Split precedes a claim, and a busy eldest/last fresh move is still searched.
    for inside,first,deep,many,free,busy,fresh,cut,speculative in product((False,True), repeat=9):
        flags=(inside,first,deep,many,free,busy,fresh,False)
        if not inside: expected=".finish"
        elif (not first or (not cut and speculative)) and deep and many and free: expected=".split"
        elif busy and not first and fresh: expected=".postpone idx (fresh - 1)"
        else: expected=".child (idx + 1) true false (-scout)"
        for tag in ([1] if busy else [0,2]):
            for tt_pv in (False,True):
                result=transition(ir,regs,header,flags,t,tag,"nonPvID",cut,tt_pv,speculative)
                checked(result==expected,"unexpected NonPV header transition")
    cfg,post,beg=decode(dump)
    for bits in product((False,True),repeat=5):
        _,higher_alpha,higher_best,_,_=bits
        pick=higher_alpha and higher_best
        checked(post_step(cfg,post,t,beg,bits,internal=True)==(False,False,higher_best,pick,pick),
                "unexpected NonPV score, best move, or back-edge")
    end,=[pc for pc,b in cfg.items() if "::Bound::classify::<NT>(" in b.end]
    split,=[split_call(b.end) for b in cfg.values() if "Thread::split(" in b.end]
    checked(split and (split[1],split[4],split[5])==("_513","305","450"),"split result lost")
    returns={
        305:["PlaceMention(_513);","_510 = copy (_513.0: score::Score);",
             "_511 = copy (_513.1: std::option::Option<square::Square>);","_512 = move (_513.2: search::counters::SearchCounters);",
             "_526 = copy _510;","_333 = move _526;","_527 = copy _511;","_332 = move _527;",
             "_529 = &mut (*_1);","_531 = &_512;","_530 = &(*_531);",
             "_528 = SearchContext::<'_>::merge_counters(move _529, move _530) -> [return: bb306, unwind: bb450];"],
        306:["_536 = &(*_6);","_535 = <Arc<threading::thread::Thread> as Deref>::deref(move _536) -> [return: bb307, unwind: bb450];"],
        307:["_534 = &(*_535);","_533 = threading::thread::Thread::should_stop_after_split(move _534) -> [return: bb308, unwind: bb450];"],
        308:["switchInt(move _533) -> [0: bb310, otherwise: bb309];"],
        309:["_0 = const score::Score::ZERO;","goto -> bb435;"],310:[f"goto -> bb{end};"],
        435:["goto -> bb438;"],438:["goto -> bb439;"],439:["goto -> bb441;"],
        441:["goto -> bb443;"],443:["goto -> bb446;"],446:["goto -> bb447;"],447:["return;"]}
    for pc, expected in returns.items():
        checked([*cfg[pc].statements,cfg[pc].end]==expected and not cfg[pc].cleanup,
                f"unsupported split result or stop after split bb{pc}")
    probe_reg=binding(source,"tt_probe_result")
    raw_reg=binding(source,"raw_eval")
    for pv_flag in (False,True):
        env={reg:name for name,reg in regs.items()}
        env.update({probe_reg:"probe",raw_reg:"noEval",regs["tt_pv"]:pv_flag})
        def finished(state,calls):
            bound=("classify","best_score","org_alpha","beta")
            slot=("slot","probe")
            write=("store","tt",slot,"board","best_score",bound,"depth","best_move","selectivity",True,pv_flag)
            checked(calls==[bound,slot,write] and state["_0"]=="best_score","wrong NonPV score storage or return")
            return True
        checked(walk(cfg,end,env,t,0,None,finished),"unknown normal NonPV tail")
    return f"""import NeuralReversi.PvsParallel.RootHeaderMir.Program

namespace NeuralReversi.PvsParallel.GeneralOwnerMir

def minSplit : Nat := {min_depth}
def maxSpeculative : Nat := {max_depth}

def canSplit (idx count depth : Nat) (cut available : Bool) : Bool :=
  (decide (1 ≤ idx) || (!cut && decide (depth ≤ maxSpeculative))) &&
    decide (minSplit ≤ depth) && decide (2 ≤ count - idx) && available

def header (idx count fresh depth : Nat) (cut available busy : Bool) (scout : Int) : HeaderAction :=
  if idx < count then
    if canSplit idx count depth cut available then .split
    else if busy && decide (idx ≠ 0) && decide (idx + 1 < fresh) then .postpone idx (fresh - 1)
    else .child (idx + 1) true false (-scout)
  else .finish

def after (alpha best score : Int) : Int × Bool × Bool :=
  if best < score then
    if alpha < score then (score, true, true) else (score, false, false)
  else (best, false, false)

end NeuralReversi.PvsParallel.GeneralOwnerMir
"""
