"""Check board, full-depth and mutable context data flow on normal PVS/worker paths."""
import re
from mir_common import checked, CALL
from extract_endgame_order_mir import code, read
from extract_header import lmr_returns_zero
from extract_split_scheduler_mir import Body, edges

CTX="search-context-{impl#0}-"
C="SearchContext::<'_>::"
F=lambda obj,index: ("field",obj,index)

class Graph(Body):
    def __init__(self,d,fn,n,roots=None):
        self.cfg=code(d,fn);self.defs={};self.roots={i:"arg"+str(i) for i in range(1,n+1)}
        self.roots.update(roots or {});self.marks={}
        for ops in self.cfg.values():
            for op in ops:
                if m:=re.fullmatch(r"(_\d+) = (.+?)(?: -> .+)?;",op):
                    self.defs.setdefault(m[1],set()).add(m[2])
    def val(self,x):
        x=re.sub(r"^(?:copy |move |&mut |&)","",x)
        if m:=re.fullmatch(r"\(\*(.+)\)",x):return self.val(m[1])
        if m:=re.fullmatch(r"(.+) as (?:u8|u32|usize|&(?:mut )?\[[^]]+\]) \((?:IntToInt|PointerCoercion\(Unsize, Implicit\))\)",x):
            return self.val(m[1])
        if m:=re.fullmatch(r"\((_\d+)\[(_\d+)\]\.(\d+): [^)]+\)",x):
            return F(("index",self.val(m[1]),self.val(m[2])),int(m[3]))
        if m:=re.fullmatch(r"\((.+)\.(\d+): [^:]+\)",x):
            val=self.val(m[1]); j=int(m[2])
            return val[j+1] if isinstance(val,tuple) and val[0]=="tuple" else F(val,j)
        if m:=re.fullmatch(r"([\w:<'_>]+) \{ (.+) \}",x):
            fields=(field.split(": ",1) for field in m[2].split(", "))
            return {n:self.val(v) for n,v in fields if not m[1].startswith("SplitRequest") or n in {"board","depth"}}
        if x.startswith("compute_lmr_reduction::<") and x.endswith(")"):
            return (x.split("(",1)[0],)
        return super().val(x)
    def calls(self):
        for pc,ops in self.cfg.items():
            if m:=re.fullmatch(CALL,ops[-1]):
                yield pc,m[2],m[3].split(", ") if m[3] else []
    def sole(self,name,at,expected):
        sites=[args for _,fn,args in self.calls() if fn==name]
        checked(len(sites)==1 and self.val(sites[0][at])==expected, f"{name} arg {at}: {sites}")
    def normal(self,pc):
        t=self.cfg[pc][-1]
        if m:=re.match(r"drop\(.+\) -> \[return: bb(\d+),",t):return [int(m[1])]
        if re.search(r" -> bb\d+;$",t) and not t.startswith("goto"):return []
        return edges(t)
    def fields(self,expected):
        result=self.val("_0")
        for k,v in expected.items():checked(result[k]==v,f"{k}: {result[k]} vs {v}")

def methods(d):
    ctx,sq,fl="arg1","arg2","arg3"; count=F(ctx,10); side=F(ctx,8);feat=F(ctx,0)
    plyOf=lambda n:F(("SubWithOverflow","const constants::INITIAL_EMPTY_COUNT",n),0)
    g=Graph(d,CTX+"new",5)
    n=("Board::empty_count",ctx)
    g.fields(dict(n_nodes="const 0_u64",side_to_move="SideToMove::Player",n_empties=n,
                  pattern_features=("PatternFeatures::new",ctx,plyOf(n))))
    g=Graph(d,"search-threading-split_point-{impl#3}-new",3)
    g.fields(dict(board=F(ctx,0),depth=F(ctx,5),side_to_move=F(fl,8),
                  p_feature=("PatternFeatures::p_feature",F(fl,0),(C+"ply",fl)),
                  o_feature=("PatternFeatures::o_feature",F(fl,0),(C+"ply",fl))))
    g=Graph(d,CTX+"from_split_task",1)
    n=("Board::empty_count",F(ctx,0))
    g.fields(dict(n_nodes="const 0_u64",side_to_move=F(ctx,1),n_empties=n,
                  pattern_features=("PatternFeatures::from_features",plyOf(n),F(ctx,5),F(ctx,6))))
    checked(Graph(d,CTX+"ply",1).val("_0")==plyOf(count),"context ply")
    g=Graph(d,CTX+"switch_players",1)
    checked(put(g)==[(1,side,("SideToMove::switch",side))],"side-to-move write")
    g=Graph(d,CTX+"increment_nodes",1)
    checked(put(g)==[(1,F(ctx,2),F(("AddWithOverflow",F(ctx,2),"const 1_u64"),0))],"increment_nodes modified context")
    g=Graph(d,"search-side_to_move-{impl#0}-switch",1)
    checked(g.cfg[0]==["_2 = discriminant(_1);","switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];"]
            and only_path(g,2,5)==[2,4] and only_path(g,3,5)==[3]
            and g.cfg[3][0]=="_0 = SideToMove::Player;" and g.cfg[4][0]=="_0 = SideToMove::Opponent;","turn toggle")
    instructions={
        "update":([(C+"increment_nodes",ctx),(C+"ply",ctx),("PatternFeatures::update",feat,sq,fl,(C+"ply",ctx),side),
                   (C+"switch_players",ctx)],[(count,F(("SubWithOverflow",count,"const 1_u32"),0))]),
        "undo":([(C+"switch_players",ctx)],[(count,F(("AddWithOverflow",count,"const 1_u32"),0))]),
        "update_pass":([(C+"increment_nodes",ctx),(C+"switch_players",ctx)],[]),
        "undo_pass":([(C+"switch_players",ctx)],[])}
    for fn,(expected,stores) in instructions.items():
        argc=3 if fn=="update" else 1
        g=Graph(d,CTX+fn,argc);pc=0;seen=set();calls={p:(c,a) for p,c,a in g.calls()};actual=[]
        while pc not in seen:
            seen.add(pc);ops=g.cfg[pc]
            checked(all(re.fullmatch(r"_\d+ = .+;|\(\(\*_1\)\.10: u32\) = .+;",s) for s in ops[:-1]),f"{fn} direct side effect")
            if pc in calls:
                c,a=calls[pc];actual.append((c,*(g.val(arg) for arg in a)))
            if ops[-1]=="return;":break
            pc,=g.normal(pc)
        checked(ops[-1]=="return;" and actual==expected and [(l,r) for _,l,r in put(g)]==stores,f"{fn}: {actual} {put(g)}")

def mutable_frontier(g,index,names):
    """Reject an unaccounted mutable borrow of the context or of its board-tracking fields."""
    arg="_"+str(index);aliases={arg};changed=True
    while changed:
        changed=False
        for local,defs in g.defs.items():
            if local not in aliases and any(x in defs for src in aliases for x in
                        ("&mut (*"+src+")","copy "+src,"move "+src)):
                aliases.add(local);changed=True
    for pc,fn,argv in g.calls():
        if any(op in ("move "+a,"copy "+a) for a in aliases for op in argv):
            checked(fn in names,f"unaccounted mutable context call bb{pc}: {fn}")
    for ops in g.cfg.values():
        for op in ops:
            if "&mut " in op or (not op.startswith("_") and " = " in op):
                target=op.split(" = ",1)[0] if not op.startswith("_") else op
                for ty in (".10: u32)","eval::pattern_feature::PatternFeatures","search::side_to_move::SideToMove"):
                    checked(ty not in target,f"unaccounted borrow/write of {ty} in {op}")

def bracket(d,name,worker=False):
    g=Graph(d,name,5 if worker else 7)
    # These locals hold the generated/claimed move. The pre-search square and flips
    # are decoded from Board::flip; owner / worker loop records come from the iterator.
    if worker:g.roots[23]="workerMove"
    else:g.roots.update({365:"preSquare",366:"preFlip",535:"loopMove"})
    tracked={C+k for k in ("clear_pv","merge_counters","prepare_pv","undo","undo_pass","update",
                            "update_pass","update_pv","update_root_move")}
    mutable_frontier(g,1,tracked | {"<EndGameStrategy as SearchStrategy>::shallow_search",
        "<SS as SearchStrategy>::shallow_search","<SS as SearchStrategy>::try_probcut",
        "enhanced_transposition_cutoff::<SS>","ordering::<impl MoveList>::evaluate_moves::<NT, SS>",
        "search::<NT, SS>","search::<NonPV, SS>","search::<PV, SS>","threading::thread::Thread::split",
        "search::<NonPV, EndGameStrategy>","search::<PV, EndGameStrategy>"})
    ctx="arg1";board=F("arg2",0) if worker else "arg2";depth=F("arg2",7) if worker else "arg3"
    seen=set();todo=[(0,None)];found=set();events={}
    while todo:
        pc,live=todo.pop()
        if (pc,live) in seen:continue
        seen.add((pc,live));ops=g.cfg[pc]
        for op in ops[:-1]:
            checked(not re.search(r"\):? ?= .+",op) or not re.search(r"\.(?:0: eval::pattern_feature::PatternFeatures|8: search::side_to_move::SideToMove|10: u32)",op.split(" = ")[0]),f"{name} direct context write")
            if not worker:checked(not op.startswith("_3 = "),f"depth overwritten in exact endgame at bb{pc}")
        term=ops[-1]
        if m:=re.fullmatch(CALL,term):
            fn=m[2];a=m[3].split(", ");v=lambda i:g.val(a[i])
            if fn in (C+"update",C+"update_pass"):
                checked(live is None and v(0)==ctx,f"{name} unbalanced update bb{pc}")
                live=(v(1),v(2)) if fn==C+"update" else "pass"
                if live!="pass": checked(isinstance(live,tuple),"move data")
            elif fn in (C+"undo",C+"undo_pass"):
                checked(v(0)==ctx and (live=="pass" if fn==C+"undo_pass" else isinstance(live,tuple)),
                        f"{name} unmatched undo bb{pc}")
                live=None
            elif fn.startswith("search::<"):
                checked(v(0)==ctx and live is not None,f"{name} recursion outside context update")
                child=v(1);cd=v(2)
                if live=="pass":
                    checked(child==("Board::switch_players",board) and cd==depth,f"pass {child} {cd}")
                    checked(fn == "search::<NT, SS>", "a pass must preserve the node type")
                    kind = ".pass"
                else:
                    sq,flip=live
                    strategy = "EndGameStrategy" if worker else "SS"
                    targets = {f"search::<{node}, {strategy}>": pv for node, pv in (("PV", "true"), ("NonPV", "false"))}
                    checked(fn in targets, f"bb{pc} unexpected child search {fn}")
                    kind = f"(.move {targets[fn]})"
                    checked(child==("Board::make_move_with_flipped",board,flip,sq),f"bb{pc} child {child}, live {live}")
                    # LMR in the generic scout subtracts compute_lmr_reduction, zero for EndGameStrategy.
                    minus=F(("SubWithOverflow",depth,"const 1_u32"),0)
                    if cd!=minus:
                        checked(isinstance(cd,tuple) and cd[0]=="field" and cd[2]==0
                             and cd[1][:2]==("SubWithOverflow",minus) and cd[1][2][0]=="compute_lmr_reduction::<NT, SS>",
                             f"bb{pc} child depth {cd}")
                found.add(pc);events[pc]=kind
            elif fn.endswith("Thread::split"):
                checked(live is None and v(1)==ctx,f"split with child context")
                obj=v(2);checked(obj["board"]==board and obj["depth"]==depth,"wrong split board/depth")
            elif fn.endswith(("::shallow_search","::static_evaluate","::evaluate")):
                checked(live is None,f"leaf entered with active child")
        if term=="return;":checked(live is None,f"{name} returns without restoring context")
        nexts=g.normal(pc)
        branch=re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];",term)
        if branch and f"{branch[1]} = const <SS as search::strategy::SearchStrategy>::IS_ENDGAME;" in ops[:-1]:
            nexts=[int(branch[3])]
        todo.extend((n,live) for n in nexts)
    checked(len(found)==(2 if worker else 5),f"unexpected child calls {found}")
    return events
def put(g):
    return [(p,g.val(l),g.val(r)) for p,ops in g.cfg.items() for stmt in ops[:-1]
            if " = " in stmt for l,r in [stmt[:-1].split(" = ",1)] if l.startswith("(")]

def only_path(g,first,last):
    p=first;out=[]
    while p!=last and p not in out:
        out.append(p);p,=g.normal(p)
    checked(p==last,"nonlinear empty-list path")
    return out

def evaluation_context(d):
    # Sorting and the root aspiration estimate both run these depth-limited evaluators
    # on the caller's mutable context. Each helper must return to the caller's board.
    specs = [
        ("search-midgame-evaluate_depth1",4,1,{57:"sq",58:"flip"}, {"evaluate_depth1","evaluate"},2),
        ("search-midgame-evaluate_depth2",4,1,{61:"move"}, {"evaluate_depth1","evaluate_depth2"},2),
        ("search-midgame-evaluate_shallow_with_tt",5,1,{92:"move"}, {"evaluate_depth1","evaluate_shallow_with_tt"},3),
        ("move_list-ordering-{impl#0}-evaluate_moves_by_search",7,2,{81:"move"},
         {"evaluate",C+"evaluate_main_with_key","shallow_search_score"},3),
        ("move_list-ordering-shallow_search_score",3,1,{},
         {"evaluate","evaluate_depth1","evaluate_depth2","evaluate_shallow_with_tt"},3),
        ("search-endgame-estimate_aspiration_base_score",2,1,{},{"evaluate","evaluate_depth2"},2)]
    for name,argc,offset,roots,reads,count in specs:
        g=Graph(d,name,argc,roots);ctx,board="arg"+str(offset),"arg"+str(offset+1)
        mutable_frontier(g,offset,reads|{C+k for k in ("update","undo","update_pass","undo_pass","increment_nodes")}
                         |{"ordering::<impl MoveList>::evaluate_moves_fast"})
        seen=set();todo=[(0,None)];sites=set()
        while todo:
            pc,live=todo.pop()
            if (pc,live) in seen:continue
            seen.add((pc,live));ops=g.cfg[pc]
            for op in ops[:-1]:
                checked(not op.startswith("((") or not any(x in op.split(" = ",1)[0] for x in
                    (".10: u32)","search::side_to_move::SideToMove","pattern_feature::PatternFeatures")),name+" field overwrite")
            if m:=re.fullmatch(CALL,ops[-1]):
                fn=m[2];args=m[3].split(", ");v=lambda i:g.val(args[i])
                if fn in (C+"update",C+"update_pass"):
                    checked(live is None and v(0)==ctx,name+" update")
                    live="pass" if fn.endswith("_pass") else (v(1),v(2))
                elif fn in (C+"undo",C+"undo_pass"):
                    checked(v(0)==ctx and (live=="pass" if fn.endswith("_pass") else isinstance(live,tuple)),name+" undo")
                    live=None
                elif fn in reads:
                    want=board if live is None else (("Board::switch_players",board) if live=="pass" else
                        ("Board::make_move_with_flipped",board,live[1],live[0]))
                    checked(v(0)==ctx and v(1)==want,f"{name} bb{pc} child/root board")
                    sites.add(pc)
            if ops[-1]=="return;":checked(live is None,name+" retained child context")
            todo.extend((n,live) for n in g.normal(pc))
        checked(len(sites)==count,f"{name} read sites {sites}")

def root_calls(d):
    a="arg1";b=F(a,0);g=Graph(d,"search-endgame-search_root",3)
    new=(C+"new",b,F(a,1),("RootMoves::new",b),F(a,2),F(a,4))
    g.sole("solve_root",0,new);g.sole("solve_root",1,a)
    g.sole("estimate_aspiration_base_score",0,new);g.sole("estimate_aspiration_base_score",1,b)
    g=Graph(d,"search-endgame-solve_root",7)
    g.sole("search::endgame::aspiration_search",0,a);g.sole("search::endgame::aspiration_search",1,F("arg2",0))
    g=Graph(d,"search-endgame-aspiration_search",5)
    for i,arg in enumerate((a,"arg2",F(a,10))):g.sole("search::<Root, EndGameStrategy>",i,arg)

def verify(d,repo):
    methods(d);root_calls(d);evaluation_context(d)
    lmr_returns_zero(read(d,"search-pvs-compute_lmr_reduction"))
    src=repo/"crates/reversi-core/src"
    checked("pub const INITIAL_EMPTY_COUNT: usize = 60;" in (src/"constants.rs").read_text(),"root ply constant")
    checked("Player = 0," in (src/"search/side_to_move.rs").read_text() and
            "Opponent = 1," in (src/"search/side_to_move.rs").read_text(),"side discriminants")
    rows=[]
    for worker,fn in ((False,'search'),(True,'search_split_point')):
        for pc,k in sorted(bracket(d,'search-pvs-'+fn,worker).items()):
            rows.append(f"  | {str(worker).lower()}, {pc} => some {k}")
    return PROGRAM.replace('{sites}','\n'.join(rows))

PROGRAM='''module
public import NeuralReversi.SearchContextState
public import NeuralReversi.PvsParallel.ExactDepth
public import NeuralReversi.EndgameSearch.Cancellation

@[expose] public section

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position GeneratedSquare play pass)
open SearchContextState (State)

inductive Step where
  | move
  | pass
  deriving DecidableEq

inductive Callee where
  | move (pv : Bool)
  | pass
  deriving DecidableEq

def recursiveSite : Bool → Nat → Option Callee
{sites}
  | _, _ => none

def childSite (worker : Bool) (pc : Nat) : Option Step :=
  (recursiveSite worker pc).map fun
    | .move _ => .move
    | .pass => .pass

def start (root : Position) : State :=
  SearchContextState.withFeatures ⟨Board.emptyCount root.val⟩ false (SearchContextState.rebuild root.val)

def rootDepth (s : State) : ExactDepth.Depth := s.empties.count

def place (s : State) (sq : Bitboard.Square) (flips : Bitboard.Bits) : State :=
  SearchContextState.placement s sq flips

def restore (s : State) : State := SearchContextState.undo s

def passTurn (s : State) : State := SearchContextState.passTurn s

def unpass (s : State) : State := SearchContextState.undoPass s

def splitCopy (s : State) : State :=
  SearchContextState.withFeatures s.empties s.opponentTurn (s.slots (SearchContextState.ply s))

/-- `EndGameStrategy::shallow_search` builds its `Empties` from the board it receives. -/
def leafState (b : Position) : EndgameSearch.Cancellation.EmptyState :=
  ⟨Empties.encode (Board.emptySquares b.val), Board.emptyCount b.val⟩

end NeuralReversi.PvsParallel.ContextPathMir
'''
