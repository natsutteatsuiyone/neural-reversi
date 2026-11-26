"""Extract shallow NWS entry, grouped iteration, four-empty leaf, and child write/undo sequence."""
import re

from mir_common import checked, split_call, PASS, blocks
from extract_header import ordinary
from extract_cache_owner import END, routing


def body(dump, name, sig):
    text=(dump/f"reversi_core.search-endgame-{name}.{PASS}.after.mir").read_text(encoding="utf-8")
    checked(text.splitlines()[:3]==[f"// MIR for `search::endgame::{name}` after {PASS}","",f"fn {name}{sig}"],
            f"unexpected recursive leaf {name}")
    raw=blocks(text)
    cfg={}
    for pc,ops in raw.items():
        filtered=[o for o in ordinary(ops) if not re.fullmatch(r"FakeRead\(ForLet\(None\), _\d+\);|AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);",o)]
        checked(filtered,f"empty block {pc}");cfg[pc]=filtered
    return cfg


SHALLOW="(_1: &mut SearchContext<'_>, _2: &Board, _3: DiscScore, _4: &mut EndGameCache<Shallow>, _5: LegalMoves) -> DiscScore {"
MOVE="(_1: &mut SearchContext<'_>, _2: &Board, _3: Square, _4: DiscScore, _5: &mut EndGameCache<Shallow>) -> DiscScore {"
GROUP="(_1: &mut SearchContext<'_>, _2: &Board, _3: LegalMoves, _4: CacheSlot<Shallow>, _5: DiscScore, _6: &mut DiscScore, _7: &mut EndGameCache<Shallow>) -> Option<DiscScore> {"
NAMES = {"Board::legal_moves":"moves","<LegalMoves as PartialEq>::eq":"same","EmptyList::count":"count",
         "<DiscScore as Add<i32>>::add":"add","<DiscScore as Sub<i32>>::sub":"sub","<DiscScore as Neg>::neg":"neg",
         "LegalMoves::is_empty":"empty","Board::switch_players":"pass","Board::solve_disc_diff":"terminal",
         "SearchContext::<'_>::increment_nodes":"nodes","shallow_search":"shallow","Board::hash":"hash",
         "EndGameCache::<Shallow>::slot":"slot","EndGameCache::<Shallow>::probe":"probe","EndGameCache::<Shallow>::store":"write",
         "stability_cutoff":"stable","SearchStats::increment_stability_cut":"stable-stat","LegalMoves::only_move":"only",
         "shallow_search_move":"child","Board::empty_squares":"blanks","Bitboard::bits":"bits","Bitboard::new":"bitboard",
         "<Bitboard as Not>::not":"not","<LegalMoves as BitAnd<Bitboard>>::bitand":"and","LegalMoves::bitboard":"as-bits",
         "LegalMoves::and_not":"andNot","shallow_search_moves":"group","EmptyList::parity":"parity",
         "LegalMoves::corners":"corners","LegalMoves::non_corners":"nonCorners","Board::make_move":"play",
         "SearchContext::<'_>::update_endgame":"remove","SearchContext::<'_>::undo_endgame":"restore","sort_last4":"sorted",
         "solve4":"four","LegalMoves::iter":"iter","<BitboardIterator as IntoIterator>::into_iter":"into",
         "<BitboardIterator as Iterator>::next":"next","<DiscScore as PartialOrd>::ge":"ge","<DiscScore as Ord>::max":"max"}
CTX,BOARD,CACHE,ALPHA,SQ,BETA="ctx","board","cache","alpha","square","beta"
CELLS,STATS="cells","stats"
LGL=("moves",BOARD)
COUNT=("count",CELLS)
INF,UNIT=127,"unit"
def val(v): return ("payload",v)
def ptr(key): return ("ref",key)


class Walker:
    """Symbolic locals with mutable lvalues for the accumulator and iterator."""
    names = NAMES
    def __init__(self, cfg, args, decide, call, stops=()):
        self.cfg,self.decide,self.call,self.stops=cfg,decide,call,stops
        self.env={f"_{i}":v for i,v in enumerate(args,1)}
        self.calls=[]
        self.call_blocks=[]
    def read(self,s):
        e=self.env
        if m:=re.fullmatch(r"(?:copy|move) (_\d+)",s):return e[m[1]]
        if m:=re.fullmatch(r"(?:copy|move) \(\*(_\d+)\)",s):return self.deref(e[m[1]])
        if m:=re.fullmatch(r"&(?:mut )?\(\*(_\d+)\)",s):return e[m[1]]
        if m:=re.fullmatch(r"&(?:mut )?(_\d+)",s):return ptr(m[1])
        if m:=re.fullmatch(r"(?:copy|move) \((_\d+)\.(\d+): (?:square::Square|&legal_moves::LegalMoves)\)",s):
            v=e[m[1]];return ("field",v,int(m[2])) if isinstance(v,tuple) and v[0]=="sorted" else v[int(m[2])]
        if m:=re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: (?:score::DiscScore|square::Square)\)",s):
            return val(e[m[1]])
        if m:=re.fullmatch(r"const (\d+)_(?:i32|u32|usize|u64)(?: as u32 \(IntToInt\))?",s): return int(m[1])
        Known={"const true":True,"const ()":UNIT,"const score::DISC_SCORE_INF":INF,
                   "&((*_1).10: empty_list::EmptyList)":CELLS,"&mut ((*_1).12: search::counters::SearchStats)":STATS,
                   "const search::endgame::shallow_search::QUADRANT_MASK":"mask","Option::<DiscScore>::None":"none"}
        if s in Known:return Known[s]
        if m:=re.fullmatch(r"Option::<DiscScore>::Some\((.+)\)",s):return ("some",self.read(m[1]))
        if m:=re.fullmatch(r"discriminant\((_\d+)\)",s):return self.decide(e[m[1]])
        if m:=re.fullmatch(r"\(move (_\d+), move (_\d+)\)",s):return (e[m[1]],e[m[2]])
        if m:=re.fullmatch(r"move (_\d+) as usize \(IntToInt\)",s):return e[m[1]]
        if m:=re.fullmatch(r"copy (_\d+)\[(_\d+)\]",s):return ("index",e[m[1]],e[m[2]])
        if m:=re.fullmatch(r"(Eq|Lt|Shl|Shr|BitAnd|BitOr)\((.+), (.+)\)",s):
            a,b=self.read(m[2]),self.read(m[3])
            if m[1]=="Lt":
                if isinstance(a,int):return a<b
                checked((a,b)==(("parity",CELLS),16),"invalid parity index");return True
            if m[1]=="Eq":return self.decide(("eq",a,b))
            return (m[1],a,b)
        raise ValueError("unsupported recursive MIR operand "+s)
    def deref(self,v):
        return self.env[v[1]] if isinstance(v,tuple) and v[0]=="ref" else v
    def run(self,start=0):
        pc=start;seen=set()
        while pc not in seen:
            if pc in self.stops and seen:return ("back",pc)
            seen.add(pc)
            *ops,t=self.cfg[pc]
            for op in ops:
                if m:=re.fullmatch(r"(?:PlaceMention\(|FakeRead\(ForIndex, )(_\d+)\);",op):
                    checked(m[1] in self.env,"missing match place");continue
                if m:=re.fullmatch(r"\(\*(_\d+)\) = (.+);",op):
                    key=self.env[m[1]];checked(key[0]=="ref","invalid mutable accumulator")
                    self.env[key[1]]=self.read(m[2]);continue
                m=re.fullmatch(r"(_\d+) = (.+);",op)
                checked(m,f"unsupported recursive MIR instruction bb{pc} {op}")
                self.env[m[1]]=self.read(m[2])
            if t=="return;":return self.env["_0"]
            if c:=split_call(t):
                fn=self.names[c[2]]; raw=tuple(self.read(s) for s in c[3].split(", ")) if c[3] else ()
                args=tuple(self.deref(a) for a in raw)
                call=(fn,*args);self.calls.append(call);self.call_blocks.append(pc)
                self.env[c[1]]=self.call(call,raw,self);pc=int(c[4]);continue
            if m:=re.fullmatch(r"goto -> bb(\d+);|false(?:Unwind|Edge) -> \[real: bb(\d+), (?:unwind|imaginary): bb\d+\];",t):
                pc=int(m[1] or m[2]);continue
            if m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",t):
                self.calls.append(("drop", self.env[m[1]]));pc=int(m[2]);continue
            if m:=re.fullmatch(r"switchInt\((.+)\) -> \[(.+)\];",t):
                v=self.read(m[1]);checked(isinstance(v,int),"unexpected symbolic branch "+str(v))
                labels=dict(re.findall(r"(\d+|otherwise): bb(\d+)",m[2]))
                pc=int(labels.get(str(int(v)),labels["otherwise"]));continue
            if m:=re.fullmatch(r'assert\(move (_\d+), ".+", .+\) -> \[success: bb(\d+), unwind: bb\d+\];',t):
                checked(self.env[m[1]] is True,"failed bounds or shift assertion");pc=int(m[2]);continue
            raise ValueError(f"unsupported recursive MIR terminator bb{pc}: {t}")
        if pc in self.stops:return ("back",pc)
        raise ValueError(f"unexpected recursive MIR cycle bb{pc}")


def groups():
    bits=("bits",("blanks",BOARD))
    horizontal=("BitOr",("BitAnd",("Shl",bits,1),0xfefefefefefefefe),("BitAnd",("Shr",bits,1),0x7f7f7f7f7f7f7f7f))
    row=("BitOr",bits,horizontal)
    adjacent=("BitOr",("BitOr",horizontal,("Shl",row,8)),("Shr",row,8))
    isolated=("and",LGL,("not",("bitboard",adjacent)))
    rest=("andNot",LGL,("as-bits",isolated));parity=("bitboard",("index","mask",("parity",CELLS)))
    odd=("and",rest,parity);even=("and",rest,("not",parity))
    return [isolated,("corners",odd),("nonCorners",odd),("corners",even),("nonCorners",even)]


def verify_sites(walker, sites, expected):
    actual = [sites[pc] for pc in walker.call_blocks if pc in sites]
    checked(actual == expected, f"recursive cache call-site order: expected {expected}, got {actual}")


def verify_node(cfg, sites):
    local=("slot",CACHE,("hash",BOARD)); probe=("probe",CACHE,local,BOARD,ALPHA); stable=("stable",BOARD,COUNT,ALPHA)
    one=("only",LGL);beta=("add",ALPHA,1); neg=("neg",beta)
    important={"shallow","child","write","probe","stable","nodes","stable-stat","group","remove","restore","four","drop"}
    for mode in ["terminal","pass","cached","stable","only",*range(6)]:
        events=[]
        def decide(v):
            if v==probe:return mode=="cached"
            if v==stable:return mode=="stable"
            if v==one:return mode=="only"
            if v[0]=="group":return mode==groups().index(v[3])
            raise ValueError(f"unexpected shallow option {v}")
        def call(c,raw,state):
            fn=c[0]
            if fn=="same":
                checked(c==("same",LGL,LGL),"debug legal move argument");return True
            if fn=="empty":
                if c[1]==LGL:return mode in ("terminal","pass")
                checked(c[1]==("moves",("pass",BOARD)),"wrong pass mask");return mode=="terminal"
            if fn=="group":
                checked(raw[5][0]=="ref","missing mutable shared score")
                state.env[raw[5][1]]=("bestAfter",c[3])
            if fn in important:events.append(c)
            return c
        w=Walker(cfg,[CTX,BOARD,ALPHA,CACHE,LGL],decide,call);actual=w.run()
        if mode=="terminal":wanted=("terminal",BOARD,COUNT);expected=[]
        elif mode=="pass":
            sub=("shallow",CTX,("pass",BOARD),neg,CACHE,("moves",("pass",BOARD)));wanted=("neg",sub);expected=[("nodes",CTX),sub]
        elif mode=="cached":wanted=val(probe);expected=[probe]
        elif mode=="stable":wanted=val(stable);expected=[probe,stable,("stable-stat",STATS)]
        elif mode=="only":
            wanted=("child",CTX,BOARD,val(one),beta,CACHE)
            expected=[probe,stable,wanted,("write",CACHE,local,BOARD,ALPHA,wanted)]
        else:
            expected=[probe,stable];best=("neg",INF)
            for i,g in enumerate(groups()):
                group=("group",CTX,BOARD,g,local,beta,best,CACHE)
                expected.append(group)
                if i==mode:wanted=val(group);break
                best=("bestAfter",g)
            else:
                wanted=best;expected.append(("write",CACHE,local,BOARD,ALPHA,best))
        checked(actual==wanted and events==expected,f"wrong shallow node calls / writes: {mode}")
        if isinstance(mode, int):
            ordinals = [1, 2, *range(5, 5 + min(mode + 1, 5))]
            if mode == 5:
                ordinals.append(10)
        else:
            ordinals = {"terminal": [], "pass": [0], "cached": [1, 2], "stable": [1, 2], "only": [1, 2, 3, 4]}[mode]
        verify_sites(w, sites, ordinals)


def verify_child(cfg, sites):
    board=("play",BOARD,SQ); alpha=("neg",BETA); slot=("slot",CACHE,("hash",board))
    probe=("probe",CACHE,slot,board,alpha);stable=("stable",board,4,alpha)
    important={"remove","restore","shallow","stable","probe","four","sorted","stable-stat","write","play","child","group","nodes","drop"}
    for mode in ("recur","cached","stable","four"):
        def decide(v):return {("eq",COUNT,4):mode!="recur",probe:mode=="cached",stable:mode=="stable"}[v]
        w=Walker(cfg,[CTX,BOARD,SQ,BETA,CACHE],decide,lambda c,_,s:c)
        result=w.run()
        calls=[board,("remove",CTX,SQ)]
        if mode=="recur":
            child=("shallow",CTX,board,alpha,CACHE,("moves",board));calls.append(child)
        else:
            calls.append(probe)
            if mode=="cached":child=val(probe)
            else:
                calls.append(stable)
                if mode=="stable":child=val(stable);calls.append(("stable-stat",STATS))
                else:
                    ordered=("sorted",CELLS)
                    child=("four",CTX,board,alpha,*(("field",ordered,i) for i in range(4)))
                    calls.extend([ordered,child,("write",CACHE,slot,board,alpha,child)])
        calls.append(("restore",CTX,SQ))
        checked(result==("neg",child) and [c for c in w.calls if c[0] in important]==calls,
                f"wrong shallow child score, restore or cache publication: {mode}")
        verify_sites(w, sites, {"recur": [3], "cached": [0, 1], "stable": [0, 1], "four": [0, 1, 2]}[mode])


def verify_group(cfg, sites):
    iterator=("into",("iter","group")); nxt=("next",iterator)
    for mode in ("empty","cut","continue"):
        sub=("child",CTX,BOARD,val(nxt),BETA,CACHE)
        def call(c,_,s):
            if c[0]=="ge":
                checked(c==("ge",sub,BETA),"incorrect group cutoff");return mode=="cut"
            return c
        def decide(v):
            checked(v==nxt,"invalid iterator discriminant");return mode!="empty"
        w=Walker(cfg,[CTX,BOARD,"group","slot",BETA,ptr("prior"),CACHE],decide,call,stops=(3,))
        w.env["prior"]="acc"
        checked(w.run()==("back",3) and w.calls==[("iter","group"),iterator],"incorrect iterator acquisition")
        w.calls=[];result=w.run(3)
        if mode=="empty":
            checked(result=="none" and w.env["prior"]=="acc" and w.calls==[nxt],"incorrect exhausted group")
        elif mode=="cut":
            diff=("sub",BETA,1)
            checked(result==("some",sub) and w.env["prior"]=="acc" and
                    w.calls==[nxt,sub,("ge",sub,BETA),diff,("write",CACHE,"slot",BOARD,diff,sub)],"wrong early cutoff publication")
        else:
            mx=("max","acc",sub)
            checked(result==("back",3) and w.env["prior"]==mx and w.env["_12"]==iterator and
                    w.calls==[nxt,sub,("ge",sub,BETA),mx],"unexpected loop state on next square")
        verify_sites(w, sites, {"empty": [], "cut": [0, 1], "continue": [0]}[mode])

def verify_sentinel(dump):
    sentinel=(dump/"reversi_core.score-DISC_SCORE_INF.built.after.mir").read_text(encoding="utf-8")
    checked(sentinel.startswith("// MIR for `score::DISC_SCORE_INF` after built\n") and
            {k:ordinary(v) for k,v in blocks(sentinel).items()}=={0:["_1 = const core::num::<impl i8>::MAX as i32 (IntToInt);",
               "_0 = DiscScore::new(move _1) -> [return: bb1, unwind: bb2];"],1:["return;"]},"wrong NWS seed")


def verify(dump,repo):
    verify_sentinel(dump)
    for name, sig, verify_path in (
        ("shallow_search", SHALLOW, verify_node),
        ("shallow_search_moves", GROUP, verify_group),
        ("shallow_search_move", MOVE, verify_child),
    ):
        sites = {pc: ordinal for ordinal, (pc, _, _) in enumerate(routing(dump, END + name))}
        verify_path(body(dump, name, sig), sites)
    t=(dump/"reversi_core.search-endgame-shallow_search-QUADRANT_MASK.built.after.mir").read_text(encoding="utf-8")
    form=(r"// MIR for `search::endgame::shallow_search::QUADRANT_MASK` after built\s+const QUADRANT_MASK: \[u64; 16\] = \{\s+"
          r"let mut _0: \[u64; 16\];\s+bb0: \{\s+_0 = \[(.+)\];\s+return;\s*\}\s*\}\s*")
    data=re.fullmatch(form,t)
    checked(data,"unknown quadrant initializer")
    values=[]
    for literal in data[1].split(", "):
        if literal=="const u64::MAX":
            values.append(2**64-1)
            continue
        m=re.fullmatch(r"const (\d+)_u64",literal);checked(m,"invalid quadrant field");values.append(int(m[1]))
    checked(len(values)==16,"wrong quadrant count")
    quad=", ".join(f"0x{n:016x}" for n in values)
    return f"""import NeuralReversi.EndgameSearch.FourOrder

namespace NeuralReversi.EndgameSearch.ShallowMir

open EndgameTreeCache (Store)
open PvsBoardTree (Position)

def masks (moves : LegalMoves.Raw) (empties : Bitboard.Bits) (parity : Fin 16) : List LegalMoves.Raw :=
  let table : Vector Bitboard.Bits 16 := #v[{quad}]
  let neighbors := ShallowMoveOrder.neighbors empties
  let isolated := LegalMoves.bitand moves (~~~neighbors)
  let rest := LegalMoves.andNot moves isolated.bits
  let odd := LegalMoves.bitand rest table[parity]
  let even := LegalMoves.bitand rest (~~~table[parity])
  [isolated, LegalMoves.corners odd, LegalMoves.nonCorners odd, LegalMoves.corners even, LegalMoves.nonCorners even]

def write (b : Position) (alpha : Int) (r : Result) : Result :=
  ⟨r.score, r.history ++ [⟨.shallow, ⟨b.val, alpha, r.score⟩⟩]⟩

inductive Stage where
  | cutoff (result : Result)
  | next (best : Int) (history : List Store)

def stage (b : Position) (beta : Int) (child : Move → List Store → Result) :
    List Move → Int → List Store → Stage
  | [], best, h => .next best h
  | m :: rest, best, h =>
    let r := child m h
    if beta ≤ r.score then .cutoff (write b (beta - 1) r)
    else stage b beta child rest (max best r.score) r.history

def groups (b : Position) (alpha : Int) (child : Move → List Store → Result) :
    List (List Move) → Int → List Store → Result
  | [], best, h => write b alpha ⟨best, h⟩
  | stageMoves :: later, best, h =>
    match stage b (alpha + 1) child stageMoves best h with
    | .cutoff r => r
    | .next best' h' => groups b alpha child later best' h'

def node (b : Position) (moves : List Move) (alpha : Int) (h : List Store)
    (other : Bool) (terminal : Int) (passed : Unit → Result) (probe stable : Option Int)
    (stages : List (List Move)) (child : Move → List Store → Result) : Result :=
  match moves with
  | [] => if other then let r := passed (); ⟨-r.score, r.history⟩ else ⟨terminal, h⟩
  | m :: rest =>
    match probe with
    | some score => ⟨score, h⟩
    | none =>
      match stable with
      | some score => ⟨score, h⟩
      | none =>
        if rest.isEmpty then write b alpha (child m h)
        else groups b alpha child stages (-127) h

def child (b : Position) (alpha : Int) (h : List Store) (count : Nat) (probe stable : Option Int)
    (solveFour : Unit → Int) (recurse : Unit → Result) : Result :=
  if count = 4 then
    match probe with
    | some score => ⟨-score, h⟩
    | none =>
      match stable with
      | some score => ⟨-score, h⟩
      | none =>
        let v := solveFour ()
        ⟨-v, (write b alpha ⟨v, h⟩).history⟩
  else let r := recurse (); ⟨-r.score, r.history⟩

end NeuralReversi.EndgameSearch.ShallowMir
"""
