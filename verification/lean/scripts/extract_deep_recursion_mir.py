"""Extract the deep NWS node, both iterator loops, poll/store sites, and the shallow/deep child frame."""
import re
from extract_cache_owner import END, routing
from extract_shallow_recursion_mir import verify_sites, Walker, NAMES, body, val, verify_sentinel
from mir_common import blocks, checked
from extract_header import ordinary
from extract_shallow_recursion_mir import CTX, BOARD, CACHE, ALPHA, SQ, BETA, STATS, LGL, EMPS

COUNT = ("count", EMPS)
NEXT = "next-empties"

CUT = "cutoff"
DEEP = ("deep-cache", CACHE)
SHALLOW = ("shallow-cache", CACHE)
ARGS = [CTX, BOARD, ALPHA, CACHE, CUT, LGL, EMPS]
BETA_NODE = ("add", ALPHA, 1)
NEG_BETA = ("neg", BETA_NODE)
POLL = ("poll", CUT)
HALTED = "stopped-score"
DEEP_SIG = "(_1: &mut SearchContext<'_>, _2: &Board, _3: DiscScore, _4: &mut EndGameCaches, _5: &C, _6: LegalMoves, _7: Empties) -> DiscScore {"
CHILD_SIG = "(_1: &mut SearchContext<'_>, _2: &Board, _3: LegalMoves, _4: Empties, _5: DiscScore, _6: &mut EndGameCaches, _7: &C) -> DiscScore {"

class DeepWalker(Walker):
    names = dict(NAMES, **{
        "<C as SplitPointCutoff>::occurred":"poll", "null_window_search::<C>":"deep", "shallow_search::<6>":"shallow",
        "search_move_nws_ec::<C>":"deep-child", "EndGameCache::<Deep>::probe":"probe",
        "EndGameCache::<Deep>::slot":"slot", "EndGameCache::<Deep>::store":"write",
        "Board::opponent":"opponent","Board::player":"player","flip::flip":"flip",
        "<Bitboard as PartialEq>::eq":"equal-bits","Board::make_move_with_flipped":"place",
        "SmallMoveList::new":"small-new","SmallMoveList::generate":"generate","SmallMoveList::iter":"small-iter",
        "<std::slice::Iter<'_, SmallMove> as IntoIterator>::into_iter":"small-into",
        "<std::slice::Iter<'_, SmallMove> as Iterator>::next":"small-next",
        "<DiscScore as PartialOrd>::le":"le","SmallMoveList::evaluate_moves":"evaluate",
        "SmallMoveList::best_first_iter":"best-iter","<impl Iterator<Item = SmallMove> + '_ as IntoIterator>::into_iter":"best-into",
        "<impl Iterator<Item = SmallMove> + '_ as Iterator>::next":"best-next",
    })
    def read(self,s):
        if m:=re.fullmatch(r"&(?:mut )?\(\(\*(_\d+)\)\.(0|1): search::endgame::cache::EndGameCache<search::endgame::cache::(Deep|Shallow)>\)",s):
            checked((m[2],m[3]) in [("0","Deep"),("1","Shallow")],"wrong cache projection")
            return (m[3].lower()+"-cache",self.env[m[1]])
        if m:=re.fullmatch(r"copy \(\((_([0-9]+)) as Some\)\.0: &?search::endgame::small_move_list::SmallMove\)",s):
            return val(self.env[m[1]])
        if m:=re.fullmatch(r"&\(\(\*(_\d+)\)\.0: board::Board\)|&\((_([0-9]+))\.0: board::Board\)",s):
            return ("board-of",self.env[m[1] or m[2]])
        if m:=re.fullmatch(r"copy \((_([0-9]+))\.(1: legal_moves::LegalMoves|3: square::Square)\)",s):
            return ("moves-of" if m[3][0]=="1" else "square-of",self.env[m[1]])
        if s=="const score::DISC_SCORE_MAX":return 64
        if s=="const search::endgame::SPLIT_CUTOFF_SCORE":return HALTED
        if m:=re.fullmatch(r"(Ge|Le)\((copy|move) (_\d+), const search::endgame::([A-Z_]+)\)",s):
            return self.decide((m[1],self.env[m[3]],m[4]))
        return super().read(s)


def verify_child(cfg, sites):
    count=("count",NEXT)
    for mode in ("stop","shallow","deep"):
        def decide(v):
            # The shallow arm asserts that it hands exactly six empties to shallow_search::<6>.
            if v==("eq",count,6):return True
            checked(v==("Le",count,"DEPTH_TO_SHALLOW_SEARCH"),"invalid routing condition")
            return mode=="shallow"
        def call(c,raw,w):return mode=="stop" if c==POLL else c
        w=DeepWalker(cfg,[CTX,BOARD,LGL,NEXT,BETA,CACHE,CUT],decide,call)
        answer=w.run()
        if mode=="stop":expected=[POLL];want=HALTED
        else:
            window=("neg",BETA)
            sub=(("shallow",CTX,BOARD,window,SHALLOW,LGL,NEXT) if mode=="shallow" else
                 ("deep",CTX,BOARD,window,CACHE,CUT,LGL,NEXT))
            want=("neg",sub)
            expected=[POLL,("nodes",CTX),count]+([count] if mode=="shallow" else [])+[window,sub,want]
        checked(answer==want and w.calls==expected,f"incorrect deep frame for {mode}")
        verify_sites(w, sites, {"stop": [], "shallow": [0], "deep": [1]}[mode])

def verify_node(cfg, sites):
    own_slot=("slot",DEEP,("hash",BOARD))
    probe=("probe",DEEP,own_slot,BOARD,ALPHA)
    stable=("stable",BOARD,COUNT,ALPHA)
    only=("only",LGL)
    sq=val(only);flips=("flip",sq,("player",BOARD),("opponent",BOARD))
    board=("place",BOARD,flips,sq)
    one=("deep-child",CTX,board,("moves",board),("without",EMPS,sq),BETA_NODE,CACHE,CUT)
    generated=("generate",("small-new",),BOARD,LGL)
    table=("generated",BOARD,LGL)
    rated=("rated",table)
    inorder=("small-into",("small-iter",table))
    best=("best-into",("best-iter",rated))
    nxt=("small-next",inorder)
    best_next=("best-next",best)
    bchild=("board-of",val(nxt));hash_next=("hash",bchild)
    probe_next=("probe",DEEP,("slot",DEEP,hash_next),bchild,NEG_BETA)
    moved=("without",EMPS,("square-of",val(best_next)))
    sub=("deep-child",CTX,("board-of",val(best_next)),("moves-of",val(best_next)),moved,BETA_NODE,CACHE,CUT)
    effects={"probe","stable","write","shallow","deep","child","group","deep-child","without","four","poll","nodes",
             "stable-stat","small-new","generate","small-iter","small-into","evaluate","best-iter","best-into","small-next","best-next","drop"}
    pick=lambda w:[c for c in w.calls if c[0] in effects]
    for mode in ("terminal","pass","hit","stable","one-wipe","one-stop","one-ok","multi-wipe","etc","ordered"):
        def decide(v):
            if v==probe:return mode=="hit"
            if v==stable:return mode=="stable"
            if v==only:return mode.startswith("one-")
            if v==("Ge",COUNT,"MIN_EC_ETC_EMPTIES"):return mode=="etc"
            raise ValueError(f"wrong node discriminant {v}")
        def call(c,raw,state):
            if c[0]=="same":checked(c==("same",LGL,LGL),"wrong legal-mask assertion");return True
            if c[0]=="sameEmpties":checked(c==("sameEmpties",EMPS,("emptiesOf",BOARD)),"wrong empties assertion");return True
            if c[0]=="empty":
                if c[1]==LGL:return mode in ("terminal","pass")
                checked(c[1]==("moves",("pass",BOARD)),"invalid switched legal moves");return mode=="terminal"
            if c[0]=="equal-bits":checked(c==("equal-bits",flips,("opponent",BOARD)),"incorrect wipeout");return mode=="one-wipe"
            if c[0]=="generate":
                checked(c==generated and raw[0][0]=="ref","wrong generated list")
                state.env[raw[0][1]]=table
                return mode!="multi-wipe"
            if c[0]=="evaluate":
                checked(c==("evaluate",table,CTX) and raw[0][0]=="ref","wrong evaluated buffer")
                state.env[raw[0][1]]=rated
            if c==POLL:return mode=="one-stop"
            return c
        w=DeepWalker(cfg,ARGS,decide,call,stops=(68,92));out=w.run()
        if mode=="terminal":wanted=("terminal",BOARD,COUNT);expected=[]
        elif mode=="pass":
            rec=("deep",CTX,("pass",BOARD),NEG_BETA,CACHE,CUT,("moves",("pass",BOARD)),EMPS)
            wanted=("neg",rec);expected=[("nodes",CTX),rec]
        elif mode=="hit":wanted=val(probe);expected=[probe]
        elif mode=="stable":wanted=val(stable);expected=[probe,stable,("stable-stat",STATS)]
        elif mode=="one-wipe":wanted=64;expected=[probe,stable]
        elif mode.startswith("one-"):
            wanted=HALTED if mode=="one-stop" else one
            expected=[probe,stable,("without",EMPS,sq),one,POLL]+([] if mode=="one-stop" else [("write",DEEP,own_slot,BOARD,ALPHA,one)])
        else:
            expected=[probe,stable,("small-new",),generated]
            if mode=="multi-wipe":wanted=64
            elif mode=="etc":wanted=("back",68);expected += [("small-iter",table),inorder]
            else:wanted=("back",92);expected += [("evaluate",table,CTX),("best-iter",rated),best]
        checked(out==wanted and pick(w)==expected,f"incorrect deep entry {mode}")
        sites_before_loop = {"terminal": [], "pass": [0], "one-stop": [1, 2, 3], "one-ok": [1, 2, 3, 4]}.get(mode, [1, 2])
        verify_sites(w, sites, sites_before_loop)
        if mode=="etc":
            base=w.env.copy()
            for path in ("exhausted","miss","weak","cut"):
                w.calls=[];w.call_blocks=[];w.env=base.copy()
                w.decide=lambda v:{nxt:path!="exhausted",probe_next:path!="miss"}[v]
                def visit(c,raw,w):
                    if c[0]=="le":checked(c==("le",val(probe_next),NEG_BETA),"wrong ETC acceptance bound");return path=="cut"
                    return call(c,raw,w)
                w.call=visit;out=w.run(68)
                if path=="exhausted":
                    checked(out==("back",92) and w.calls==[nxt,("neg",127),("evaluate",table,CTX),("best-iter",rated),best] and w.env["_194"]==("neg",127),"incorrect ETC exhaust")
                else:
                    expected=[nxt,hash_next,("slot",DEEP,hash_next),NEG_BETA,probe_next]
                    if path!="miss":expected += [NEG_BETA,("le",val(probe_next),NEG_BETA)]
                    good=("neg",val(probe_next))
                    if path=="cut":expected.append(good)
                    checked(w.calls==expected and out==(good if path=="cut" else ("back",68)) and w.env["_166"]==inorder and w.env["_152"]==table,"bad ETC iteration")
                verify_sites(w, sites, [] if path == "exhausted" else [5, 6])
        if mode=="ordered":
            base=w.env.copy()
            checked(base["_194"]==("neg",127),"wrong initial best score")
            for path in ("continue","empty","empty-stop","cut","cut-stop"):
                w.calls=[];w.call_blocks=[];w.env=base.copy();w.env["_194"]="prior"
                w.decide=lambda v:{best_next:not path.startswith("empty")}[v]
                def visit(c,raw,w):
                    if c[0]=="ge":checked(c==("ge",sub,BETA_NODE),"incorrect score cutoff");return path.startswith("cut")
                    if c==POLL:return path.endswith("-stop")
                    return c
                w.call=visit;out=w.run(92)
                expected=[best_next]+([] if path.startswith("empty") else [moved,sub,("ge",sub,BETA_NODE)])
                if path=="continue":
                    mx=("max","prior",sub)
                    checked(out==("back",92) and w.calls==expected+[mx] and w.env["_194"]==mx and w.env["_202"]==best,"incorrect ordered iteration")
                else:
                    result="prior" if path.startswith("empty") else sub
                    expected += [("drop",best),("drop",best),POLL]
                    if not path.endswith("-stop"):expected += [("write",DEEP,own_slot,BOARD,ALPHA,result)]
                    checked(w.calls==expected and out==(HALTED if path.endswith("-stop") else result),"incorrect parent publication")
                issued = [] if path.startswith("empty") else [7]
                if path not in ("continue", "empty-stop", "cut-stop"):
                    issued.append(8)
                verify_sites(w, sites, issued)

def constants(dump):
    data={}
    for mod,name in (("search-endgame","DEPTH_TO_SHALLOW_SEARCH"),("search-endgame","MIN_EC_ETC_EMPTIES"),
                     ("search-endgame","SPLIT_CUTOFF_SCORE"),("score","DISC_SCORE_MAX")):
        source=(dump/f"reversi_core.{mod}-{name}.built.after.mir").read_text(encoding="utf-8")
        checked(source.splitlines()[0]==f"// MIR for \x60{mod.replace('-', '::')}::{name}\x60 after built","wrong constant owner")
        code={k:ordinary(v) for k,v in blocks(source).items()}
        if name in ("DEPTH_TO_SHALLOW_SEARCH","MIN_EC_ETC_EMPTIES"):
            m=re.fullmatch(r"_0 = const (\d+)_u32;",code[0][0]);checked(m and code=={0:[m[0],"return;"]},"wrong unsigned constant")
        else:
            m=re.fullmatch(r"_0 = DiscScore::new\(const (-?\d+)_i32\) -> \[return: bb1, unwind: bb2\];",code[0][0])
            checked(m and code=={0:[m[0]],1:["return;"]},"wrong score constant")
        data[name]=int(m[1])
    return data

def check(dump):
    verify_sentinel(dump)
    for name, sig, verify_path in (
        ("search_move_nws_ec", CHILD_SIG, verify_child),
        ("null_window_search", DEEP_SIG, verify_node),
    ):
        sites = {pc: ordinal for ordinal, (pc, _, _) in enumerate(routing(dump, END + name))}
        verify_path(body(dump, name, sig), sites)
    return constants(dump)


def verify(dump, repo):
    data=check(dump)
    result=PROGRAM
    for tag,name in [("@STOP@","SPLIT_CUTOFF_SCORE"),("@SHALLOW@","DEPTH_TO_SHALLOW_SEARCH"),("@MAX@","DISC_SCORE_MAX")]:
        result=result.replace(tag,str(data[name]))
    return result

def render_thresholds(dump):
    """Render the endgame thresholds shared by the extracted and hand-written models."""
    return THRESHOLDS.replace("@ETC@",str(constants(dump)["MIN_EC_ETC_EMPTIES"]))

THRESHOLDS="""/-! Endgame search thresholds extracted from the rustc MIR of `search::endgame` by
`verification/lean/scripts/extract_deep_recursion_mir.py`. Regenerate with
`check_endgame_mir.py --write` instead of editing. -/

namespace NeuralReversi.EndgameSearch.Thresholds

/-- `MIN_EC_ETC_EMPTIES`: the fewest empties at which the deep search tries enhanced cutoffs. -/
def minEcEtcEmpties : Nat := @ETC@

end NeuralReversi.EndgameSearch.Thresholds
"""

PROGRAM="""import NeuralReversi.EndgameSearch.Cancellation
import NeuralReversi.EndgameSearch.Thresholds

namespace NeuralReversi.EndgameSearch.DeepRecursionMir

open EndgameTreeCache (Address Store entryFor)
open Cancellation (EmptyState Outcome State)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare play)
open Flip.Dispatch (TargetConfig)

def stopped : Int := @STOP@

def negate (r : Outcome) : Outcome := { r with score := r.score.map Neg.neg }

def executeChild (sq : Bitboard.Square) (beta : Int) (input : State)
    (shallow : EmptyState → Int → List Store → Result) (deep : Int → State → Outcome) : Outcome :=
  let polled := { input with polls := input.polls.after }
  if input.polls.occurred then ⟨none, polled⟩
  else
    let placed := { polled with empties := input.empties.remove sq }
    let out := if placed.empties.count.toNat ≤ @SHALLOW@ then
      let r := shallow placed.empties (-beta) placed.history
      ⟨some (-r.score), {placed with history := r.history}⟩
    else negate (deep (-beta) placed)
    { out with state := {out.state with empties := out.state.empties.restore sq} }

def finish (b : Position) (alpha best : Int) (state : State) : Outcome :=
  let polled := { state with polls := state.polls.after }
  if state.polls.occurred then ⟨none, polled⟩
  else ⟨some best, { polled with history := state.history ++ [⟨.deep, ⟨b.val, alpha, best⟩⟩] }⟩

def ordered {Move : Type} (child : Move → State → Outcome) (beta : Int) :
    List Move → Int → State → Int × State
  | [], best, state => (best, state)
  | m :: tail, best, state =>
    let out := child m state
    let score := out.score.getD stopped
    if beta ≤ score then (score, out.state) else ordered child beta tail (max best score) out.state

def early (fc : FlipConfig) (paired : Bool) (b : Position) (s : State) (addr : Address) (a : Int)
    (first : Bitboard.Square) (rest : List Bitboard.Square) : Option Int :=
  match EndgameCache.probe (entryFor addr s.history .deep b.val) b.val a with
  | some score => some score
  | none =>
    match EndgameStability.tryCutoff b.val s.empties.count.toNat ⟨a⟩ with
    | some score => some score.value
    | none =>
      let wipe := EndgameCutoffs.wipesOut fc b.val
      if rest.isEmpty then if wipe first then some @MAX@ else none
      else
        let generated := MoveList.SmallGenerate.generate paired wipe (first :: rest)
        if generated.1 then
          if Thresholds.minEcEtcEmpties ≤ s.empties.count.toNat then
            let boards := generated.2.map fun sq => Board.place b.val
              (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq
            EndgameCache.enhancedCutoff (entryFor addr s.history .deep) a boards
          else none
        else some @MAX@

def moves (b : Position) (a : Int) (state : State) (first : Move) (rest orderedMoves : List Move)
    (child : Move → State → Outcome) : Outcome :=
  let result := if rest.isEmpty then
    let r := child first state
    (r.score.getD stopped, r.state)
  else ordered child (a + 1) orderedMoves (-127) state
  finish b a result.1 result.2

end NeuralReversi.EndgameSearch.DeepRecursionMir
"""


