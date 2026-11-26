"""Decode the normal source paths from the root constructor through the initialized MoveArray slice."""
import re

from check_outer_state import eval_body
from extract_endgame_order_mir import read
from extract_pvs_context_mir import Graph, F
from extract_root_wipeout import constants
from mir_common import run_extractor_cli, checked, split_call

MOVE = "move_list-{impl#2}-"
ARRAY = "move_list-move_array-{impl#0}-"
ROOT = "search-root_move-"
ITER = "<std::slice::Iter<'_, move_list::Move> as "
NEXT = ITER + "Iterator>::next"
INTO = ITER + "IntoIterator>::into_iter"
LOCK = "std::sync::Mutex::<Vec<RootMove>>::new"
VEC = "Vec::<RootMove>::with_capacity"
PUSH = "Vec::<RootMove>::push"
ATOMIC = "Atomic::<usize>::new"
ARG = "arg1"
FILL = "MoveList::fill_in_place::<false>"


def trace(g, switches):
    pc = 0
    calls = []
    seen_switches = {}
    switch_sites = {}
    for _ in range(90):
        *stmts, end = g.cfg[pc]
        for stmt in stmts:
            checked(re.fullmatch(r"_\d+ = .+;",stmt) is not None, f"unexpected store at bb{pc}: {stmt}")
        if end == "return;":
            return calls
        if m := split_call(end):
            names = m[2]
            calls.append((names, *[g.val(k) for k in m[3].split(", ") if k]))
            pc = int(m[4])
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[(.+)\];",end):
            tag = g.val(m[1])
            values = switches[tag]
            checked(switch_sites.setdefault(tag,pc) == pc,"loop test did not return to its header")
            count = seen_switches.get(tag,0)
            checked(count < len(values), f"invalid repeated switch {tag}")
            seen_switches[tag] = count+1
            cases = dict(x.split(": bb") for x in m[2].split(", "))
            value = str(int(values[count]))
            pc = int(cases.get(value,cases["otherwise"]))
        else:
            edges = g.normal(pc)
            checked(len(edges) == 1, f"unexpected terminating edge {end}")
            pc = edges[0]
    raise ValueError("source failed to complete constructor loop")


def simple(d, name, nargs, calls, output):
    g = Graph(d,name,nargs)
    checked(trace(g,{}) == calls, name + " normal calls")
    checked(g.val("_0") == output, name + " result " + str(g.val("_0")))


def render(d, repo):
    field = F(ARG,0)
    simple(d,"board-{impl#1}-legal_moves",1,[("LegalMoves::new",ARG)],("LegalMoves::new",ARG))
    legal = ("Bitboard::legal_moves",("Board::player",ARG),("Board::opponent",ARG))
    g = Graph(d,"legal_moves-{impl#0}-new",1)
    bounds = ("Le",("Bitboard::count",legal),"const legal_moves::LegalMoves::MAX_COUNT")
    legal_calls = trace(g,{"const true":[True],"const false":[False],bounds:[True]})
    checked(legal_calls[:3] == [legal[1],legal[2],legal] and legal_calls[3:] in ([],[("Bitboard::count",legal)]) and
            g.val("_0") == ("LegalMoves",legal), "Board legal mask or bound")
    legal = ("Board::legal_moves",ARG)
    simple(d,MOVE+"new",1,[legal,("MoveList::with_moves",ARG,legal)],("MoveList::with_moves",ARG,legal))
    uninit = ("MaybeUninit::<MoveList>::uninit",)
    ptr = ("MaybeUninit::<MoveList>::as_mut_ptr",uninit)
    wrapped = ("MaybeUninit::<MoveList>::assume_init",uninit)
    simple(d,MOVE+"with_moves",2,[uninit,ptr,(FILL,ARG,"arg2",ptr),wrapped],wrapped)
    simple(d,MOVE+"count",1,[("MoveArray::len",field)],("MoveArray::len",field))
    simple(d,ARRAY+"len",1,[],F(ARG,1))
    getptr = ("core::slice::<impl [MaybeUninit<move_list::Move>]>::as_ptr",field)
    cast = ("std::ptr::const_ptr::<impl *const MaybeUninit<move_list::Move>>::cast::<move_list::Move>",getptr)
    simple(d,ARRAY+"as_ptr",1,[getptr,cast],cast)
    width = int(re.search(r"\.0: \[std::mem::MaybeUninit<move_list::Move>; (\d+)\]",read(d,ARRAY+"as_ptr"))[1])
    maximum = int(re.search(r"pub const MAX_COUNT: usize = (\d+);",(repo/"crates/reversi-core/src/legal_moves.rs").read_text(encoding="utf-8"))[1])
    checked(maximum == width,"LegalMoves bound differs from MoveArray capacity")
    slicer = Graph(d,ARRAY+"as_slice",1)
    prefix = slicer.val("_0")
    ptr = ("MoveArray::as_ptr",ARG)
    length = ("MoveArray::len",ARG)
    checked(prefix in ((f"{p}::from_raw_parts::<'_, move_list::Move>",ptr,length) for p in ("std::slice","slice")),"pointer or initialized length")
    checked(trace(slicer,{}) == [ptr,length,prefix],"slice constructor")
    simple(d,"move_list-move_array-{impl#3}-deref",1,[("MoveArray::as_slice",ARG)],("MoveArray::as_slice",ARG))
    it = ("core::slice::<impl [move_list::Move]>::iter",field)
    simple(d,MOVE+"iter",1,[("<MoveArray as Deref>::deref",field),it],it)

    # Recover pushed fields and the reachable write in each iterator cycle.
    neg = ("<Score as Neg>::neg","const score::Score::INF")
    empty = ("Vec::<Square>::new",)
    g = Graph(d,ROOT+"{impl#0}-new",1)
    checked(trace(g,{}) == [neg,neg,empty],"fresh score or PV constructor")
    fields = g.val("_0")
    terms = {ARG:"sq",neg:"negativeInfinity",empty:"[]","const 0_i64":"0"}
    fresh = [terms[fields[k]] for k in ("sq","score","average_score","mean_squared_score","pv")]
    g = Graph(d,ROOT+"{impl#1}-new",1)
    moves = ("MoveList::new",ARG)
    count = ("MoveList::count",moves)
    vec = (VEC,count)
    it = ("MoveList::iter",moves)
    iterator = (INTO,it)
    nxt = (NEXT,iterator)
    new = ("RootMove::new",F(("unSome",nxt),2))
    guard = ("discriminant",nxt)
    built = g.val("_0")
    index = built["pv_idx"]
    checked(built["moves"] == (LOCK,vec) and index[0] == ATOMIC,"wrong root vector or index")
    i = re.fullmatch(r"const (\d+)_usize",index[1])
    checked(i is not None,"nonconstant new root index")
    for n in (0,1,2):
        path = trace(g,{guard:[True]*n + [False]})
        expected = [moves,count,vec,it,iterator]+[op for _ in range(n) for op in (nxt,new,(PUSH,vec,new))]+[nxt,(LOCK,vec),index]
        checked(path == expected, "root-vector iterator, square, push or publication")
    guard = ("std::sync::Mutex::<Vec<RootMove>>::lock",field)
    unwrapped = ("Result::<std::sync::MutexGuard<\'_, Vec<RootMove>>, PoisonError<std::sync::MutexGuard<\'_, Vec<RootMove>>>>::unwrap",guard)
    deref = ("<std::sync::MutexGuard<\'_, Vec<RootMove>> as Deref>::deref",unwrapped)
    count = ("Vec::<RootMove>::len",unwrapped)
    simple(d,ROOT+"{impl#1}-count",1,[guard,unwrapped,deref,count],count)
    # The sole repeated live switch is the slice iterator discriminant above.
    for active in (False,True):
        for empty_moves in (False,True):
            path = d / "reversi_core.search-endgame-search_root.SimplifyCfg-initial.after.mir"
            actions,out = eval_body(path,["task","last","thread"],active,empty_moves)
            expected=([], "noLegal") if empty_moves else (["solve_root"],"solved")
            checked((actions,out) == expected,"wrong direct root entry branch")

    inf = constants(repo.resolve())["const score::Score::INF"]
    return f"""import NeuralReversi.PvsParallel.PvEntryMir
import NeuralReversi.RootMoveStorage

namespace NeuralReversi.AspirationMir.NativeRootBirth

open PvsParallel

def capacity : Nat := {width}
def negativeInfinity : Int := -{inf}

def fresh (sq : Bitboard.Square) : RootMoveStorage.Row :=
  ⟨{", ".join(fresh)}⟩

def boot (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : PvsBoardTree.Position)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) : Option (Nat × List RootMoveStorage.Row) := do
  let list := PvEntryMir.generated mc fc b paired mem
  let n ← list.len
  if n ≤ capacity then
    let rows ← (List.finRange n).mapM fun i => (list.slots i.val).map fun mv => fresh mv.sq
    return ({i[1]},rows)
  else none

end NeuralReversi.AspirationMir.NativeRootBirth
"""


def main():
    run_extractor_cli(
        __doc__,
        render,
        check_message="generated root constructor differs",
    )


if __name__=="__main__":
    main()

