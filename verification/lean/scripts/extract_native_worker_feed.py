"""Verify the published immutable iterator, relaxed ticket update and worker child call sites."""
import re
from mir_common import checked
from extract_cache_owner import routing
from extract_pvs_context_mir import Graph,F,only_path,put

ITER="move_list-iterator-{impl#1}-"
ARRAY="move_list-move_array-{impl#0}-"
WORKER="search-pvs-search_split_point"

def verify(d,repo):
    a="arg1";j="arg2";rc=F(a,0);store=F(rc,0)
    Graph(d,ITER+"from_offset",2).fields({"move_list":a,"current":("Atomic::<usize>::new",j)})
    pub=Graph(d,"search-threading-thread-{impl#1}-initialize_split_point",4)
    deref=F("arg4",6);offset=F("arg4",7);init=("ConcurrentMoveIterator::from_offset",deref,offset)
    pub.sole("SplitPoint::set_move_iter",0,"arg2");pub.sole("SplitPoint::set_move_iter",1,init)
    stored=("SyncUnsafeCell::<Option<ConcurrentMoveIterator>>::get",F(a,2))
    setter=Graph(d,"search-threading-split_point-{impl#5}-set_move_iter",2)
    getter=Graph(d,"search-threading-split_point-{impl#5}-move_iter",1)
    checked(put(setter)==[(1,stored,("Option::<ConcurrentMoveIterator>::Some",j))] and
        getter.val("_0")==("Option::<&ConcurrentMoveIterator>::unwrap",("Option::<ConcurrentMoveIterator>::as_ref",stored)),
        "publisher and borrowers must share the initialized iterator cell")
    g=Graph(d,ITER+"next",1)
    array_place="&(((*_1).0: move_list::MoveList).0: move_list::move_array::MoveArray)"
    checked(g.defs["_3"]==g.defs["_21"]=={array_place},"length and ticket access refer to different arrays")
    g.roots.update({3:store,21:store})
    checked(g.val("_2")==("MoveArray::len",store),"iterator length not from immutable records")
    call,=[(pc,fn,args) for pc,fn,args in g.calls() if fn.startswith("Atomic::<usize>::try_update::")]
    pc,fn,args=call
    checked(pc==1 and g.cfg[pc][-1].startswith("_7 = Atomic::<usize>::try_update") and [g.val(x) for x in args[:3]]==[F(a,1),"std::sync::atomic::Ordering::Relaxed",
        "std::sync::atomic::Ordering::Relaxed"],"wrong atomic counter or ordering")
    checked(len(g.defs["_11"])==1 and re.fullmatch(r"\{closure@.+?\} \{ len: move _12 \}",next(iter(g.defs["_11"])))
        and args[3]=="move _11" and
        g.val("_12")==("MoveArray::len",store),"update closure captures wrong bound")
    checked(g.defs["_6"]=={"Result::<usize, usize>::ok(move _7)"}
      and g.defs["_5"]=={"<Option<usize> as Try>::branch(move _6)"}
      and g.cfg[4][-1]=="switchInt(move _13) -> [0: bb7, 1: bb6, otherwise: bb5];"
      and g.defs["_17"]=={"copy ((_5 as Continue).0: usize)"} and
      g.defs["_4"]=={"copy _17"} and g.defs["_13"]=={"discriminant(_5)"},"RMW result arm or old cursor changed")
    checked(only_path(g,0,4)==[0,1,2,3] and only_path(g,7,12)==[7,10,11] and only_path(g,6,12)==[6,8,9]
      and g.defs["_0"]=={"Option::<(&move_list::Move, usize)>::Some(move _18)",
        "<Option<(&move_list::Move, usize)> as FromResidual<Option<!>>>::from_residual(move _16)"}
      and g.defs["_14"]=={"copy ((_5 as Break).0: std::option::Option<!>)"} and g.defs["_16"]=={"copy _14"},
      "normal iterator control/return")
    g.roots[4]="old"
    checked(g.val("_18")==("tuple",("MoveArray::get_unchecked_ref",store,"old"),
        F(("AddWithOverflow","old","const 1_usize"),0)),"ticket must return record[old] and old+1")
    clo=Graph(d,ITER+"next-{closure#0}",2)
    checked(clo.val("_3")==("Lt",j,F(a,0)) and re.fullmatch(r"\{closure@.+\} \{ index: move _7 \}",
        next(iter(clo.defs["_6"]))) and clo.val("_7")==j and
        re.fullmatch(r"core::bool::<impl bool>::then::<usize, \{closure@.+\}>\(move _3, move _6\)",next(iter(clo.defs["_0"]))),
        "atomic callback must offer an increment only inside storage")
    checked(Graph(d,ITER+"next-{closure#0}-{closure#0}",1).val("_0")==F(("AddWithOverflow",F(a,0),"const 1_usize"),0),
        "atomic callback increments a different index")
    ptr=("std::ptr::const_ptr::<impl *const MaybeUninit<move_list::Move>>::cast::<move_list::Move>",
         ("core::slice::<impl [MaybeUninit<move_list::Move>]>::as_ptr",F(a,0)))
    checked(Graph(d,ARRAY+"as_ptr",1).val("_0")==ptr and Graph(d,ARRAY+"len",1).val("_0")==F(a,1),"array storage")
    checked(Graph(d,ARRAY+"get_unchecked_ref",2).val("_0")==("std::ptr::const_ptr::<impl *const move_list::Move>::add",
        ("MoveArray::as_ptr",a),j),"record offset changed")
    cfg=Graph(d,WORKER,5)
    cfg.sole("ConcurrentMoveIterator::next",0,"arg3")
    checked(cfg.cfg[5][-1].startswith("_20 = ConcurrentMoveIterator::next") and cfg.cfg[6]==[
        "_22 = discriminant(_20);","switchInt(move _22) -> [1: bb7, otherwise: bb98];"] and only_path(cfg,7,8)==[7]
        and cfg.defs["_23"]=={"copy (((_20 as Some).0: (&move_list::Move, usize)).0: &move_list::Move)"}
        and cfg.defs["_24"]=={"copy (((_20 as Some).0: (&move_list::Move, usize)).1: usize)"},
        "worker must receive record and ordinal from the same successful ticket")
    sites=routing(d,WORKER)
    bypc={pc:fn for pc,fn,args in cfg.calls()};pcs={}
    for ordinal,(pc,callee,flow) in enumerate(sites):
        checked(callee=="pvs" and flow=="forward","unexpected nested worker callee")
        specialization=bypc[pc]
        checked(specialization in ("search::<NonPV, EndGameStrategy>","search::<PV, EndGameStrategy>"),"wrong specialized child")
        pcs[specialization.startswith("search::<PV")]=(pc,ordinal)
    checked(len(sites)==len(pcs)==2,"expected scout and conditional wide site")
    scout,wide=pcs[False],pcs[True]
    return PROGRAM.replace("{scoutpc}",str(scout[0])).replace("{scout}",str(scout[1])).replace(
        "{widepc}",str(wide[0])).replace("{wide}",str(wide[1]))

PROGRAM='''import NeuralReversi.MoveList.Concurrent
import NeuralReversi.Move.Model

namespace NeuralReversi.PvsParallel.NativeWorkerFeed

def site (pv : Bool) : Nat × Nat := if pv then ({widepc}, {wide}) else ({scoutpc}, {scout})

def ticket (rc : List Move.Move) (old : BitVec w) : Option (Move.Move × Nat) := do
  let (index, ordinal) ← MoveList.Concurrent.next rc.length old
  let mv ← rc[index]?
  return (mv,ordinal.toNat)

end NeuralReversi.PvsParallel.NativeWorkerFeed
'''
