"""Replay the occupied seed of a public one-empty search after an earlier search on the same engine."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from check_occupied_sources import instrument as instrument_atomics
from check_serial_calls_trace import REPO, capture, replace_once
from normalize_pvs_trace import require


GAME = [44,43,18,29,51,34,30,22,26,38,33,50,37,32,40,59,24,53,45,19,31,10,52,20,46,61,25,17,
        12,39,41,21,47,54,42,48,11,3,2,1,58,9,55,63,15,16,23,7,13,14,5,6,60,62,8,0,49,57,56]

FIXTURE = """
use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use crate::square::Square;
pub fn run(root: &std::path::Path) {
    let resources = crate::engine::Engine::from_eval(Arc::new(Eval::with_weight_files(
        Some(&root.join(EVAL_FILE_NAME)),Some(&root.join(EVAL_SM_FILE_NAME))).unwrap()),0,Some(4));
    let engine = resources.open();
    println!("endian {}",cfg!(target_endian="big") as u8);
    let mut board = Board::new();
    for n in GAME {
        let sq = Square::from_u8(n).unwrap();
        assert!(board.legal_moves().iter().any(|m| m == sq));
        board = board.make_move(sq);
    }
    assert_eq!(board.empty_count(),1);
    for (i,level) in [Level::with_depths(1,[1,1,0,0,0]),Level::perfect()].into_iter().enumerate() {
        println!("session {i}");
        let result = resources.search(engine, &board,&SearchRunOptions::with_level(level).multi_pv(true));
        let SearchResult::BestMove {sq,score,n_nodes,pv_line,depth,selectivity,is_endgame,pv_moves,..} = result else { panic!() };
        assert_eq!(pv_moves.len(),1);
        assert_eq!(pv_line.len(),1);
        assert_eq!(pv_moves[0].pv_line.len(),1);
        println!("public {i} {} {} {n_nodes} {depth} {} {} {} {} {} {}", sq.index(),score.value(),selectivity.as_u8(),
            is_endgame as u8,pv_line[0].index(),pv_moves[0].sq.index(),pv_moves[0].score.value(),pv_moves[0].pv_line[0].index());
    }
}
"""


def instrument(src):
    instrument_atomics(src)
    tt = src / "transposition_table.rs"
    text = replace_once(tt.read_text(), "assert_eq!(SOURCE_THREAD.get(), Some(&id));", "assert_eq!(SOURCE_THREAD.get_or_init(|| id), &id);")
    text += """
impl TranspositionTable {
    pub(crate) fn warm_seed_read(&self, b: &Board, key: u64, kind: &str) -> TTProbeResult<'_> {
        let base = self.entries.as_ptr() as usize;
        let first = self.get_cluster_idx(key);
        let cell = &self.entries[first];
        let addr = cell as *const _ as usize;
        let worker = source_worker();
        if kind == "seed" {
            println!("entry_image {worker} {base} {first} {addr} {} {} {} {}",
                cell.seq.load(Ordering::Relaxed),cell.player.load(Ordering::Relaxed),
                cell.opponent.load(Ordering::Relaxed),cell.data.load(Ordering::Relaxed));
        }
        println!("probe {kind} {worker} {base} {} {} {} {key}",self.cluster_count,b.player().bits(),b.opponent().bits());
        WINDOW_TRACE.set(Some((worker,base)));
        let got = self.probe(b,key);
        WINDOW_TRACE.set(None);
        let index = (got.slot().0 as *const _ as usize - base) / mem::size_of::<TTEntry>();
        println!("returned_probe {kind} {worker} {index} {} {}",got.data().map_or(0,TTEntryData::to_u64),
            got.data().is_some() as u8);
        got
    }
}
"""
    anchor = "        slot.0.save(board, data);"
    text = replace_once(text,anchor,"""        println!("store {} {} {} {} {} {} {} {} {} {} {} {}",source_worker(),slot.0 as *const _ as usize,
            board.player().bits(),board.opponent().bits(),score.value(),bound as u8,depth,
            best_move.map_or(64,|m| m.index()),selectivity.as_u8(),data.generation(),is_endgame as u8,is_pv as u8);
""" +anchor)
    tt.write_text(text)
    root = src / "search.rs"
    text = root.read_text()
    anchor = "fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {"
    text = replace_once(text, anchor, anchor + '\n    println!("root {}",thread.idx());')
    anchor = "RootPhase::Endgame { last } => endgame::search_root(task, last, thread),"
    root.write_text(replace_once(text, anchor, """RootPhase::Endgame { last } => {
            let r = endgame::search_root(task,last,thread);
            println!("endrootback {}",thread.idx());
            r
        },"""))
    thread = src / "search/threading/thread.rs"
    anchor = "                    let result = search::search_root(task, &self);"
    thread.write_text(replace_once(thread.read_text(),anchor,anchor+'\n                    println!("rootback {}",self.idx);'))
    context = src / "search/context.rs"
    anchor = "        self.root_moves.update(sq, score, is_pv, self.pv());"
    context.write_text(replace_once(context.read_text(),anchor, """
        let pv = self.pv().iter().map_while(|q| *q).map(|q| q.index().to_string()).collect::<Vec<_>>().join(",");
        println!("pv {} {} {}",sq.index(),is_pv as u8,pv);
""" + anchor))
    end = src / "search/endgame.rs"
    text = replace_once(end.read_text(),"    let board = task.board;",'    println!("endroot {}",thread.idx());\n    let board = task.board;')
    text = replace_once(text,"    solve_root(\n","    let r = solve_root(\n")
    text = replace_once(text,"        Selectivity::Level1,\n        last,\n    )\n}","""        Selectivity::Level1,
        last,
    );
    println!("solveback {}",thread.idx());
    r
}""")
    anchor = "    let time_manager = task.time_manager.as_deref();"
    text = replace_once(text,anchor, """    println!("solve {} {} {} {} {} {} {}",thread.idx(),base_score.value(),task.multi_pv as u8,
        min_reported.as_u8(),last.as_u8(),completed.last.is_some() as u8,warm_row(ctx));
""" + anchor)
    anchor = "            let score = aspiration_search(ctx, board, &mut alpha, &mut beta, thread);"
    text = replace_once(text,anchor,'            println!("window {}",thread.idx());\n'+anchor+'\n            println!("endwindow {}",thread.idx());')
    text += """
pub(super) fn warm_row(ctx: &SearchContext) -> String {
    let rows = ctx.root_moves.snapshot();
    assert_eq!(rows.len(),1);
    let r = &rows[0];
    let pv = r.pv.iter().map(|q| q.index().to_string()).collect::<Vec<_>>().join(",");
    format!("{} {} {} {} {}",r.sq.index(),r.score.value(),r.average_score.value(),r.mean_squared_score,
        if pv.is_empty() { "-".to_owned() } else { pv })
}
"""
    anchor = "    let base_score = estimate_aspiration_base_score(&mut ctx, &board);"
    text = replace_once(text, anchor, anchor + '\n    println!("base {} {}",thread.idx(),base_score.value());')
    anchor = "    if let Some(tt_data) = ctx.tt.probe(board, board.hash()).data()"
    text = replace_once(text, anchor, '    let raw = ctx.tt.warm_seed_read(board,board.hash(),"seed");\n    println!("seedraw {} {} {}",board.player().bits(),board.opponent().bits(),raw.data().is_some() as u8);\n    if let Some(tt_data) = raw.data()')
    anchor = "        return tt_data.score();"
    text = replace_once(text, anchor, '        println!("accepted_seed {}",tt_data.score().value());\n' + anchor)
    end.write_text(text)
    pvs = src / "search/pvs.rs"
    text = pvs.read_text()
    anchor = "            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);"
    text = replace_once(text,anchor,anchor+'\n            println!("childback {} {}",thread.idx(),(-score).value());')
    anchor = "    let org_alpha = alpha;"
    text = replace_once(text, anchor, anchor + """
    println!("pvs {} {} {} {} {} {} {} {} {} {}",thread.idx(),NT::ROOT_NODE as u8,NT::PV_NODE as u8,
        depth,ctx.selectivity.as_u8(),alpha.value(),beta.value(),cut_node as u8,board.player().bits(),board.opponent().bits());
    if NT::ROOT_NODE { println!("row {} {}",ctx.root_moves.pv_idx(),endgame::warm_row(ctx)); }""")
    anchor = "            return SS::evaluate(ctx, board);"
    text = replace_once(text, anchor, """            let s = SS::evaluate(ctx,board);
            println!("leaf {} {}",thread.idx(),s.value());
            return s;""")
    anchor = "    let tt_probe_result = ctx.tt.probe(board, tt_key);"
    text = replace_once(text, anchor,'    let tt_probe_result = ctx.tt.warm_seed_read(board,tt_key,"parent");')
    anchor = "    let bound = Bound::classify::<NT>(best_score, org_alpha, beta);"
    text = replace_once(text, anchor, anchor + """
    println!("chosen {} {} {} {} {} {} {} {}",thread.idx(),board.player().bits(),board.opponent().bits(),
        depth,ctx.selectivity.as_u8(),best_score.value(),bound as u8,best_move.map_or(64,|s| s.index()));""")
    text = replace_once(text, "    best_score\n}", '    println!("back {} {}",thread.idx(),best_score.value());\n    best_score\n}')
    pvs.write_text(text)


def replay(lines, folder):
    from check_occupied_sources import render, sample_fields
    # Reserve separate event IDs for the four cold initialization writes.
    rows = [["init"] for _ in range(4)] + [line.split() for line in lines]
    events = [".engine (.other 0)" for _ in rows]
    sites = {"cas": ".cas", "releaseFence": ".fence", "player": ".field .player",
             "opponent": ".field .opponent", "metadata": ".field .metadata", "publish": ".finish"}
    labels = {"A": (".sequence",".acquire"), "D": ("(.payload .metadata)",".relaxed"),
              "P": ("(.payload .player)",".relaxed"), "O": ("(.payload .opponent)",".relaxed"), "Q": (".sequence",".relaxed")}
    roots, readers, stores, writes, read = [], [], [], [], []
    sessions, arguments, publications = [], [], []
    stack, active = [], None
    pos = lambda p,o: f"(⟨⟨{int(p)},{int(o)}⟩,rfl⟩ : PvsBoardTree.Position)"
    def li(xs):
        return "[" + ",".join(map(str,xs)) + "]"
    def pv(s):
        return li([] if s == "-" else map(int,s.split(",")))
    def row(v):
        sq,score,mean,square,pvline = v
        return f"⟨{sq},({score}),({mean}),({square}),{pv(pvline)}⟩"
    def pop(name):
        tag,start = stack.pop()
        require(tag == name,"wrong native return")
        return start
    for t,(kind,*v) in enumerate(rows):
        if kind == "init":
            continue
        if kind == "endian":
            endian = str(bool(int(v[0]))).lower()
            continue
        if kind == "session":
            require(not stack and not active,"earlier public run still active")
            current = int(v[0])
            sessions.append({})
        elif kind == "entry_image":
            w,base,slot,ptr,seq,p,o,d = map(int,v)
            if current == 0:
                original = (base,slot,ptr)
                init = [seq,p,o,d]
                require(ptr == base + slot * 32,"unexpected TT slot address")
                require(init == [0]*4,"earlier slot was occupied at start")
            require(w == 0 and original == (base,slot,ptr),"second public search allocated a different TT")
        elif kind == "probe":
            tag,w,base,n,p,o,key = v
            require(int(w) == 0 and int(base) == original[0] and active is None,"wrong probe allocation/worker")
            if not readers:
                coord = (int(p),int(o),int(n),int(key))
            require(coord == tuple(map(int,(p,o,n,key))),"wrong root board/cluster")
            active = dict(tag=tag,start=t,loads=[])
            readers.append(active)
            events[t] = ".enter 0 args"
        elif kind == "op":
            who,code,slot,value = v
            require(active and who == "0","unscoped TT reader instruction")
            if code == "F":
                op = ".fence .acquire"
            else:
                require(int(slot) == original[1],"TT load on different slot")
                loc, mode = labels[code]
                op = f".load address {loc} {mode}"
                active["loads"].append(t)
                read.append((t,loc,mode,int(value)))
            events[t] = f".issue 0 ({op}) (label {t}) {value}"
        elif kind == "returned_probe":
            tag,who,idx,value,hit = v
            require(active and active["tag"] == tag and int(idx) == original[1],"wrong TT boundary")
            answer = f".hit {int(idx)%2} (Packing.unpack endian {value})" if int(hit) else f".miss {int(idx)%2}"
            events[t] = f".leave 0 (.probe ({answer}))"
            active.update(answer=answer,finish=t,metadata=int(value))
            if tag == "seed":
                sessions[current]["seed"] = active
            active = None
        elif kind == "seedraw":
            p,o,hit = map(int,v)
            r = sessions[current]["seed"]
            require((p,o)==coord[:2] and hit == int(r["metadata"]!=0),"returned probe and receiver differ")
            sessions[current]["receiver"] = t
            events[t] = f'.seedRegister 0 b ({r["answer"]})'
        elif kind == "base":
            sessions[current]["base"] = int(v[1])
        elif kind == "accepted_seed":
            sessions[current]["accepted"] = int(v[0])
        elif kind in ("root","endroot","window"):
            name = {"root":"root","endroot":"endRoot","window":"endWindow"}[kind]
            stack.append((name,t))
            events[t] = f".engine (.localCall 0 (.entry {t} 0 .{name} .none none) none)"
        elif kind == "solve":
            w,base,multi,minimum,cap,saved,*rr = v
            require((w,multi,minimum,saved)==("0","1","1","0") and int(base)==sessions[current]["base"],"bad solve header")
            sessions[current]["cap"] = int(cap)
            stack.append(("solve",t))
            events[t]=f".solveEnter 0 {t} 0 .none ⟨b,BitVec.ofInt 32 ({base}),[{row(rr)}],none,true,{minimum},{cap}⟩"
        elif kind == "pvs":
            w,root,pvflag,d,lvl,a,z,cut,p,o=map(int,v)
            require(w == 0 and pvflag == 1 and cut == 0,"unexpected PVS routine")
            if root:
                require(stack[-1][0]=="endWindow" and rows[t+1][:2]==["row","0"] and (p,o)==coord[:2],"invalid root call")
                rr = row(rows[t+1][2:])
                root_open=t
                nodes = f"⟨b.val,{d},⟨BitVec.ofInt 32 ({a})⟩,⟨BitVec.ofInt 32 ({z})⟩,true,{lvl},true,false,some 0⟩"
                events[t]=f".rootEnter 0 {t} 0 .none {nodes} [{rr}]"
                roots.append([current,t,lvl,a,z])
            else:
                require(stack[-1] == ("pvs",root_open) and d == 0,"expected the only native PV child")
                nodes = f"⟨{pos(p,o)}.val,{d},⟨BitVec.ofInt 32 ({a})⟩,⟨BitVec.ofInt 32 ({z})⟩,true,{lvl},true,false,none⟩"
                events[t]=f".engine (.localCall 0 (.entry {t} 8 .pvs .none none) (some {nodes}))"
            stack.append(("pvs",t))
        elif kind in ("leaf","chosen"):
            value = int(v[1] if kind=="leaf" else v[5])
            events[t] = f".engine (.localCall 0 (.scored {t} ⟨BitVec.ofInt 32 ({value})⟩) none)"
            if kind=="chosen":
                require(stack[-1] == ("pvs",root_open),"root register chosen outside root")
                require(tuple(map(int,(v[0],v[1],v[2],v[3],v[4],v[6],v[7]))) ==
                        (0,*coord[:2],1,roots[-1][2],3,4),"selected register belongs to another root")
                chosen = t
        elif kind == "pv":
            sq,valid,line = v
            require(valid=="1" and stack[-1] == ("pvs",root_open),"invalid native root PV")
            arguments.append((root_open,int(sq),pv(line)))
        elif kind == "store":
            w,addr,p,o,score,bd,d,move,lvl,gen,phase,tt_pv = map(int,v)
            require((w,addr,p,o,bd,d,move,phase,tt_pv) == (0,original[2],*coord[:2],3,1,4,1,1) and stack[-1] == ("pvs",root_open),
                    "physical TT::store did not follow this root's chosen bound")
            require(rows[chosen][6] == str(score) and roots[-1][2] == lvl,"store differs from chosen root score/level")
            store = dict(chosen=chosen,score=score,move=move,level=lvl,gen=gen)
        elif kind == "call":
            w,addr,old,p,o,packed=map(int,v)
            require((w,addr,p,o)==(0,original[2],*coord[:2]),"writer allocated a different slot")
            require(rows[t-1][0]=="store","seqlock writer outside root TT::store")
            store.update(old=old,packed=packed,start=t,pc=[])
            idx=len(stores)
            stores.append(store)
            events[t]=f".engine (.beginWrite 0 {idx} (operand {idx}))"
        elif kind in sites:
            who,addr,*rest=map(int,v)
            require(who==0 and addr==original[2] and len(store["pc"])<6,"wrong publication instructions")
            if kind=="cas":
                expected,new,actual=rest
                require(expected==store["old"]==actual and new==expected|1,"failed TT writer")
                value=new
            else:
                value=rest[0]
            store["pc"].append(kind)
            writes.append((t,idx,sites[kind],value))
            events[t]=f".engine (.atomic 0 {idx})"
        elif kind in ("childback","back","endwindow","solveback","endrootback","rootback"):
            name={"childback":"pvs","back":"pvs","endwindow":"endWindow","solveback":"solve",
                  "endrootback":"endRoot","rootback":"root"}[kind]
            first=pop(name)
            result=f"(some ⟨BitVec.ofInt 32 ({v[1]})⟩)" if name=="pvs" else "none"
            events[t]=f".engine (.localCall 0 (.leave {t} {first} .{name} none {result}) none)"
        elif kind=="public":
            publications.append(list(map(int,v)))
        elif kind!="row":
            raise AssertionError(kind)
    require(not active and not stack and [s["cap"] for s in sessions]==[2,5] and len(stores)==len(roots)==7,"incomplete public history")
    require(all(s["pc"]==list(sites) for s in stores),"unobserved native writer PC")
    require(sessions[1]["accepted"]==sessions[1]["base"] and "accepted" not in sessions[0],"seed path")
    def by_time(fn):
        return "(match q.time with " + " ".join(f"| {t} => {li(fn(t,l,w))}" for t,l,m,w in read)+" | _ => [])"
    def history(loc):
        key={".sequence":(".cas",".finish"),"(.payload .player)":(".field .player",),
             "(.payload .opponent)":(".field .opponent",),"(.payload .metadata)":(".field .metadata",)}[loc]
        return [(at,value) for at,_,pc,value in writes if pc in key]
    def possible(t,l,w,ordered):
        ws=history(l)
        lower=sum(at<t for at,_ in ws) if ordered else 0
        return [n for n,val in enumerate([0]+[v for _,v in ws]) if n>=lower and val==w]
    sel=sessions[1]["seed"]
    fields=dict(ENDIAN=endian,ALLOCATION=original[0],SLOT=original[1],INITIALSEQ=init[0],INITIALPLAYER=init[1],
        INITIALOPPONENT=init[2],INITIALMETADATA=init[3],PLAYER=coord[0],OPPONENT=coord[1],CLUSTERS=coord[2],KEY=coord[3],
        OPENING=sel["start"],FINISH=sel["finish"],FIRSTLOAD=sel["loads"][0],INDICES="_|("*len(stores)+"k"+")"*len(stores),
        SITES="|".join(["rfl"]*len(writes)),PROFILE=by_time(lambda t,l,w: possible(t,l,w,True)),
        VALUECANDIDATES=by_time(lambda t,l,w:possible(t,l,w,False)),EVENTS=li(events),
        LOADS=li([f"⟨{t},{l},{m},{sum(at<t for at,_ in history(l))}⟩" for t,l,m,w in read]),
        WRITES=li([f"({t},⟨{k},{pc}⟩,{v})" for t,k,pc,v in writes]),WRITERS="\n".join(
            f'  | {i} => ⟨address,⟨{s["old"]},by decide⟩,.store (some {s["chosen"]}) b (Packing.unpack endian {s["packed"]}),true,fun pc => {s["start"]+1}+pc⟩'
            for i,s in enumerate(stores)))
    fields.update(PUBLICPROOF=(Path(__file__).parent/"serial_calls/PublicWarmSeed.lean.in").read_text(),GAME=li(GAME),
        SELECTED=sel["metadata"],RECEIVER=sessions[1]["receiver"],BASE=sessions[1]["base"],COLD=sessions[0]["base"],
        PVINPUTS="\n".join(f"  | {t}, {q} => {line}" for t,q,line in arguments),
        SAVES="\n".join(f'  | {i} => save {i} {s["old"]} ({s["score"]}) {s["move"]} {s["level"]} {s["gen"]}' for i,s in enumerate(stores)),
        LASTEVENT=stores[1]["start"]+6,CHOSEN=stores[1]["chosen"],LASTRETURN=next(i for i,r in enumerate(rows) if i>stores[1]["start"] and r[0]=="back"))
    require(publications[1]==[1,4,10240,5,1,5,1,4,4,10240,4], "unexpected second exact public report")
    fields.update(RESULT=publications[1][2],NODES=publications[1][3],PUBSQUARE=publications[1][1],
       PUBLICPV=publications[1][7],PUBDEPTH=publications[1][4],PUBLEVEL=publications[1][5])
    fields.update(sample_fields([(t,pc,v) for t,_,pc,v in writes],sorted([(s["start"]+1,".sequence",s["old"]) for s in stores]+[(t,l,w) for t,l,m,w in read]),True))
    render(folder,fields,"PublicWarmSeed.lean")
    print(f"Two public searches, {len(stores)} internal stores, slot {original[1]}. "
          f"The second store supplied the occupied seed {sessions[1]['base']}. "
          f"Checked 14 scored frames and reproduced E1 {publications[1][2]} in {publications[1][3]} nodes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-public-warm-seed-") as tmp:
        folder = Path(tmp)
        fixture = folder / "warm_seed.rs"
        fixture.write_text(FIXTURE.replace("GAME",str(GAME)))
        binary = capture(folder,args.target_dir.resolve(),instrument,fixture=fixture,serial_trace=False)
        log = subprocess.check_output([binary,REPO],text=True)
        if args.output:
            args.output.write_text(log)
        replay(log.splitlines(),folder)


if __name__ == "__main__":
    main()
