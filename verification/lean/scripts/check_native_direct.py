"""Run a four-worker public one-empty search and replay its cold TT seed and every root trial."""

import argparse
import re
from pathlib import Path
import subprocess
import tempfile

from check_reader_window import instrument as instrument_reads
from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


RUST = """
use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use std::path::Path;
pub fn run(root: &Path) {
    let resources = crate::engine::Engine::from_eval(Arc::new(Eval::with_weight_files(
        Some(&root.join(EVAL_FILE_NAME)), Some(&root.join(EVAL_SM_FILE_NAME))).unwrap()), 8, Some(4));
    let search = resources.open();
    let board = Board::from_bitboards(0xffffffdfbfffffff,0x40000000);
    let result = resources.search(search, &board, &SearchRunOptions::with_level(Level::perfect()).multi_pv(true));
    let SearchResult::BestMove {sq, score, n_nodes, pv_line, depth, selectivity, is_endgame, pv_moves, ..} = result else { panic!("no best move") };
    assert_eq!(pv_line.len(),1);
    assert_eq!(pv_moves.len(),1);
    assert_eq!(pv_moves[0].pv_line.len(),1);
    println!("public {} {} {n_nodes} {depth} {} {} {} {} {} {} {}", sq.index(),score.value(),selectivity.as_u8(),
        is_endgame as u8,pv_line[0].index(),pv_moves.len(),pv_moves[0].sq.index(),pv_moves[0].score.value(),pv_moves[0].pv_line[0].index());
}
"""


def instrument(src):
    instrument_reads(src)
    table = src / "transposition_table.rs"
    text = table.read_text() + '''
impl TranspositionTable {
    pub(crate) fn direct_probe(&self, board: &Board, key: u64) -> TTProbeResult<'_> {
        let worker = DIRECT_WORKER.get().unwrap();
        let base = self.entries.as_ptr() as usize;
        println!("probe {worker} {} {} {base} {} {key}", board.player().bits(),board.opponent().bits(), self.cluster_count);
        WINDOW_TRACE.set(Some((worker,base)));
        let r = self.probe(board, key);
        WINDOW_TRACE.set(None);
        let slot = ((r.slot().0 as *const TTEntry as usize - base) / mem::size_of::<TTEntry>()) % CLUSTER_SIZE;
        println!("finish {worker} {slot} {}", r.data().map_or(0,TTEntryData::to_u64));
        r
    }
}
thread_local! { pub(crate) static DIRECT_WORKER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
'''
    text = replace_once(text, "        if !score.is_finite() {", '        println!("unexpected_store");\n        if !score.is_finite() {')
    table.write_text(text)
    root = src / "search.rs"
    text = root.read_text()
    anchor = "fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {"
    text = replace_once(text, anchor, anchor + '''
    crate::transposition_table::DIRECT_WORKER.set(Some(thread.idx()));
    println!("root {}",thread.idx());''')
    anchor = "RootPhase::Endgame { last } => endgame::search_root(task, last, thread),"
    root.write_text(replace_once(text, anchor, """RootPhase::Endgame { last } => {
            let result = endgame::search_root(task,last,thread);
            println!("endrootback {}",thread.idx());
            result
        },"""))
    thread = src / "search/threading/thread.rs"
    anchor = "                    let result = search::search_root(task, &self);"
    thread.write_text(replace_once(thread.read_text(), anchor, anchor + '\n                    println!("rootback {}",self.idx);'))
    context = src / "search/context.rs"
    anchor = "        self.root_moves.update(sq, score, is_pv, self.pv());"
    context.write_text(replace_once(context.read_text(),anchor, """
        let pv = self.pv().iter().map_while(|q| *q).map(|q| q.index().to_string()).collect::<Vec<_>>().join(",");
        println!("pv {} {} {}",sq.index(),is_pv as u8,pv);
""" + anchor))
    end = src / "search/endgame.rs"
    text = end.read_text()
    text = replace_once(text, "    let board = task.board;", '''    println!("endroot {}",thread.idx());
    let board = task.board;''')
    text = replace_once(text, "    if let Some(tt_data) = ctx.tt.probe(board, board.hash()).data()", '''    let probed = ctx.tt.direct_probe(board,board.hash());
    println!("seed {} {} {}", board.player().bits(),board.opponent().bits(),probed.data().is_some() as u8);
    if let Some(tt_data) = probed.data()''')
    text = replace_once(text, "    midgame::evaluate(ctx, board)\n}", '''    let base = midgame::evaluate(ctx, board);
    println!("eval {}",base.value());
    base
}''')
    text = replace_once(text, "    solve_root(\n", "    let result = solve_root(\n")
    text = replace_once(text, "        Selectivity::Level1,\n        last,\n    )\n}", """        Selectivity::Level1,
        last,
    );
    println!("solveback {}",thread.idx());
    result
}""")
    anchor = "    let time_manager = task.time_manager.as_deref();"
    text = replace_once(text, anchor, '''    println!("solve {} {} {} {} {} {} {}",thread.idx(),base_score.value(),task.multi_pv as u8,
        min_reported.as_u8(),last.as_u8(),completed.last.is_some() as u8,direct_row(ctx));
''' + anchor)
    text = replace_once(text, "    let mut delta = ASPIRATION_DELTA;", '''    println!("window {}",thread.idx());
    let mut delta = ASPIRATION_DELTA;''')
    anchor = "        if thread.is_search_aborted() {\n            return score;"
    text = replace_once(text, anchor, '''        println!("back {} {}",thread.idx(),score.value());
''' + anchor)
    anchor = "            let score = aspiration_search(ctx, board, &mut alpha, &mut beta, thread);"
    text = replace_once(text, anchor, anchor + '\n            println!("endwindow {}",thread.idx());')
    text += '''
pub(super) fn direct_row(ctx: &SearchContext) -> String {
    let rows = ctx.root_moves.snapshot();
    assert_eq!(rows.len(),1);
    let row = &rows[0];
    let pv = row.pv.iter().map(|q| q.index().to_string()).collect::<Vec<_>>().join(",");
    format!("{} {} {} {} {}",row.sq.index(),row.score.value(),row.average_score.value(),
        row.mean_squared_score,if pv.is_empty() { "-".to_string() } else { pv })
}
'''
    end.write_text(text)
    pvs = src / "search/pvs.rs"
    text = pvs.read_text()
    anchor = "    let org_alpha = alpha;"
    text = replace_once(text, anchor, anchor + '''
    assert!(NT::ROOT_NODE && SS::IS_ENDGAME);
    println!("trial {} {} {} {} {} {} {} {} {} {}",thread.idx(),board.player().bits(),board.opponent().bits(),depth,
        ctx.selectivity.as_u8(),ctx.root_moves.pv_idx(),alpha.value(),beta.value(),cut_node as u8,endgame::direct_row(ctx));''')
    anchor = "        return Score::MAX;"
    text = replace_once(text, anchor, '''        println!("choice {} {} {}",thread.idx(),sq.index(),Score::MAX.value());
''' + anchor)
    pvs.write_text(text)


def replay(log, directory):
    events, stack, trials = [], [], 0
    pv_inputs = []
    score_sites = []
    public = None
    endgame_board = (0xffffffdfbfffffff, 0x40000000)
    seed = None

    def root_row(fields):
        sq, score, avg, mean, pv = fields
        return f'⟨{int(sq)},({int(score)}),({int(avg)}),({int(mean)}),{squares(pv)}⟩'

    def squares(pv):
        return '[' + ('' if pv == '-' else ','.join(str(int(n)) for n in pv.split(','))) + ']'

    def check_worker(fields):
        require(int(fields[0]) == 0, 'unexpected actor on direct root route')

    def push(kind):
        stack.append((kind, len(events)))

    def pop(kind):
        node, opening = stack.pop()
        require(kind == node, 'mismatched native return')
        return opening

    for line in log.splitlines():
        kind, *v = line.split()
        t = len(events)
        if kind in ('root','endroot','window'):
            check_worker(v)
            node = {'root':'root','endroot':'endRoot','window':'endWindow'}[kind]
            push(node)
            events.append(f'enter {t} .{node}')
        elif kind == 'probe':
            check_worker(v)
            p, o, allocation, clusters, key = map(int, v[1:])
            require((p,o) == endgame_board, 'root probe board mismatch')
            events.append(f'.enter 0 ⟨.probe (Cutoffs.key board),{allocation},{clusters},{key}⟩')
        elif kind == 'op':
            check_worker(v)
            name, slot, value = v[1:]
            addr = f'⟨{allocation},{int(slot)}⟩'
            op = {'A':f'.load {addr} .sequence .acquire','D':f'.load {addr} (.payload .metadata) .relaxed',
                  'F':'.fence .acquire','Q':f'.load {addr} .sequence .relaxed'}[name]
            events.append(f'.issue 0 ({op}) 99 {int(value)}')  # Unobserved RF placeholder, overwritten by Lean.
        elif kind == 'finish':
            check_worker(v)
            slot, packed = map(int,v[1:])
            require(packed == 0, 'expected cold TT miss')
            events.append(f'.leave 0 (.probe (.miss {slot}))')
        elif kind == 'seed':
            p, o, hit = map(int,v)
            require((p,o) == endgame_board and not hit, 'seed register differs from actual probe')
            receiver = t
            events.append(f'.seedRegister 0 board (.miss {slot})')
        elif kind == 'eval':
            seed = int(v[0])
        elif kind == 'solve':
            check_worker(v)
            base, multi, minimum, last, saved = map(int,v[1:6])
            require((multi,minimum,last,saved) == (1,1,5,0) and base == seed,'unexpected native solve header')
            push('solve')
            events.append(f'.solveEnter 0 {t} 0 .none ⟨board,BitVec.ofInt 32 ({base}),[{root_row(v[6:])}],none,true,{minimum},{last}⟩')
        elif kind == 'trial':
            check_worker(v)
            p, o, depth, level, idx, alpha, beta, cut = map(int,v[1:9])
            require((p,o) == endgame_board and not cut and idx == 0 and stack[-1][0] == 'endWindow','bad root call')
            push('pvs')
            events.append(f'trial {t} {depth} {level} {idx} ({alpha}) ({beta}) false [{root_row(v[9:])}]')
            trials += 1
        elif kind == 'pv':
            sq, is_pv, line = v
            require(stack[-1][0] == 'pvs' and int(is_pv) == 1, 'root PV update outside an active trial')
            pv_inputs.append((stack[-1][1],int(sq),squares(line)))
        elif kind in ('solveback','endrootback','rootback'):
            check_worker(v)
            node = {'solveback':'solve','endrootback':'endRoot','rootback':'root'}[kind]
            events.append(f'exit {t} {pop(node)} .{node} none')
        elif kind == 'choice':
            check_worker(v)
            sq, score = map(int,v[1:])
            score_sites.append((t,score))
            require(sq == 37, 'unexpected wipeout square')
            events.append(f'.engine (.localCall 0 (.scored {t} (word ({score}))) none)')
        elif kind == 'back':
            check_worker(v)
            value = int(v[1])
            score_sites.append((t,value))
            events.append(f'exit {t} {pop("pvs")} .pvs (some (word ({value})))')
        elif kind == 'endwindow':
            check_worker(v)
            events.append(f'exit {t} {pop("endWindow")} .endWindow none')
        elif kind == 'public':
            public = list(map(int, v))
        else:
            raise AssertionError(f'unexpected native event {kind}')
    require(public is not None and not stack and len(pv_inputs) == trials, 'incomplete direct run')
    sq, score, nodes, depth, level, endgame, pv, count, alt, alt_score, alt_pv = public
    require(endgame and count == 1 and trials == 5 and level == 5, 'missing public row or level')
    fields = dict(INDICES='_|('*(len(events))+'i'+')'*len(events),
                  NODES=nodes,DEPTH=depth,LEVEL=level,TRIALS=trials,RECEIVER=receiver,BASE=seed,
                  PVINPUTS='\n'.join(f'  | {opened}, {q} => {line}' for opened,q,line in pv_inputs),
                  BEST=f'⟨{sq},({score}),[{pv}]⟩',ALTERNATIVE=f'⟨{alt},({alt_score}),[{alt_pv}]⟩')
    template = (Path(__file__).parent / 'serial_calls/NativeDirect.lean.in').read_text()
    def verify(name, changed):
        fields["BAD"] = str(changed).lower()
        fields["EVENTS"] = "[\n  " + ",\n  ".join(events) + "]"
        source = directory / name
        conclusion = "" if changed else """
example : report.best.score = 256 * EndgameTreeCache.discValue mc fc board.val ∧
    MultiRows.ReportExact mc fc board report := by
  have ranked (last : Nat) : MultiPublic.unlimited.ordered last := by
    constructor
    · intro i _ step
      exact Nat.le_refl _
    · intro step
      simp [MultiPublic.unlimited]
  have proved := PhysicalCalls.from_checked_frames mc fc raw board RECEIVER (fun _ => (BASE))
    (fun ok => feed (raw.execution ok)) true MultiPublic.unlimited 5 false (fun _ => none)
    sorting NODES (fun _ => .noLegalMove) report TRIALS (by decide) (ranked _) replayed rfl rfl
  exact ⟨proved.1,proved.2 rfl⟩
"""
        program = template.replace("CONSEQUENCE", conclusion)
        source.write_text(re.sub(r'\b[A-Z]+\b', lambda m: str(fields.get(m[0], m[0])), program))
        subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", str(source)], cwd=LEAN, check=True)

    verify("NativeDirect.lean", False)
    require(len(score_sites) == 2 * trials, "missing score/return receipt")
    for i, original in score_sites[-2:]:
        events[i] = replace_once(events[i], f"word ({original})", f"word ({original-256})")
    verify("NativeDirectWrongScore.lean", True)
    print(f'Public direct replay matched {trials} root attempts, cold seed {seed}, and move {sq} score {score}. An altered exact choice and return were rejected.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-native-direct-") as tmp:
        folder=Path(tmp)
        rs=folder / "direct.rs"
        rs.write_text(RUST)
        binary=capture(folder,args.target_dir.resolve(),instrument,fixture=rs,serial_trace=False)
        run=subprocess.run([binary,REPO],text=True,capture_output=True,check=True)
        if args.output:
            args.output.write_text(run.stdout)
        replay(run.stdout,folder)


if __name__=="__main__":
    main()
