"""Replay an ETC cutoff from the four-worker public hard-20 #166 Multi-PV search."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import FIXTURES, LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


CLOCK = '''
static ETC_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(crate) static ETC_CAPTURED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! { pub(crate) static ETC_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) fn etc_tick() -> u64 { ETC_CLOCK.fetch_add(1, Ordering::Relaxed) }
pub(crate) fn etc_data(data: Option<TTEntryData>) -> String {
    data.map_or_else(|| "miss".to_owned(), |e| format!("{} {} {} {} {} {} {}",
        e.score().value(), e.best_move().map_or(255, |sq| sq.index()), e.bound() as u8, e.depth(),
        e.selectivity().as_u8(), e.generation(), u8::from(e.is_endgame()) | (u8::from(e.is_pv()) << 1)))
}
'''


def instrument(src):
    table = src / 'transposition_table.rs'
    text = table.read_text() + CLOCK
    text = replace_once(text, '    /// Returns the first entry index of the cluster selected by `key`.', '''
    pub(crate) fn etc_allocation(&self) -> usize { self.entries.as_ptr() as usize }
    pub(crate) fn etc_clusters(&self) -> u64 { self.cluster_count }
    pub(crate) fn etc_slot(&self, key: u64, slot: TTSlot<'_>) -> usize {
        let first = self.get_cluster_idx(key);
        (0..2).find(|i| std::ptr::eq(&self.entries[first + i],slot.0)).unwrap()
    }
    /// Returns the first entry index of the cluster selected by `key`.''')
    text = replace_once(text, '        if !score.is_finite() {', '''        if ETC_ACTIVE.get() {
            println!("ETC store {} {} {} {} {} {} {} {} {} {} {}", etc_tick(), board.player().bits(), board.opponent().bits(), self.etc_allocation(),
                self.etc_slot(board.hash(),slot),score.value(),bound as u8,depth,best_move.unwrap().index(),selectivity.as_u8(),u8::from(is_endgame));
        }
        if !score.is_finite() {''')
    text = replace_once(text, '        slot.0.save(board, data);', '''        slot.0.save(board, data);
        if ETC_ACTIVE.get() { println!("ETC stored {}", etc_tick()); }''')
    table.write_text(text)
    pvs = src / 'search/pvs.rs'
    text = pvs.read_text()
    start = text.index('pub fn search<')
    end = text.index('    let all_node', start)
    header = text[start:end]
    wrapper = header.replace('mut depth', 'depth').replace('mut alpha', 'alpha') + '''
    let watching = SS::IS_ENDGAME && !NT::PV_NODE && depth >= 14 && ctx.selectivity == Selectivity::None && !ETC_CAPTURED.load(Relaxed);
    let opened = if watching { etc_tick() } else { 0 };
    let result = etc_search::<NT, SS>(ctx, board, depth, alpha, beta, thread, cut_node, opened);
    if ETC_ACTIVE.replace(false) {
        println!("ETC return {} {} {} {} {}", etc_tick(),thread.idx(),board.player().bits(),board.opponent().bits(),result.value());
    }
    result
}
'''
    inner = header.replace('pub fn search<', 'fn etc_search<').replace(') -> Score {', '    opened: u64,\n) -> Score {')
    text = text[:start] + wrapper + inner + text[end:]
    text = replace_once(text, 'use std::sync::Arc;', '''use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use crate::transposition_table::{etc_tick,etc_data,ETC_CAPTURED,ETC_ACTIVE,TTEntryData};''')
    anchor = '    let tt_probe_result = ctx.tt.probe(board, tt_key);'
    text = replace_once(text, anchor, '    let begin = if opened != 0 {etc_tick()} else {0};\n'+anchor+'\n    let end = if opened != 0 {etc_tick()} else {0};')
    anchor = '                tt_probe_result.slot(),\n                tt_pv,\n            ) {'
    text = replace_once(text, anchor, anchor.replace('            )', '''                (opened,thread.idx(),begin,end,beta.value(),cut_node,tt_probe_result.data(),tt_key),
            )'''))
    anchor = '    let etc_depth = depth - 1;'
    text = replace_once(text, '    tt_pv: bool,\n) -> Option<Score> {\n'+anchor, '''    tt_pv: bool,
    audit: (u64,usize,u64,u64,i32,bool,Option<TTEntryData>,u64),
) -> Option<Score> {
    let (opened,worker,begin,end,beta,cut,parent_data,parent_key) = audit;
    let tracking = opened != 0 && !ETC_CAPTURED.load(Relaxed);
    let mut reads = Vec::new();
'''+anchor)
    anchor = '        if let Some(etc_tt_data) = ctx.tt.lookup(&next, etc_tt_key)'
    text = replace_once(text, anchor, '''        let opening = if tracking {etc_tick()} else {0};
        let observed = ctx.tt.lookup(&next, etc_tt_key);
        if tracking { reads.push((opening,etc_tick(),mv.sq,next,ctx.tt.etc_allocation(),ctx.tt.etc_clusters(),etc_tt_key,observed)); }
        if let Some(etc_tt_data) = observed''')
    anchor = '                ctx.tt.store(\n                    tt_slot,'
    text = replace_once(text, anchor, '''                if tracking && reads.len() >= 2 && !ETC_CAPTURED.swap(true,Relaxed) {
                    let chosen = etc_tick();
                    ETC_ACTIVE.set(true);
                    println!("ETC entry {} {} {} {} {} {} {} {} {} {}",u8::from(cfg!(target_endian = "big")),opened,worker,board.player().bits(),board.opponent().bits(),depth,alpha.value(),beta,cut as u8,ctx.selectivity.as_u8());
                    println!("ETC probe {} {} {} {} {} {} {}",begin,end,ctx.tt.etc_allocation(),ctx.tt.etc_clusters(),parent_key,ctx.tt.etc_slot(parent_key,tt_slot),etc_data(parent_data));
                    for (a,z,sq,next,table,clusters,key,data) in &reads {
                        println!("ETC lookup {} {} {} {} {} {} {} {} {}",a,z,sq.index(),next.player().bits(),next.opponent().bits(),table,clusters,key,etc_data(*data));
                    }
                    println!("ETC chosen {} {} {}",chosen,mv.sq.index(),score.value());
                }
'''+anchor)
    pvs.write_text(text)


TEMPLATE = '''import NeuralReversi.PvsParallel.Interleaving.JournalScout

namespace NeuralReversi.PvsParallel.Interleaving.NativeEtcReplay
open PvsBoardTree (Position legalSquares)
open TranspositionTable hiding Score
open ScoreRefresh (Score)

structure Lookup where
  opening : Nat
  finish : Nat
  square : Nat
  board : Position
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  data : Option EntryData

structure Store where
  atTime : Nat
  board : Position
  allocation : Nat
  slot : Nat
  score : Int
  bound : Nat
  depth : Depth
  square : Nat
  level : Selectivity
  endgame : Bool

structure Trace where
  endian : Bool
  opened : Nat
  worker : Nat
  board : Position
  depth : Depth
  alpha : Int
  beta : Int
  cut : Bool
  level : Selectivity
  probeOpen : Nat
  probeClose : Nat
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  slot : Fin 2
  parent : Option EntryData
  lookups : List Lookup
  choice : Nat
  square : Nat
  chosen : Int
  saved : Store
  stored : Nat
  returned : Nat
  returnWorker : Nat
  returnBoard : Position
  returnWord : Int

private def word (value : Int) : Score := ⟨BitVec.ofInt 32 value⟩
private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

private def Trace.rawProbe (t : Trace) : ProbeRetry.Result := match t.parent with
  | none => .miss t.slot | some e => .hit t.slot e

private def Trace.tape (t : Trace) : Joint.Tape Store := Id.run do
  let w := t.worker
  let args : NativeFrames.Args := ⟨t.board.val,t.depth,word t.alpha,word t.beta,true,t.level,false,t.cut,none⟩
  let lookups := t.lookups.flatMap fun r => [(r.opening,Joint.Event.enter w ⟨.lookup r.board,r.allocation,r.clusters,r.key⟩),(r.finish,.leave w (.lookup r.data))]
  let events : List (Nat × Joint.Event Store) := [(t.opened,.engine (.localCall w (.entry t.opened 0 .pvs .none none) (some args))),
    (t.probeOpen,.enter w ⟨.probe (Cutoffs.key t.board),t.allocation,t.clusters,t.key⟩),(t.probeClose,.leave w (.probe t.rawProbe))] ++ lookups ++
    [(t.choice,.engine (.localCall w (.scored t.choice (word t.chosen)) none)),(t.saved.atTime,.engine (.beginWrite w 0 t.saved)),
      (t.stored,.engine (.other w)),(t.returned,.engine (.localCall t.returnWorker (.leave t.returned t.opened .pvs none (some (word t.returnWord))) none))]
  return (List.range (t.returned-t.opened+1)).map fun i => (events.lookup (t.opened+i)).getD (.engine (.other (w+1)))

private def Trace.writer (t : Trace) : Option (CallJournal.Writer Store) := do
  let prefixEvents := t.tape.take (t.saved.atTime-t.opened+1)
  let st ← CallJournal.run mc fc calls false (prefixEvents.map Joint.Event.engineEvent)
  st.writing t.worker

private def accepts (t : Trace) : Bool := Option.isSome (do
  let es := t.tape
  let choice := t.choice-t.opened
  let done_ := t.returned-t.opened
  let (begin,close,parent) ← JournalPrefix.parentAt es t.worker 0 choice t.board
  guard (begin+t.opened = t.probeOpen ∧ close+t.opened = t.probeClose ∧ parent = t.rawProbe ∧
    (JournalScout.parentCuts false t.depth t.level (word t.beta) parent).toOption = some false)
  guard (t.key == ReaderWindow.engineKey t.endian (.probe (Cutoffs.key t.board)))
  guard (t.lookups.all (fun r =>
    (ReaderWindow.Args.mk (.lookup r.board) r.allocation r.clusters r.key).inTable t.endian t.allocation t.clusters))
  let (windows,some hit) ← JournalScout.scan mc fc t.board ⟨t.depth,t.level,true,⟨(word t.alpha).raw⟩⟩
    (legalSquares mc t.board.val) (JournalScout.events es t.worker close choice) | none
  guard (windows.length = t.lookups.length ∧ windows.length ≥ 2 ∧ hit.1.val = t.square ∧ hit.2 = t.chosen)
  guard ((windows.zip t.lookups).all (fun (w,r) => w.opening+t.opened == r.opening && w.finish+t.opened == r.finish && w.move.val.val == r.square && w.word == r.data))
  let writer ← t.writer
  let owner ← writer.owner
  let raw := writer.raw
  guard (owner.stamp = choice ∧ owner.word.raw.toInt = hit.2 ∧ owner.board = raw.board ∧ owner.depth = raw.depth ∧
    owner.level = raw.level ∧ owner.endgame = raw.endgame ∧ raw.endgame ∧ raw.bound = 1 ∧ raw.slot = t.slot.val ∧
    raw.allocation = t.allocation ∧ raw.square = hit.1.val ∧ raw.score = owner.word.raw.toInt)
  let frames ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let [r] := frames.allRows | none
  guard (r.worker = t.returnWorker ∧ r.worker = t.worker ∧ r.chosen = owner.stamp ∧ r.score = t.returnWord ∧
    r.entry.request.board = t.returnBoard ∧ t.returnBoard = owner.board ∧ r.returned = some done_ ∧
    t.saved.atTime < t.stored ∧ t.stored < t.returned ∧ owner.level = Cutoffs.exactSelectivity ∧
    owner.depth = Board.emptyCount t.board.val ∧ owner.depth.toNat ≥ 14 ∧ owner.beta.raw.toInt = owner.alpha.raw.toInt+1)
  pure ())

CAPTURE

private def swapped (t : Trace) : Trace := match t.lookups with
  | a::b::xs => {t with lookups := {b with opening := a.opening,finish := a.finish}::{a with opening := b.opening,finish := b.finish}::xs}
  | _ => t

private def premature := {capture with choice := capture.saved.atTime,saved := {capture.saved with atTime := capture.choice}}

#guard ((capture.writer.bind (·.owner)).map (·.stamp)) = some (capture.choice-capture.opened)
#guard premature.writer.isSome && (premature.writer.bind (·.owner)).isNone
#guard accepts capture
#guard !(accepts (swapped capture))
#guard !(accepts {capture with chosen := capture.chosen+1, saved := {capture.saved with score := capture.chosen+1},returnWord := capture.chosen+1})
#guard !(accepts {capture with key := capture.key+1})
#guard !(accepts {capture with lookups := capture.lookups.modifyHead (fun r => {r with allocation := r.allocation+1})})
#guard !(accepts {capture with lookups := capture.lookups.modifyHead (fun r => {r with key := r.key+1})})
#guard !(accepts premature)
#guard !(accepts {capture with saved := {capture.saved with slot := 1-capture.saved.slot}})
#guard !(accepts {capture with returnWord := capture.returnWord+1})

end NeuralReversi.PvsParallel.Interleaving.NativeEtcReplay
'''


def position(player, opponent):
    return f'⟨⟨{int(player)},{int(opponent)}⟩,by unfold Board.wellFormed; decide⟩'


def data(fields):
    if fields == ['miss']:
        return 'none'
    require(len(fields) == 7, 'malformed native TT entry')
    value, *rest = map(int, fields)
    return '(some ⟨BitVec.ofInt 16 (' + str(value) + '),' + ','.join(map(str, rest)) + '⟩)'


def replay(lines, directory):
    rows = [line.split()[1:] for line in lines]
    kinds = [row[0] for row in rows]
    reads = [row[1:] for row in rows if row[0] == 'lookup']
    require(kinds == ['entry', 'probe', *(['lookup'] * len(reads)), 'chosen', 'store', 'stored', 'return'], 'incomplete native ETC trace')
    endian, opened, worker, p, o, depth, lo, hi, cut, level = map(int, rows[0][1:])
    begin, end, allocation, clusters, key, slot = map(int, rows[1][1:7])
    chosen, square, value = map(int, rows[-4][1:])
    stored_at, sp, so, sa, ss, sv, sb, sd, sm, sl, phase = map(int, rows[-3][1:])
    stored = int(rows[-2][1])
    returned, rw, rp, ro, rv = map(int, rows[-1][1:])
    stamps = [opened,begin,end,*[int(stamp) for row in reads for stamp in row[:2]],chosen,stored_at,stored,returned]
    require(all(x < y for x, y in zip(stamps,stamps[1:])), 'native instrumentation did not report increasing event stamps')
    bit = lambda x: 'true' if x else 'false'
    items = [f'⟨{int(a)},{int(b)},{int(sq)},{position(cp,co)},{int(alloc)},{int(count)},{int(k)},{data(words)}⟩' for a,b,sq,cp,co,alloc,count,k,*words in reads]
    fixture = f'''private def capture : Trace := ⟨{bit(endian)},{opened},{worker},{position(p,o)},{depth},({lo}),({hi}),{bit(cut)},{level},
      {begin},{end},{allocation},{clusters},{key},{slot},{data(rows[1][7:])},[{','.join(items)}],{chosen},{square},({value}),
      ⟨{stored_at},{position(sp,so)},{sa},{ss},({sv}),{sb},{sd},{sm},{sl},{bit(phase)}⟩,{stored},{returned},{rw},{position(rp,ro)},({rv})⟩'''
    source = directory / 'NativeEtcReplay.lean'
    source.write_text(TEMPLATE.replace('CAPTURE', fixture))
    subprocess.run(['lake', 'env', 'lean', '-DwarningAsError=true', str(source)], cwd=LEAN, check=True)
    print(f'hard-20 #166, worker {worker}, {depth} empties, {len(reads)} child lookups, chosen {value} at {chosen}; store {stored_at}–{stored}, frame returned at {returned}.')
    print('Lean rejected a wrong parent or child hash, a foreign table, a swapped lookup, a consistent wrong score, a premature store, a different slot, and a changed normal return.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='reversi-native-etc-') as tmp:
        directory = Path(tmp)
        binary = capture(directory, args.target_dir.resolve() if args.target_dir else directory / 'target', instrument, fixture=FIXTURES / 'parallel_tickets.rs', serial_trace=False)
        native = subprocess.run([binary, REPO], text=True, capture_output=True, check=True)
        lines = [line for line in native.stdout.splitlines() if line.startswith('ETC ')]
        require(bool(lines), 'no multi-lookup exact ETC cutoff in hard-20 #166')
        if args.output:
            args.output.write_text('\n'.join(lines) + '\n')
        replay(lines, directory)


if __name__ == '__main__':
    main()
