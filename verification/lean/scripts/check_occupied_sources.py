"""Replay an occupied probe after two equal-word publications to the same native TT slot."""

import argparse
import re
from pathlib import Path
import subprocess
import tempfile

from check_reader_window import instrument as instrument_reader
from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


RUST = """
static SOURCE_THREAD: std::sync::OnceLock<std::thread::ThreadId> = std::sync::OnceLock::new();
fn source_worker() -> usize {
    let id = std::thread::current().id();
    assert_eq!(SOURCE_THREAD.get(), Some(&id));
    0
}
pub(crate) fn occupied_sources() {
    SOURCE_THREAD.set(std::thread::current().id()).unwrap();
    let b = Board::from_bitboards(0xffffffdfbfffffff, 0x40000000);
    let key = b.hash();
    let table = TranspositionTable::new(0);
    let start = table.entries.as_ptr() as usize;
    let slot = table.probe(&b,key).slot().0;
    let index = (slot as *const _ as usize - start) / mem::size_of::<TTEntry>();
    println!("initial {} {} {start} {index} {} {} {} {} {} {} {key} {} {}", source_worker(),
        cfg!(target_endian="big") as u8,slot.seq.load(Ordering::Relaxed),slot.player.load(Ordering::Relaxed),
        slot.opponent.load(Ordering::Relaxed),slot.data.load(Ordering::Relaxed),b.player().bits(),b.opponent().bits(),
        table.cluster_count,slot as *const _ as usize);
    for _ in 0..2 {
        table.store(TTSlot(slot),&b,Score::from_raw(16384),Bound::Exact,1,Some(crate::square::Square::F5),
            Selectivity::None,true,true);
    }
    window_read(&table,&b,key,source_worker(),true,false);
}
"""


def instrument(src):
    instrument_reader(src)
    table = src / "transposition_table.rs"
    code = table.read_text()
    a = code.index("    fn seqlock_write(")
    b = code.index("        true\n", a)
    writer = code[a:b]
    # The successful CAS's return register proves that each following store was reached.
    begin = "        if self\n            .seq\n            .compare_exchange("
    writer = replace_once(writer, begin, '''        let addr = self as *const _ as usize;
        println!("call {} {addr} {expected_seq} {player} {opponent} {}",source_worker(),data.to_u64());
        let outcome = self.seq.compare_exchange(''')
    writer = replace_once(writer, '''            )
            .is_err()
        {''', '''            );
        println!("cas {} {addr} {expected_seq} {} {}",source_worker(),expected_seq|1,outcome.unwrap());
        if outcome.is_err() {''')
    require("Ordering::AcqRel,\n                Ordering::Relaxed," in writer, "write CAS orders changed")
    sites = [("fence(Ordering::Release);", "releaseFence", "0"),
             ("self.player.store(player, Ordering::Relaxed);", "player", "player"),
             ("self.opponent.store(opponent, Ordering::Relaxed);", "opponent", "opponent"),
             ("self.data.store(data.to_u64(), Ordering::Relaxed);", "metadata", "data.to_u64()"),
             (".store(expected_seq.wrapping_add(2), Ordering::Release);", "publish", "expected_seq.wrapping_add(2)")]
    for anchor, tag, word in sites:
        writer = replace_once(writer, anchor, anchor + f'\n        println!("{tag} {{}} {{addr}} {{}}",source_worker(),{word});')
    code = code[:a] + writer + code[b:]
    code = replace_once(code, 'println!("op {worker} {opcode} {index} {word}");',
                        'assert_eq!(worker,source_worker()); println!("op {worker} {opcode} {index} {word}");')
    table.write_text(code + RUST)


def replay(log, folder):
    rows = [row.split() for row in log.splitlines()]
    tag, worker, endian, alloc, idx, seq, initial_p, initial_o, initial_meta, p, o, key, clusters, ptr = rows[0]
    require(tag == "initial" and worker == "0", "missing physical slot image")
    require(int(ptr) == int(alloc) + 32 * int(idx), "unexpected physical slot layout")
    events = [".engine (.other 0)"]
    fields = dict(ALLOCATION=alloc, SLOT=idx, ENDIAN=str(bool(int(endian))).lower(), INITIALSEQ=seq,
                  INITIALPLAYER=initial_p, INITIALOPPONENT=initial_o, INITIALMETADATA=initial_meta,
                  PLAYER=p, OPPONENT=o, KEY=key, CLUSTERS=clusters)
    pcs = {"cas": ".cas", "releaseFence": ".fence", "player": ".field .player",
           "opponent": ".field .opponent", "metadata": ".field .metadata", "publish": ".finish"}
    loads = {"A": (".sequence", ".acquire", 4), "D": ("(.payload .metadata)", ".relaxed", 2),
             "P": ("(.payload .player)", ".relaxed", 2), "O": ("(.payload .opponent)", ".relaxed", 2),
             "Q": (".sequence", ".relaxed", 4)}
    writers, writes, selected = [], [], []
    numeric_writes, numeric_reads = [], []
    order = ["call", *pcs, "call", *pcs, "enter", *(["op"] * 6), "leave"]
    require([r[0] for r in rows[1:]] == order, "incomplete publication or occupied snapshot")
    require([r[2] for r in rows if r[0] == "op"] == ["A", "D", "P", "O", "F", "Q"], "unexpected physical read")
    for t, (kind, who, *args) in enumerate(rows[1:], 1):
        require(who == worker, "worker changed during the synchronous native run")
        if kind == "call":
            addr, old, player, opponent, packed = args
            require((addr, player, opponent) == (ptr, p, o), "writer arguments changed")
            k = len(writers)
            writers.append(f"  | {k} => ⟨address,⟨{int(old)},by decide⟩,.store none b "
                           f"(Packing.unpack endian {int(packed)}),true,fun pc => {t+1} + pc⟩")
            events.append(f".engine (.beginWrite 0 {k} (operand {k}))")
        elif kind in pcs:
            addr, *words = args
            require(addr == ptr, "write reached another allocation")
            if kind == "cas":
                old, new, returned = map(int, words)
                require(returned == old and new == old | 1, "unexpected CAS operands or outcome")
                numeric_reads.append((t,".sequence",old))
                value = new
            else:
                value = int(words[0])
            numeric_writes.append((t,pcs[kind],value))
            writes.append(f"({t},⟨{k},{pcs[kind]}⟩,{value})")
            events.append(f".engine (.atomic 0 {k})")
        elif kind == "enter":
            require(args == ["probe", p, o, clusters, key], "reader board/cluster key differs")
            fields["OPENING"] = t
            events.append(".enter 0 args")
        elif kind == "op":
            code, slot, word = args
            if code == "F":
                op = ".fence .acquire"
            else:
                require(slot == idx, "reader addressed a different slot")
                loc, mode, index = loads[code]
                numeric_reads.append((t,loc,int(word)))
                selected.append(f"⟨{t},{loc},{mode},{index}⟩")
                op = f".load address {loc} {mode}"
            events.append(f".issue 0 ({op}) (label {t}) {int(word)}")
        elif kind == "leave":
            method, slot, data = args
            require(method == "probe" and int(slot) == int(idx) % 2 and int(data) != 0, "native probe missed")
            fields["FINISH"] = t
            events.append(f".leave 0 (.probe (.hit {slot} (Packing.unpack endian {data})))")
    def ls(items):
        return "[\n  " + ",\n  ".join(items) + "]"
    fields.update(PROFILE="[q.source]", FIRSTLOAD=fields["OPENING"]+1,VALUECANDIDATES="[1,2]",PUBLICPROOF="",
                  INDICES="_|(" * len(writers) + "k" + ")" * len(writers), SITES="|".join("rfl" for _ in writes),
                  WRITERS="\n".join(writers), WRITES=ls(writes), LOADS=ls(selected), EVENTS=ls(events))
    fields.update(sample_fields(numeric_writes,numeric_reads,False))
    render(folder, fields, "OccupiedSources.lean")
    print(f"Occupied probe at allocation {alloc}, slot {idx}: two completed publications, five physical loads. "
          "Both seq loads identified modification 4; player, opponent and metadata identified modification 2. "
          "Removing the worker ordering retained both equal-word publications for each payload field.")


def sample_fields(writes, reads, front):
    locs = [".sequence", "(.payload .player)", "(.payload .opponent)", "(.payload .metadata)"]
    def loc_of(pc):
        if pc in (".cas", ".finish"):
            return ".sequence"
        if pc.startswith(".field "):
            return "(.payload " + pc[7:] + ")"
        return None
    histories = [(l, [(t,v) for t,pc,v in writes if loc_of(pc) == l]) for l in locs]
    return dict(FRONT=str(front).lower(),RANKS=1+max(len(h) for _,h in histories),
        HISTORIES="\n".join(f"    | {l} => ["+",".join(f"({t},{v})" for t,v in h)+"]" for l,h in histories),
        SAMPLELOADS="["+",".join(f"⟨{t},{l},{v}⟩" for t,l,v in reads)+"]")


def render(folder, fields, name):
    template = (Path(__file__).parent / "serial_calls/OccupiedSources.lean.in").read_text()
    output = folder / name
    output.write_text(re.sub(r'\b[A-Z]+\b', lambda m: str(fields.get(m[0],m[0])), template.replace("PUBLICPROOF",fields["PUBLICPROOF"])))
    subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", str(output)], cwd=LEAN, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-occupied-sources-") as tmp:
        folder = Path(tmp)
        fixture = folder / "occupied.rs"
        fixture.write_text('pub fn run(_: &std::path::Path) { crate::transposition_table::occupied_sources(); }\n')
        binary = capture(folder, args.target_dir.resolve(), instrument, fixture=fixture, serial_trace=False)
        log = subprocess.check_output([binary, REPO], text=True)
        if args.output:
            args.output.write_text(log)
        replay(log, folder)


if __name__ == "__main__":
    main()
