"""Replay writes and a cross-search TT hit from two normal four-worker exact Multi-PV requests.

Requires the native evaluation weights, hard-20.obf, and a built Interleaving.JournalResident.
Only the temporary engine is instrumented. All four whole-table snapshots are taken outside
Search::run. Instrumentation affects scheduling; callback timestamps do not establish happens-before.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require

FIXTURE = Path(__file__).parent / "serial_calls" / "warm_tt.rs"
ZERO = (0, 0, 0, 0)


def instrument(src):
    path = src / "transposition_table.rs"
    source = path.read_text()
    claim = "        // We now hold exclusive write access (sequence is odd)."
    source = replace_once(source, claim,
        '        println!("claim {} {} {:?}", std::ptr::from_ref(self).addr(), expected_seq, std::thread::current().id());\n' + claim)
    finish = "            .store(expected_seq.wrapping_add(2), Ordering::Release);"
    source = replace_once(source, finish, finish +
        '\n        println!("finish {} {} {} {} {}", std::ptr::from_ref(self).addr(), expected_seq, player, opponent, data.to_u64());')
    for i in range(2):
        ret = f"return TTProbeResult::Hit {{\n                            data: tt_data,\n                            slot: TTSlot(entry{i}),"
        source = replace_once(source, ret, f"""if WARM_READER.load(Ordering::Relaxed) {{
                            println!("hit {{}} {{}} {{}} {{}} {{}}", std::ptr::from_ref(entry{i}).addr(), _seq, player, opponent, tt_data.to_u64());
                        }}
                        """ + ret)
    path.write_text(source + """
static WARM_READER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
impl TranspositionTable {
    pub(super) fn warm_dump(&self, snapshot: usize) {
        println!("image {} {} {} {}", snapshot, self.entries.as_ptr().addr(), self.entries.len(), std::mem::size_of::<TTEntry>());
        for (i, e) in self.entries.iter().enumerate() {
            let seq = e.seq.load(Ordering::Relaxed);
            let p = e.player.load(Ordering::Relaxed);
            let o = e.opponent.load(Ordering::Relaxed);
            let d = e.data.load(Ordering::Relaxed);
            if (seq | p | o | d) != 0 {
                println!("word {i} {seq} {p} {o} {d}");
            }
        }
        WARM_READER.store(snapshot == 2, Ordering::Relaxed);
    }
}
""")


def parse(output):
    sizes, images, results = [], [], []
    claims, finishes = [dict(), dict()], [dict(), dict()]
    hits = []
    for line in output.splitlines():
        tag, *args = line.split()
        if tag == "image":
            index, base, count, stride = map(int, args)
            require(index == len(images), "missing snapshot")
            sizes.append((base, count, stride))
            images.append({})
        elif tag == "word":
            index, *words = map(int, args)
            base, count, stride = sizes[-1]
            addr = base + index * stride
            require(0 <= index < count and addr not in images[-1] and len(words) == 4, "invalid slot")
            images[-1][addr] = tuple(words)
        elif tag == "result":
            results.append(tuple(map(int, args)))
        else:
            require(len(images) in (1, 3), "activity outside Search::run")
            phase = len(images) // 2
            addr, seq = map(int, args[:2])
            base, count, stride = sizes[-1]
            require(0 <= addr-base < count*stride and (addr-base) % stride == 0 and seq % 2 == 0,
                    "invalid physical address or sequence")
            key = (addr, seq)
            if tag == "claim":
                require(key not in claims[phase], "duplicate successful claim")
                claims[phase][key] = args[2]
            elif tag == "finish":
                require(key not in finishes[phase], "duplicate release")
                finishes[phase][key] = (seq + 2, *map(int, args[2:]))
            else:
                require(tag == "hit" and phase == 1, "unexpected event")
                hits.append((addr, (seq, *map(int, args[2:]))))
    require(len(images) == 4 and len(set(sizes)) == 1 and not images[0], "changed TT allocation or nonempty start")
    require(results == [(0, 1536, 12), (1, 1536, 12)], "missing exact public results")
    require(images[1] == images[2], "prior final and current initial physical words differ")
    for n in range(2):
        require(claims[n].keys() == finishes[n].keys(), "successful writer missing its release")
        current = images[2*n].copy()
        for (addr, expected), words in sorted(finishes[n].items()):
            require(current.get(addr, ZERO)[0] == expected, "claim skips a generation")
            current[addr] = words
        require(current == images[2*n+1], "final TT differs from released payloads")
    require(all(len(set(writers.values())) > 1 for writers in claims), "expected multiple writer threads in each search")
    warm = {a for a, words in hits if images[1].get(a) == words}
    require(bool(warm), "second request never probed an unchanged first-session slot")
    return images, sizes[0], claims, warm


def replay(folder, out):
    images, (base, count, stride), claims, warm = parse(out)
    addr = min(warm)
    seq, player, opponent, data = images[1][addr]
    # Use actual claim sequence numbers for this slot as call identifiers.
    keys = sorted(s for a, s in claims[0] if a == addr)
    require(bool(keys) and seq == keys[-1] + 2, "missing captured release for warm slot")
    records = []
    for row in out.split("image 1 ", 1)[0].splitlines():
        tag, *args = row.split()
        if tag in ("claim", "finish") and int(args[0]) == addr:
            pc = "cas" if tag == "claim" else "finish"
            records.append(f".writer ⟨{int(args[1])},.{pc}⟩")
    require(len(records) == 2 * len(keys), "sampled writer records do not match")
    script = folder / "Warm.lean"
    script.write_text(f"""import NeuralReversi.PvsParallel.Interleaving.JournalResident
open NeuralReversi.PvsParallel
open NativeTableWrites.Journal (Record)
open Interleaving.JournalResident (checkDrained)
def observed : List Record := [{",".join(records)}]
#guard checkDrained (fun _ => true) observed
#guard !(checkDrained (fun _ => true) observed.dropLast)
""")
    subprocess.run(["lake", "env", "lean", str(script)], cwd=LEAN, check=True)
    # Flip just the player key of the same snapshot in the captured next-session input.
    index = (addr-base) // stride
    before, next_ = out.split("image 2 ", 1)
    old = f"word {index} {seq} {player} {opponent} {data}\n"
    require(old in next_, "missing next-session slot")
    corrupted = before + "image 2 " + next_.replace(old, f"word {index} {seq} {player ^ 1} {opponent} {data}\n", 1)
    try:
        parse(corrupted)
    except ValueError as err:
        require(str(err) == "prior final and current initial physical words differ", str(err))
    else:
        raise ValueError("accepted altered player key")
    print(f"2 × 12 OBF scores, best +1536, all {count} slots matched at the boundary; "
          f"successful writes {len(claims[0])}/{len(claims[1])} on "
          f"{len(set(claims[0].values()))}/{len(set(claims[1].values()))} writer threads. "
          f"Second search hit {len(warm)} unchanged prior slots.")
    print(f"Prior final = next initial at allocation {base:#x}, slot {index}: "
          f"seq={seq} player={player:#018x} opponent={opponent:#018x} data={data:#018x}; "
          f"Lean replayed {len(keys)} successful writes. Rejected changed key and missing finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-warm-tt-") as tmp:
        folder = Path(tmp)
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        binary = capture(folder, target, instrument, fixture=FIXTURE, serial_trace=False)
        replay(folder, subprocess.check_output([str(binary), str(REPO)], cwd=folder, text=True))
