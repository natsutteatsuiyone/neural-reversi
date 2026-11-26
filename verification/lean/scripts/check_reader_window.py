"""Replay the Rust TT's per-thread instructions and returned words with the Lean reader window."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


def instrument(src, weakened=False):
    path = src / "transposition_table.rs"
    source = path.read_text()
    left = source.index("    fn try_load_snapshot(")
    right = source.index("    /// Stores a result", left)
    reader = source[left:right]
    if weakened:
        first, lookup = reader.split("    fn read_for_lookup(", 1)
        first = replace_once(first, "let seq1 = self.seq.load(Ordering::Acquire);",
                            "let seq1 = self.seq.load(Ordering::Relaxed);")
        reader = first + "    fn read_for_lookup(" + lookup
    opcodes = {("seq", "Acquire"): "A", ("seq", "Relaxed"): "Q", ("data", "Relaxed"): "D",
               ("player", "Relaxed"): "P", ("opponent", "Relaxed"): "O"}

    def record(load):
        register, field, ordering = load.groups()
        require((field, ordering) in opcodes, f"unrecognized reader load: {field} {ordering}")
        return load[0] + f' window_op(Some(self), "{opcodes[field, ordering]}", {register});'

    reader, loads = re.subn(r"let (\w+) = self\.(\w+)\.load\(Ordering::(\w+)\);", record, reader)
    require(loads == 11 and reader.count("fence(Ordering::Acquire);") == 3, "TT reader load/fence sites changed")
    reader = reader.replace("fence(Ordering::Acquire);", 'fence(Ordering::Acquire); window_op(None, "F", 0);')
    source = source[:left] + reader + source[right:]
    write = source.index("    fn seqlock_write(")
    fence = source.index("        fence(Ordering::Release);", write)
    source = source[:fence] + "        window_hold_claim();\n" + source[fence:]
    require("self.generation.load(Ordering::Relaxed) & TTEntry::GENERATION_MASK" in source,
            "generation is no longer a masked relaxed byte load")
    probe = source.index("    pub fn probe(")
    end = source.index("    /// Stores a finite search score", probe)
    body = replace_once(source[probe:end], "let generation = self.generation();",
                        'let generation = self.generation(); window_op(None, "G", generation as u64);')
    body = replace_once(body, "std::hint::spin_loop();", 'std::hint::spin_loop(); window_op(None, "S", 0);')
    path.write_text(source[:probe] + body + source[end:] + (Path(__file__).parent / "serial_calls/reader_window.rs").read_text())


def rows(lines, endian):
    items = []
    finishes = []
    for line in lines:
        kind, who, *args = line.split()
        worker = int(who)
        op = "none"
        word = 0
        marker = ".other"
        if kind == "enter":
            method, p, o, count, key = args
            b = f"(⟨⟨{int(p)},{int(o)}⟩,rfl⟩ : PvsBoardTree.Position)"
            m = f".probe (Cutoffs.key {b})" if method == "probe" else f".lookup {b}"
            marker = f".enter ⟨{m},0,{int(count)},{int(key)}⟩"
        elif kind == "leave":
            method, slot, data = args
            raw = f"(Packing.unpack {endian} {int(data)})"
            if method == "probe":
                answer = f".hit {int(slot)} {raw}" if int(data) else f".miss {int(slot)}"
                marker = f".leave (.probe ({answer}))"
            else:
                marker = f".leave (.lookup {'(some ' + raw + ')' if int(data) else 'none'})"
            finishes.append(len(items))
        else:
            require(kind == "op", "unknown instruction row")
            code, slot, value = args
            addr = f"⟨0,{int(slot)}⟩"
            op = {"A": f".load {addr} .sequence .acquire", "Q": f".load {addr} .sequence .relaxed",
                  "D": f".load {addr} (.payload .metadata) .relaxed", "P": f".load {addr} (.payload .player) .relaxed",
                  "O": f".load {addr} (.payload .opponent) .relaxed", "F": ".fence .acquire",
                  "G": ".generation 0", "S": ".spin"}[code]
            op = f"some ({op})"
            word = int(value)
        items.append(f"⟨{worker},{marker},{op},{word}⟩")
    return items, finishes


def run(binary, folder, weakened):
    out = subprocess.check_output([str(binary), str(REPO)], cwd=folder, text=True, timeout=45).splitlines()
    endian = out.pop(0).split()
    require(endian in (["endian", "false"], ["endian", "true"]), "missing native endianness")
    endian = endian[1]
    items, finishes = rows(out, endian)
    require(len(finishes) == 13, "missing reader returns")
    for code in ("A", "Q", "D", "P", "O", "G", "F", "S"):
        require(any(x.startswith("op ") and x.split()[2] == code for x in out), f"missing {code} instruction")
    # Workers 2, 3 and 4 run one call each before, during and after a real CAS writer.
    empty, held, published = ([line.split() for line in out if line.split()[1] == str(w)] for w in (2, 3, 4))
    first_code = "Q" if weakened else "A"
    require([row[2] for row in empty[1:-1]] == [first_code, "D", "F", "Q"] and empty[-1][-2:] == ["0", "0"],
            "empty metadata path did not return the first slot")
    round_ = [first_code, first_code, "D", "P", "O", "F", "Q", "S"]
    require([row[2] for row in held[1:-1]] == round_ * 3 and held[-1][-2:] == ["1", "0"],
            "held odd writer did not exhaust retries and select the stable second slot")
    require(all(int(row[4]) & 1 for row in held[1:-1:len(round_)]), "first slot was not odd on every retry")
    byteorder = "big" if endian == "true" else "little"
    published_score = int.from_bytes(int(published[-1][4]).to_bytes(8, byteorder)[:2], byteorder, signed=True)
    require(published[-1][3] == "0" and published_score == 768 and int(published[1][4]) == int(held[1][4]) + 1,
            "post-publication probe did not read the new sequence and score from slot zero")
    require(int(empty[1][3]) > 0, "fixture did not address a nonzero cluster")
    first_fence = next(i for i, line in enumerate(out) if line.startswith("op ") and line.split()[2] == "F")
    return_check = next(i for i in finishes if int(out[i].split()[-1]) != 0)
    script = """import NeuralReversi.PvsParallel.Interleaving.ReaderWindow
set_option maxRecDepth 4096
open NeuralReversi NeuralReversi.PvsParallel NeuralReversi.TranspositionTable
open PvsParallel.FrameTableReads.Operations.Execution (Instruction)

structure Row where
  worker : Nat
  mark : Interleaving.ReaderWindow.Mark
  op : Option Instruction
  word : BitVec 64
"""
    script += "def sample : Array Row := #[\n" + ",\n".join(items) + "\n]\n"
    script += f"""
def passed (rows : Array Row) (end_ : Nat) := Id.run do
  let at_ : Nat → Row := fun i => rows[i]?.getD ⟨0,.other,none,0⟩
  return (Interleaving.ReaderWindow.check (fun i => (at_ i).worker) (fun i => (at_ i).mark)
    (fun i => (at_ i).op) (fun _ i => (at_ i).word) (fun _ i => (at_ i).word.setWidth 8)
    {endian} end_ end_).isSome
def finishes : List Nat := {finishes}
"""
    if weakened:
        empty_end = next(i for i in finishes if out[i].split()[1] == "2")
        script += f"#guard !(passed sample {empty_end})\n"
    else:
        first_load = next(i for i, line in enumerate(out) if line.startswith("op "))
        script += f"""#guard finishes.all (passed sample)
def first := sample[{first_load}]
#guard !(finishes.all (passed (sample.set! {first_load} {{first with op := some (.load ⟨0,0⟩ .sequence .acquire)}})))
def fence := sample[{first_fence}]
#guard !(finishes.all (passed (sample.set! {first_fence} {{fence with op := none}})))
#guard !(finishes.all (passed (sample.set! {first_fence} {{fence with worker := 1-fence.worker}})))
def hit := sample[{return_check}]
#guard !(passed (sample.set! {return_check} {{hit with mark := .leave (.probe (.miss 0))}}) {return_check})
"""
    source = folder / "NativeReaderWindow.lean"
    source.write_text(script)
    subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", str(source)], cwd=LEAN, check=True)
    if weakened:
        print("PASS: changing the Rust snapshot's initial Acquire to Relaxed rejects the empty-slot probe")
    else:
        print(f"PASS: {len(finishes)} native returns, {len(items) - 2 * len(finishes)} instructions, "
              "empty slot, 3 held-writer retries, post-release hit, 4 negative controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        fixture = folder / "fixture.rs"
        fixture.write_text("pub fn run(_: &std::path::Path) { crate::transposition_table::run_reader_window(); }\n")
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        for weakened in (False, True):
            build = folder / ("weakened" if weakened else "baseline")
            build.mkdir()
            binary = capture(build, target, lambda src: instrument(src, weakened), fixture=fixture, serial_trace=False)
            run(binary, build, weakened)


if __name__ == "__main__":
    main()
