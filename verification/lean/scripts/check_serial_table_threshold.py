"""Compare live exact PVS and Lean TT images across the 14/13/12-empty dispatch boundary."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from check_serial_calls_trace import FIXTURES, LEAN, REPO, capture
from check_serial_table_trace import instrument as instrument_table
from normalize_pvs_trace import require


def instrument(src):
    instrument_table(src)
    with (src / "transposition_table.rs").open("a") as table:
        table.write("""
impl TranspositionTable {
    pub(crate) fn threshold_snapshot(&self) {
        for (i, entry) in self.entries.iter().enumerate() {
            eprintln!("CELL {} {}", i, entry.trace_words());
        }
    }
}
""")


def observations(lines):
    history, slots, expected = [], {}, []
    phases, cuts = [], []
    for line in lines:
        tag, *args = line.split()
        if tag == "ALLOC":
            base, clusters, stride = map(int, args)
            require(clusters == 16, "unexpected table layout")
        elif tag == "BEGIN":
            name, clear = args
            phases.append(name)
            count = 0
            snapshot = {}
            if clear == "1":
                history, slots = [], {}
        elif tag == "FRAME":
            count += 1
        elif tag in ("CUT", "ETC"):
            cuts.append((name, tag))
        elif tag == "STORE":
            address = int(args[1])
            offset = address - base
            require(0 <= offset < 2*clusters*stride and offset % stride == 0, "store outside TT")
            slot = offset // stride
            before = tuple(map(int, args[-4:]))
        elif tag == "AFTER":
            after = tuple(map(int, args))
            if before != after:
                slots[slot] = after
                _, player, opponent, data = after
                history.append(f"WRITE {player} {opponent} {data}")
        elif tag == "CELL":
            index, *words = map(int, args)
            snapshot[index] = tuple(words)
        elif tag == "SCORE":
            require(len(snapshot) == 2 * clusters, "incomplete native slot snapshot")
            for i in range(2 * clusters):
                require(snapshot[i] == slots.get(i, (0, 0, 0, 0)), f"untraced native write to slot {i}")
            expected.append(f"RESULT {count} {args[0]}")
            expected.extend(history)
            for index, words in sorted(snapshot.items()):
                if words == (0, 0, 0, 0):
                    continue
                expected.append(f"SLOT {index} " + " ".join(map(str, words)))
    require(phases == ["root", "scout", "hit", "seed", "etc"], "incomplete fixture")
    require(cuts == [("hit", "CUT"), ("etc", "ETC")], "missing warm direct or child cutoff")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="reusable Cargo target directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="serial-table-threshold-") as tmp:
        folder = Path(tmp)
        binary = capture(folder, (args.target_dir or folder / "target").resolve(),
                         instrument, fixture=FIXTURES / "table_threshold.rs")
        env = {k: v for k, v in os.environ.items() if k != "REVERSI_TABLE_FAULT"}
        trace = subprocess.run([binary, REPO], env=env, stderr=subprocess.PIPE, text=True, check=True)
        actual = observations(trace.stderr.splitlines())
        runner = folder / "Compare.lean"
        shutil.copyfile(FIXTURES / "TableThreshold.lean.in", runner)
        subprocess.run(["lake", "build", "NeuralReversi.PvsParallel.SerialTable.Trial"], cwd=LEAN, check=True)
        lean = subprocess.run(["lake", "env", "lean", "--run", runner], cwd=LEAN,
                              check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
        require(len(lean) == len(actual), f"length differs: Lean {len(lean)} vs Rust {len(actual)}")
        for i, (lhs, rhs) in enumerate(zip(lean, actual)):
            require(lhs == rhs, f"observation {i}: Lean {lhs} vs Rust {rhs}")
        print("Five searches matched activation count, score, ordered packed publications and resulting slots.")


if __name__ == "__main__":
    main()
