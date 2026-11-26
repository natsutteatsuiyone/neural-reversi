"""Check actual serial endgame cache storage, probes and bounds on the normal Rust search path.

    python3 verification/lean/scripts/check_serial_cache_trace.py

The temporary fixture uses the checkout's normal evaluation weights. An optional
--target-dir can reuse the scratch dependency build of check_serial_calls_trace.py.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from normalize_pvs_trace import require

EMPTY = (0, 0, -65, 65)
MISS = -(1 << 31)
MASK = (1 << 64) - 1
NINE = (0x81afcddf13010000, 0x7e503220ecfe7c24)


def board_hash(player, opponent):
    # Board::hash serializes the board to native bytes; rapidhash reads little-endian.
    if sys.byteorder != "little":
        player, opponent = (int.from_bytes(x.to_bytes(8, "big"), "little") for x in (player, opponent))
    def fold(x, y):
        product = x * y
        return (product >> 64) ^ (product & MASK)
    secret = 0x8bb84b93962eacc9
    return fold(player ^ secret, opponent ^ fold(0x4b33a62ed433d4a3, secret) ^ 16)


def instrument(src):
    path = src / "search/endgame/cache.rs"
    text = path.read_text()
    for fn, alias in (("new", "allocate"), ("probe", "probe"), ("store", "store")):
        text = replace_once(text, f"pub fn {fn}(", f"fn serial_{alias}(")
    path.write_text(text + (FIXTURES / "cache.rs").read_text())
    endgame = src / "search/endgame.rs"
    trial = "let score = aspiration_search(ctx, board, &mut alpha, &mut beta, thread);"
    operands = "Arc::as_ptr(thread) as usize, board.player().bits(), board.opponent().bits(), pv_idx, selectivity as u8"
    start = f'eprintln!("RUNG {{}} {{}} {{}} {{}} {{}}", {operands});'
    end = f'eprintln!("RUNG_END {{}} {{}} {{}} {{}} {{}}", {operands});'
    endgame.write_text(replace_once(endgame.read_text(), trial, f"{start}\n            {trial}\n            {end}"))
    with endgame.open("a") as out:
        out.write("""
impl EndGameCaches {
    pub(in crate::search) fn serial_checkout(&mut self, owner: usize) -> &mut Self {
        eprintln!("BORROW {} {} {}", owner, self.ec.serial_base(), self.shallow.serial_base());
        self
    }
}
""")
    thread = src / "search/threading/thread.rs"
    borrow = "unsafe { &mut *self.endgame_caches.get() }"
    thread.write_text(replace_once(thread.read_text(), borrow, borrow + ".serial_checkout(self as *const _ as usize)"))
    pvs = src / "search/pvs.rs"
    mark = 'eprintln!("ENTER '
    pvs.write_text(replace_once(pvs.read_text(), mark, 'eprintln!("WORKER {}", Arc::as_ptr(thread) as usize); ' + mark))


@dataclass
class Table:
    kind: str
    size: int
    stride: int
    cells: dict = field(default_factory=dict)
    writers: dict = field(default_factory=dict)


def check(lines):
    tables, stack, counts, bounds = {}, [], Counter(), []
    opened = 0
    worker, lease = None, None
    using_cache = set()
    rung = None
    nine = Counter()
    nine_ladder, nine_roots = [], {"cold": [], "public": []}
    for line in lines:
        event, *args = line.split()
        if event == "RUNG":
            require(rung is None and not stack, "overlapping aspiration ladder calls")
            rung = tuple(map(int, args))
            if rung[1:3] == NINE and rung[3] == 0:
                nine_ladder.append(rung[4])
        elif event == "RUNG_END":
            require(tuple(map(int, args)) == rung and not stack, "ladder exit is not paired with its entry")
            rung = None
        elif event == "ALLOC":
            kind, base, size, stride = args
            base, size, stride = int(base), int(size), int(stride)
            require(kind in ("Deep", "Shallow") and base not in tables and size > 0 and stride == 18, line)
            tables[base] = Table(kind, size, stride)
            counts["allocated"] += 1
        elif event == "FREE":
            base = int(args[0])
            if lease and base in lease[2:]:
                lease = None
            del tables[base]
        elif event == "WORKER":
            worker, = map(int, args)
        elif event == "ENTER":
            require(worker is not None, "missing physical PVS worker")
            opened += 1
            stack.append((opened, args, worker))
            worker = None
        elif event == "BORROW":
            owner, deep, shallow = map(int, args)
            require(tables[deep].kind == "Deep" and tables[shallow].kind == "Shallow", "lease has wrong tables")
            if stack:
                require(owner == stack[-1][2], "cache receiver differs from active PVS worker")
            lease = (stack[-1][0] if stack else 0, owner, deep, shallow)
            if stack and tuple(map(int, stack[0][1][:2])) == NINE and stack[0][1][7] == "0" and not nine["cold leases"]:
                require(not tables[deep].cells and not tables[shallow].cells, "direct full-window trial started warm")
                nine["cold leases"] += 1
        elif event == "EXIT":
            _, entry, _ = stack.pop()
            if not stack:
                if tuple(map(int, entry[:2])) == NINE:
                    public = entry[7] == "1"
                    mode = "public" if public else "cold"
                    require((rung is not None) == public, "public and direct nine-empty roots were conflated")
                    if public:
                        require(rung[1:] == (*NINE, 0, 5), "observed a different public root rung")
                    nine_roots[mode].append((int(entry[5]), int(entry[6]), int(args[0])))
                # Include the root window too. The minimax oracle checks fail-low,
                # fail-high and strictly interior returns using the same interface.
                bounds.append(" ".join([*entry[:2], *entry[5:7], args[0]]))
        elif event in ("LOAD", "SAVE"):
            kind, *numbers = args
            base, address, i, key, p, o, alpha, word, *raw = map(int, numbers)
            tab = tables[base]
            require(tab.kind == kind and key == board_hash(p, o) and i == (key * tab.size) >> 64
                    and address == base + tab.stride*i, "wrong cache identity, hash, index or address")
            cell = tab.cells.get(i, EMPTY)
            require(tuple(raw[:4]) == cell and p & o == 0, "resident bytes differ from last physical store")
            resident_p, resident_o, low, high = cell
            match = (p, o) == (resident_p, resident_o)
            frame = stack[-1][0] if stack else 0
            require(lease and lease[0] == frame and lease[2 if kind == "Deep" else 3] == base,
                    "cache operation outside its activation's physical table lease")
            mode = None
            if frame:
                using_cache.add(frame)
                root = stack[0][1]
                if tuple(map(int, root[:2])) == NINE:
                    mode = "public" if root[7] == "1" else "cold"
                    nine[mode + " " + event] += 1
            if rung and rung[1:3] == NINE and rung[3] == 0 and 1 <= rung[4] < 5:
                nine["prior-rung " + event] += 1
            if event == "LOAD":
                chosen = MISS
                side = None
                if match and low > alpha:
                    chosen, side = low, 0
                elif match and high <= alpha:
                    chosen, side = high, 1
                require(len(raw) == 4 and word == chosen, "native probe disagrees with reconstructed slot")
                if chosen != MISS:
                    counts[kind + " hits"] += 1
                    writer, origin = tab.writers.get((i, side), (None, None))
                    if mode == "public" and origin and rung and origin[:4] == rung[:4] and 1 <= origin[4] < rung[4]:
                        nine["prior-rung hits at None"] += 1
                    if frame and writer and frame != writer:
                        counts[kind + " cross-frame hits"] += 1
                    if frame and writer == 0:
                        counts["warm-to-exact hits"] += 1
            else:
                if not match:
                    if cell != EMPTY:
                        counts["replacements"] += 1
                    low, high = -65, 65
                    tab.writers.pop((i, 0), None)
                    tab.writers.pop((i, 1), None)
                else:
                    counts["merges"] += 1
                if word > alpha:
                    if low < word:
                        low = word
                        tab.writers[i, 0] = (frame,rung)
                elif word < high:
                    high = word
                    tab.writers[i, 1] = (frame,rung)
                expected = (p, o, low, high)
                require(tuple(raw[4:]) == expected, "stored bytes disagree with NWS bound update")
                tab.cells[i] = expected
            if event == "SAVE" or word != MISS:
                bounds.append(f"{p} {o} {256*alpha} {256*(alpha+1)} {256*word}")
            counts[event] += 1
            if frame:
                counts["exact " + event] += 1
    require(not stack and not tables and rung is None, "unterminated activation, allocation or rung")
    require(nine_ladder == [1, 2, 3, 4, 5] and nine_roots["cold"] == [(-32767, 32767, 7168)],
            "missing cold internal PV or public nine-empty ladder")
    require(nine_roots["public"] and nine_roots["public"][-1][2] == 7168
            and nine_roots["public"][0][:2] != nine_roots["cold"][0][:2],
            "public nine-empty trial did not inherit its own aspiration window")
    require(nine["cold leases"] and nine["cold SAVE"] and nine["prior-rung SAVE"] and nine["prior-rung hits at None"],
            f"missing cold writes or a public hit from this search's earlier rungs: {nine}")
    require(all(counts[k] for k in ("Deep hits", "Shallow hits", "merges", "replacements",
                                     "Deep cross-frame hits", "Shallow cross-frame hits", "warm-to-exact hits")), str(counts))
    counts["exact frames using cache"] = len(using_cache)
    return counts, bounds, nine, nine_roots


def run(binary, folder, fault=None, record=None):
    env = os.environ.copy()
    for key in ("REVERSI_TRACE_MUTATE_INACTIVE", "REVERSI_TRACE_REOPEN", "REVERSI_TRACE_CACHE_FAULT"):
        env.pop(key, None)
    if fault:
        env["REVERSI_TRACE_CACHE_FAULT"] = fault
    # Stream the physical rows to a file so a larger fixture cannot exhaust memory.
    trace = folder / "cache.trace"
    with trace.open("w") as dest:
        result = subprocess.run([str(binary), str(REPO)], cwd=folder, stderr=dest, env=env)
    require(result.returncode == 0, f"engine exited with {result.returncode}")
    with trace.open() as rows:
        result = check(rows)
    if record:
        shutil.copyfile(trace, record)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="Optional temporary Cargo target cache")
    parser.add_argument("--record", type=Path, help="Keep the complete validated baseline Rust transcript")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-serial-cache-") as tmp:
        folder = Path(tmp)
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        binary = capture(folder, target, extra_instrument=instrument)
        counts, bounds, nine, roots = run(binary, folder, record=args.record)
        print(counts, flush=True)
        print(f"Nine empties: {dict(nine)}, roots={roots}", flush=True)
        oracle = folder / "minimax"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES / "oracle.rs"), "-o", str(oracle)],
                       cwd=REPO, check=True)
        subprocess.run([str(oracle)], input="\n".join(bounds) + "\n", text=True, check=True)

        for fault, expected in (("forget", "stored bytes disagree with NWS bound update"),
                                ("miss", "native probe disagrees with reconstructed slot")):
            try:
                run(binary, folder, fault)
            except ValueError as err:
                require(str(err) == expected, f"{fault}: unexpected failure {err}")
            else:
                raise ValueError(f"accepted native cache mutation {fault}")
        print("Rejected a dropped stored bound and a suppressed physical hit.")
