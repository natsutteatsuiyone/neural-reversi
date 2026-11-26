"""Trace physical shared TT writers and reads in native, one-worker public endgame searches."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from check_serial_cache_trace import board_hash
from normalize_pvs_trace import require


def instrument_slots(src):
    table = src / "transposition_table.rs"
    t = table.read_text()
    t = replace_once(t, "pub fn new(mb_size:", "fn trace_new(mb_size:")
    t = replace_once(t, "pub fn probe(&self,", "fn trace_probe(&self,")
    t = replace_once(t, "            return Some(data);\n        }\n\n        let entry1",
                     "            self.trace_lookup(board, key, entry0, Some(data));\n"
                     "            return Some(data);\n        }\n\n        let entry1")
    lookup = "entry1.read_for_lookup(board_player, board_opponent)"
    t = replace_once(t, lookup, f"let data = {lookup}; self.trace_lookup(board, key, entry1, data); data")
    save = "slot.0.save(board, data);"
    fields = "self.entries.as_ptr() as usize, slot.trace_addr(), board.player().bits(), board.opponent().bits()"
    # Capture the requested (unpacked) store operands, then sample the resulting physical words.
    t = replace_once(t, save, "let before = slot.0.trace_words();\n"
                     + "let fresh = self.trace_probe(board, board.hash()).slot();\n"
                     + 'eprintln!("STORE {} {} {} {} {} {} {} {} {} {} {} {} {} {}", '
                     + fields + ", score.value(), best_move.map_or(255, |s| s as u8), bound as u8, depth, "
                     + "selectivity as u8, self.generation(), is_endgame as u8, is_pv as u8, fresh.trace_addr(), before);\n"
                     + 'if is_endgame && data.bound == Bound::Exact as u8 && data.score != 0 '
                     + '&& std::env::var("REVERSI_TABLE_FAULT").as_deref() == Ok("store") {'
                     + 'slot.0.save(board, TTEntryData { score: -data.score, ..data }); } '
                     + 'else if std::env::var("REVERSI_TABLE_FAULT").as_deref() == Ok("reprobe") '
                     + '{ fresh.0.save(board, data); } else {'
                     + save + '}\n eprintln!("AFTER {}", slot.0.trace_words());')
    penalty = "slot.0\n            .penalize(board.player().bits(), board.opponent().bits(), expected);"
    t = replace_once(t, penalty, 'eprintln!("PENALTY {} {} {} {} {} {}", ' + fields
                     + ", expected.to_u64(), slot.0.trace_words());\n" + penalty
                     + '\n eprintln!("AFTER {}", slot.0.trace_words());')
    table.write_text(t + (FIXTURES / "table.rs").read_text())



def instrument(src):
    instrument_slots(src)
    pvs = src / "search/pvs.rs"
    t = pvs.read_text()
    # Suppress the link-snapshot workload of the shared fixture, and collect all ladder stages.
    old = "let tracing = SS::IS_ENDGAME && ctx.selectivity == Selectivity::None;"
    require(t.count(old) == 2, "PVS wrapper no longer has two tracing predicates")
    t = t.replace(old, "let tracing = false;")
    call = "let result = observed_search::<NT, SS>(ctx, board, depth, alpha, beta, thread, cut_node);"
    t = replace_once(t, call, 'eprintln!("FRAME {} {} {} {} {} {} {} {} {}", board.player().bits(), '
                     + "board.opponent().bits(), depth, alpha.value(), beta.value(), ctx.selectivity as u8, "
                     + "SS::IS_ENDGAME as u8, NT::PV_NODE as u8, NT::ROOT_NODE as u8);\n"
                     + call + '\n eprintln!("END {}", result.value());')
    eligible = "if tt_data.can_cut(beta, depth, ctx.selectivity, SS::IS_ENDGAME) {"
    t = replace_once(t, eligible, eligible[:-1] + "|| (SS::IS_ENDGAME && ctx.selectivity == Selectivity::None "
                     + '&& std::env::var("REVERSI_TABLE_FAULT").as_deref() == Ok("ordinal") '
                     + "&& tt_data.can_cut(beta, depth, tt_data.selectivity(), true)) {")
    cut = "return tt_data.score();"
    t = replace_once(t, cut, 'eprintln!("CUT {} {}", tt_probe_result.slot().trace_addr(), tt_data.trace_raw());\n'
                     + 'if SS::IS_ENDGAME && ctx.selectivity == Selectivity::None '
                     + '&& std::env::var("REVERSI_TABLE_FAULT").as_deref() == Ok("cut") { return -tt_data.score(); }\n' + cut)
    store = "ctx.tt.store(\n                    tt_slot,"
    t = replace_once(t, store, 'eprintln!("ETC {} {} {} {} {} {} {}", mv.sq.index(), '
                     + "next.player().bits(), next.opponent().bits(), etc_tt_data.trace_raw(), "
                     + "alpha.value(), score.value(), tt_slot.trace_addr());\n                " + store)
    t = replace_once(t, "return Some(score);", 'if SS::IS_ENDGAME && ctx.selectivity == Selectivity::None '
                     + '&& std::env::var("REVERSI_TABLE_FAULT").as_deref() == Ok("etc") { return Some(-score); }\n'
                     + "                return Some(score);")
    pvs.write_text(t)


def run(binary, trace, fault=False):
    env = os.environ.copy()
    env.pop("REVERSI_TABLE_FAULT", None)
    if fault:
        env["REVERSI_TABLE_FAULT"] = fault
    with trace.open("w") as dest:
        subprocess.run([str(binary), str(REPO)], env=env, stderr=dest, check=True)


def unpack(raw):
    b = raw.to_bytes(8, sys.byteorder)
    return (int.from_bytes(b[:2], sys.byteorder, signed=True), *b[2:])


def pack(score, move, bound, depth, level, age, flags):
    return int.from_bytes(score.to_bytes(2, sys.byteorder, signed=True)
                          + bytes((move, bound, depth, level, age, flags)), sys.byteorder)


def play(p, o, sq):
    bit = 1 << sq
    require((p | o) & bit == 0, "ETC occupied move")
    flips = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        x, y, line = sq % 8 + dx, sq // 8 + dy, 0
        while 0 <= x < 8 and 0 <= y < 8:
            at = 1 << (8*y+x)
            if not (o & at):
                if p & at:
                    flips |= line
                break
            line |= at
            x, y = x+dx, y+dy
    require(flips, "ETC move failed independent flip check")
    return o ^ flips, (p | bit) ^ flips


@dataclass
class Frame:
    data: tuple
    path: tuple
    probe: tuple = ()
    lookup: tuple = ()
    cutoff: int | None = None
    stored: int | None = None
    etc_store: tuple = ()
    overwritten: bool = False


def check(rows, coverage=True):
    empty = (0, 0, 0, 0)
    tables, cells, authors = {}, {}, {}
    stack, counts, bounds, levels, completed = [], Counter(), [], [], set()
    current, pending = (), ()
    trial = 0

    def origin():
        return (counts["runs"], trial, stack[-1].path if stack else ())

    def link(writer, prefix):
        require(writer is not None, prefix + " has no originating write")
        r, t, path = writer
        rr, tt, here = origin()
        counts[prefix] += 1
        if r != rr:
            counts[prefix + " across public runs"] += 1
        elif t != tt:
            counts[prefix + " across trials"] += 1
            if t and levels[t-1] == levels[tt-1]:
                counts[prefix + " across aspiration retries"] += 1
        elif path and here and path[:len(here)] != here and here[:len(path)] != path:
            counts[prefix + " between sibling subtrees"] += 1

    def oracle(p, o, score, bound):
        a, b = {1: (score-1, score), 2: (score, score+1), 3: (-32767, 32767)}[bound]
        bounds.append(f"{p} {o} {a} {b} {score}")

    def select(base, p, o, key, generation):
        n, stride = tables[base]
        addr = base + 2*((key*n) >> 64)*stride
        slots = [addr, addr+stride]
        for a in slots:
            s = cells.get(a, empty)
            if s[3] and s[1:3] == (p, o):
                return a, s[3]
            if generation is not None and not s[3]:
                return a, 0
        if generation is None:
            return slots[1], 0
        def priority(a):
            _, _, _, depth, _, gen, _ = unpack(cells[a][3])
            return depth - 8*((generation-gen) & 127)
        return min(slots, key=priority), 0

    for line in rows:
        event, *tokens = line.split()
        v = tuple(map(int, tokens))
        f = stack[-1] if stack else None
        if event == "ALLOC":
            base, n, stride = v
            require(base not in tables and n > 0 and stride == 32, "unexpected production TT allocation")
            tables[base] = (n, stride)
        elif event == "RUN":
            require(not current and not stack and not pending, "overlapping Search::run")
            counts["runs"] += 1
            current = v
        elif event == "PUBLIC":
            require(current and not stack and not pending, "unterminated physical search")
            oracle(*current, v[0], 3)
            current = ()
        elif event == "FRAME":
            counts["frames"] += 1
            if not stack:
                require(current == v[:2] and v[6:] == (1, 1, 1), "unexpected trial entry point")
                trial += 1
                levels.append(v[5])
            stack.append(Frame(v, (f.path if f else ()) + (counts["frames"],)))
        elif event == "END":
            require(f and not f.etc_store and not pending, "unfinished ETC parent publication")
            if f.cutoff is not None:
                require(v == (f.cutoff,), "returned score differs from the authoring TT entry")
            if f.stored is not None:
                require(v == (f.stored,), "published score differs from the authoring frame return")
            completed.add(f.path[-1])
            stack.pop()
        elif event in ("PROBE", "LOOK"):
            base, p, o, key = v[:4]
            require(key == board_hash(p, o), "native TT hash differs from both board words")
            actual, raw = v[-2:]
            require((actual, raw) == select(base, p, o, key, v[4] if event == "PROBE" else None),
                    f"{event} differs from reconstructed bucket or metadata")
            writer = authors.get(actual) if raw else None
            if raw:
                link(writer, "hits")
            if f:
                if event == "PROBE" and f.data[:2] == (p, o):
                    f.probe = (actual, raw, writer)
                    if raw and f.data[5:8] == (5, 1, 0):
                        score, _, bound, depth, level, _, flags = unpack(raw)
                        if flags & 1 and depth >= f.data[2] and level < 5 and bound & (1 if score >= f.data[4] else 2):
                            counts["otherwise usable lower-level probes at None"] += 1
                if event == "LOOK":
                    f.lookup = (p, o, raw, writer)
            counts[event] += 1
        elif event in ("STORE", "PENALTY"):
            base, addr, p, o = v[:4]
            n, stride = tables[base]
            cluster = ((board_hash(p, o)*n) >> 64) * 2
            require(addr in (base+cluster*stride, base+(cluster+1)*stride), "store to wrong physical cluster")
            s = cells.get(addr, empty)
            require(not pending and v[-4:] == s, "pre-write bytes disagree with previous publication")
            seq, pp, oo, old = s
            old_score, old_move, _, old_d, old_l, old_g, old_flags = unpack(old)
            if event == "STORE":
                score, move, bound, depth, lvl, gen, end, pv, fresh = v[4:-4]
                require(f and f.probe and addr == f.probe[0] and f.stored is None,
                        "publication is not at its issuing frame's probe slot")
                require(fresh == select(base, p, o, board_hash(p, o), gen)[0],
                        "diagnostic native re-probe disagrees with both reconstructed slots")
                if f.overwritten:
                    counts["parent stores after different-board descendant writes"] += 1
                    if lvl == 5:
                        counts["parent stores after descendant writes at None"] += 1
                        counts["original hits overwritten by descendants at None"] += int(bool(f.probe[1]))
                    if addr != fresh:
                        counts["different re-probe destination after descendant writes"] += 1
                fp, fo, fd, alpha, beta, fl, fe, fpv, _ = f.data
                require((p, o, depth, lvl, end) == (fp, fo, fd, fl, fe)
                        and bound == (1 if score >= beta else 3 if fpv and alpha < score else 2)
                        and pv == int(bool(fpv or unpack(f.probe[1])[6] & 2)),
                        "store metadata differs from the issuing call and its window")
                f.stored = score
                if f and f.etc_store:
                    require((addr, p, o, score, move, bound, depth, lvl, end, pv) == f.etc_store,
                            "ETC wrote different parent slot, board or bound")
                    f.etc_store = ()
                same = old and (pp, oo) == (p, o)
                write = bound == 3 or not same or depth+2 >= old_d or (lvl > old_l and depth+4 >= old_d) or gen != old_g
                use_move = old_move if bound != 3 and same and move == 255 else move
                new = pack(score, use_move, bound, depth, lvl, gen & 127, end+2*pv)
                if write and end and lvl == 5:
                    oracle(p, o, score, bound)
                    counts["oracle exact-level stores"] += 1
            else:
                expected, = v[4:-4]
                write = old == expected and (pp, oo) == (p, o) and old_d > 0
                new = pack(old_score, old_move, unpack(old)[2], old_d-1, old_l, old_g, old_flags) if write else old
            pending = (addr, (seq+2, p, o, new) if write else s, write)
            counts[event] += 1
        elif event == "AFTER":
            addr, want, write = pending
            require(v == want, "physical TT publication differs from requested bound")
            cells[addr] = v
            if write:
                authors[addr] = origin()
                for ancestor in stack[:-1]:
                    if ancestor.probe and ancestor.probe[0] == addr and ancestor.data[:2] != v[1:3]:
                        ancestor.overwritten = True
            pending = ()
        elif event in ("CUT", "ETC"):
            p, o, depth, a, b, lvl, end, pv, _ = f.data
            require(not pv and f.probe and end, "unexpected consumer of PVS cutoff")
            addr, probe, author = f.probe
            if event == "CUT":
                require(v == (addr, probe) and probe, "direct cut differs from this frame's own probe")
                data = unpack(probe)
                score, _, bound, d, l, _, flags = data
                require(d >= depth and l >= lvl and flags & 1 and bound & (1 if score >= b else 2),
                        "TT hit cannot cut the actual window")
            else:
                sq, cp, co, raw, alpha, score, dest = v
                require(f.lookup and f.lookup[:3] == (cp, co, raw) and raw and dest == addr,
                        "ETC return differs from its physical child lookup or parent probe")
                author = f.lookup[3]
                cs, _, bound, d, l, _, flags = unpack(raw)
                require(play(p, o, sq) == (cp, co) and flags & 1 and d >= depth-1 and l >= lvl
                        and bound in (2, 3) and score == -cs and alpha == a and score > a,
                        "ETC source is not an eligible upper/exact bound on the legal child")
                f.etc_store = (addr, p, o, score, sq, 1, depth, lvl, end, int(bool(unpack(probe)[6] & 2)))
                if lvl == 5:
                    oracle(cp, co, cs, bound)
            link(author, event)
            require(author[2] and author[2][-1] in completed, "source frame has not returned")
            if lvl == 5:
                link(author, event + " exact")
                bounds.append(f"{p} {o} {a} {b} {score}")
            f.cutoff = score
        else:
            raise ValueError(f"unexpected trace event {event}")
    require(not current and not stack and not pending, "unterminated search or writer")
    if coverage:
        for name in ("CUT exact across public runs", "ETC exact across public runs",
                     "CUT across aspiration retries", "CUT exact between sibling subtrees", "oracle exact-level stores",
                     "otherwise usable lower-level probes at None", "original hits overwritten by descendants at None",
                     "different re-probe destination after descendant writes"):
            require(counts[name], f"fixture did not exercise {name}")
    return counts, bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--record", type=Path, help="Save the validated baseline trace")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-serial-table-") as tmp:
        folder = Path(tmp)
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        binary = capture(folder, target, instrument, fixture=FIXTURES / "table_fixture.rs")
        trace = folder / "events"
        run(binary, trace)
        with trace.open() as rows:
            counts, bounds = check(rows)
        if args.record:
            shutil.copyfile(trace, args.record)
        oracle_bin = folder / "check-bounds"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES / "oracle.rs"), "-o", str(oracle_bin)],
                       cwd=REPO, check=True)
        subprocess.run([str(oracle_bin)], input="\n".join(bounds) + "\n", text=True, check=True)
        for fault, expected in (("store", "physical TT publication"), ("reprobe", "physical TT publication"),
                                ("ordinal", "TT hit cannot cut"),
                                ("cut", "returned score differs"), ("etc", "returned score differs")):
            run(binary, trace, fault)
            try:
                with trace.open() as rows:
                    check(rows, coverage=False)
            except ValueError as error:
                require(expected in str(error), f"unrelated {fault} failure: {error}")
            else:
                raise ValueError(f"accepted incorrect native {fault}")
        print(dict(sorted(counts.items())))
        print("Rejected wrong stored score, re-probed destination, lower-selectivity contamination, wrong direct and ETC returns.")
