"""Observe successful and unsuccessful endgame ProbCut attempts during public one-worker solves."""

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from normalize_pvs_trace import require


def instrument(src):
    pvs = src / "search/pvs.rs"
    s = pvs.read_text()
    predicate = "let tracing = SS::IS_ENDGAME && ctx.selectivity == Selectivity::None;"
    require(s.count(predicate) == 2, "PVS fixture tracing predicate changed")
    s = s.replace(predicate, "let tracing = false;")
    pvs.write_text(s)
    instrument_decisions(src)


def instrument_decisions(src):
    pvs = src / "search/pvs.rs"
    s = pvs.read_text()
    cut = "ctx.stats.increment_probcut_cut();"
    pvs.write_text(replace_once(s, cut, cut + '\n if SS::IS_ENDGAME { eprintln!("USED {} {} {} {}", '
                                + 'board.player().bits(), board.opponent().bits(), ctx.selectivity as u8, score.value()); }'))
    end = src / "search/endgame.rs"
    s = end.read_text()
    begin = "if !ctx.selectivity.is_enabled() {"
    start = 'eprintln!("TRY {} {} {} {} {} {}", board.player().bits(), board.opponent().bits(), depth, '
    s = replace_once(s, begin, start + "ctx.selectivity as u8, beta.value(), cut_node as u8);\n    " + begin
                     + '\n        eprintln!("DISABLED");')
    cap = "if pc_beta >= Score::MAX || eval < window.screen_beta {"
    s = replace_once(s, cap, 'eprintln!("WINDOW {} {} {}", eval.value(), pc_beta.value(), window.screen_beta.value());\n    ' + cap)
    verify = "let score = midgame::evaluate_depth2(ctx, board, pc_beta - 1, pc_beta);"
    s = replace_once(s, verify, verify + '\n    eprintln!("VERIFIED {}", score.value());')
    call = "let result = probcut_decision(ctx, board, depth, beta, cut_node);"
    s = replace_once(s, call, call + '\n    eprintln!("RESULT {} {}", ctx.selectivity as u8, '
                     + 'result.map_or(i32::MIN, Score::value));')
    end.write_text(s)


def check(rows, wins=None):
    attempt = window = verified = used = None
    counts = Counter()
    nearest = {}
    for line in rows:
        event, *tokens = line.split()
        v = tuple(map(int, tokens))
        if used is not None:
            require(event == "USED" and v == used, "positive endgame decision did not return beta from PVS")
            if wins is not None:
                wins.append(used)
            used = None
            continue
        if event in ("RUN", "PUBLIC"):
            require(attempt is None, "unfinished attempt at public boundary")
        elif event == "TRY":
            require(attempt is None and v[2] == 64-(v[0]|v[1]).bit_count() and v[2] >= 13, "invalid endgame decision")
            attempt = v
        elif event == "DISABLED":
            require(attempt and attempt[3] == 5 and window is None, "disabled decision at a weaker level")
            counts["disabled at None"] += 1
            attempt = None
        elif event == "WINDOW":
            require(attempt and 1 <= attempt[3] <= 4 and window is None, "calibration outside selective call")
            window = v
        elif event == "VERIFIED":
            ev, pc, screen = window
            require(ev >= screen and pc < 16384 and verified is None, "verification outside its screen")
            verified, = v
        elif event == "RESULT":
            p, o, depth, level, beta, cut = attempt
            ev, pc, screen = window
            require((verified is not None) == (ev >= screen and pc < 16384), "missing verified return")
            won = verified is not None and verified >= pc
            require(v == (level, beta if won else -(1 << 31)), "changed selectivity or wrong optional beta")
            if won:
                used = (p, o, level, beta)
            kind = "verified cut" if won else "verification missed" if verified is not None else "screened"
            counts[kind] += 1
            counts[(level, kind)] += 1
            if verified is not None and (kind not in nearest or abs(verified-pc) < nearest[kind][0]):
                nearest[kind] = (abs(verified-pc), p, o, depth, level, beta, cut, ev, pc, screen, verified)
            attempt = window = verified = None
        else:
            raise ValueError(event)
    require(attempt is None and used is None and all(counts[k] for k in
            ("verified cut", "verification missed", "screened", "disabled at None")), "missing decision coverage")
    require(all(row[0] <= 256 for row in nearest.values()), "missing observations near the verification threshold")
    return counts, nearest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-probcut-") as tmp:
        folder = Path(tmp)
        binary = capture(folder, args.target_dir.resolve() if args.target_dir else folder / "target",
                         instrument, fixture=FIXTURES / "table_fixture.rs")
        trace = args.record or folder / "probcut.trace"
        with trace.open("w") as stream:
            subprocess.run([str(binary), str(REPO)], stderr=stream, check=True)
        wins = []
        with trace.open() as stream:
            counts, nearest = check(stream, wins)
        print({k:v for k,v in counts.items() if isinstance(k, str)})
        print("Nearest verification results (distance, player, opponent, depth, level, beta, cut, eval, pc_beta, screen_beta, verified):",
              nearest)
        oracle = folder / "minimax"
        subprocess.run(["rustc", "--edition=2024", "-O", str(FIXTURES/"oracle.rs"), "-o", str(oracle)], cwd=REPO, check=True)
        positions = sorted({(p,o) for p,o,_,_ in wins})
        questions = "".join(f"{p} {o} -32767 32767 0\n" for p,o in positions)
        result = subprocess.run([str(oracle),"--values"], input=questions, text=True, stdout=subprocess.PIPE, check=True)
        exact = {(p,o):s for line in result.stdout.splitlines() for p,o,s in [map(int,line.split())]}
        require(set(exact) == set(positions), "incomplete independent minimax for weak results")
        unsupported = [(p,o,lvl,beta,exact[p,o]) for p,o,lvl,beta in wins if beta > exact[p,o]]
        print(f"{len(wins)} weak returns on {len(positions)} boards; {len(unsupported)} exceed independent minimax.")
        print("Distinct (player, opponent, level, returned lower bound, minimax):", sorted(set(unsupported)))
