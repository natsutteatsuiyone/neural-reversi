"""Compare each Multi-PV trial, warm TT, worker caches and all published rows against Lean."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from check_midgame_child_effects import instrument as instrument_images
from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require

CASES = Path(__file__).resolve().parent / "serial_calls"


def instrument(src):
    instrument_images(src)
    end = src / "search/endgame.rs"
    text = replace_once(end.read_text(), "    midgame::evaluate(ctx, board)\n}",
                        '    let raw = midgame::evaluate(ctx, board); println!("EVAL {}", raw.value()); raw\n}')
    run = "search::<Root, EndGameStrategy>(ctx, board, n_empties, *alpha, *beta, thread, false);"
    text = replace_once(text, run, run + """
        println!("TRIAL {} {} {} {} {}", ctx.root_moves.pv_idx(), ctx.selectivity as u8,
            alpha.value(), beta.value(), score.value());
        for row in ctx.root_moves.snapshot() {
            println!("ROW {} {} {} {}", row.sq.index(), row.score.value(),
                row.average_score.value(), row.mean_squared_score);
        }
        ctx.tt.midgame_image();
        thread.endgame_caches().midgame_image();""")
    end.write_text(text)
    roots = src / "search/root_move.rs"
    reset = "        for rm in moves.iter_mut().skip(pv_idx) {"
    roots.write_text(replace_once(roots.read_text(), reset, """
        if pv_idx == 1 && std::env::var_os("REVERSI_MULTI_TOUCH_PREFIX").is_some() {
            moves[0].average_score += Score::from_raw(1);
        }
""" + reset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-multi-pv-") as tmp:
        target = args.target_dir.resolve() if args.target_dir else Path(tmp) / "target"
        binary = capture(Path(tmp), target, instrument, fixture=CASES / "public_multi.rs")
        rust = subprocess.run([binary, REPO], capture_output=True, text=True, check=True)
        rows = rust.stdout.splitlines()
        seeds = [r.split()[1] for r in rows if r.startswith("EVAL ")]
        require(len(seeds) == 1, "later runs must reuse the exact root entry from the previous cap")
        actual = [r for r in rows if not r.startswith("EVAL ")]
        trials = [r for r in actual if r.startswith("TRIAL ")]
        require(len(trials) > 3 * sum(range(1, 6)), "expected at least one aspiration retry")
        require(sorted(actual[-3:]) == ["PV 12 -5120", "PV 15 -2560", "PV 5 -2560"],
                "unexpected exact published move scores")
        subprocess.run(["lake", "build", "NeuralReversi.AspirationMir.SerialTable.MultiPublic"], cwd=LEAN, check=True)
        lean = subprocess.run(["lake", "env", "lean", "--run", CASES / "PublicMulti.lean.in", seeds[0]],
                              cwd=LEAN, capture_output=True, text=True, check=True)
        expected = lean.stderr.splitlines()
        require(actual == expected, "Rust and Lean Multi-PV trials or public rows differ")
        mutant = subprocess.run([binary, REPO], capture_output=True, text=True, check=True,
                                env={**os.environ, "REVERSI_MULTI_TOUCH_PREFIX": "1"})
        changed = [r for r in mutant.stdout.splitlines() if not r.startswith("EVAL ")]
        require(len(changed) == len(expected), "mutation changed the number of events")
        differences = [(a, b) for a, b in zip(changed, expected) if a != b]
        require(differences and all(a.startswith("ROW ") and b.startswith("ROW ") for a, b in differences),
                "expected fixed-prefix damage with unchanged search trials and published scores")
        print(f"Five warm Multi-PV calls and {len(trials)} trials matched. Fixed-prefix mutation rejected.")


if __name__ == "__main__":
    main()
