"""Compare a one-worker midgame child and its retained TT and both cache images with Lean."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require

CASES = Path(__file__).resolve().parent / "serial_calls"


def instrument(src):
    with (src / "transposition_table.rs").open("a") as out:
        out.write("""
impl TranspositionTable {
    pub(crate) fn midgame_image(&self) {
        for (i, s) in self.entries.iter().enumerate() {
            let (seq, p, o, d) = (s.seq.load(Ordering::Relaxed), s.player.load(Ordering::Relaxed),
                                  s.opponent.load(Ordering::Relaxed), s.data.load(Ordering::Relaxed));
            if seq | p | o | d != 0 { println!("TT {i} {seq} {p} {o} {d}"); }
        }
    }
}
""")
    with (src / "search/endgame/cache.rs").open("a") as out:
        out.write("""
impl<K> EndGameCache<K> {
    pub(super) fn midgame_image(&self, name: &str) {
        println!("CACHE {name} {}", self.table.len());
        for (i, &RawEntry {player, opponent, lower, upper}) in self.table.iter().enumerate() {
            if player != 0 || opponent != 0 || lower != -65 || upper != 65 {
                println!("CELL {name} {i} {player} {opponent} {lower} {upper}");
            }
        }
    }
}
""")
    with (src / "search/endgame.rs").open("a") as out:
        out.write("""
impl EndGameCaches {
    pub(in crate::search) fn midgame_image(&self) {
        self.ec.midgame_image("deep");
        self.shallow.midgame_image("shallow");
    }
}
""")
    path = src / "search/pvs.rs"
    text = replace_once(path.read_text(), "return tt_data.score();",
                        'eprintln!("MIDGAME_HIT"); return tt_data.score();')
    text = replace_once(text, "ctx.tt.penalize(tt_probe_result.slot(), board, tt_data);",
                        'eprintln!("MIDGAME_PENALTY"); ctx.tt.penalize(tt_probe_result.slot(), board, tt_data);')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-midgame-child-") as tmp:
        target = args.target_dir.resolve() if args.target_dir else Path(tmp) / "target"
        binary = capture(Path(tmp), target, instrument, fixture=CASES / "midgame_child.rs")
        rust = subprocess.run([binary, REPO], capture_output=True, text=True, check=True)
        require(rust.stderr.splitlines().count("MIDGAME_HIT") == 1, "warm midgame TT cutoff missing")
        require(rust.stderr.splitlines().count("MIDGAME_PENALTY") == 1, "wrong-direction bound was not penalized")
        interrupted, rest = rust.stdout.split("SCORE false", 1)
        require(interrupted.startswith("SCORE true 0 0\n") and "CELL " in interrupted and "TT " not in interrupted,
                "stopped midgame parent must retain its promoted child cache and skip its TT write")
        require("TT " in rest, "resumed midgame search did not store in the table")
        subprocess.run(["lake", "build", "NeuralReversi.PvsParallel.SerialTable.MidgameNative"], cwd=LEAN, check=True)
        lean = subprocess.run(["lake", "env", "lean", "--run", CASES / "MidgameChild.lean.in"],
                              cwd=LEAN, capture_output=True, text=True, check=True)
        require(lean.stdout.splitlines() == rust.stdout.splitlines(), "midgame score, TT or worker cache image differs")
        print("Interrupted child, resumed search, warm cutoff and depth penalty matched scores, every TT slot and both worker cache images.")


if __name__ == "__main__":
    main()
