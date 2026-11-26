"""Check per-thread endgame cache reuse across two public searches on one pool."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from check_serial_calls_trace import REPO, capture, replace_once


def instrument(src):
    strategy = src / "search/strategy.rs"
    code = strategy.read_text()
    before = "let score_d = if n_empties > endgame::DEPTH_TO_SHALLOW_SEARCH {"
    code = replace_once(code, before, """let prior_hit = if (5..=endgame::DEPTH_TO_SHALLOW_SEARCH).contains(&n_empties) {
            caches.shallow.probe(caches.shallow.slot(board.hash()), board, alpha_d)
        } else {
            None
        };
        """ + before)
    after = "let score = Score::from_disc_diff(score_d);"
    code = replace_once(code, after,
                        "super::serial_calls_fixture::observe(ctx, board, alpha_d, alpha, beta, prior_hit, score_d);\n        " + after)
    strategy.write_text(code)
    cache = src / "search/endgame/cache.rs"
    point = "let entry = unsafe { self.table.get_unchecked_mut(slot.0) };"
    cache.write_text(replace_once(cache.read_text(), point,
        "crate::search::serial_calls_fixture::stored(slot.0, board, alpha, score);\n        " + point))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="Optional external scratch Cargo target cache")
    args = parser.parse_args()
    fixture = Path(__file__).with_name("serial_calls") / "warm_local_cache.rs"
    with tempfile.TemporaryDirectory(prefix="nr-warm-local-") as scratch:
        folder = Path(scratch)
        target = args.target_dir.resolve() if args.target_dir else folder / "target"
        binary = capture(folder, target, instrument, fixture=fixture, serial_trace=False)
        tape = folder / "boundary-returns.txt"
        with tape.open("w") as output:
            subprocess.run([str(binary), str(REPO)], check=True, cwd=folder, stdout=output)
        project = REPO / "verification/lean"
        module = "NeuralReversi.PvsParallel.Interleaving.JournalLocalCacheFixture"
        subprocess.run(["lake", "build", module], check=True, cwd=project)
        runner = folder / "ReplayWarmCache.lean"
        runner.write_text(f"""import {module}

def main (args : List String) : IO Unit := do
  let [file] := args | throw (IO.userError "expected one trace file")
  {module}.replayFile file
""")
        subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", "--run", str(runner), str(tape)],
                       check=True, cwd=project)


if __name__ == "__main__":
    main()
