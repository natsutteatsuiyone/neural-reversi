"""Follow the exact PVS root entry through a generated wipeout to its return."""

from pathlib import Path
import re
import subprocess
import tempfile
import threading

from mir_common import checked, run_extractor_cli


FILENAME = "reversi_core.search-pvs-search.SimplifyCfg-initial.after.mir"
_CONSTANTS_LOCK = threading.Lock()
_CONSTANTS_CACHE = {}


def seed_constants(repo, values):
    with _CONSTANTS_LOCK:
        _CONSTANTS_CACHE[Path(repo).resolve()] = dict(values)


def constants(repo):
    key = Path(repo).resolve()
    with _CONSTANTS_LOCK:
        cached = _CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    prefix = key / "crates/reversi-core/src"
    source = f'''#[path = r#"{prefix / 'score.rs'}"#] mod score;
#[path = r#"{prefix / 'search/node_type.rs'}"#] mod node_type;
use node_type::{{NodeType, Root}};
use score::Score;
fn main() {{ println!("{{}} {{}} {{}} {{}} {{}}", Root::ROOT_NODE, Root::PV_NODE, Score::MAX.value(), Score::ZERO.value(), Score::INF.value()); }}
'''
    with tempfile.TemporaryDirectory() as tmp:
        rust, exe = Path(tmp) / "constants.rs", Path(tmp) / "constants.exe"
        rust.write_text(source, encoding="utf-8")
        subprocess.run(["rustc", "--edition=2024", "-Adead_code", str(rust), "-o", str(exe)], cwd=key, check=True)
        root, pv, maximum, zero, infinity = subprocess.check_output([str(exe)], text=True).split()
    values = {
        "const <NT as search::node_type::NodeType>::ROOT_NODE": int(root == "true"),
        "const <NT as search::node_type::NodeType>::PV_NODE": int(pv == "true"),
        "const score::Score::MAX": int(maximum), "const score::Score::ZERO": int(zero),
        "const score::Score::INF": int(infinity), "const true": 1, "const false": 0,
    }
    with _CONSTANTS_LOCK:
        _CONSTANTS_CACHE[key] = values
    return values.copy()


def extract(mir, values):
    source = mir.read_text(encoding="utf-8")
    checked(source.startswith("// MIR for `search::pvs::search` after SimplifyCfg-initial\n"), "wrong function")
    bodies = {int(m[1]): [line.strip() for line in m[2].splitlines()] for m in
              re.finditer(r"^    bb(\d+): \{\n((?:        [^\n]+\n)+)    \}", source, re.M)}
    checked(len(bodies) == len(re.findall(r"^    bb\d+: \{\s*$", source, re.M)), "malformed block")
    def binding(name):
        matches = re.findall(r"^\s*debug " + name + r" => (_\d+);$", source, re.M)
        checked(len(matches) == 1, f"missing or ambiguous {name}")
        return matches[0]

    probes = [num for num, lines in bodies.items() if " = MoveList::wipeout_move(" in lines[-1]]
    checked(len(probes) == 1, "expected one wipeout probe")
    pc, seen = 0, set()
    env = {binding("ctx"): "ctx_ref", binding("board"): "board_ref", binding("depth"): "positive_depth",
           binding("alpha"): "alpha", binding("beta"): "beta"}
    reset = False
    updates = []
    units = set(re.findall(r"let (?:mut )?(_\d+): \(\);", source))

    def operand(text):
        if text in values:
            return values[text]
        if m := re.fullmatch(r"(?:move|copy) (_\d+)", text):
            return env[m[1]]
        raise ValueError(f"unrecognized operand {text}")

    while pc not in seen:
        seen.add(pc)
        lines = bodies[pc]
        for line in lines[:-1]:
            if re.fullmatch(r"(?:StorageLive|StorageDead)\(_\d+\);", line):
                continue
            if m := re.fullmatch(r"FakeRead\(ForLet\(None\), (_\d+)\);", line):
                checked(m[1] in env, f"unknown let operand: {line}")
                continue
            if m := re.fullmatch(r"PlaceMention\((_\d+)\);", line):
                checked(m[1] in env, f"unknown value {line}")
                continue
            m = re.fullmatch(r"(_\d+) = (.*);", line)
            checked(m, f"unsupported bb{pc} statement: {line}")
            var, expr = m.groups()
            if expr == "const ()":
                checked(var in units, f"unit write of {var}")
                env[var] = "unit"
            elif n := re.fullmatch(r"&(_\d+)", expr):
                checked(env[n[1]] == "list", f"borrowed the wrong MoveList: {expr}")
                env[var] = "list_ref"
            elif n := re.fullmatch(r"&(?:mut )?\(\*(_\d+)\)", expr):
                checked(env[n[1]] in ("ctx_ref", "board_ref"), f"unexpected reborrow: {expr}")
                env[var] = env[n[1]]
            elif n := re.fullmatch(r"&\(\*\(\(\*(_\d+)\)\.(\d+): &([^\)]+)\)\)", expr):
                checked(env[n[1]] == "ctx_ref", "wrong owner for a context field")
                fields = {("1", "search::root_move::RootMoves"): "rows_ref",
                          ("5", "transposition_table::TranspositionTable"): "table_ref"}
                env[var] = fields[(n[2], n[3])]
            elif n := re.fullmatch(r"Eq\(move (_\d+), const 0_(u32|usize)\)", expr):
                checked((env[n[1]], n[2]) in (("positive_depth", "u32"), ("positive_count", "usize")),
                        f"positive entry guard does not apply to {expr}")
                env[var] = 0
            elif n := re.fullmatch(r"discriminant\((_\d+)\)", expr):
                checked(env[n[1]] == "some_square", f"wrong discriminant source: {expr}")
                env[var] = 1
            elif n := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: square::Square\)", expr):
                checked(env[n[1]] == "some_square", f"wrong square source: {expr}")
                env[var] = "square"
            else:
                env[var] = operand(expr)
        term = lines[-1]
        if m := re.fullmatch(r"(_\d+) = (.+)\((.*)\) -> \[return: bb(\d+), unwind: bb\d+\];", term):
            var, call, args, nxt = m.groups()
            params = [operand(arg) for arg in args.split(", ")]
            if call == "SearchContext::<'_>::update_root_move":
                checked(params[:2] == ["ctx_ref", "square"] and len(params) == 4 and
                        type(params[2]) is int and params[3] in (0, 1), f"unexpected update {params}")
                checked(reset, "observed a root write without resetting the suffix")
                updates.append((params[2], bool(params[3])))
                env[var] = "unit"
            elif call == "RootMoves::reset_scores_from_pv_idx":
                checked(params == ["rows_ref"] and not updates, "invalid root reset")
                reset = True
                env[var] = "unit"
            else:
                ops = {
                    "SearchContext::<'_>::prepare_pv": (["ctx_ref"], "unit"),
                    "Board::hash": (["board_ref"], "board_hash"),
                    "TranspositionTable::prefetch": (["table_ref", "board_hash"], "unit"),
                    "Board::legal_moves": (["board_ref"], "legal"),
                    "RootMoves::moves_bb_from_pv_idx": (["rows_ref"], "suffix"),
                    "<LegalMoves as BitAnd<Bitboard>>::bitand": (["legal", "suffix"], "selected"),
                    "LegalMoves::is_empty": (["selected"], 0),
                    "MoveList::with_moves": (["board_ref", "selected"], "list"),
                    "MoveList::count": (["list_ref"], "positive_count"),
                    "MoveList::wipeout_move": (["list_ref"], "some_square"),
                }
                checked(call in ops, f"unsupported call during wipeout exit: {call}")
                arguments, result = ops[call]
                checked(params == arguments, f"bb{pc}: wrong operands of {call}: {params}")
                env[var] = result
            pc = int(nxt)
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];", term):
            checked(isinstance(env[m[1]], int), f"noninteger switch: {term}")
            pc = int(m[3] if env[m[1]] == int(m[2]) else m[4])
        elif m := re.fullmatch(r"falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];|goto -> bb(\d+);", term):
            pc = int(m[1] or m[2])
        elif term == "return;":
            checked(reset and probes[0] in seen and type(env["_0"]) is int,
                    "unexpected return without the root reset, cached wipeout, and a defined score")
            return updates, env["_0"]
        else:
            raise ValueError(f"unsupported bb{pc} terminator: {term}")
    raise ValueError(f"cycle at bb{pc}")


def render(mir, repo):
    writes, result = extract(mir, constants(repo))
    payload = ", ".join(f"(square, {score}, {str(pv).lower()})" for score, pv in writes)
    return f"""module
public import NeuralReversi.Bitboard.SquareModel

@[expose] public section

namespace NeuralReversi.PvsParallel

def rootWipeoutExit (square : Bitboard.Square) : Prod (List (Prod Bitboard.Square (Prod Int Bool))) Int :=
  ([{payload}], {result})

end NeuralReversi.PvsParallel
"""


def main():
    run_extractor_cli(
        __doc__,
        lambda dump, repo: render(dump / FILENAME, repo),
        filters="pvs::search",
        check_message=lambda p: f"MIR program differs from {p}",
    )


if __name__ == "__main__":
    main()
