"""Regenerate or check the normal control flow of the endgame aspiration MIR."""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

from mir_common import (
    Block,
    PASS,
    checked,
    dump_mir,
    normal_target,
    parse_blocks as blocks,
    run_extractor_cli,
    split_call,
)


DRIVER = "search::endgame::aspiration_search"
WIDENER = "search::aspiration::widen_aspiration_window"
DRIVER_SIGNATURE = "fn search::endgame::aspiration_search(_1: &mut SearchContext<'_>, _2: &Board, _3: &mut Score, _4: &mut Score, _5: &Arc<threading::thread::Thread>) -> Score {"
WIDENER_SIGNATURE = "fn widen_aspiration_window(_1: Score, _2: &mut Score, _3: &mut Score, _4: Score) -> bool {"


# The output instruction represents all the listed operand preparations and the call.
# A local live across blocks keeps its MIR register (_6 delta, _7 depth, _11 score,
# _19 abort flag, _21 dereferenced thread, _25 widen flag).
CALLS = {
    "EmptyList::count": (
        "prepare", ["_6 = const search::endgame::ASPIRATION_DELTA;", "_8 = &((*_1).10: empty_list::EmptyList);"],
        "_7", "move _8"
    ),
    "search::<Root, EndGameStrategy>": (
        "query", ["_12 = &mut (*_1);", "_13 = &(*_2);", "_14 = copy _7;", "_15 = copy (*_3);",
                  "_16 = copy (*_4);", "_17 = &(*_5);"],
        "_11", "move _12, move _13, move _14, move _15, move _16, move _17, const false"
    ),
    "<Arc<threading::thread::Thread> as Deref>::deref": (
        "go", ["_22 = &(*_5);"], "_21", "move _22"
    ),
    "threading::thread::Thread::is_search_aborted": (
        "poll", ["_20 = &(*_21);"], "_19", "move _20"
    ),
    "widen_aspiration_window": (
        "widen", ["_26 = copy _11;", "_27 = &mut (*_3);", "_28 = &mut (*_4);", "_29 = copy _6;"],
        "_25", "move _26, move _27, move _28, move _29"
    ),
    "<Score as AddAssign>::add_assign": (
        "double", ["_32 = &mut _6;", "_33 = copy _6;"], "_31", "move _32, move _33"
    ),
}


def driver(body):
    ops = []
    for bb in range(len(body)):
        block = body[bb]
        s, t = block.statements, block.end
        if block.cleanup:
            checked(block == Block([], "resume;", True), f"bb{bb}: unexpected cleanup")
            op = ".unwind"
        elif m := split_call(t):
            checked(m[2] in CALLS, f"bb{bb}: unknown call {m[2]}")
            name, assignments, output, operands = CALLS[m[2]]
            checked((s, m[1], m[3]) == (assignments, output, operands), f"bb{bb}: {m[2]} data flow differs")
            op = f".{name} {normal_target(body, m[4], m[5])}"
        elif m := re.fullmatch(r"switchInt\(move (_19|_25)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", t):
            checked(not s, f"bb{bb}: unexpected branch assignments")
            name = "branchAbort" if m[1] == "_19" else "branchWiden"
            op = f".{name} {normal_target(body, m[2])} {normal_target(body, m[3])}"
        elif m := re.fullmatch(r"falseUnwind -> \[real: bb(\d+), unwind: bb(\d+)\];", t):
            checked(not s, f"bb{bb}: unexpected loop entry")
            op = f".go {normal_target(body, m[1], m[2])}"
        elif m := re.fullmatch(r"goto -> bb(\d+);", t):
            checked(s in ([], ["_0 = copy _11;"]), f"bb{bb}: unknown data transfer {s}")
            name = "saveScore" if s else "go"
            op = f".{name} {normal_target(body, m[1])}"
        else:
            checked(not s and t == "return;", f"bb{bb}: unknown terminal {t}")
            op = ".returned"
        ops.append(op)
    return ops


@dataclass(frozen=True)
class Ref:
    place: str


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()

    def lean(self):
        if not self.args:
            return "." + self.op
        return "(." + self.op + " " + " ".join(arg.lean() for arg in self.args) + ")"


SCORE_OPS = {
    "<Score as PartialOrd>::le": ("le", 2),
    "<Score as PartialOrd>::ge": ("ge", 2),
    "<Score as Sub>::sub": ("sub", 2),
    "<Score as Add>::add": ("add", 2),
    "<Score as Neg>::neg": ("neg", 1),
    "<Score as Ord>::min": ("min", 2),
    "<Score as Ord>::max": ("max", 2),
}


def place(text, env):
    if re.fullmatch(r"_\d+", text):
        return text
    if m := re.fullmatch(r"\(\*(_\d+)\)", text):
        ref = env[m[1]]
        checked(isinstance(ref, Ref), f"dereference of {ref}")
        return ref.place
    raise ValueError(f"unknown place {text}")


def operand(text, env):
    if text in ("const true", "const false"):
        return text == "const true"
    if text == "const score::Score::INF":
        return Expr("inf")
    if m := re.fullmatch(r"&(.*)", text):
        return Ref(place(m[1], env))
    if m := re.fullmatch(r"(copy|move) (.*)", text):
        return env[place(m[2], env)]
    raise ValueError(f"unknown operand {text}")


def widen_paths(body):
    env = {"_1": Expr("score"), "_2": Ref("alpha"), "alpha": Expr("alpha"),
           "_3": Ref("beta"), "beta": Expr("beta"), "_4": Expr("delta")}
    visited = set()

    def walk(bb, env, guards, ancestors):
        checked(bb not in ancestors, f"widen loop at bb{bb}")
        visited.add(bb)
        ancestors = ancestors | {bb}
        block = body[bb]
        for statement in block.statements:
            m = re.fullmatch(r"(.+) = (.+);", statement)
            checked(m, f"unknown assignment {statement}")
            env[place(m[1], env)] = operand(m[2], env)
        t = block.end
        if t == "return;":
            checked(isinstance(env["_0"], bool), "non-boolean return")
            yield guards, env["alpha"], env["beta"], env["_0"]
        elif m := re.fullmatch(r"goto -> bb(\d+);", t):
            yield from walk(normal_target(body, m[1]), env, guards, ancestors)
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", t):
            test = env[m[1]]
            checked(isinstance(test, Expr) and test.op in ("le", "ge"), f"unknown comparison {test}")
            for decision, target in [(True, m[3]), (False, m[2])]:
                yield from walk(normal_target(body, target), env.copy(), guards + [(test, decision)], ancestors)
        elif m := split_call(t):
            checked(m[2] in SCORE_OPS, f"unknown widen operation {m[2]}")
            op, arity = SCORE_OPS[m[2]]
            args = [operand(arg, env) for arg in m[3].split(", ")]
            checked(len(args) == arity, f"bad arity for {m[2]}")
            if op in ("le", "ge"):
                checked(all(isinstance(arg, Ref) for arg in args), f"comparison expected references {args}")
                args = [env[arg.place] for arg in args]
            checked(all(isinstance(arg, Expr) and arg.op not in ("le", "ge") for arg in args), f"bad operands {args}")
            env[m[1]] = Expr(op, tuple(args))
            yield from walk(normal_target(body, m[4], m[5]), env, guards, ancestors)
        else:
            raise ValueError(f"unsupported widen terminator {t}")

    paths = list(walk(0, env, [], set()))
    for bb, block in body.items():
        checked(bb in visited or block == Block([], "resume;", True), f"outside modeled paths: bb{bb}")
    return paths


def score_constants(repo):
    endgame = (repo / "crates/reversi-core/src/search/endgame.rs").read_text(encoding="utf-8")
    constants = re.findall(r"^const ASPIRATION_DELTA: Score = [^;]+;", endgame, re.M)
    checked(len(constants) == 1, "missing or duplicate ASPIRATION_DELTA")
    score = repo / "crates/reversi-core/src/score.rs"
    source = f'#[path = r#"{score}"#] mod score;\nuse score::{{Score, DiscScore}};\n{constants[0]}\n'
    source += 'fn main() { println!("{} {}", ASPIRATION_DELTA.value(), Score::INF.value()); }\n'
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        fixture, binary = folder / "constants.rs", folder / "constants.exe"
        fixture.write_text(source, encoding="utf-8")
        subprocess.run(["rustc", "--edition=2024", "-Adead_code", str(fixture), "-o", str(binary)], cwd=repo, check=True)
        values = subprocess.check_output([str(binary)], text=True).split()
    checked(len(values) == 2, "expected delta and infinity")
    return tuple(map(int, values))


def render(dump, repo):
    def read(name, signature):
        path = dump / ("reversi_core." + name.replace("::", "-") + f".{PASS}.after.mir")
        return blocks(path, name, signature)

    program = driver(read(DRIVER, DRIVER_SIGNATURE))
    paths = widen_paths(read(WIDENER, WIDENER_SIGNATURE))
    delta, infinity = score_constants(repo)
    checked(256 <= delta <= 8192, f"ASPIRATION_DELTA is {delta}, but the termination proof "
            "(delta_bounds in AspirationMir/Termination.lean) needs one to 32 discs (256..8192)")
    rows = []
    for guards, alpha, beta, result in paths:
        conditions = ", ".join(f"({guard.lean()}, {str(value).lower()})" for guard, value in guards)
        rows.append(f"([{conditions}], {alpha.lean()}, {beta.lean()}, {str(result).lower()})")
    return "\n".join([
        "import NeuralReversi.AspirationMir.Syntax", "",
        "namespace NeuralReversi.AspirationMir", "",
        f"def initialDelta : Int := {delta}", f"def infinity : Int := {infinity}", "",
        "def driverCode : List DriverOp := [", "  " + ",\n  ".join(program) + "]", "",
        "def widenCode : List WidenPath := [", "  " + ",\n  ".join(rows) + "]", "",
        "end NeuralReversi.AspirationMir", "",
    ])


def extract(repo):
    with dump_mir(repo, "aspiration_search|widen_aspiration_window") as dump:
        return render(dump, repo)


def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="aspiration_search|widen_aspiration_window",
        check_message=lambda p: f"generated MIR program differs from {p}",
    )


if __name__ == "__main__":
    main()
