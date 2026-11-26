"""Check EndGameStrategy's normal MIR returns at 0..4 empties and full depth."""

import re

from mir_common import (
    run_extractor_cli,
    PASS,
    parse_blocks as blocks,
    checked,
    normal_target,
    split_call,
)


def constant(dump, name):
    path = dump / f"reversi_core.search-endgame-{name}.built.after.mir"
    pattern = (rf"// MIR for `search::endgame::{name}` after built\s+"
               rf"const {name}: u32 = \{{\s+let mut _0: u32;\s+bb0: \{{\s+"
               r"_0 = const (\d+)_u32;\s+return;\s*\}\s*\}\s*")
    match = re.fullmatch(pattern, path.read_text(encoding="utf-8"))
    checked(match, f"unsupported MIR for constant {name}")
    return int(match[1])


def read_body(dump, repo):
    source = (repo / "crates/reversi-core/src/search/strategy.rs").read_text(encoding="utf-8")
    lines = [i for i, line in enumerate(source.splitlines(), 1)
             if line == "impl SearchStrategy for EndGameStrategy {"]
    checked(len(lines) == 1, "ambiguous EndGameStrategy impl")
    signature = ("(_1: &mut SearchContext<'_>, _2: &Board, _3: u32, _4: Score, _5: Score, "
                 "_6: &Arc<threading::thread::Thread>) -> Score {")
    pattern = (r"// MIR for `search::(strategy::<impl at crates[\\/]reversi-core[\\/]src[\\/]"
               rf"search[\\/]strategy.rs:{lines[0]}:1: {lines[0]}:40>::shallow_search)` after {PASS}")
    candidates = []
    for path in dump.glob(f"reversi_core.search-strategy-*-shallow_search.{PASS}.after.mir"):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        if match := re.fullmatch(pattern, header):
            name = match[1]
            candidates.append(blocks(path, "search::" + name, "fn " + name + signature))
    checked(len(candidates) == 1, "missing or ambiguous shallow_search MIR")
    return candidates[0]


EMPTIES = ("emptiesOf", "board")

# Each supported call is a leaf in the extracted path. Receivers and arguments
# are symbolic terms. This preserves the actual argument order at every use.
CALLS = {
    "Score::to_disc_diff": "discAlpha",
    "<Arc<threading::thread::Thread> as Deref>::deref": "worker",
    "threading::thread::Thread::endgame_caches": "caches",
    "Empties::new": "emptiesOf",
    "Empties::count": "count",
    "Board::empty_squares": "emptySquares",
    "odd_quadrants": "odd",
    "Board::final_disc_diff": "terminal",
    "Board::player": "player",
    "Board::legal_moves": "legalMoves",
    "shallow_search::<5>": "shallow5",
    "shallow_search::<6>": "shallow6",
    "Empties::first::<1>": "first",
    "Empties::first::<2>": "first",
    "Empties::first::<3>": "first",
    "sort_last4": "sort4",
    "solve::solve1": "solve1",
    "solve2": "solve2",
    "solve3": "solve3",
    "solve4": "solve4",
    "Score::from_disc_diff": "scaled",
}


def path_result(body, count, nws_depth, shallow_depth, extra=None):
    env = dict(_1="ctx", _2="board", _3=count, _4="alpha", _5="beta", _6="thread")
    observed = []

    def local(name):
        checked(name in env, f"use of undefined local {name}")
        return env[name]

    def get(source):
        if extra is not None:
            handled, value = extra.operand(source, local, observed)
            if handled:
                return value
        if m := re.fullmatch(r"(?:copy|move) (_\d+)", source):
            return local(m[1])
        if m := re.fullmatch(r"&(?:mut )?\(\*(_\d+)\)", source):
            return local(m[1])
        if source == "copy ((*_1).2: u64)":
            return "nodes"
        if m := re.fullmatch(r"&mut \(\(\*(_\d+)\)\.1: search::endgame::cache::EndGameCache<search::endgame::cache::Shallow>\)", source):
            checked(local(m[1]) == ("caches", ("worker", "thread")), "unexpected cache owner")
            return ("shallowCache", local(m[1]))
        if m := re.fullmatch(r"copy \((_\d+)\.([0-3]): square::Square\)", source):
            checked(local(m[1])[0] == "sort4", "unexpected tuple projection")
            return ("sorted", int(m[2]))
        if m := re.fullmatch(r"copy (_\d+)\[(\d+) of (\d+)\]", source):
            checked(local(m[1]) == ("first", EMPTIES) and int(m[3]) == count, "unexpected first-squares projection")
            return ("square", int(m[2]))
        if source == "Option::<TTSlot<'_>>::None":
            return "none"
        if m := re.fullmatch(r"discriminant\((_\d+)\)", source):
            checked(local(m[1]) == "none", "unexpected discriminant")
            return 0
        if m := re.fullmatch(r"(Eq|Gt)\(move (_\d+), const search::endgame::(DEPTH_TO_NWS|DEPTH_TO_SHALLOW_SEARCH)\)", source):
            val = local(m[2])
            limit = {"DEPTH_TO_NWS": nws_depth, "DEPTH_TO_SHALLOW_SEARCH": shallow_depth}[m[3]]
            checked(type(val) is int, "comparison did not read depth or empty count")
            return int(val == limit if m[1] == "Eq" else val > limit)
        raise ValueError(f"unknown operand {source}")

    bb = 0
    visited = set()
    while True:
        checked(bb not in visited, f"loop on fixed-size path at bb{bb}")
        visited.add(bb)
        block = body[bb]
        for statement in block.statements:
            if mention := re.fullmatch(r"PlaceMention\((_\d+)\);", statement):
                local(mention[1])
                continue
            assign = re.fullmatch(r"(_\d+) = (.+);", statement)
            checked(assign, f"unexpected statement at bb{bb}: {statement}")
            env[assign[1]] = get(assign[2])
        term = block.end
        if term == "return;":
            return local("_0"), observed
        if m := split_call(term):
            name = (extra.calls.get(m[2]) if extra is not None else None) or CALLS.get(m[2])
            checked(name is not None, f"unexpected call at bb{bb}: {m[2]}")
            call = (name, *(get(arg) for arg in m[3].split(", ")))
            observed.append(call)
            env[m[1]] = (count if call == ("count", EMPTIES) else
                         extra.response(call) if extra is not None else call)
            bb = normal_target(body, m[4], m[5])
        elif m := re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];", term):
            bb = normal_target(body, m[1] or m[2])
        elif m := re.fullmatch(r"switchInt\((copy|move) (_\d+)\) -> \[([^\]]+)\];", term):
            cases = [re.fullmatch(r"(\d+|otherwise): bb(\d+)", case) for case in m[3].split(", ")]
            checked(all(cases), f"malformed switch at bb{bb}")
            targets = {case[1]: case[2] for case in cases}
            checked(len(targets) == len(cases) and "otherwise" in targets, "malformed switch arms")
            value = local(m[2])
            checked(type(value) is int, f"unknown switch value {value}")
            bb = normal_target(body, targets.get(str(value), targets["otherwise"]))
        elif extra is not None and (next_bb := extra.checked_target(body, term, local)) is not None:
            bb = next_bb
        else:
            raise ValueError(f"unsupported terminator at bb{bb}: {term}")


def expected(count):
    alpha = ("discAlpha", "alpha")
    worker = ("worker", "thread")
    calls = [alpha, worker, ("caches", worker), EMPTIES, ("count", EMPTIES)]
    if count == 0:
        result = ("terminal", "board")
    elif count in (5, 6):
        moves = ("legalMoves", "board")
        calls.append(moves)
        result = (f"shallow{count}", "ctx", "board", alpha, ("shallowCache", ("caches", worker)), moves, EMPTIES)
    elif count == 4:
        odd = ("odd", ("emptySquares", "board"))
        calls.extend([("emptySquares", "board"), odd, ("sort4", EMPTIES, odd)])
        result = ("solve4", "ctx", "board", alpha, *(("sorted", i) for i in range(4)))
    else:
        calls.append(("first", EMPTIES))
        squares = [("square", i) for i in range(count)]
        board = "board"
        if count == 1:
            board = ("player", "board")
            calls.append(board)
        result = (f"solve{count}", "ctx", board, alpha, *squares)
    scaled = ("scaled", result)
    return scaled, [*calls, result, scaled]


def verify(dump, repo):
    body = read_body(dump, repo)
    nws_depth = constant(dump, "DEPTH_TO_NWS")
    shallow_depth = constant(dump, "DEPTH_TO_SHALLOW_SEARCH")
    arms = []
    for n in range(5):
        actual = path_result(body, n, nws_depth, shallow_depth)
        checked(actual == expected(n), f"{n} empties: MIR call sequence or return differs: {actual!r}")
        arms.append(actual[0][1][0])
    return "\n".join(["namespace NeuralReversi.EndgameSearch.Direct.Mir", "",
                      "inductive Arm where", "  | terminal | solve1 | solve2 | solve3 | solve4", "",
                      "def dispatch : Nat → Option Arm",
                      *(f"  | {n} => some .{name}" for n, name in enumerate(arms)),
                      "  | _ => none", "", "end NeuralReversi.EndgameSearch.Direct.Mir", ""])


def main():
    run_extractor_cli(
        __doc__,
        verify,
        filters="shallow_search|DEPTH_TO_SHALLOW_SEARCH|DEPTH_TO_NWS",
        check_message="generated fixed-leaf program differs",
        check_ok_message="MIR fixed endgame returns verified for 0, 1, 2, 3, 4 empties at full depth",
        exclude_alloc_bytes=False,
    )


if __name__ == "__main__":
    main()

