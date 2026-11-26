"""Verify the split point's three atomic fields and their score and square representations."""

import argparse
from pathlib import Path
import re

from check_move_iterator_mir import RELAXED, call, compare, graph
from mir_common import checked


def check(dump, repo):
    suffix = ".SimplifyCfg-initial.after.mir"

    def mir(name, prefix="reversi_core.search-threading-split_point-{impl#0}-"):
        return (dump / (prefix + name + suffix)).read_text(encoding="utf-8").replace("\\", "/")

    for name, index, ty, construct in [
        ("alpha", 1, "i32", "Score::from_raw"),
        ("best_score", 2, "i32", "Score::from_raw"),
        ("best_move", 3, "u8", "Square::from_u8"),
    ]:
        member = f"&((*_1).{index}: std::sync::atomic::Atomic<{ty}>)"
        compare(name, graph(mir(name)), [
            [f"_3 = {member};", f"_4 = {RELAXED};", call(f"_2 = Atomic::<{ty}>::load(move _3, move _4)", 1, 3)],
            [call(f"_0 = {construct}(move _2)", 2, 3)], ["return;"], ["unwind"]])
        setter = mir("set_" + name)
        if ty == "i32":
            args = ["_5 = Score::value(move _6)"]
            order = "_7"
        else:
            inner = mir("set_best_move-{closure#0}")
            closure = re.search(r"\(_1: (\{closure@[^{}]+/split_point\.rs:\d+:\d+: \d+:\d+\}), _2: Square\)", inner)
            checked(closure, "best move encoder closure identity")
            compare("best move encoder", graph(inner), [
                ["_3 = copy _2;", "_4 = discriminant(_3);", "_0 = move _4 as u8 (IntToInt);", "return;"]])
            args = [f"_7 = {closure[1]};",
                    f"_5 = Option::<Square>::map_or::<u8, {closure[1]}>(move _6, const core::num::<impl u8>::MAX, move _7)"]
            order = "_8"
        compare("set_" + name, graph(setter), [
            [f"_4 = {member};", "_6 = copy _2;", *args[:-1], call(args[-1], 1, 3)],
            [f"{order} = {RELAXED};", call(f"_3 = Atomic::<{ty}>::store(move _4, move _5, move {order})", 2, 3)],
            ["_0 = const ();", "return;"], ["unwind"]])

    square = (repo / "crates/reversi-core/src/square.rs").read_text(encoding="utf-8")
    definition = re.search(r"#\[repr\(u8\)\]\s*(?:#\[[^\]]+\]\s*)*pub enum Square \{([^}]+)\}", square)
    checked(definition and [s.strip() for s in definition[1].split(",") if s.strip()] ==
            [f"{file}{rank}" for rank in range(1, 9) for file in "ABCDEFGH"], "unexpected square discriminants")
    paths = list(dump.glob("reversi_core.square-*-from_u8_unchecked" + suffix))
    checked(len(paths) == 1, "missing square decoder")
    body = graph(paths[0].read_text(encoding="utf-8"))
    checked(body[0] in ([f"_3 = const {flag};", "switchInt(move _3) -> [0: bb6, otherwise: bb1];"]
                        for flag in ("true", "false")) and
            body[1] == ["_6 = copy _1;", "_5 = Lt(move _6, const 64_u8);",
                        "switchInt(move _5) -> [0: bb3, otherwise: bb2];"] and
            body[2] == ["_4 = const ();", "_2 = const ();", "goto -> bb7;"] and
            body[6] == ["_2 = const ();", "goto -> bb7;"] and
            body[7] == ["_19 = copy _1;", call("_0 = std::intrinsics::transmute::<u8, Square>(move _19)", 8, 9)]
            and body[8] == ["return;"] and body[9] == ["unwind"], "decoder on a valid square")
    safe = list(dump.glob("reversi_core.square-*-from_u8" + suffix))
    checked(len(safe) == 1, "expected one checked square decoder")
    compare("Square::from_u8", graph(safe[0].read_text(encoding="utf-8")), [
        ["_3 = copy _1;", "_2 = Lt(move _3, const 64_u8);", "switchInt(move _2) -> [0: bb3, otherwise: bb1];"],
        ["_5 = copy _1;", call("_4 = Square::from_u8_unchecked(move _5)", 2, 5)],
        ["_0 = Option::<Square>::Some(move _4);", "goto -> bb4;"],
        ["_0 = Option::<Square>::None;", "goto -> bb4;"], ["return;"], ["unwind"]])

    score = "reversi_core.score-{impl#5}-"
    compare("Score::value", graph(mir("value", score)), [["_0 = copy (_1.0: i32);", "return;"]])
    make = graph(mir("from_raw", score))
    normal = [bb for bb, code in make.items() if "return;" in code]
    checked(len(normal) == 1 and re.fullmatch(r"(_\d+) = copy _1; _0 = Score\(move \1\); return;",
            " ".join(make[normal[0]])),
            "all normal Score::from_raw exits must return the original signed word")
    checked(all(not line.startswith("_1 =") and not re.search(r"&(?:raw )?mut _1\b", line)
                for block in make.values() for line in block), "Score::from_raw changes its raw input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    opts = parser.parse_args()
    check(opts.mir, opts.repo)
    print("verified split point alpha, best score and move words")
