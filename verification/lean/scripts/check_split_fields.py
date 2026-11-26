"""Match split-point field loads, stores, initialization and the final owner reads to rustc MIR."""

import re

from check_split_field_words import check as check_words
from mir_common import checked
from extract_split_activation_mir import drop_of, field, lock
from extract_split_scheduler_mir import Body, owner_fields


def check(dump, repo):
    check_words(dump, repo)
    init = Body(dump, "initialize_split_point", ["thread", "slot", "ctx", "req"])
    rust = (repo / "crates/reversi-core/src/search/threading/thread.rs").read_text(encoding="utf-8")
    request = re.search(r"pub struct SplitRequest<'a> \{([^}]+)\}", rust)
    checked(request and re.findall(r"\bpub (\w+):", request[1]) ==
            ["board","alpha","beta","best_score","best_move","depth","move_list","move_count","node_type","cut_node"],
            "split request field map")
    state = ("SplitPoint::state", "slot")
    ctor = "ConcurrentMoveIterator::from_offset"
    init.call(ctor, [field("req", 6), field("req", 7)], "iterator")
    pc, guard = lock(init, "SplitPoint::lock", "slot")
    init.marks[pc] = ["lock"]
    init.call("SplitPoint::set_move_iter", ["slot", (ctor, field("req", 6), field("req", 7))], "install")
    for name, index in [("best_score", 3), ("best_move", 4), ("alpha", 1)]:
        init.call("SplitPointState::set_" + name, [state, field("req", index)], name)
    for pc in drop_of(init, guard):
        init.marks[pc] = ["unlock"]
    checked(init.projection() == ["iterator", "lock", "install", "best_score", "best_move", "alpha", "unlock"],
            "split request initialization and lock scope")
    owner, _ = owner_fields(dump)
    checked(owner == ["init", "idle", "finalize", "best", "picked", "returned"], "split return order")
