"""Exercise the slot selection, field operands, order and return tuple of Thread::split."""

from extract_split_scheduler_mir import owner_fields
from mir_mutants import patched_mir, replace_once


def check(dump):
    owner = "split"
    init = "initialize_split_point"
    cases = [
        (owner, [("_50 = move _42;", "_50 = move _46;")]),
        (owner, [("_0 = (move _46, move _48,", "_0 = (move _46, move _46,")]),
        (owner, [("_49 = &(*_31);", "_49 = &(*_1);")]),
        (owner, [("take_counters_after_finished(move _43) -> [return: bb18", "take_counters_after_finished(move _43) -> [return: bb19"),
                 ("best_move(move _49) -> [return: bb20", "best_move(move _49) -> [return: bb18"),
                 ("best_score(move _47) -> [return: bb19", "best_score(move _47) -> [return: bb20")]),
        (init, [("_72 = copy (_4.3: score::Score);", "_72 = copy (_4.1: score::Score);")]),
        (init, [("_75 = copy (_4.4: std::option::Option<square::Square>);", "_75 = Option::<Square>::None;")]),
        (init, [("_78 = copy (_4.1: score::Score);", "_78 = copy (_4.3: score::Score);")]),
        (init, [("set_best_score(move _71, move _72) -> [return: bb34", "set_best_score(move _71, move _72) -> [return: bb35")]),
    ]
    expected = owner_fields(dump)
    sources = {}
    for name in (owner, init):
        source, = dump.glob("*-" + name + ".SimplifyCfg-initial.after.mir")
        sources[name] = (source, source.read_text(encoding="utf-8"))
    for name, edits in cases:
        path, changed = sources[name]
        for old, new in edits:
            changed = replace_once(changed, old, new, label=name)
        with patched_mir(path, changed):
            try:
                if owner_fields(dump) == expected:
                    raise AssertionError(f"accepted {name} mutation: {edits}")
            except ValueError:
                pass
    return len(cases)
