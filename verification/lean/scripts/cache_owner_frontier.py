"""Enumerate direct callers in a full default-feature SimplifyCfg-initial dump."""

import mmap
import re

from cache_owner_mir import PASS, POOL, THREAD, WORKER
from mir_common import checked, iter_mir_texts
from cache_owner_queue import EXECUTE, LAUNCH


_FRONTIER_MARKERS = (
    ".13: std::cell::SyncUnsafeCell<search::endgame::EndGameCaches>",
    ".6: std::sync::Mutex<std::option::Option<std::sync::Arc<dyn for",
    " -> ",
    "PointerCoercion(ReifyFnPointer",
)


def lines(path):
    # The embedded compressed networks form individual byte-string lines longer than 300 MB.
    with path.open("rb") as file, mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as src:
        pos = 0
        while pos < len(src):
            end = src.find(b"\n",pos)
            if end < 0:
                end = len(src)
            if end-pos > 32768:
                checked(re.match(rb"        _\d+ = const b\"",src[pos:pos+48]) and
                        src[end-2:end] == b'\";', f"{path.name}: unrecognized long MIR line")
            else:
                yield src[pos:end].decode("utf-8").rstrip("\r")
            pos = end+1


def inspect(stem, source, targets, nodes):
    external = {
        (LAUNCH,"ThreadPool::start_helpers"),
        (POOL+"create_main_thread-{closure#0}",THREAD+"::main_thread_loop"),
        (POOL+"create_worker_threads-{closure#0}",THREAD+"::idle_loop"),
        (POOL+"start_helpers",THREAD+"::book_helper_job"),
        (EXECUTE,"ThreadPool::start_thinking"),
    }
    auxiliary = {f for _,f in external}
    def sensitive(fn):
        names = [fn]
        for prefix in ("search::endgame::", "endgame::", "search::pvs::", "pvs::", "search::"):
            if fn.startswith(prefix):
                names.append(fn.removeprefix(prefix))
        if "::endgame_caches" in fn:
            return True
        return any(targets(name) or name in auxiliary for name in names)
    observed = set()
    normal = False
    for line in source:
        if line.startswith("    bb"):
            normal = bool(re.fullmatch(r"    bb\d+: \{",line))
        if not normal or not line.startswith("        "):
            continue
        if ".13: std::cell::SyncUnsafeCell<search::endgame::EndGameCaches>" in line:
            checked(stem==WORKER+"endgame_caches",f"{stem}: untracked direct cache projection")
        if ".6: std::sync::Mutex<std::option::Option<std::sync::Arc<dyn for" in line:
            checked(stem in {WORKER+"idle_loop",WORKER+"book_helper_job"},f"{stem}: untracked helper mailbox")
        if " -> " not in line:
            if "PointerCoercion(ReifyFnPointer" in line:
                rhs = line.split(" = ",1)[-1].split(" as ",1)[0]
                checked(not sensitive(rhs),f"{stem}: indirect protected function {rhs}")
            continue
        # Include normal calls, diverging calls and tail calls irrespective of unwind edge.
        m = re.match(r"        (?:_\d+ = |tailcall )(.+?)\((?:move |copy |const |\))",line)
        if not m:
            continue
        fn = m[1]
        if sensitive(fn):
            if stem not in nodes:
                checked((stem,fn) in external,f"{stem}: untracked caller of {fn}")
                observed.add((stem,fn))
            else:
                checked(targets(fn) is not None,f"{stem}: call missing from receiver projection: {fn}")
    return observed


def frontier(dump, targets, nodes):
    files = list(iter_mir_texts(dump, PASS, prefix="reversi_core."))
    names = {p.name.removeprefix("reversi_core.").removesuffix(PASS) for p, _ in files}
    required = {"board-{impl#1}-make_move", "eval-{impl#0}-with_weight_files", "probcut-MIDGAME_TABLE"}
    checked(required <= names and set(nodes) <= names, "ownership frontier needs whole-crate MIR")
    seen = set()
    for path, text in files:
        if not any(marker in text for marker in _FRONTIER_MARKERS):
            continue
        stem = path.name.removeprefix("reversi_core.").removesuffix(PASS)
        seen.update(inspect(stem, text.splitlines(), targets, nodes))
    checked(len(seen) == 5, "external engine entry set changed")
