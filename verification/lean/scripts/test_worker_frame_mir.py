"""Run small counterfactual mutations against the actual emitted scheduler functions."""

from extract_worker_frame_mir import ANY_FIELD, KINDS, METHODS, PL, SUFFIX, TH, render
from mir_mutants import check_mir_projection, run_mir_test_cli


CASES = [
    (TH + "can_join", "_0 = AtomicBitSet::test(move _32, move _35)", "_0 = AtomicBitSet::test(move _35, move _32)"),
    (TH + "can_join", "_19 = Eq(move _20, const 0_usize);", "_19 = Ne(move _20, const 0_usize);"),
    (TH + "can_join", "_0 = const false;", "_0 = const true;"),
    (TH + "can_join", "_10 = &((*_1).20:", "_10 = &mut ((*_1).20:"),
    (TH + "can_join", "_11 = std::sync::atomic::Ordering::Relaxed;", "_11 = ExportAlias(move _10);"),
    (TH + "run_helper_job", "FakeRead(ForLet(None), _3);", "_0 = ExportGuard(move _3);"),
    (PL + "create_main_thread", "spawn::<", "store::<"),
    (TH + "abandon_active_split", "const true,", "const false,"),
    (TH + "run_helper_job-{impl#0}-drop", "const false,", "const true,"),
    # A new syntactic parent for the protected helper entry changes the generated graph.
    (TH + "can_join", "_24 = threading::thread::Thread::deepest_split_point(",
     "_24 = threading::thread::Thread::run_helper_job("),
]


def _include(path):
    source = path.read_text(encoding="utf-8")
    name = path.name.removesuffix(SUFFIX)
    return (
        name in KINDS
        or bool(ANY_FIELD.search(source))
        or "FinishJob::<" in source
        or any("Thread::" + method in source for method in METHODS)
        or name in (TH + "new", PL + "create_main_thread", PL + "create_worker_threads")
    )


def check(dump, expected):
    return check_mir_projection(
        dump, expected, render, SUFFIX, CASES, file_filter=_include, expected_errors=(ValueError, KeyError)
    )


if __name__ == "__main__":
    run_mir_test_cli(__doc__, check, "{count} mutations detected")
