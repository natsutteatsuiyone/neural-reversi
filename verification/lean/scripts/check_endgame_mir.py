"""Regenerate the endgame proof inputs with the workspace compiler and compare them."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path

from mir_common import PASS, dump_mir, preload_mir_texts, start_before_mir_cleanup

import check_cache_owner_mutations
import check_deep_recursion_mir_regression
import check_endgame_order_mir_regression
import check_fixed_leaf_mir_regression
import check_midgame_children_mir
import check_midgame_handoff_mir
import check_move_iterator_mir
import check_cutoff_poll_mir
import check_move_generation_mir
import check_result_guards_mir
import check_outer_abort_mir
import check_outer_state
import check_serial_move_list_mir
import check_serial_leaf_restore_mir
import check_serial_child_calls_mir
import check_serial_score_words_mir
import check_serial_table_slot_mir
import check_endgame_cache_cutoff_mir
import check_endgame_probcut_mir
import check_serial_worker_mir
import check_serial_level_mir
import check_native_worker_feed_mutants
import check_pvs_context_mir_mutants
import check_shallow_recursion_mir_regression
import extract_aspiration_mir
import extract_cache
import extract_cache_budgets
import check_cache_call_sites
import extract_cache_effect_sites
import extract_cache_owner
import extract_deep_recursion_mir
import extract_dispatch_origin_mir
import extract_endgame_order_mir
import extract_fixed_leaf_mir
import extract_frame
import extract_general_nonpv_mir
import extract_general_owner_mir
import extract_guard_receiver_mir
import extract_header
import extract_native_worker_feed
import extract_nonpv_leaf_mir
import extract_nws_leaf_mir
import extract_outer
import extract_owner_sites
import extract_pool_roster_mir
import extract_pv_entry_mir
import extract_pv_serial_mir
import extract_publication_mir
import extract_pvs_context_mir
import extract_root_wipeout
import extract_root_birth
import extract_root_rows
import extract_stop_returns
import extract_root_update
import extract_shallow_recursion_mir
import extract_single
import extract_slot_frame_mir
import extract_split_activation_mir
import extract_split_rebooking_mir
import extract_split_worker_mir
import extract_worker_poll_mir
import check_worker_poll_mir_mutants
import check_stop_returns_mutants
import extract_split_scheduler_mir
import check_split_owner_mutants
import check_split_spinlock
import extract_tt_reader_mir
import extract_tt_probe_mir
import extract_tt_lookup_mir
import extract_tt_duplicate
import extract_tt_save
import extract_tt_store
import extract_tt_writer
import extract_worker_frame_mir
import test_dispatch_origin_mir
import test_guard_receiver_mir
import test_pool_roster_mir
import test_slot_frame_mir
import test_worker_frame_mir


# Values that the Lean proofs case-split on. Tuning one of them needs proof changes, not only --write.
PROOF_DEPTHS = {"DEPTH_TO_NWS": 12, "DEPTH_TO_SHALLOW_SEARCH": 6}
# EndGameStrategy thresholds, as offsets from DEPTH_TO_NWS.
PROOF_OFFSETS = {"MIN_PROBCUT_DEPTH": 1, "MIN_ETC_DEPTH": 2}
FAST_ORDER_FULL_DEPTH_EMPTIES = 13


def check_proof_depths(dump, repo):
    for name, value in PROOF_DEPTHS.items():
        actual = extract_fixed_leaf_mir.constant(dump, name)
        if actual != value:
            raise SystemExit(f"{name} is {actual}, but the Lean endgame proofs assume {value}")
    for name, offset in PROOF_OFFSETS.items():
        actual = extract_general_nonpv_mir.plus_threshold(dump, repo, name)
        if actual != PROOF_DEPTHS["DEPTH_TO_NWS"] + offset:
            raise SystemExit(f"{name} is {actual}, but the Lean endgame proofs assume DEPTH_TO_NWS + {offset}")
    fast = [n for n, depth in enumerate(extract_endgame_order_mir.endgame_sort_depths(dump)) if n < depth]
    if fast != list(range(FAST_ORDER_FULL_DEPTH_EMPTIES + 1)):
        raise SystemExit("ENDGAME_MIN_SORT_DEPTH changes where full-depth searches use the fast move order, "
                         f"but the Lean proofs assume it through {FAST_ORDER_FULL_DEPTH_EMPTIES} empties")


_WORKER_SEEDED_REPO = None


def _warm_rustc_probes(pool, repo):
    """Launch the 5 repo-only rustc probes so they overlap with `dump_mir`."""
    return (
        pool.submit(extract_root_wipeout.constants, repo),
        pool.submit(extract_single.check_score_operators, repo),
        pool.submit(extract_header.specialization, repo),
        pool.submit(extract_aspiration_mir.score_constants, repo),
        pool.submit(check_move_iterator_mir.compile_atomic, repo),
    )


def _collect_probe_state(repo, probe_futures):
    consts_fut, ops_fut, spec_fut, asp_fut, atomic_fut = probe_futures
    consts = consts_fut.result()
    ops_fut.result()
    spec_fut.result()
    asp_consts = asp_fut.result()
    atomic_fut.result()
    return (repo, consts, asp_consts)


def _apply_probe_state(probe_state):
    global _WORKER_SEEDED_REPO
    repo, consts, asp_consts = probe_state
    if _WORKER_SEEDED_REPO == repo:
        return
    extract_root_wipeout.seed_constants(repo, consts)
    extract_single.seed_score_operators(repo)
    extract_header.seed_specialization(repo)
    extract_aspiration_mir.seed_score_constants(repo, asp_consts)
    check_move_iterator_mir.seed_atomic_checked(repo)
    _WORKER_SEEDED_REPO = repo


def _worker_warmup():
    return True


def _run_pre_task(task_id, dump, repo, probe_state):
    _apply_probe_state(probe_state)
    if task_id == "proof_depths":
        check_proof_depths(dump, repo)
        return None
    if task_id == "iterator":
        check_move_iterator_mir.check_iterator(dump)
        check_move_iterator_mir.compile_atomic(repo)
        return "verified concurrent move iterator and relaxed try_update"
    if task_id == "spinlock":
        check_split_spinlock.check(dump)
        return f"checked split mutex acquisition, guard drop, and {check_split_spinlock.check_mutants(dump)} MIR mutants"
    if task_id == "cutoff_poll":
        check_cutoff_poll_mir.check(dump)
        return None
    if task_id == "midgame_children":
        return f"checked midgame recursive calls and rejected {check_midgame_children_mir.check(dump, repo)} MIR mutants"
    if task_id == "midgame_handoff":
        return f"checked timed midgame handoff and rejected {check_midgame_handoff_mir.check(dump)} MIR mutants"
    if task_id == "result_guards":
        check_result_guards_mir.check(dump)
        return f"checked public result guards and rejected {check_result_guards_mir.check_mutants(dump)} MIR mutants"
    if task_id == "outer_abort":
        cases = check_outer_abort_mir.check(dump)
        mutants = check_outer_abort_mir.check_mutants(dump)
        return f"checked {cases} polled outer cases and rejected {mutants} MIR mutants"
    if task_id == "outer_state":
        check_outer_state.check(dump)
        return (f"checked completion-state construction and rejected "
                f"{check_outer_state.check_mutants(dump)} MIR mutants")
    if task_id == "move_generation":
        check_move_generation_mir.check(dump)
        return f"checked move generation and rejected {check_move_generation_mir.check_mutants(dump)} MIR mutants"
    if task_id == "serial_leaf_restore":
        return f"checked fixed solver context effects and rejected {check_serial_leaf_restore_mir.check(dump)} MIR mutants"
    if task_id == "serial_child_calls":
        return f"checked serial recursive calls and rejected {check_serial_child_calls_mir.check(dump, repo)} MIR mutants"
    if task_id == "serial_score_words":
        return f"checked signed recursive scores and rejected {check_serial_score_words_mir.check(dump)} MIR mutants"
    if task_id == "serial_table_slot":
        return f"checked retained PVS and ETC table slots and rejected {check_serial_table_slot_mir.check(dump)} MIR mutants"
    if task_id == "endgame_cache_cutoff":
        return f"checked endgame_cache_cutoff and fail_low_score MIR and rejected {check_endgame_cache_cutoff_mir.check(dump)} MIR mutants"
    if task_id == "endgame_probcut":
        return f"checked selective ProbCut and rejected {check_endgame_probcut_mir.check(dump, repo)} MIR mutants"
    if task_id == "serial_level":
        return f"checked selectivity guards in serial thirteen-empty searches and rejected {check_serial_level_mir.check(dump, repo)} MIR mutants"
    if task_id == "serial_move_list":
        return f"rejected {check_serial_move_list_mir.check_mutants(dump)} serial move-list MIR mutants"
    raise ValueError(f"unknown pre task: {task_id}")


_EXTRACTOR_ORDER = (
    "AspirationMir/Program.lean",
    "AspirationMir/NativeRootBirth/Program.lean",
    "PvsParallel/RootStorageMir/Program.lean",
    "PvsParallel/NativeStopReturns/Program.lean",
    "OuterMir/Program.lean",
    "OuterMir/PolledProgram.lean",
    "OuterMir/CompletionProgram.lean",
    "EndgameSearch/Direct/Mir/Program.lean",
    "EndgameSearch/CacheMir/Program.lean",
    "EndgameSearch/CacheMir/Budgets.lean",
    "EndgameSearch/CacheOwner/Program.lean",
    "EndgameSearch/CacheHistory/NativeTrace/Sites.lean",
    "EndgameSearch/ShallowMir/Program.lean",
    "EndgameSearch/DeepRecursionMir/Program.lean",
    "EndgameSearch/Thresholds.lean",
    "PvsParallel/NonPvMir/Program.lean",
    "PvsParallel/DeepLeafMir/Program.lean",
    "PvsParallel/PvEntryMir/Program.lean",
    "PvsParallel/PvSerialMir/Program.lean",
    "PvsParallel/GeneralNonPvMir/Program.lean",
    "PvsParallel/NativeFrames/OwnerSites/Program.lean",
    "PvsParallel/GeneralOwnerMir/Program.lean",
    "PvsParallel/EndgameOrderMir/Program.lean",
    "MoveList/OrderWeights.lean",
    "PvsParallel/SerialTable/MidgameShallow/Masks.lean",
    "PvsParallel/ContextPathMir/Program.lean",
    "PvsParallel/WipeoutExit.lean",
    "PvsParallel/SingleExit.lean",
    "PvsParallel/RootFrameMir/Program.lean",
    "PvsParallel/RootHeaderMir/Program.lean",
    "PvsParallel/RootUpdateMir/Program.lean",
    "PvsParallel/SplitWorkerMir/Program.lean",
    "PvsParallel/WorkerPollMir/Program.lean",
    "PvsParallel/SplitWorkerMir/FieldProgram.lean",
    "PvsParallel/NativeWorkerFeed/Program.lean",
    "PvsParallel/SplitSchedulerMir/Program.lean",
    "PvsParallel/SplitActivationMir/Program.lean",
    "PvsParallel/SplitRebookingMir/Program.lean",
    "PvsParallel/WorkerFrameMir/Program.lean",
    "PvsParallel/SlotFrameMir/Program.lean",
    "PvsParallel/SlotFrameMir/Pool.lean",
    "PvsParallel/SlotFrameMir/DispatchProgram.lean",
    "PvsParallel/SlotFrameMir/ReceiverProgram.lean",
    "PublicationMir/Program.lean",
    "TranspositionTable/ReaderMir/Program.lean",
    "TranspositionTable/ReaderMir/ProbeProgram.lean",
    "TranspositionTable/ReaderMir/Lookup.lean",
    "TranspositionTable/Duplicate/Program.lean",
    "TranspositionTable/WriterMir/Program.lean",
    "TranspositionTable/WriterMir/Origins.lean",
    "TranspositionTable/WriterMir/Store.lean",
)

# Extractors that scan the entire crate-wide SimplifyCfg-initial dump; run on the main
# process so they share the single preloaded in-memory MIR cache without re-reading huge files.
_CRATE_WIDE_EXTRACTORS = {
    "PvsParallel/WorkerFrameMir/Program.lean",
    "PvsParallel/SlotFrameMir/Program.lean",
    "PvsParallel/SlotFrameMir/Pool.lean",
    "PvsParallel/SlotFrameMir/DispatchProgram.lean",
    "PvsParallel/SlotFrameMir/ReceiverProgram.lean",
}


def _run_extractor_task(name, dump, repo, probe_state):
    _apply_probe_state(probe_state)
    pvs = dump / extract_root_wipeout.FILENAME
    reduction = dump / "reversi_core.search-pvs-compute_lmr_reduction.SimplifyCfg-initial.after.mir"
    dispatch = {
        "AspirationMir/Program.lean": lambda: extract_aspiration_mir.render(dump, repo),
        "AspirationMir/NativeRootBirth/Program.lean": lambda: extract_root_birth.render(dump, repo),
        "PvsParallel/RootStorageMir/Program.lean": lambda: extract_root_rows.render(dump, repo),
        "PvsParallel/NativeStopReturns/Program.lean": lambda: extract_stop_returns.render(dump),
        "OuterMir/Program.lean": lambda: extract_outer.render(dump, repo),
        "OuterMir/PolledProgram.lean": lambda: extract_outer.render_polled(dump, repo),
        "OuterMir/CompletionProgram.lean": lambda: check_outer_state.render_completion(dump, repo),
        "EndgameSearch/Direct/Mir/Program.lean": lambda: extract_fixed_leaf_mir.verify(dump, repo),
        "EndgameSearch/CacheMir/Program.lean": lambda: extract_cache.render(dump, repo),
        "EndgameSearch/CacheMir/Budgets.lean": lambda: extract_cache_budgets.render(dump, repo),
        "EndgameSearch/CacheOwner/Program.lean": lambda: extract_cache_owner.render(dump, repo),
        "EndgameSearch/CacheHistory/NativeTrace/Sites.lean": lambda: extract_cache_effect_sites.render(dump, repo),
        "EndgameSearch/ShallowMir/Program.lean": lambda: extract_shallow_recursion_mir.verify(dump, repo),
        "EndgameSearch/DeepRecursionMir/Program.lean": lambda: extract_deep_recursion_mir.verify(dump, repo),
        "EndgameSearch/Thresholds.lean": lambda: extract_deep_recursion_mir.render_thresholds(dump),
        "PvsParallel/NonPvMir/Program.lean": lambda: extract_nonpv_leaf_mir.verify(dump, repo),
        "PvsParallel/DeepLeafMir/Program.lean": lambda: extract_nws_leaf_mir.verify(dump, repo),
        "PvsParallel/PvEntryMir/Program.lean": lambda: extract_pv_entry_mir.verify(dump, repo),
        "PvsParallel/PvSerialMir/Program.lean": lambda: extract_pv_serial_mir.verify(dump, repo),
        "PvsParallel/GeneralNonPvMir/Program.lean": lambda: extract_general_nonpv_mir.verify(dump, repo),
        "PvsParallel/NativeFrames/OwnerSites/Program.lean": lambda: extract_owner_sites.render(dump),
        "PvsParallel/GeneralOwnerMir/Program.lean": lambda: extract_general_owner_mir.verify(dump, repo),
        "PvsParallel/EndgameOrderMir/Program.lean": lambda: extract_endgame_order_mir.verify(dump, repo),
        "MoveList/OrderWeights.lean": lambda: extract_endgame_order_mir.render_weights(dump),
        "PvsParallel/SerialTable/MidgameShallow/Masks.lean": lambda: check_midgame_children_mir.render_masks(dump),
        "PvsParallel/ContextPathMir/Program.lean": lambda: extract_pvs_context_mir.verify(dump, repo),
        "PvsParallel/WipeoutExit.lean": lambda: extract_root_wipeout.render(pvs, repo),
        "PvsParallel/SingleExit.lean": lambda: extract_single.render(pvs, repo),
        "PvsParallel/RootFrameMir/Program.lean": lambda: extract_frame.render(pvs, repo),
        "PvsParallel/RootHeaderMir/Program.lean": lambda: extract_header.render(pvs, reduction, repo),
        "PvsParallel/RootUpdateMir/Program.lean": lambda: extract_root_update.render(dump, repo),
        "PvsParallel/SplitWorkerMir/Program.lean": lambda: extract_split_worker_mir.render(dump, repo),
        "PvsParallel/WorkerPollMir/Program.lean": lambda: extract_worker_poll_mir.render(dump, repo),
        "PvsParallel/SplitWorkerMir/FieldProgram.lean": lambda: extract_split_worker_mir.render(dump, repo, fields=True),
        "PvsParallel/NativeWorkerFeed/Program.lean": lambda: extract_native_worker_feed.verify(dump, repo),
        "PvsParallel/SplitSchedulerMir/Program.lean": lambda: extract_split_scheduler_mir.render(dump, repo),
        "PvsParallel/SplitActivationMir/Program.lean": lambda: extract_split_activation_mir.render(dump),
        "PvsParallel/SplitRebookingMir/Program.lean": lambda: extract_split_rebooking_mir.render(dump, repo),
        "PvsParallel/WorkerFrameMir/Program.lean": lambda: extract_worker_frame_mir.render(dump),
        "PvsParallel/SlotFrameMir/Program.lean": lambda: extract_slot_frame_mir.render(dump),
        "PvsParallel/SlotFrameMir/Pool.lean": lambda: extract_pool_roster_mir.render(dump),
        "PvsParallel/SlotFrameMir/DispatchProgram.lean": lambda: extract_dispatch_origin_mir.render(dump),
        "PvsParallel/SlotFrameMir/ReceiverProgram.lean": lambda: extract_guard_receiver_mir.render(dump),
        "PublicationMir/Program.lean": lambda: extract_publication_mir.render(dump, repo),
        "TranspositionTable/ReaderMir/Program.lean": lambda: extract_tt_reader_mir.render(dump, repo),
        "TranspositionTable/ReaderMir/ProbeProgram.lean": lambda: extract_tt_probe_mir.render(dump, repo),
        "TranspositionTable/ReaderMir/Lookup.lean": lambda: extract_tt_lookup_mir.render(dump, repo),
        "TranspositionTable/Duplicate/Program.lean": lambda: extract_tt_duplicate.render(dump, repo),
        "TranspositionTable/WriterMir/Program.lean": lambda: extract_tt_writer.render(dump, repo),
        "TranspositionTable/WriterMir/Origins.lean": lambda: extract_tt_save.render(dump, repo),
        "TranspositionTable/WriterMir/Store.lean": lambda: extract_tt_store.render(dump, repo),
    }
    return dispatch[name]()


def _run_post_task(task_id, dump, repo, probe_state, baseline=None):
    _apply_probe_state(probe_state)
    if task_id == "owner_sites":
        return f"checked {extract_owner_sites.check_mutants(dump)} recursive call-site mutants"
    if task_id == "cache_call_sites":
        return f"checked {check_cache_call_sites.check(dump)} endgame cache call-site mutants"
    if task_id == "tt_probe":
        return f"checked {extract_tt_probe_mir.check_mutants(dump)} cluster probe MIR mutants"
    if task_id == "tt_duplicate":
        return f"checked {extract_tt_duplicate.check_mutants(dump, repo)} TT allocation and copy MIR mutants"
    if task_id == "split_child":
        return f"checked {extract_split_worker_mir.check_child_mutants(dump, repo, baseline=baseline)} split child operand and return MIR mutants"
    if task_id == "split_guard":
        return f"checked {extract_split_worker_mir.check_guard_mutants(dump, repo, baseline=baseline)} worker guard lifetime MIR mutants"
    if task_id == "split_owner":
        return f"checked {check_split_owner_mutants.check(dump)} split owner field and return MIR mutants"
    if task_id == "worker_poll":
        return f"checked {check_worker_poll_mir_mutants.check(dump, repo, baseline=baseline)} worker-loop MIR mutants"
    if task_id == "stop_returns":
        return f"checked {check_stop_returns_mutants.check(dump, baseline=baseline)} PVS cancellation MIR mutants"
    if task_id == "cache_owner":
        return f"checked {check_cache_owner_mutations.check(dump, repo)} cache owner routing and entry mutants"
    if task_id == "deep_rec":
        return f"checked {check_deep_recursion_mir_regression.check(dump, repo, baseline=baseline)} deep recursion MIR mutants"
    if task_id == "endgame_order":
        return f"checked {check_endgame_order_mir_regression.check(dump, repo, baseline=baseline)} endgame ordering MIR mutants"
    if task_id == "fixed_leaf":
        return f"checked {check_fixed_leaf_mir_regression.check(dump, repo)} fixed-leaf MIR mutants"
    if task_id == "shallow_rec":
        return f"checked {check_shallow_recursion_mir_regression.check(dump, repo, baseline=baseline)} shallow recursion MIR mutants"
    if task_id == "native_worker_feed":
        return f"checked {check_native_worker_feed_mutants.check(dump, repo, baseline=baseline)} worker ticket and return-site MIR mutants"
    if task_id == "pvs_context":
        return f"checked {check_pvs_context_mir_mutants.check(dump, repo, baseline=baseline)} search context MIR mutants"
    proj_modules = {
        "PvsParallel/WorkerFrameMir/Program.lean": test_worker_frame_mir,
        "PvsParallel/SlotFrameMir/Program.lean": test_slot_frame_mir,
        "PvsParallel/SlotFrameMir/Pool.lean": test_pool_roster_mir,
        "PvsParallel/SlotFrameMir/DispatchProgram.lean": test_dispatch_origin_mir,
        "PvsParallel/SlotFrameMir/ReceiverProgram.lean": test_guard_receiver_mir,
    }
    if task_id in proj_modules:
        return f"checked {proj_modules[task_id].check(dump, baseline)} projection mutants for {task_id}"
    raise ValueError(f"unknown post task: {task_id}")


def check_inputs(dump, repo, write=False, *, probe_futures=None, proc_pool=None, owns_dump=False):
    own_pool = proc_pool is None
    workers = min(8, max(2, os.cpu_count() or 4))
    if probe_futures is None:
        with ThreadPoolExecutor(max_workers=5) as probe_pool:
            probe_state = _collect_probe_state(repo, _warm_rustc_probes(probe_pool, repo))
    else:
        probe_state = _collect_probe_state(repo, probe_futures)
    _apply_probe_state(probe_state)

    if own_pool:
        proc_pool = ProcessPoolExecutor(max_workers=workers)
    try:
        # Wave 1: dispatch all non-crate-wide pre-checks, extractors, and early mutants to worker
        # processes immediately while the main process preloads the crate-wide MIR cache.
        worker_pre_order = (
            "iterator", "spinlock", "cutoff_poll", "midgame_children", "midgame_handoff",
            "result_guards", "outer_abort", "outer_state", "move_generation",
            "serial_leaf_restore", "serial_child_calls", "serial_score_words",
            "serial_table_slot", "endgame_cache_cutoff", "endgame_probcut", "serial_level", "serial_move_list",
        )
        depths_fut = proc_pool.submit(_run_pre_task, "proof_depths", dump, repo, probe_state)
        # Submit heaviest tasks first (longest-job-first scheduling) so critical path finishes earliest.
        heavy_first_pre = ("serial_child_calls", "outer_abort", "serial_table_slot", "midgame_children", "endgame_probcut")
        pre_submit_order = heavy_first_pre + tuple(t for t in worker_pre_order if t not in heavy_first_pre)
        pre_futs = {
            tid: proc_pool.submit(_run_pre_task, tid, dump, repo, probe_state)
            for tid in pre_submit_order
        }
        # Submit extractors that unblock baseline-dependent mutants first!
        priority_extractors = (
            "PvsParallel/EndgameOrderMir/Program.lean",
            "MoveList/OrderWeights.lean",
            "PvsParallel/SplitWorkerMir/Program.lean",
            "PvsParallel/GeneralOwnerMir/Program.lean",
            "PvsParallel/PvSerialMir/Program.lean",
            "EndgameSearch/DeepRecursionMir/Program.lean",
            "EndgameSearch/Thresholds.lean",
            "EndgameSearch/ShallowMir/Program.lean",
            "PvsParallel/ContextPathMir/Program.lean",
            "PvsParallel/NativeStopReturns/Program.lean",
            "PvsParallel/WorkerPollMir/Program.lean",
            "PvsParallel/NativeWorkerFeed/Program.lean",
        )
        ext_submit_order = priority_extractors + tuple(
            n for n in _EXTRACTOR_ORDER
            if n not in _CRATE_WIDE_EXTRACTORS and n not in priority_extractors
        )
        ext_futs = {
            name: proc_pool.submit(_run_extractor_task, name, dump, repo, probe_state)
            for name in ext_submit_order
        }
        early_post_ids = (
            "owner_sites", "cache_call_sites", "tt_probe", "tt_duplicate",
            "split_owner", "cache_owner", "fixed_leaf",
        )
        post_futs = {
            tid: proc_pool.submit(_run_post_task, tid, dump, repo, probe_state)
            for tid in early_post_ids
        }

        # Main process: load crate-wide SimplifyCfg-initial.after.mir once into memory, then kick
        # off background deletion of unused .before.mir files (leaving all .after.mir intact).
        preload_mir_texts(dump, f".{PASS}.after.mir", max_workers=min(12, max(4, os.cpu_count() or 4)))
        if owns_dump:
            start_before_mir_cleanup(dump)
        local_programs = {
            "PvsParallel/WorkerFrameMir/Program.lean": _run_extractor_task(
                "PvsParallel/WorkerFrameMir/Program.lean", dump, repo, probe_state
            )
        }
        with ThreadPoolExecutor(max_workers=2) as local_pool:
            wf_proj_fut = local_pool.submit(
                _run_post_task,
                "PvsParallel/WorkerFrameMir/Program.lean",
                dump,
                repo,
                probe_state,
                local_programs["PvsParallel/WorkerFrameMir/Program.lean"],
            )
            extract_cache_owner.check_full_frontier(dump)
            check_serial_worker_mir.check(dump)
            serial_worker_msg = (
                f"checked single-worker pool state and rejected "
                f"{check_serial_worker_mir.check_mutants(dump)} MIR mutants"
            )
            for name in (
                "PvsParallel/SlotFrameMir/Program.lean",
                "PvsParallel/SlotFrameMir/Pool.lean",
                "PvsParallel/SlotFrameMir/DispatchProgram.lean",
                "PvsParallel/SlotFrameMir/ReceiverProgram.lean",
            ):
                prog = _run_extractor_task(name, dump, repo, probe_state)
                local_programs[name] = prog
                post_futs[name] = proc_pool.submit(_run_post_task, name, dump, repo, probe_state, prog)
            worker_frame_proj_msg = wf_proj_fut.result()

        # Submit baseline-dependent post-mutants strictly from the main thread once their
        # priority extractors complete (never inside ExecutorManagerThread callbacks).
        sw_baseline = ext_futs["PvsParallel/SplitWorkerMir/Program.lean"].result()
        baseline_posts = (
            ("split_child", sw_baseline),
            ("split_guard", sw_baseline),
            ("endgame_order", (
                ext_futs["PvsParallel/EndgameOrderMir/Program.lean"].result(),
                ext_futs["MoveList/OrderWeights.lean"].result(),
            )),
            ("deep_rec", (
                ext_futs["EndgameSearch/DeepRecursionMir/Program.lean"].result(),
                ext_futs["EndgameSearch/Thresholds.lean"].result(),
            )),
            ("stop_returns", ext_futs["PvsParallel/NativeStopReturns/Program.lean"].result()),
            ("worker_poll", ext_futs["PvsParallel/WorkerPollMir/Program.lean"].result()),
            ("shallow_rec", ext_futs["EndgameSearch/ShallowMir/Program.lean"].result()),
            ("native_worker_feed", ext_futs["PvsParallel/NativeWorkerFeed/Program.lean"].result()),
            ("pvs_context", ext_futs["PvsParallel/ContextPathMir/Program.lean"].result()),
        )
        for tid, baseline in baseline_posts:
            post_futs[tid] = proc_pool.submit(_run_post_task, tid, dump, repo, probe_state, baseline)

        # Collect Wave 1 pre-checks in deterministic order.
        depths_fut.result()
        for tid in worker_pre_order:
            msg = pre_futs[tid].result()
            if msg is not None:
                print(msg)
            if tid == "serial_level":
                print(serial_worker_msg)
        print("verified split cutoff flag publication and cached stop checks")

        # Collect all 51 extractors in deterministic order and verify/write files.
        programs = dict(local_programs)
        inputs = []
        modules = repo / "verification/lean/NeuralReversi"
        for name in _EXTRACTOR_ORDER:
            regenerated = programs[name] if name in programs else ext_futs[name].result()
            programs[name] = regenerated
            inputs.append((name, regenerated))

        for name, regenerated in inputs:
            path = modules / name
            if not path.exists() or path.read_text(encoding="utf-8") != regenerated:
                if not write:
                    raise SystemExit(f"compiled Rust differs from {name} (rerun with --write to regenerate it)")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(regenerated, encoding="utf-8", newline="\n")
                print(f"regenerated {name}")
                continue
            print(f"verified {name}")

        post_order = (
            "owner_sites", "cache_call_sites", "tt_probe", "tt_duplicate",
            "split_child", "split_guard", "split_owner", "worker_poll",
            "stop_returns", "cache_owner", "deep_rec", "endgame_order",
            "fixed_leaf", "shallow_rec", "native_worker_feed", "pvs_context",
            "PvsParallel/WorkerFrameMir/Program.lean",
            "PvsParallel/SlotFrameMir/Program.lean",
            "PvsParallel/SlotFrameMir/Pool.lean",
            "PvsParallel/SlotFrameMir/DispatchProgram.lean",
            "PvsParallel/SlotFrameMir/ReceiverProgram.lean",
        )
        for tid in post_order:
            if tid == "PvsParallel/WorkerFrameMir/Program.lean":
                print(worker_frame_proj_msg)
            else:
                print(post_futs[tid].result())
    finally:
        if own_pool:
            proc_pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--mir", type=Path, help="reuse a dump from the same compiler and checkout")
    parser.add_argument("--write", action="store_true", help="rewrite proof inputs that differ instead of failing")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.mir:
        check_inputs(args.mir, repo, args.write)
        return

    # Emit every crate-wide SimplifyCfg-initial body plus the `built` MIR of the 4 modules
    # containing checked constants (omitting eval's embedded-weight `built` copies and unused passes).
    names = (
        PASS,
        "built&search::",
        "built&move_list::",
        "built&score::",
        "built&transposition_table::",
    )
    workers = min(6, max(2, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=2) as probe_pool, ProcessPoolExecutor(max_workers=workers) as proc_pool:
        probe_futures = _warm_rustc_probes(probe_pool, repo)
        warmup_futs = [proc_pool.submit(_worker_warmup) for _ in range(workers)]
        with dump_mir(repo, names) as dump:
            for fut in warmup_futs:
                fut.result()
            check_inputs(dump, repo, args.write, probe_futures=probe_futures, proc_pool=proc_pool, owns_dump=True)


if __name__ == "__main__":
    main()
