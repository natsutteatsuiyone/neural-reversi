"""Regenerate the endgame proof inputs with the workspace compiler and compare them."""

import argparse
from pathlib import Path

from mir_common import dump_mir

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
import check_serial_move_list_mir
import check_serial_leaf_restore_mir
import check_serial_child_calls_mir
import check_serial_score_words_mir
import check_serial_table_slot_mir
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


def check_inputs(dump, repo, write=False):
    check_proof_depths(dump, repo)
    modules = repo / "verification/lean/NeuralReversi"
    pvs = dump / extract_root_wipeout.FILENAME
    reduction = dump / "reversi_core.search-pvs-compute_lmr_reduction.SimplifyCfg-initial.after.mir"
    check_move_iterator_mir.check_iterator(dump)
    check_move_iterator_mir.compile_atomic(repo)
    print("verified concurrent move iterator and relaxed try_update")
    check_split_spinlock.check(dump)
    print(f"checked split mutex acquisition, guard drop, and {check_split_spinlock.check_mutants(dump)} MIR mutants")
    check_cutoff_poll_mir.check(dump)
    print(f"checked midgame recursive calls and rejected {check_midgame_children_mir.check(dump, repo)} MIR mutants")
    print(f"checked timed midgame handoff and rejected {check_midgame_handoff_mir.check(dump)} MIR mutants")
    check_result_guards_mir.check(dump)
    print(f"checked public result guards and rejected {check_result_guards_mir.check_mutants(dump)} MIR mutants")
    check_move_generation_mir.check(dump)
    print(f"checked move generation and rejected {check_move_generation_mir.check_mutants(dump)} MIR mutants")
    print(f"checked fixed solver context effects and rejected {check_serial_leaf_restore_mir.check(dump)} MIR mutants")
    print(f"checked serial recursive calls and rejected {check_serial_child_calls_mir.check(dump, repo)} MIR mutants")
    print(f"checked signed recursive scores and rejected {check_serial_score_words_mir.check(dump)} MIR mutants")
    print(f"checked retained PVS and ETC table slots and rejected {check_serial_table_slot_mir.check(dump)} MIR mutants")
    print(f"checked selective ProbCut and rejected {check_endgame_probcut_mir.check(dump, repo)} MIR mutants")
    check_serial_worker_mir.check(dump)
    print(f"checked selectivity guards in serial thirteen-empty searches and rejected {check_serial_level_mir.check(dump, repo)} MIR mutants")
    print(f"checked single-worker pool state and rejected {check_serial_worker_mir.check_mutants(dump)} MIR mutants")
    print(f"rejected {check_serial_move_list_mir.check_mutants(dump)} serial move-list MIR mutants")
    print("verified split cutoff flag publication and cached stop checks")
    extract_cache_owner.check_full_frontier(dump)
    inputs = [
        ("AspirationMir/Program.lean", extract_aspiration_mir.render(dump, repo)),
        ("AspirationMir/NativeRootBirth/Program.lean", extract_root_birth.render(dump, repo)),
        ("PvsParallel/RootStorageMir/Program.lean", extract_root_rows.render(dump, repo)),
        ("PvsParallel/NativeStopReturns/Program.lean", extract_stop_returns.render(dump)),
        ("OuterMir/Program.lean", extract_outer.render(dump, repo)),
        ("EndgameSearch/Direct/Mir/Program.lean", extract_fixed_leaf_mir.verify(dump, repo)),
        ("EndgameSearch/CacheMir/Program.lean", extract_cache.render(dump, repo)),
        ("EndgameSearch/CacheMir/Budgets.lean", extract_cache_budgets.render(dump, repo)),
        ("EndgameSearch/CacheOwner/Program.lean", extract_cache_owner.render(dump, repo)),
        ("EndgameSearch/CacheHistory/NativeTrace/Sites.lean", extract_cache_effect_sites.render(dump, repo)),
        ("EndgameSearch/ShallowMir/Program.lean", extract_shallow_recursion_mir.verify(dump, repo)),
        ("EndgameSearch/DeepRecursionMir/Program.lean", extract_deep_recursion_mir.verify(dump, repo)),
        ("EndgameSearch/Thresholds.lean", extract_deep_recursion_mir.render_thresholds(dump)),
        ("PvsParallel/NonPvMir/Program.lean", extract_nonpv_leaf_mir.verify(dump, repo)),
        ("PvsParallel/DeepLeafMir/Program.lean", extract_nws_leaf_mir.verify(dump, repo)),
        ("PvsParallel/PvEntryMir/Program.lean", extract_pv_entry_mir.verify(dump, repo)),
        ("PvsParallel/PvSerialMir/Program.lean", extract_pv_serial_mir.verify(dump, repo)),
        ("PvsParallel/GeneralNonPvMir/Program.lean", extract_general_nonpv_mir.verify(dump, repo)),
        ("PvsParallel/NativeFrames/OwnerSites/Program.lean", extract_owner_sites.render(dump)),
        ("PvsParallel/GeneralOwnerMir/Program.lean", extract_general_owner_mir.verify(dump, repo)),
        ("PvsParallel/EndgameOrderMir/Program.lean", extract_endgame_order_mir.verify(dump, repo)),
        ("MoveList/OrderWeights.lean", extract_endgame_order_mir.render_weights(dump)),
        ("PvsParallel/SerialTable/MidgameShallow/Masks.lean", check_midgame_children_mir.render_masks(dump)),
        ("PvsParallel/ContextPathMir/Program.lean", extract_pvs_context_mir.verify(dump, repo)),
        ("PvsParallel/WipeoutExit.lean", extract_root_wipeout.render(pvs, repo)),
        ("PvsParallel/SingleExit.lean", extract_single.render(pvs, repo)),
        ("PvsParallel/RootFrameMir/Program.lean", extract_frame.render(pvs, repo)),
        ("PvsParallel/RootHeaderMir/Program.lean", extract_header.render(pvs, reduction, repo)),
        ("PvsParallel/RootUpdateMir/Program.lean", extract_root_update.render(dump, repo)),
        ("PvsParallel/SplitWorkerMir/Program.lean", extract_split_worker_mir.render(dump, repo)),
        ("PvsParallel/WorkerPollMir/Program.lean", extract_worker_poll_mir.render(dump, repo)),
        ("PvsParallel/SplitWorkerMir/FieldProgram.lean", extract_split_worker_mir.render(dump, repo, fields=True)),
        ("PvsParallel/NativeWorkerFeed/Program.lean", extract_native_worker_feed.verify(dump, repo)),
        ("PvsParallel/SplitSchedulerMir/Program.lean", extract_split_scheduler_mir.render(dump, repo)),
        ("PvsParallel/SplitActivationMir/Program.lean", extract_split_activation_mir.render(dump)),
        ("PvsParallel/SplitRebookingMir/Program.lean", extract_split_rebooking_mir.render(dump, repo)),
        ("PvsParallel/WorkerFrameMir/Program.lean", extract_worker_frame_mir.render(dump)),
        ("PvsParallel/SlotFrameMir/Program.lean", extract_slot_frame_mir.render(dump)),
        ("PvsParallel/SlotFrameMir/Pool.lean", extract_pool_roster_mir.render(dump)),
        ("PvsParallel/SlotFrameMir/DispatchProgram.lean", extract_dispatch_origin_mir.render(dump)),
        ("PvsParallel/SlotFrameMir/ReceiverProgram.lean", extract_guard_receiver_mir.render(dump)),
        ("PublicationMir/Program.lean", extract_publication_mir.render(dump, repo)),
        ("TranspositionTable/ReaderMir/Program.lean", extract_tt_reader_mir.render(dump, repo)),
        ("TranspositionTable/ReaderMir/ProbeProgram.lean", extract_tt_probe_mir.render(dump, repo)),
        ("TranspositionTable/ReaderMir/Lookup.lean", extract_tt_lookup_mir.render(dump, repo)),
        ("TranspositionTable/Duplicate/Program.lean", extract_tt_duplicate.render(dump, repo)),
        ("TranspositionTable/WriterMir/Program.lean", extract_tt_writer.render(dump, repo)),
        ("TranspositionTable/WriterMir/Origins.lean", extract_tt_save.render(dump, repo)),
        ("TranspositionTable/WriterMir/Store.lean", extract_tt_store.render(dump, repo)),
    ]
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
    print(f"checked {extract_owner_sites.check_mutants(dump)} recursive call-site mutants")
    print(f"checked {check_cache_call_sites.check(dump)} endgame cache call-site mutants")
    print(f"checked {extract_tt_probe_mir.check_mutants(dump)} cluster probe MIR mutants")
    print(f"checked {extract_tt_duplicate.check_mutants(dump, repo)} TT allocation and copy MIR mutants")
    print(f"checked {extract_split_worker_mir.check_child_mutants(dump, repo)} split child operand and return MIR mutants")
    print(f"checked {extract_split_worker_mir.check_guard_mutants(dump, repo)} worker guard lifetime MIR mutants")
    print(f"checked {check_split_owner_mutants.check(dump)} split owner field and return MIR mutants")
    print(f"checked {check_worker_poll_mir_mutants.check(dump, repo)} worker-loop MIR mutants")
    print(f"checked {check_stop_returns_mutants.check(dump)} PVS cancellation MIR mutants")
    print(f"checked {check_cache_owner_mutations.check(dump, repo)} cache owner routing and entry mutants")
    print(f"checked {check_deep_recursion_mir_regression.check(dump, repo)} deep recursion MIR mutants")
    print(f"checked {check_endgame_order_mir_regression.check(dump, repo)} endgame ordering MIR mutants")
    print(f"checked {check_fixed_leaf_mir_regression.check(dump, repo)} fixed-leaf MIR mutants")
    print(f"checked {check_shallow_recursion_mir_regression.check(dump, repo)} shallow recursion MIR mutants")
    print(f"checked {check_native_worker_feed_mutants.check(dump, repo)} worker ticket and return-site MIR mutants")
    print(f"checked {check_pvs_context_mir_mutants.check(dump, repo)} search context MIR mutants")
    programs = dict(inputs)
    for name, module in (
        ("PvsParallel/WorkerFrameMir/Program.lean", test_worker_frame_mir),
        ("PvsParallel/SlotFrameMir/Program.lean", test_slot_frame_mir),
        ("PvsParallel/SlotFrameMir/Pool.lean", test_pool_roster_mir),
        ("PvsParallel/SlotFrameMir/DispatchProgram.lean", test_dispatch_origin_mir),
        ("PvsParallel/SlotFrameMir/ReceiverProgram.lean", test_guard_receiver_mir),
    ):
        print(f"checked {module.check(dump, programs[name])} projection mutants for {name}")


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

    names = (
        "aspiration_search", "widen_aspiration_window", "try_load_snapshot", "from_u64",
        "ROOT_NODE", "pvs::compute_lmr_reduction", "search::result", "search::root_move",
        "search_split_point", "pvs::claim_child", "MIN_CLAIM_DEPTH", extract_frame.FILTER,
        "SPECULATIVE_SPLIT_MAX_DEPTH", "MIN_SPLIT_DEPTH", "node_type",
        "endgame::solve_root", "endgame::search_root", "line_completed", "begin_phase", "pv_count",
        "ENDGAME_LADDER", "util::bitset", "search::threading::thread", "search::threading::split_point",
        "assign_helpers_to_split_point", "shallow_search", "null_window_search", "search_move_nws_ec",
        "MIN_EC_ETC_EMPTIES", "DEPTH_TO_SHALLOW_SEARCH", "DISC_SCORE_INF", "DISC_SCORE_MAX", "SPLIT_CUTOFF_SCORE",
        "DEPTH_STEP_THRESHOLD", "DEPTH1_MOVE_ORDER_MASKS", "IIR_MIN_DEPTH", "IS_ENDGAME", "DEPTH_TO_SHALLOW", "search::aspiration",
        "ENDGAME_TRACKING_MIN_SELECTIVITY", "from_raw", "transposition_table", "try_probcut", "MIN_ETC_DEPTH", "MIN_PROBCUT_DEPTH", "is_enabled",
        "search::endgame::cache", "util::mul_hi64", "for_thread_count", extract_nonpv_leaf_mir.FILTER,
        "SimplifyCfg-initial", extract_endgame_order_mir.FILTER,
    )
    with dump_mir(repo, names) as dump:
        check_inputs(dump, repo, args.write)


if __name__ == "__main__":
    main()
