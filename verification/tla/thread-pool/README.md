# ThreadPool lifecycle and shutdown model (TLA+)

Bounded PlusCal/TLA+ model check of `ThreadPool` startup, Lazy SMP <-> YBWC search mode switching, condvar sleep/wakeup, and `Arc<ThreadPool>` shutdown (including last-reference drops on the external caller, main thread, and a late-joining worker thread).

See [`../README.md`](../README.md) for prerequisites, shared runner behavior, CI coverage, and `trace.py`.

## Run

```bash
python3 verification/tla/thread-pool/check.py            # default bounds (N=3, MaxSearches=2)
python3 verification/tla/thread-pool/check.py --deep     # 3 alternating searches (N=3, MaxSearches=3)
python3 verification/tla/thread-pool/check.py --mutants  # verify every mutant fails its target property
python3 verification/tla/thread-pool/check.py --all      # clean + deep + all mutants
```

## Model overview

| Model element (`ThreadPool.tla`) | Rust implementation |
| --- | --- |
| `C_WaitReady`, `ready`, `poolReady` | `ThreadPool::init`, `wait_for_threads_ready`, `all_threads_ready` ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)), `Thread::idle_loop` ready store ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)) |
| `C_Loop` / `C_AfterSend`, `msgQueue`, `thinking`, `resultReady` | `ThreadPool::start_thinking`, `Message::{StartThinking, Exit}` ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)), `Search::execute_search` ([`search.rs`](../../../crates/reversi-core/src/search.rs)) |
| `M_Loop` / `M_Recv` / `M_FinishSearch` / `M_SendResult` | `Thread::main_thread_loop` ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)) |
| `M_Dispatch` / `M_StopHelpers` / `M_WaitHelpers` / `W_RunHelper` | `ThreadPool::{start_helpers, stop_helpers}` ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)), helper job methods ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)), `Helpers::{start, stop}` ([`lazy_smp.rs`](../../../crates/reversi-core/src/search/midgame/lazy_smp.rs)) |
| `"MidToEnd"` phase transition | `midgame::search_root` stopping Lazy SMP helpers before handing off to `endgame::solve_root` ([`midgame.rs`](../../../crates/reversi-core/src/search/midgame.rs)) |
| `M_StartSplit` / `M_MainSplitDone` / `M_WaitSplit` / `W_RunSplit` | `assign_helpers_to_split_point` ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)), split-point lifecycle ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)) |
| `W_TryLateJoin` / `W_DropLateJoin`, `holdsRef`, `OtherRefs` | `Thread::try_late_join` upgrading `Weak<ThreadPool>` after clearing `helpers_mask` ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)) |
| `W_SleepCheck` / `W_Wait`, `sleeping` | `Thread::idle_loop` condvar wait, `Thread::notify_one` ([`thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs)), `ThreadPool::notify_all` ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)) |
| `RunExit` (`EX_Enter`..`EX_Done`), `joining` | `Drop for ThreadPool` -> `ThreadPool::exit` with the `handle.thread().id() != current` self-join guard ([`pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs)) |

Across up to `MaxSearches` consecutive searches (`"Midgame"`, `"Endgame"`, or `"MidToEnd"`), the model exercises all three `Arc<ThreadPool>` last-reference drop paths: external caller, main thread (when the caller dropped its ref mid-search), and a worker thread holding an upgraded `Weak<ThreadPool>` ref from `try_late_join`.

## Checked properties

### Safety invariants

- `TypeOK`: all state variables stay within their declared domains (including `Len(msgQueue) <= 2`).
- `ReadyBeforeSearch`: once `poolReady` holds or any search has been dispatched, every thread `0..N-1` has signaled `ready[t] = TRUE`.
- `NoHelperSplitOverlap`: no worker simultaneously holds or executes Lazy SMP and YBWC work, and no YBWC split point is active while any worker has a pending or active Lazy SMP helper job.
- `NoExitWhileRefHeld`: `ThreadPool::exit` never starts while any `Arc<ThreadPool>` reference is held (`exiting => RefCount = 0`).
- `NoUseAfterExit`: once `exiting` begins, `thinking` and `spActive` are false and no worker holds or runs Lazy SMP or YBWC work.
- `NoSelfJoin`: no pool thread attempts to `join()` itself (`\A t \in Threads : joining[t] # t`).

### Liveness and deadlock

- Deadlock freedom (`check_deadlock=True`): all processes terminate without hanging on condvar waits, channel receives, or thread joins.
- `EventuallyExits == <>AllExited`: under weak fairness (`fair process`), every pool thread eventually reaches `exited[t] = TRUE`.

## Mutants

| Mutant | Simulated regression | Expected catch |
| --- | --- | --- |
| `skip_wait_ready` | Omits `wait_for_threads_ready` | `ReadyBeforeSearch` |
| `skip_stop_helpers_before_split` | Skips `stop_helpers` before a `"MidToEnd"` split | `NoHelperSplitOverlap` |
| `exit_before_zero_refs` | Caller calls `exit()` with `RefCount > 0` | `NoExitWhileRefHeld` |
| `drop_pool_before_clear_thinking` | Main drops `pool` before clearing `thinking` | `NoUseAfterExit` |
| `drop_last_ref_self_join` | Omits the `j # self` self-join guard in `RunExit` | `NoSelfJoin` |
| `exit_skips_notify` | Skips `notify_all()` in `RunExit` (sleeping worker lost wakeup) | `Deadlock` |
| `exit_skips_main_msg` | Skips `Message::Exit` send in `RunExit` (main stuck in `recv`) | `Deadlock` |
| `worker_ignores_exit_flag` | Worker omits outer `!exit` check in `idle_loop` | `Temporal` (`EventuallyExits`) |

## Scope and assumptions

- **Memory ordering**: PlusCal steps are sequentially consistent and do not model weak-memory reorderings.
- **Outside this model**: nested YBWC split-point slots and panic unwinding (covered in [`../split-point/README.md`](../split-point/README.md)), and search abort generations / timer threads (covered in [`../stop-abort/README.md`](../stop-abort/README.md)).
