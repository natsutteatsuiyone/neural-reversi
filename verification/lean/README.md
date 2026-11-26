# Lean 4 proofs

Machine-checked Lean 4 proofs for hand-written models of `crates/reversi-core`.
The suite uses the Lean toolchain pinned in [`lean-toolchain`](lean-toolchain) with no external Lean dependencies ([`lake-manifest.json`](lake-manifest.json)).

Most proofs cover hand-written models; correspondence to `crates/reversi-core` rests on weight-free Rust differential checks (`difftest/` and `revmodel`), MIR checks against the pinned Rust compiler, engine unit tests, and source review.
For bounded concurrency protocol model checking, see [`verification/tla/README.md`](../tla/README.md).

## Run

Run from the repository root with `elan`, Python 3.10 or later, and the Rust toolchain in [`rust-toolchain.toml`](../../rust-toolchain.toml) on `PATH`:

```bash
python3 verification/lean/check.py                 # full suite + Rust differential checks (must print PASS)
python3 verification/lean/check.py --lean-only     # skip Rust differential checks
cargo make verify-lean                             # wrapper around check.py
```

[`check.py`](check.py) executes the following steps and exits `0` only if all pass:

1. Regenerates `NeuralReversi/AuditModules.lean` and `NeuralReversi/AuditImports.lean` via [`scripts/audit_modules.py`](scripts/audit_modules.py) `--write` and verifies [`Audit.lean`](Audit.lean) and [`lakefile.toml`](lakefile.toml) cover every `.lean` file.
2. Scans every `.lean` file with [`scripts/check_policy.py`](scripts/check_policy.py) for forbidden proof shortcuts and runs its regression checks.
3. Runs `lake build NeuralReversi revmodel` (with `-DwarningAsError=true -M4096`), including [`Audit.lean`](Audit.lean), [`CSimpAuditRegression.lean`](CSimpAuditRegression.lean), and [`AuditOwnershipRegression.lean`](AuditOwnershipRegression.lean).
4. Runs [`scripts/check_policy_lean_regression.py`](scripts/check_policy_lean_regression.py) alongside the Rust differential checks.
5. Unless `--lean-only` is set, builds and runs [`difftest/`](difftest/Cargo.toml) (without `.zst` neural weights) and checks its outputs against [`revmodel`](DiffMain.lean), then runs `crates/reversi-core/build.rs` for little- and big-endian targets to verify generated pattern-table bytes and stale-output cleanup.

### CI and platform-specific checks

- [`.github/workflows/verify-lean.yml`](../../.github/workflows/verify-lean.yml) runs `verification/lean/check.py` on Linux x86-64 whenever verification files or `reversi-core` sources change.
- **Wasm SIMD128**: with `wasm32-unknown-unknown` and Node.js installed, run `python3 verification/lean/check.py --wasm-pattern-features` to verify the `simd128` pattern fixture against Lean.
- **AArch64 NEON / SHA3**: on an AArch64 host with SHA3, run `check.py` with `RUSTFLAGS="-Ctarget-feature=+sha3"` (`neon-sha3`) and `RUSTFLAGS="-Ctarget-feature=-sha3"` (`neon`).
- **Weight-backed engine tests**: run `cargo test --locked -p reversi-core` with the local `.zst` weight files in the repository root.

## Endgame MIR and replay checks

### Compiler MIR checks

Parts of the exact endgame and parallel search proofs consume control flow extracted from the pinned Rust compiler.
With Cargo dependencies available, regenerate and compare all MIR proof inputs and MIR mutation regressions from the repository root (also checked on Windows and on Linux ARM64 in [`.github/workflows/test.yml`](../../.github/workflows/test.yml)):

```bash
python3 verification/lean/scripts/check_endgame_mir.py
```

After a deliberate change, `--write` rewrites the proof inputs that differ instead of failing; rebuild the Lean proofs afterwards. The endgame move-ordering square table and its six weights are extracted into [`MoveList/OrderWeights.lean`](NeuralReversi/MoveList/OrderWeights.lean), which the extracted and hand-written ordering models share. Tuning those values needs only this regeneration unless a weight becomes negative (`weights_nonneg`) or a key no longer fits below the TT hint (`keys_fit`). `MIN_EC_ETC_EMPTIES` is shared the same way through [`EndgameSearch/Thresholds.lean`](NeuralReversi/EndgameSearch/Thresholds.lean), and the endgame sort-depth table can change as long as full-depth searches keep the fast move order through 13 empties. The parallel split and claim depths (above 5), the boundary-store node threshold (at least 1), the endgame cache budgets (at least one 18-byte entry each), the depth schedule of `evaluate_moves_by_search` (it must stay a 0, 1, 2 step function) and `ASPIRATION_DELTA` (one to 32 discs) also need only regeneration; outside these ranges the script stops and names the proof that needs the range. So do the midgame depth-one visit tiers in [`MidgameShallow/Masks.lean`](NeuralReversi/PvsParallel/SerialTable/MidgameShallow/Masks.lean), as long as they partition the board (`mask_partition`). No model reads the endgame ProbCut fit, `DEPTH_STEP_THRESHOLD`, or the midgame IIR, LMR and table ProbCut constants. The midgame aspiration constants are hard-coded in [`MidgameAspiration/Model.lean`](NeuralReversi/MidgameAspiration/Model.lean), and `check_midgame_handoff_mir.py` rejects a change to them. `DEPTH_TO_NWS` and `DEPTH_TO_SHALLOW_SEARCH` must stay 12 and 6, and `MIN_PROBCUT_DEPTH` and `MIN_ETC_DEPTH` must stay `DEPTH_TO_NWS + 1` and `+ 2`: the proofs case-split on these depths, and the script stops first when they differ.

Key proof entry points covered by these models:

- **Serial exact endgame (single-PV and Multi-PV)**: aspiration window convergence, five-level selectivity ladder from a board, repeated requests across TT generation wraparound, timed midgame-to-endgame handoff, retries after interrupted searches, and Multi-PV ordering ([`FullLadder.lean`](NeuralReversi/AspirationMir/SerialTable/FullLadder.lean), [`Repeated.lean`](NeuralReversi/AspirationMir/SerialTable/Repeated.lean), [`MidgameSession.lean`](NeuralReversi/AspirationMir/SerialTable/MidgameSession.lean), [`MultiHistory.lean`](NeuralReversi/AspirationMir/SerialTable/MultiHistory.lean), [`ExactScore.lean`](NeuralReversi/PublicationMir/ExactScore.lean)). Limits of these statements:
  - The engine contracts in `FullLadder.lean`, `Repeated.lean`, `MidgameSession.lean` and `MultiHistory.lean` take an accepted LRAT certificate for the 35-move mobility bound as a hypothesis. Neither `check.py` nor CI generates one; the [optional certificate check](#optional-extra-checks) does.
  - For an interrupted search, the proofs show only that the TT and caches stay valid, so that later searches remain exact. What the interrupted search itself publishes is not covered: `ExactScore.Contract` leaves its `RowScores` obligation to the caller, and no theorem discharges it for that path.
  - The timed handoff stops only between selectivity levels. The searches inside a level never poll the stop flag (`frames … false` in `MidgameSession.Contract`).
- **Concurrent YBWC and physical TT replay**: helper interleavings, stale alpha reads, cutoffs, mask release sequences, nested splits, and cross-session TT retention ([`Contracts.lean`](NeuralReversi/PvsParallel/Interleaving/Contracts.lean), [`Nested.lean`](NeuralReversi/PvsParallel/Interleaving/Nested.lean), [`Entry.lean`](NeuralReversi/PvsParallel/Interleaving/Entry.lean), [`PhysicalCalls.lean`](NeuralReversi/PvsParallel/Interleaving/PhysicalCalls.lean), [`JournalResident.lean`](NeuralReversi/PvsParallel/Interleaving/JournalResident.lean), and the [Quint -> TLC -> Lean witness suite](../tla/lean-witnesses/README.md)). Limits of these statements:
  - `Contracts.lean` states split soundness over an abstract split join. `Nested.lean` and `PhysicalCalls.lean` link it to the owner's physical `best_score` load and the TT memory.
  - The end-to-end `PhysicalCalls.CheckedContract` is a trace check: when the Boolean coverage check accepts a captured trace, the published score is exact. A rejected trace satisfies it vacuously.
  - The traces that instantiate it (`check_native_direct.py`, `check_public_warm_seed.py`) contain no splits, so no fixture yet shows the split hypotheses holding together in one run. The split pieces are replayed separately (`check_native_split_owner.py`, `check_native_split_alpha.py`, `check_warm_tt.py`).

### Native trace and fixture drivers

Optional drivers under [`scripts/`](scripts/) instrument a temporary copy of `reversi-core`, run concrete searches with the local `.zst` evaluation weights, and replay traces against Lean (including negative controls):

| Group | Scripts (`verification/lean/scripts/`) |
| --- | --- |
| Serial calls, caches, TT, and Multi-PV | `check_serial_calls_trace.py`, `check_serial_cache_trace.py`, `check_serial_cache_stack.py`, `check_serial_table_trace.py`, `check_serial_table_threshold.py`, `check_midgame_child_effects.py`, `check_endgame_probcut_trace.py`, `check_public_fourteen_stack.py`, `check_public_multi.py` |
| Concurrent YBWC, splits, and TT replay | `check_split_interleaving.py`, `check_reader_window.py`, `check_warm_tt.py`, `check_native_etc.py`, `check_native_split_owner.py`, `check_native_split_alpha.py`, `check_native_ancestor_stop.py`, `check_native_mask_generation.py`, `check_occupied_sources.py`, `check_native_direct.py`, `check_public_warm_seed.py`, `check_warm_local_cache.py` |

Build the corresponding Lean modules with `lake build` first if not already built. Drivers accepting `--target-dir <dir>` (required for `check_native_*`, `check_occupied_sources.py`, and `check_public_warm_seed.py`) can share a reusable Cargo scratch directory such as `target/verify-lean/native-target`.

### Optional extra checks

- **35-move mobility LRAT certificate** ([`NeuralReversi/LegalMoves/Encoding.lean`](NeuralReversi/LegalMoves/Encoding.lean)): generate and verify the UNSAT certificate (~418 MB) using the CaDiCaL binary bundled with Lean. The serial engine contracts above take its acceptance as a hypothesis:

  ```bash
  (
    cd verification/lean
    python3 scripts/audit_modules.py --write || exit
    lake build revmodel || exit
    work=../../target/verify-lean
    mkdir -p "$work"
    .lake/build/bin/revmodel move-bound-cnf 35 > "$work/mobility-bound-35.cnf" || exit
    solver="$(lean --print-prefix)/bin/cadical"
    status=0
    "$solver" --lrat --binary "$work/mobility-bound-35.cnf" "$work/mobility-bound-35.lrat" || status=$?
    test "$status" -eq 20
  ) &&
  python3 verification/lean/check.py --move-bound-certificate target/verify-lean/mobility-bound-35.lrat
  ```

- **Instrumented exact-solve PVS transcript**: normalize and replay a captured trace through [`scripts/normalize_pvs_trace.py`](scripts/normalize_pvs_trace.py) and `revmodel pvs-trace-check` (pass `--pvs-threshold-depth <empties>` to override the default 8-empty threshold):

  ```bash
  python3 verification/lean/check.py --lean-only --pvs-trace target/verify-lean/pvs-child-board-native.trace
  ```

- **Local weight-file lengths**: compare decoded lengths of local weight files against the Lean stream model:

  ```bash
  cargo run --quiet --release --locked --manifest-path verification/lean/difftest/Cargo.toml \
    --target-dir target/verify-lean/target --features weight-stream --bin weight_stream_cases \
    -- eval-e6bbc4f6.zst eval_sm-e6bbc4f6.zst > target/verify-lean/weight-files.txt
  verification/lean/.lake/build/bin/revmodel weight-stream-cases target/verify-lean/weight-files.txt
  ```

## Where things live

- `NeuralReversi/`: Lean 4 models and proofs.
- [`NeuralReversi/Registry.lean`](NeuralReversi/Registry.lean): Registered properties (run `verification/lean/.lake/build/bin/revmodel registry` for JSON).
- [`check.py`](check.py), [`DiffMain.lean`](DiffMain.lean), [`difftest/`](difftest/Cargo.toml): Main verification driver and weight-free Rust differential fixtures.
- [`scripts/`](scripts/): Audit/policy checkers, MIR extractors, and native trace replay drivers.

## Scope and assumptions

- **Outside Lean**: target ISA code generation (AVX2, AVX-512, BMI2, NEON, Wasm SIMD128), `unsafe` memory layout and pointer provenance, `const` table generation outside `difftest/`, OS allocator and large-page APIs, and crates other than `reversi-core`.
- **Trace and concurrency premises**: concurrency, split-point, PVS, TT, evaluation-cache, PV, and deadline proofs assume matched atomic read-from, publication, and call histories. Relating every compiled Rust execution to these primitive trace events remains open.
- **Key side conditions**:
  - AArch64 NEON-SHA3 move generation and batch `flip2`–`flip4` proofs require disjoint bitboards (enforced in Rust by `Board::from_bitboards`).
  - Input-layer and small-network row bounds require every pattern lane to be below its pattern size (checked per evaluation in debug builds only).
  - `EvalCache` key uniqueness requires at least `2^18` entries.
  - ProbCut full-game bounds are conditional on the shallow overestimate fitting the verification margin.
  - `AlignedBuffer` drop proofs cover normal drops and a single unwind, not a double panic or `panic = "abort"`.

## Proof policy and maintenance

- **Axioms**: [`Audit.lean`](Audit.lean) restricts every project declaration to Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`).
- **Forbidden shortcuts**: [`scripts/check_policy.py`](scripts/check_policy.py) and [`AuditCore.lean`](NeuralReversi/AuditCore.lean) reject `sorry`, `sorryAx`, `admit`, `axiom`, `native_decide`, `bv_decide`, `ofReduceBool`, `ofReduceNat`, `trustCompiler`, `implemented_by`, `extern`, `unsafe`, and `partial` (allowing only compiler-generated `._unsafe_rec` and `ctorIdx._impl`).
- **Compiler replacements**: [`Audit.lean`](Audit.lean) permits only the reviewed `[csimp]` entries in [`NeuralReversi/CSimpAudit.lean`](NeuralReversi/CSimpAudit.lean).
- **Fresh clone / editing `.lean` files**: when adding, renaming, or removing `.lean` files, update `roots` in [`lakefile.toml`](lakefile.toml) if adding a top-level root and rerun `python3 verification/lean/check.py`. To run `lake build` directly on a fresh clone before `check.py`, first generate the untracked audit files:

  ```bash
  python3 verification/lean/scripts/audit_modules.py --write
  ```

- When changing a covered `reversi-core` file, update the matching Lean model and rerun `python3 verification/lean/check.py` (and `check_endgame_mir.py` for endgame/search changes).
