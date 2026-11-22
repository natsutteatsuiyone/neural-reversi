# Lean proofs

Machine-checked Lean 4 proofs about hand-written models of `crates/reversi-core`.
The suite uses the Lean version pinned in [`lean-toolchain`](lean-toolchain) with no external Lean dependencies ([`lake-manifest.json`](lake-manifest.json)).

Lean proves properties of hand-written models, not the compiled Rust binary.
Correspondence to `crates/reversi-core` rests on the live weight-free Rust differential checks (`difftest/` and `revmodel`), native unit tests in `crates/reversi-core`, and source review.
For bounded interleaving checks of split points, see the [TLA+ split-point model](../tla/split-point/README.md).

## Run

Run from the repository root with `elan`, Python 3.10 or later, and the Rust toolchain in [`rust-toolchain.toml`](../../rust-toolchain.toml) on `PATH`:

```bash
python3 verification/lean/check.py              # full suite and Rust differential checks (must print PASS)
python3 verification/lean/check.py --lean-only  # skip Rust differential checks
cargo make verify-lean                          # wrapper around check.py
```

[`check.py`](check.py) runs the following checks. After the Lean build succeeds, the policy regressions run alongside the Rust differential checks. The command exits `0` only if all requested checks pass.

1. [`scripts/audit_modules.py`](scripts/audit_modules.py) `--write` generates `NeuralReversi/AuditModules.lean` and `NeuralReversi/AuditImports.lean` (untracked) from every `.lean` file, and checks that [`Audit.lean`](Audit.lean) and the `NeuralReversi` roots in [`lakefile.toml`](lakefile.toml) cover them.
2. [`scripts/check_policy.py`](scripts/check_policy.py) scans every `.lean` file outside comments and non-interpolated literals for forbidden proof shortcuts.
3. `lake build NeuralReversi revmodel` builds the library and [`revmodel`](DiffMain.lean) with `-DwarningAsError=true -M4096` and runs [`Audit.lean`](Audit.lean), [`CSimpAuditRegression.lean`](CSimpAuditRegression.lean), and [`AuditOwnershipRegression.lean`](AuditOwnershipRegression.lean).
4. [`scripts/check_policy_lean_regression.py`](scripts/check_policy_lean_regression.py) checks the pinned Lean version and runs the policy and audit regression fixtures.
5. Unless `--lean-only` is set, [`difftest/`](difftest/Cargo.toml) generates test cases from the live `reversi-core` sources without `.zst` neural weights and `revmodel` checks them against the Lean models.

[`.github/workflows/verify-lean.yml`](../../.github/workflows/verify-lean.yml) runs `check.py` on Linux x86-64 when `verification/lean/**` or a covered `reversi-core` source changes.
The AArch64 NEON backend is chosen at compile time from `target_feature`, and the default `aarch64-unknown-linux-gnu` target leaves `sha3` off even on a CPU that has it. On an AArch64 host with SHA3, run `check.py` once with `RUSTFLAGS="-Ctarget-feature=+sha3"` and once with `RUSTFLAGS="-Ctarget-feature=-sha3"`, and confirm that the move generation line reports `neon-sha3` in the first run and `neon` in the second.
To also run the weight-backed engine unit tests, run `cargo test --locked -p reversi-core` with the workspace `.zst` weight files in place.

### Check the 35-move LRAT certificate

[`NeuralReversi/LegalMoves/Encoding.lean`](NeuralReversi/LegalMoves/Encoding.lean) proves a CNF reduction for disjoint boards with 36 or more legal moves and the soundness of its LRAT checker.
To generate and verify the UNSAT certificate (about 418 MB) with the CaDiCaL binary bundled in the Lean toolchain, run these commands from the repository root:

```sh
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

## Where things live

- [`NeuralReversi/Registry.lean`](NeuralReversi/Registry.lean) lists every registered property (run `verification/lean/.lake/build/bin/revmodel registry` for JSON).
- `NeuralReversi/` holds the Lean models and proofs.
- [`check.py`](check.py), [`DiffMain.lean`](DiffMain.lean), [`difftest/`](difftest/Cargo.toml), and [`.github/workflows/verify-lean.yml`](../../.github/workflows/verify-lean.yml) define the live differential checks and covered `reversi-core` files.

## Scope and assumptions

- **Outside Lean.** Target ISA code generation (AVX2, AVX-512, BMI2, NEON, Wasm SIMD128), `unsafe` memory layout and pointer provenance, `const` table generation outside `difftest/`, OS allocator and large-page APIs, and crates other than `reversi-core`.
- **Trace premises.** Concurrency, split-point, PVS, transposition-table, evaluation-cache, PV, and deadline proofs assume matched atomic read-from, publication, and call histories. Automated trace extraction from native Rust runs remains open.
- **Key side conditions.** AArch64 NEON-SHA3 move generation and batch `flip2` through `flip4` proofs require disjoint bitboards (enforced in Rust by `Board::from_bitboards`). Input-layer and small-network row bounds require every pattern lane below its pattern size; `PatternFeatures::update` keeps this only for moves that fit the stored position, and Rust checks it per evaluation in debug builds only. `EvalCache` key uniqueness requires at least `2^18` entries. ProbCut full-game bounds are conditional on the shallow overestimate fitting the verification margin. `AlignedBuffer` drop proofs cover normal drops and a single unwind, not a double panic or `panic = "abort"`.

## Proof policy and maintenance

- **Axioms.** [`Audit.lean`](Audit.lean) restricts every project declaration to Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`).
- **Forbidden shortcuts.** [`scripts/check_policy.py`](scripts/check_policy.py) and [`AuditCore.lean`](NeuralReversi/AuditCore.lean) reject `sorry`, `sorryAx`, `admit`, `axiom`, `native_decide`, `bv_decide`, `ofReduceBool`, `ofReduceNat`, `trustCompiler`, `implemented_by`, `extern`, `unsafe`, and `partial`, allowing only Lean's compiler-generated `._unsafe_rec` and `ctorIdx._impl`.
- **Compiler replacements.** [`Audit.lean`](Audit.lean) permits only the reviewed `[csimp]` entries listed in [`NeuralReversi/CSimpAudit.lean`](NeuralReversi/CSimpAudit.lean).

When you add, rename, or remove `.lean` files under `verification/lean/`, update `roots` in [`lakefile.toml`](lakefile.toml) if you added a top-level root, then rerun `python3 verification/lean/check.py`; it regenerates the audit inventory.
To run `lake build` or open the project in an editor without `check.py` (for example on a fresh clone), generate the inventory first:

```bash
python3 verification/lean/scripts/audit_modules.py --write
```

When you change a covered `reversi-core` file, update the matching Lean model and rerun `python3 verification/lean/check.py` on each affected backend.
