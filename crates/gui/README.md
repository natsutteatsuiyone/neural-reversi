# Reversi GUI

A cross-platform desktop application for playing Reversi (Othello) against a strong AI opponent. Built with Tauri v2 and React.

## Development

### Prerequisites

1. Rust toolchain (version specified in `rust-toolchain.toml`)
2. Bun
3. Neural network weight files in the repository root
   - Download from the GitHub releases page
   - Required for AI functionality

### Setup

1. Clone the repository
2. Ensure weight files are present in the root directory
3. Install dependencies:

```bash
cd crates/gui
bun install
# or with npm: npm install
```

### Running in Development Mode

```bash
# Full Tauri development mode with hot reload
bun tauri dev

# Frontend-only development (without Tauri backend)
bun dev
```

### Building for Production

```bash
# Build for current platform
bun tauri build

# Frontend-only production build
bun run build
```

### Testing

```bash
# Single run
bun run test

# Watch mode
bun run test:watch

# Tauri desktop E2E (Windows/Linux)
bun run e2e
```

> Note: use `bun run test`, not `bun test` — the latter invokes Bun's
> built-in test runner, which cannot run this vitest suite.

The E2E suite runs the built Tauri desktop binary through WebDriverIO and
`tauri-driver`. Install the system driver once before running it:

```bash
cargo install tauri-driver --locked
```

On Windows, the suite uses `msedgedriver.exe` from `PATH` when available and
falls back to the local `edgedriver` package. Set `TAURI_E2E_NATIVE_DRIVER` to
force a specific driver path. By default the suite builds into
`../../target/e2e/debug` so it does not overwrite a running development binary.
Use `TAURI_E2E_APP` to run against an existing binary, `TAURI_E2E_SKIP_BUILD=1`
to skip the automatic debug build, or `TAURI_E2E_TARGET_DIR` to override the
isolated build directory.
