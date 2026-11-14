# Reversi GUI

A native Rust desktop application for playing and analyzing Reversi (Othello), built with Native Shell (NSH) 0.5.1. The `gui` package and binary live in `crates/gui/Cargo.toml`.

## Architecture and functionality

NSH renders the board, game controls, move history, analysis, solver, and settings directly. The Rust game model owns positions and history. A background engine worker calls `reversi-core` directly and delivers search progress/results over channels, keeping searches off the UI thread. There is no browser frontend, IPC bridge, JavaScript runtime, or substitute AI implementation.

- Play against the engine as either color, or play locally with two people.
- Choose level-based or time-limited play; stop and resume engine activity.
- Navigate move history, undo/redo, and copy game transcripts.
- Set up positions manually or import transcripts and board strings.
- Inspect move hints, search progress, principal variations, and game analysis.
- Explore positions in the solver with selectable search modes and selectivity.
- Adjust language, engine/search settings, and view application and third-party license information.

The native interface retains the original neutral palette, compact toolbar, vertically stacked player cards, tabbed analysis area, and header settings menu. The top-down board uses the original frame/grid proportions and cached material textures baked from the former Three.js scene, with native placement/flip animations; the setup editor retains its separate flat preview. AI turns run automatically. **Stop** commits the best move found so far. Left/right arrow keys browse the retained history when no game search is active; a new move after undo replaces the discarded continuation. Hints can be interrupted by playing a move.

During a flip, the stone image follows the animated frame's width and height around the cell center; its intrinsic square image size must not participate in normal-flow layout.

Modal backgrounds use a translucent dim overlay, and the floating history toolbar uses a near-opaque panel. Board coordinates embed the original Noto Sans glyphs, while setup previews and player cards retain the original flat stone materials.

Setup input validates transcripts and board strings as they are edited. Invalid text shows its localized error without discarding the last valid preview, and prevents starting that position. Slider controls support pointer dragging and the original arrow, Home/End, and Page Up/Down keyboard behavior.

The solver has its own position history and completed-result cache; history navigation includes automatic passes in the preceding move. Stop/resume, candidate selection, and selectivity changes invalidate older search results immediately. Exiting restores the live game's position, history, and analysis. The analysis chart uses the black player's perspective and links back to the move history. Desktop panel proportions are adjusted by dragging the vertical and horizontal dividers and are saved automatically. Below 1024px, the sidebar stacks beneath the board.

The WebAssembly application in `crates/web` is separate and continues to use Bun/Vite.

## Prerequisites

- Rust via [rustup](https://rustup.rs/), plus the native build tools for your OS.
- Download `eval-*.zst` and `eval_sm-*.zst` from the [weights releases](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/latest), placing them in the repository root **before compiling**. The engine embeds its weights in the executable; distributed binaries do not need separate weight files.
- A graphical desktop session to launch the application. Headless UI tests exercise the retained element tree and event handling, but do not verify native window rendering.

### Windows

Use Windows 10 version 1809 or later, the MSVC Rust toolchain, and Visual Studio Build Tools with the Desktop development with C++ workload and Windows SDK. NSH uses Win32, Direct2D, and DirectWrite. The GUI does not create web views. The backend includes WebView2 bindings, but this application does not need a WebView2 runtime for its interface.

### Linux

NSH requires GTK 4.14 or later. On Ubuntu 24.04, install:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang cmake pkg-config libclang-dev libssl-dev \
  libgtk-4-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

CI installs the same dependencies through `.github/actions/setup-rust`. Other distributions need the equivalent development packages and GTK version. NSH's Linux backend links GStreamer even though the GUI does not use media controls. The GUI disables NSH's optional `webview` feature, so WebKitGTK development packages are not required.

At runtime, use an X11 or Wayland desktop session with GTK 4.14 or later and GStreamer. The binary is dynamically linked to system libraries. It is not a self-contained AppImage and may not run on systems older than its build environment.

### macOS

Use macOS 14 or later and install Xcode's command line tools with `xcode-select --install`. NSH uses AppKit, Core Animation, and CoreText. The distribution task currently targets Apple Silicon (`aarch64-apple-darwin`). It does not provide Intel macOS builds. Packaging additionally uses the system `codesign`, `ditto`, and `hdiutil` commands.

## Development

Run these commands from the repository root:

```bash
cargo run -p gui
cargo run -p gui --release
```

Build the native executable without launching a window:

```bash
cargo build -p gui --release
```

The result is `target/release/gui.exe` on Windows and `target/release/gui` on Linux/macOS. No frontend dependency installation or development server is needed.

### Checks and tests

```bash
cargo check -p gui --all-targets
cargo test -p gui --all-targets
cargo clippy -p gui --all-targets -- -D warnings
cargo fmt --all -- --check
```

CI tests the workspace, including the GUI, on Windows and Linux. Workspace linting also includes the GUI. The tests cover the Rust model, engine integration, and NSH headless UI flows. Headless tests do not verify native drawing, fonts, text input, or clipboard integration. To check those behaviors, launch `cargo run -p gui`, play a legal move, start and stop a search, navigate history, copy a transcript, and open setup/settings on an actual desktop. macOS builds and native rendering need separate verification.

## Configuration persistence

Settings are saved to `neural-reversi/native-settings.json` beneath the user's application data directory:

- Windows: `%APPDATA%\neural-reversi\native-settings.json`.
- Linux: `$XDG_DATA_HOME/neural-reversi/native-settings.json`, or `~/.local/share/neural-reversi/native-settings.json` when unset.
- macOS: `~/Library/Application Support/neural-reversi/native-settings.json`.

Saved preferences include game mode, AI level or time limit, hint/analysis levels, hash size, analysis visibility, language, solver mode/selectivity, and panel sizes. Settings are not a saved-game file. On first load, the native application imports the neighboring legacy `settings.json` when available (including the former versioned frontend settings envelope), leaving that file untouched. Invalid settings or filesystem errors are surfaced by the application rather than silently discarding preferences. The automatic language setting uses the OS locale.

## Distribution and platform notes

Install [cargo-make](https://github.com/sagiegurari/cargo-make), then build on the destination OS:

```bash
cargo make build-dist gui windows x86-64-v3
cargo make build-dist gui linux x86-64-v3
cargo make build-dist gui macos apple-m1
```

The task builds `cargo build --release -p gui --bin gui --target ...` and writes versioned artifacts to `dist/`:

- Windows: `neural-reversi-gui-<version>-windows-x86_64-<tier>.exe`.
- Linux: `neural-reversi-gui-<version>-linux-x86_64_<tier>`.
- macOS: `Neural Reversi.app` and `neural-reversi-gui-<version>-aarch64-apple-darwin.dmg`. The app contains the executable, an `Info.plist`, and `assets/icon.icns`; the disk image includes an Applications shortcut.

`native` targets the build machine's CPU. `portable` builds the x86-64-v2/v3/v4 variants on Windows/Linux and Apple Silicon on macOS. Select an executable compatible with the destination CPU; v3 needs AVX2 and v4 needs AVX-512. Platform SDKs and cross-compilers are not installed by this task, so selecting `all` platforms is not a substitute for building on each OS.

macOS bundles are ad-hoc signed, **not Developer ID signed or notarized**. Gatekeeper may require explicit approval of downloaded builds. Release signing/notarization and Windows installer/signing are not part of the distribution task. Native build CI is not evidence that window rendering was tested on every GPU or display server.

### Third-party licenses

Install [cargo-about](https://github.com/EmbarkStudios/cargo-about), then regenerate the Rust dependency notices:

```bash
cargo make licenses-gui
```

This runs cargo-about against `crates/gui/Cargo.toml` and writes `crates/gui/THIRD_PARTY_LICENSES.txt`. There is no JavaScript runtime or license bundle. The retained Lucide SVG assets have their upstream license in `assets/icons/LICENSE`; the embedded Noto Sans font is covered by the SIL Open Font License in `assets/fonts/LICENSE`. The application's Third-Party tab includes both notices alongside the Rust dependency notices. Regenerate the Rust notices after changing dependencies before distributing the GUI.
