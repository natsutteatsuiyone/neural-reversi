# Reversi GUI

A native Rust desktop application for playing and analyzing Reversi (Othello), built with GPUI Kit 0.6.6 (GPUI 0.3.6 and gpui-component 0.6.6). The `gui` package and binary live in `crates/gui/Cargo.toml`.

## Architecture and functionality

The GPUI window renders the board, game controls, move history, analysis, solver, and settings directly. The Rust game model owns positions and history. A background engine worker calls `reversi-core` directly and delivers search progress/results over channels, keeping searches off the UI thread. There is no browser frontend, IPC bridge, JavaScript runtime, or substitute AI implementation.

- Play against the engine as either color, or play locally with two people.
- Choose level-based or time-limited play; stop and resume engine activity.
- Navigate move history, undo/redo, and copy game transcripts.
- Set up positions manually or import transcripts and board strings.
- Inspect move hints, search progress, principal variations, and game analysis.
- Explore positions in the solver with selectable search modes and selectivity.
- Adjust language, engine/search settings, and view application and third-party license information.

The native interface retains the original neutral palette, compact toolbar, vertically stacked player cards, tabbed analysis area, and header settings menu. The top-down board uses the original frame/grid proportions and cached material textures baked from the former Three.js scene, with native placement/flip animations; the setup editor retains its separate flat preview. On launch, an AI-first game waits for **Start**; subsequent AI turns run automatically. **Stop** commits the best move found so far. Left/right arrow keys browse the retained history when no game search is active; a new move after undo replaces the discarded continuation. Hints can be interrupted by playing a move.

During a flip, the stone image follows the animated frame's width and height around the cell center; its intrinsic square image size must not participate in normal-flow layout.

GPUI has no backdrop blur, so modal backgrounds use a translucent dim overlay and the floating history toolbar a near-opaque panel. Board coordinates embed the original Noto Sans glyphs, while setup previews and player cards retain the original flat stone materials.

Setup input validates transcripts and board strings as they are edited. Invalid text shows its localized error without discarding the last valid preview, and prevents starting that position. Slider controls support pointer dragging and the original arrow, Home/End, and Page Up/Down keyboard behavior.

The solver has its own position history and completed-result cache; history navigation includes automatic passes in the preceding move. Stop/resume, candidate selection, and selectivity changes invalidate older search results immediately. Exiting restores the live game's position, history, and analysis. The analysis chart uses the black player's perspective and links back to the move history. Desktop panel proportions are adjusted by dragging the vertical and horizontal dividers and are saved automatically. Below 1024px, the sidebar stacks beneath the board.

The WebAssembly application in `crates/web` is separate and continues to use Bun/Vite.

## Prerequisites

- Rust via [rustup](https://rustup.rs/), plus the native build tools for your OS.
- Download `eval-*.zst` and `eval_sm-*.zst` from the [weights releases](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/latest), placing them in the repository root **before compiling**. The engine embeds its weights in the executable; distributed binaries do not need separate weight files.
- A graphical desktop session and a compatible GPU/graphics driver to launch the application. Headless compilation and model/engine tests do not exercise window rendering.

### Windows

Use the MSVC Rust toolchain and Visual Studio Build Tools with the Desktop development with C++ workload and Windows SDK. Release builds need the SDK's `fxc.exe` shader compiler; set `GPUI_FXC_PATH` to its full path if GPUI cannot discover the installed SDK. GPUI uses the native Windows graphics backend; a working GPU driver is required. WebView2 and browser drivers are not dependencies.

### Linux

GPUI Kit enables GPUI's X11, Wayland, and Fontconfig/font-kit integration. On Ubuntu 24.04/Debian-family systems, install:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang cmake pkg-config libclang-dev \
  libasound2-dev libfontconfig1-dev libfreetype6-dev libglib2.0-dev \
  libssl-dev libwayland-dev libx11-xcb-dev libxkbcommon-x11-dev \
  libxcb-shape0-dev libxcb-xfixes0-dev libvulkan-dev
```

CI installs the same dependencies through `.github/actions/setup-rust`. Other distributions need the equivalent development packages. At runtime, use an X11 or Wayland session, installed fonts, the Vulkan loader, and a compatible Vulkan driver (for example, the appropriate Mesa or vendor driver). A display server alone is not sufficient if its graphics driver cannot support GPUI. The binary is dynamically linked to system libraries; it is not a self-contained AppImage and may not run on systems older than its build environment. GTK/WebKit development packages are not required for the GUI.

### macOS

Install full Xcode and select its developer directory with `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`. GPUI Kit enables GPUI's runtime shader compilation, so the build does not invoke `xcrun metal`. A Metal-capable macOS environment is required. The distribution task currently targets Apple Silicon (`aarch64-apple-darwin`); Intel macOS builds are not provided by that task. Packaging additionally uses the system `codesign`, `ditto`, and `hdiutil` commands.

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

CI checks and tests the native package on Windows, Linux, and macOS; workspace linting also includes it. Tests exercise Rust model and engine behavior, not a WebDriver/browser surface. For a UI smoke check, launch `cargo run -p gui`, play a legal move, start and stop a search, navigate history, and open setup/settings on an actual desktop.

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
