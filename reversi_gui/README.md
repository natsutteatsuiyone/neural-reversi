# Reversi GUI

A cross-platform desktop application for playing Reversi (Othello) against a strong AI opponent. Built with Tauri v2 and React.

## Development

### Prerequisites

1. Rust toolchain (version specified in `rust-toolchain.toml`)
2. Node.js and Bun (or npm)
3. Neural network weight files in the repository root
   - Download from the GitHub releases page
   - Required for AI functionality

### Setup

1. Clone the repository
2. Ensure weight files are present in the root directory
3. Install dependencies:

```bash
cd reversi_gui
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
bun build
```
