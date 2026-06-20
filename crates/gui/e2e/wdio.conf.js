import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const e2eDir = fileURLToPath(new URL(".", import.meta.url));
const guiDir = path.resolve(e2eDir, "..");
const repoRoot = path.resolve(guiDir, "../..");
const executableName = process.platform === "win32" ? "gui.exe" : "gui";
const e2eTargetDir = process.env.TAURI_E2E_TARGET_DIR
  ? path.resolve(guiDir, process.env.TAURI_E2E_TARGET_DIR)
  : path.resolve(repoRoot, "target", "e2e");
const defaultApplication = path.resolve(e2eTargetDir, "debug", executableName);
const application = process.env.TAURI_E2E_APP
  ? path.resolve(guiDir, process.env.TAURI_E2E_APP)
  : defaultApplication;
const specPattern = path.join(e2eDir, "specs/**/*.e2e.js").replaceAll(path.sep, "/");

// The specs assert the launch auto-start behaviour, which depends on the
// persisted game mode. To control it deterministically WITHOUT touching the
// developer's real config, the e2e build overrides the Tauri identifier (see
// `tauri.e2e.conf.json`, merged via `--config`). The app then resolves its
// data dir (tauri-plugin-store, BaseDirectory::AppData) to a dedicated
// `<app data dir>/<e2e identifier>/` that we can seed and wipe freely.
//
// This is why we do not override env vars instead: on Windows the data dir
// comes from SHGetKnownFolderPath (FOLDERID_RoamingAppData), which ignores
// APPDATA, so an env override would silently read the developer's real
// settings. A separate identifier isolates the data dir on every platform.
const e2eConfigPath = path.join(e2eDir, "tauri.e2e.conf.json");
const E2E_IDENTIFIER = JSON.parse(readFileSync(e2eConfigPath, "utf8")).identifier;
const PREBUILT_IDENTIFIER_ENV = "TAURI_E2E_PREBUILT_IDENTIFIER";
const usesPrebuiltApplication =
  Boolean(process.env.TAURI_E2E_APP) || process.env.TAURI_E2E_SKIP_BUILD === "1";

let windowsRoamingAppData = null;

function resolveWindowsRoamingAppData() {
  if (windowsRoamingAppData) return windowsRoamingAppData;

  const powershell = process.env.SystemRoot
    ? path.join(
        process.env.SystemRoot,
        "System32",
        "WindowsPowerShell",
        "v1.0",
        "powershell.exe",
      )
    : "powershell.exe";
  const result = spawnSync(
    powershell,
    [
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      "[Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)",
    ],
    { encoding: "utf8", windowsHide: true },
  );

  const roaming = (result.stdout ?? "").trim();
  if (result.status === 0 && roaming) {
    windowsRoamingAppData = roaming;
    return windowsRoamingAppData;
  }

  const stderr = (result.stderr ?? "").trim();
  const details = result.error?.message ?? (stderr || `PowerShell exited with ${result.status}`);
  throw new Error(`Failed to resolve Windows roaming AppData for Tauri E2E: ${details}`);
}

function e2eDataDir() {
  if (process.platform === "win32") {
    return path.join(resolveWindowsRoamingAppData(), E2E_IDENTIFIER);
  }

  const dataHome = process.env.XDG_DATA_HOME ?? path.join(homedir(), ".local", "share");
  return path.join(dataHome, E2E_IDENTIFIER);
}

// Seed a deterministic launch: ai-black so the AI plays black (moves first) and
// the launch starts paused; level 1 so the AI's move after Start resolves fast
// in the unoptimized e2e build. Other settings fall back to their defaults.
const E2E_SETTINGS = JSON.stringify({ gameMode: "ai-black", aiMode: "level", aiLevel: 1 });

function seedE2ESettings() {
  const dir = e2eDataDir();
  // Start from a clean, isolated profile. This dir belongs to the e2e
  // identifier only, so wiping it never affects the developer's real data.
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, "settings.json"), E2E_SETTINGS, "utf8");
}

function cleanupE2ESettings() {
  rmSync(e2eDataDir(), { recursive: true, force: true });
}

let tauriDriver = null;
let nativeWebDriver = null;

function executableNames(name) {
  if (process.platform !== "win32") return [name];

  const extensions = (process.env.PATHEXT ?? ".EXE;.CMD;.BAT;.COM")
    .split(";")
    .filter(Boolean);
  const lowerName = name.toLowerCase();

  if (extensions.some((extension) => lowerName.endsWith(extension.toLowerCase()))) {
    return [name];
  }

  return extensions.map((extension) => `${name}${extension.toLowerCase()}`);
}

function findOnPath(name) {
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    if (!directory) continue;

    for (const executable of executableNames(name)) {
      const candidate = path.join(directory, executable);
      if (existsSync(candidate)) return candidate;
    }
  }

  return null;
}

function resolveTauriDriver() {
  if (process.env.TAURI_DRIVER) {
    return path.resolve(guiDir, process.env.TAURI_DRIVER);
  }

  const pathDriver = findOnPath("tauri-driver");
  if (pathDriver) return pathDriver;

  const cargoDriver = path.join(
    homedir(),
    ".cargo",
    "bin",
    process.platform === "win32" ? "tauri-driver.exe" : "tauri-driver",
  );
  if (existsSync(cargoDriver)) return cargoDriver;

  throw new Error(
    "tauri-driver was not found. Install it with `cargo install tauri-driver --locked`, " +
      "or set TAURI_DRIVER to the tauri-driver executable path.",
  );
}

function tauriDriverArgs() {
  if (!nativeWebDriver) return [];
  return ["--native-driver", nativeWebDriver];
}

function verifyNativeWebDriver() {
  if (process.env.TAURI_E2E_NATIVE_DRIVER) {
    const nativeDriver = path.resolve(guiDir, process.env.TAURI_E2E_NATIVE_DRIVER);
    if (!existsSync(nativeDriver)) {
      throw new Error(`TAURI_E2E_NATIVE_DRIVER does not exist: ${nativeDriver}`);
    }
    nativeWebDriver = nativeDriver;
    return;
  }

  if (process.platform === "win32" && findOnPath("msedgedriver")) {
    return;
  }

  if (process.platform === "win32") {
    const localEdgeDriver = path.join(guiDir, "node_modules", ".bin", "edgedriver.exe");
    if (existsSync(localEdgeDriver)) {
      nativeWebDriver = localEdgeDriver;
      return;
    }

    throw new Error(
      "msedgedriver.exe was not found on PATH. Install the Microsoft Edge Driver " +
        "matching your Edge version, run `bun install`, or set TAURI_E2E_NATIVE_DRIVER to its path.",
    );
  }

  if (process.platform === "linux" && !findOnPath("WebKitWebDriver")) {
    throw new Error(
      "WebKitWebDriver was not found on PATH. Install your distribution's WebKit WebDriver package.",
    );
  }
}

function verifySupportedPlatform() {
  if (process.platform === "darwin") {
    throw new Error(
      "Tauri desktop WebDriver E2E is not supported on macOS by tauri-driver. " +
        "Run this suite on Windows or Linux.",
    );
  }
}

function buildApplication() {
  if (process.env.TAURI_E2E_SKIP_BUILD === "1") return;
  if (process.env.TAURI_E2E_APP) return;

  // `--config` merges the e2e identifier override so the built app isolates its
  // data dir from the developer's real install.
  const result = spawnSync(
    "bun",
    ["run", "tauri", "build", "--debug", "--no-bundle", "--config", e2eConfigPath],
    {
      cwd: guiDir,
      env: {
        ...process.env,
        CARGO_TARGET_DIR: e2eTargetDir,
      },
      stdio: "inherit",
      shell: true,
    },
  );

  if (result.status !== 0) {
    throw new Error(`Failed to build the Tauri application for E2E testing (exit ${result.status}).`);
  }
}

function verifyPrebuiltApplicationIdentifier() {
  if (!usesPrebuiltApplication) return;
  if (process.env[PREBUILT_IDENTIFIER_ENV] === E2E_IDENTIFIER) return;

  const prebuiltMode = process.env.TAURI_E2E_APP ? "TAURI_E2E_APP" : "TAURI_E2E_SKIP_BUILD=1";
  const prebuiltEnvName = process.env.TAURI_E2E_APP ? "TAURI_E2E_APP" : "TAURI_E2E_SKIP_BUILD";
  const actualIdentifier = process.env[PREBUILT_IDENTIFIER_ENV]
    ? `Got ${PREBUILT_IDENTIFIER_ENV}=${process.env[PREBUILT_IDENTIFIER_ENV]}. `
    : "";

  throw new Error(
    `${prebuiltMode} uses a prebuilt Tauri binary, so the e2e harness cannot apply ` +
      `${path.relative(guiDir, e2eConfigPath)}. The launch specs seed only the isolated ` +
      `${E2E_IDENTIFIER} profile; a binary with another identifier would read the wrong ` +
      `settings and may touch the developer profile. ${actualIdentifier}` +
      `Unset ${prebuiltEnvName} so the harness builds with --config, or build the binary ` +
      `with the e2e config and set ${PREBUILT_IDENTIFIER_ENV}=${E2E_IDENTIFIER}.`,
  );
}

function verifyApplicationExists() {
  if (existsSync(application)) return;

  throw new Error(
    `Tauri application binary was not found at ${application}. ` +
      "Run `bun run tauri build --debug --no-bundle` or set TAURI_E2E_APP.",
  );
}

function closeTauriDriver() {
  if (tauriDriver) {
    tauriDriver.kill();
    tauriDriver = null;
  }
}

function onShutdown(fn) {
  const cleanup = () => {
    try {
      fn();
    } finally {
      process.exit();
    }
  };

  process.once("SIGINT", cleanup);
  process.once("SIGTERM", cleanup);
  process.once("exit", fn);
}

verifySupportedPlatform();
const tauriDriverPath = resolveTauriDriver();
verifyNativeWebDriver();

export const config = {
  runner: "local",
  host: "127.0.0.1",
  port: 4444,
  specs: [specPattern],
  maxInstances: 1,
  logLevel: "error",
  bail: 0,
  waitforTimeout: 10000,
  connectionRetryTimeout: 120000,
  connectionRetryCount: 1,
  capabilities: [
    {
      maxInstances: 1,
      "tauri:options": {
        application,
      },
    },
  ],
  reporters: ["spec"],
  framework: "mocha",
  mochaOpts: {
    ui: "bdd",
    timeout: 60000,
  },
  onPrepare: () => {
    buildApplication();
    verifyPrebuiltApplicationIdentifier();
    verifyApplicationExists();
    seedE2ESettings();
    // Launcher-only tidy-up if onComplete never runs. Wiping the isolated e2e
    // data dir is idempotent and never touches the developer's real data.
    process.once("exit", cleanupE2ESettings);
  },
  onComplete: () => {
    cleanupE2ESettings();
  },
  beforeSession: async () => {
    tauriDriver = spawn(tauriDriverPath, tauriDriverArgs(), {
      stdio: [null, process.stdout, process.stderr],
    });

    tauriDriver.on("error", (error) => {
      console.error("tauri-driver error:", error);
      process.exit(1);
    });

    await new Promise((resolve) => setTimeout(resolve, 500));

    if (tauriDriver.exitCode !== null) {
      throw new Error(`tauri-driver exited before the WebDriver session started (${tauriDriver.exitCode}).`);
    }
  },
  afterSession: () => {
    closeTauriDriver();
  },
};

onShutdown(closeTauriDriver);
