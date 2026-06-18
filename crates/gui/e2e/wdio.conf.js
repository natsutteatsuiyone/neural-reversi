import { existsSync } from "node:fs";
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

  const result = spawnSync("bun", ["run", "tauri", "build", "--debug", "--no-bundle"], {
    cwd: guiDir,
    env: {
      ...process.env,
      CARGO_TARGET_DIR: e2eTargetDir,
    },
    stdio: "inherit",
    shell: true,
  });

  if (result.status !== 0) {
    throw new Error(`Failed to build the Tauri application for E2E testing (exit ${result.status}).`);
  }
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
    verifyApplicationExists();
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
