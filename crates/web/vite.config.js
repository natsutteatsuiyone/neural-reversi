import { defineConfig } from "vite";
import { resolve, sep } from "path";
import { cpSync, existsSync, readFileSync } from "fs";
import wasm from "vite-plugin-wasm";

function serveWasm() {
  return {
    name: "serve-wasm",
    configureServer(server) {
      const root = resolve(import.meta.dirname);
      server.middlewares.use((req, res, next) => {
        if (req.url?.endsWith(".wasm")) {
          let requested;
          try {
            requested = resolve(root, decodeURIComponent(req.url).slice(1));
          } catch {
            res.statusCode = 400;
            res.end();
            return;
          }

          if (!requested.startsWith(root + sep)) {
            res.statusCode = 404;
            res.end();
            return;
          }

          try {
            const data = readFileSync(requested);
            res.setHeader("Content-Type", "application/wasm");
            res.end(data);
            return;
          } catch {
            res.statusCode = 404;
            res.end();
            return;
          }
        }
        next();
      });
    },
  };
}

function reloadOnWasmChange() {
  return {
    name: "reload-on-wasm-change",
    configureServer(server) {
      const wasmPath = resolve(import.meta.dirname, "pkg/web_bg.wasm");
      server.watcher.add(wasmPath);
      server.watcher.on("change", (file) => {
        if (file === wasmPath) {
          server.ws.send({ type: "full-reload" });
        }
      });
    },
  };
}

function copyWasmPackages() {
  const packageDirs = ["pkg", "pkg-relaxed"];

  return {
    name: "copy-wasm-packages",
    writeBundle() {
      for (const dir of packageDirs) {
        const src = resolve(import.meta.dirname, dir);
        if (!existsSync(src)) {
          console.warn(`[copy-wasm-packages] ${dir} not found; skipping copy.`);
          continue;
        }
        cpSync(src, resolve(import.meta.dirname, "dist", dir), {
          recursive: true,
        });
      }
    },
  };
}

export default defineConfig({
  plugins: [serveWasm(), reloadOnWasmChange(), copyWasmPackages(), wasm()],
  resolve: {
    alias: {
      "/pkg": resolve(import.meta.dirname, "pkg"),
    },
  },
  build: {
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
      },
      output: {
        // Enable hash-based cache busting for all assets
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash].[ext]",
      },
    },
  },
  server: {
    port: 8080,
    host: "127.0.0.1",
  },
  worker: {
    format: "es",
    plugins: () => [wasm()],
  },
});
