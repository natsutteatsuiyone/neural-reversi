import { defineConfig } from 'vite';
import { resolve } from 'path';
import { cpSync, existsSync, readFileSync } from 'fs';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

function serveWasm() {
  return {
    name: 'serve-wasm',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.endsWith('.wasm')) {
          const filePath = resolve(__dirname, req.url.slice(1));
          try {
            const data = readFileSync(filePath);
            res.setHeader('Content-Type', 'application/wasm');
            res.end(data);
            return;
          } catch {
            // fall through to next middleware
          }
        }
        next();
      });
    }
  };
}

function copyWasmPackages() {
  const packageDirs = ['pkg', 'pkg-relaxed'];

  return {
    name: 'copy-wasm-packages',
    writeBundle() {
      for (const dir of packageDirs) {
        const src = resolve(__dirname, dir);
        if (!existsSync(src)) {
          console.warn(`[copy-wasm-packages] ${dir} not found; skipping copy.`);
          continue;
        }
        cpSync(src, resolve(__dirname, 'dist', dir), {
          recursive: true,
        });
      }
    }
  };
}

export default defineConfig({
  plugins: [
    serveWasm(),
    copyWasmPackages(),
    wasm(),
    topLevelAwait()
  ],
  resolve: {
    alias: {
      '/pkg': resolve(__dirname, 'pkg')
    }
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: process.env.BENCHMARK === 'true'
        ? {
            main: resolve(__dirname, 'index.html'),
            benchmark: resolve(__dirname, 'benchmark.html')
          }
        : {
            main: resolve(__dirname, 'index.html')
          },
      output: {
        // Enable hash-based cache busting for all assets
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  server: {
    port: 8080,
    host: '127.0.0.1'
  },
  worker: {
    format: 'es',
    plugins: () => [
      wasm(),
      topLevelAwait()
    ]
  }
});
