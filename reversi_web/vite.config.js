import { defineConfig } from 'vite';
import { resolve } from 'path';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait()
  ],
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
