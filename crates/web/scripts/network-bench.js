#!/usr/bin/env bun
/**
 * Raw neural-network forward benchmark CLI for WebAssembly.
 *
 * Usage:
 *   bun scripts/network-bench.js
 *   bun scripts/network-bench.js --iterations 50000 --warmup 1000
 */

import { parseArgs } from "util";
import { importPreferredWasmModule } from "../wasm-loader.js";

const DEFAULT_ITERATIONS = 10000;
const DEFAULT_WARMUP_ITERATIONS = 1000;

const { values } = parseArgs({
  options: {
    iterations: { type: "string", short: "n", default: String(DEFAULT_ITERATIONS) },
    warmup: { type: "string", short: "w", default: String(DEFAULT_WARMUP_ITERATIONS) },
    help: { type: "boolean", short: "h" },
  },
});

if (values.help) {
  console.log(`
Network forward benchmark CLI (WASM)

Usage:
  bun scripts/network-bench.js [options]

Options:
  -n, --iterations   Timed outer iterations (default: ${DEFAULT_ITERATIONS})
  -w, --warmup       Warmup outer iterations (default: ${DEFAULT_WARMUP_ITERATIONS})
  -h, --help         Show help message

Examples:
  bun scripts/network-bench.js
  bun scripts/network-bench.js --iterations 50000 --warmup 1000
`);
  process.exit(0);
}

function parsePositiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInteger(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return parsed;
}

function formatNumber(value) {
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function formatOpsPerSecond(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M ops/s`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K ops/s`;
  return `${value.toFixed(2)} ops/s`;
}

async function main() {
  const iterations = parsePositiveInteger(values.iterations, "iterations");
  const warmup = parseNonNegativeInteger(values.warmup, "warmup");
  const { module, relaxedSimd } = await importPreferredWasmModule({
    relaxedPath: "./pkg-node-relaxed/web.js",
    fallbackPath: "./pkg-node/web.js",
  });
  const { BenchmarkRunner } = module;

  console.log("Loading evaluation network...");
  console.log(`Wasm SIMD: ${relaxedSimd ? "relaxed-simd" : "simd128"}`);
  const runner = new BenchmarkRunner();
  const positions = runner.network_forward_positions();

  if (warmup > 0) {
    console.log(`Warmup: ${formatNumber(warmup * positions)} evaluations`);
    runner.run_network_forward(warmup);
  }

  const totalEvaluations = iterations * positions;
  console.log(`Running: ${formatNumber(totalEvaluations)} evaluations`);

  const start = performance.now();
  const checksum = runner.run_network_forward(iterations);
  const elapsedMs = performance.now() - start;

  const avgUs = (elapsedMs * 1000) / totalEvaluations;
  const opsPerSecond = totalEvaluations / (elapsedMs / 1000);

  console.log("\n## Network Forward\n");
  console.log(`- Positions per iteration: ${positions}`);
  console.log(`- Iterations: ${formatNumber(iterations)}`);
  console.log(`- Total evaluations: ${formatNumber(totalEvaluations)}`);
  console.log(`- Total time: ${elapsedMs.toFixed(2)}ms`);
  console.log(`- Average time: ${avgUs.toFixed(3)}us/eval`);
  console.log(`- Throughput: ${formatOpsPerSecond(opsPerSecond)}`);
  console.log(`- Checksum: ${checksum}`);

  runner.free();
  process.exit(0);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
