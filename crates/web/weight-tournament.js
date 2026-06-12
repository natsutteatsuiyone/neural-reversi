#!/usr/bin/env bun
/**
 * Winner-stays tournament CLI for WebAssembly weight files.
 *
 * Usage:
 *   bun weight-tournament.js <weights-dir> --opening-file <openings.txt>
 */

import { readdirSync, readFileSync } from "fs";
import { basename, join, resolve } from "path";
import { parseArgs } from "util";
import { importPreferredWasmModule } from "./wasm-loader.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "opening-file": { type: "string", short: "o" },
    help: { type: "boolean", short: "h" },
  },
});

function usage(exitCode = 0) {
  console.log(`
WASM weight tournament CLI (1-ply, winner-stays)

Usage:
  bun weight-tournament.js <weights-dir> [options]

Options:
  -o, --opening-file  Opening file in match-runner format
  -h, --help          Show this help message

Examples:
  bun weight-tournament.js <weights-dir> --opening-file <openings.txt>
  bun weight-tournament.js ../../weights --opening-file ../../openings.txt
`);
  process.exit(exitCode);
}

if (values.help) {
  usage(0);
}

if (positionals.length !== 1) {
  usage(1);
}

function readOpeningFile(path) {
  const text = readFileSync(resolve(path), "utf-8");
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
}

function readWeightFiles(dir) {
  const root = resolve(dir);
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".zst"))
    .map((entry) => ({
      name: entry.name,
      path: join(root, entry.name),
      bytes: new Uint8Array(readFileSync(join(root, entry.name))),
    }))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

function snapshotResult(result) {
  const snapshot = {
    winner: result.winner,
    engine1Score: result.engine1_score,
  };
  result.free();
  return snapshot;
}

function formatSigned(value) {
  return value > 0 ? `+${value}` : String(value);
}

function compareMatchResult(stats) {
  if (stats.engine1Wins > stats.engine2Wins) return "engine1";
  if (stats.engine2Wins > stats.engine1Wins) return "engine2";
  if (stats.engine1Score > 0) return "engine1";
  if (stats.engine1Score < 0) return "engine2";
  return "draw";
}

function playMatch(WeightMatchRunner, champion, challenger, openings) {
  const runner = new WeightMatchRunner(champion.bytes, challenger.bytes);
  const stats = {
    engine1Wins: 0,
    engine2Wins: 0,
    draws: 0,
    engine1Score: 0,
  };

  for (const opening of openings) {
    for (const engine1IsBlack of [true, false]) {
      const result = snapshotResult(runner.play_game(engine1IsBlack, opening));
      stats.engine1Score += result.engine1Score;

      if (result.winner === "engine1") {
        stats.engine1Wins += 1;
      } else if (result.winner === "engine2") {
        stats.engine2Wins += 1;
      } else {
        stats.draws += 1;
      }
    }
  }

  runner.free();
  return {
    ...stats,
    winner: compareMatchResult(stats),
  };
}

function winnerName(result, champion, challenger) {
  if (result.winner === "engine1") return champion.name;
  if (result.winner === "engine2") return challenger.name;
  return "Draw";
}

async function main() {
  const weightsDir = positionals[0];
  const weights = readWeightFiles(weightsDir);
  if (weights.length < 2) {
    throw new Error("weights-dir must contain at least two .zst files");
  }

  const openings = values["opening-file"] ? readOpeningFile(values["opening-file"]) : [""];
  if (openings.length === 0) {
    throw new Error("no playable openings found");
  }

  const { module, relaxedSimd } = await importPreferredWasmModule({
    relaxedPath: "./pkg-node-relaxed/web.js",
    fallbackPath: "./pkg-node/web.js",
  });
  const { WeightMatchRunner } = module;
  if (!WeightMatchRunner) {
    throw new Error("WeightMatchRunner is missing; rebuild with `bun run build:wasm:node`");
  }

  const totalComparisons = weights.length - 1;
  const gamesPerComparison = openings.length * 2;

  console.log(`Wasm SIMD: ${relaxedSimd ? "relaxed-simd" : "simd128"}`);
  console.log(`Weights: ${weights.length}`);
  console.log(`Openings: ${openings.length}`);
  console.log(`Comparisons: ${totalComparisons}`);
  console.log(`Games: ${totalComparisons * gamesPerComparison}`);
  console.log("Tie policy: exact drawn comparisons keep the incumbent.\n");

  let champion = weights[0];

  for (let i = 1; i < weights.length; i += 1) {
    const challenger = weights[i];
    const result = playMatch(WeightMatchRunner, champion, challenger, openings);
    const previousChampion = champion;

    if (result.winner === "engine2") {
      champion = challenger;
    }

    const line =
      `[${i}/${totalComparisons}] ${basename(previousChampion.name)} vs ${basename(challenger.name)}: ` +
      `${result.engine1Wins}-${result.engine2Wins}-${result.draws}, ` +
      `score ${formatSigned(result.engine1Score)}; winner ${winnerName(result, previousChampion, challenger)}; ` +
      `champion ${champion.name}`;
    console.log(line);
  }

  console.log("\n## Result\n");
  console.log(`Strongest: ${champion.name}`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
