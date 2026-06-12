#!/usr/bin/env bun
/**
 * Simple weight-file match CLI for the WebAssembly engine.
 *
 * Usage:
 *   bun weight-match.js ../../eval_wasm-e6bbc4f6.zst ../../eval_wasm-test1.zst
 *   bun weight-match.js weight-a.zst weight-b.zst --opening-file openings.txt
 */

import { readFileSync } from "fs";
import { basename, resolve } from "path";
import { parseArgs } from "util";
import { importPreferredWasmModule } from "./wasm-loader.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "opening-file": { type: "string", short: "o" },
    details: { type: "boolean" },
    "show-moves": { type: "boolean" },
    name1: { type: "string" },
    name2: { type: "string" },
    help: { type: "boolean", short: "h" },
  },
});

function usage(exitCode = 0) {
  console.log(`
WASM weight match CLI (1-ply)

Usage:
  bun weight-match.js <engine1-weight.zst> <engine2-weight.zst> [options]

Options:
  -o, --opening-file  Opening file in match-runner format
      --details       Print one row per game
      --show-moves    Include full move sequences with --details
      --name1         Display name for engine 1
      --name2         Display name for engine 2
  -h, --help          Show this help message

Examples:
  bun weight-match.js ../../eval_wasm-e6bbc4f6.zst ../../eval_wasm-test1.zst
  bun weight-match.js a.zst b.zst --opening-file ../../openings.txt
`);
  process.exit(exitCode);
}

if (values.help) {
  usage(0);
}

if (positionals.length !== 2) {
  usage(1);
}

function readWeightFile(path) {
  return new Uint8Array(readFileSync(resolve(path)));
}

function readOpeningFile(path) {
  const text = readFileSync(resolve(path), "utf-8");
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
}

function formatSigned(value) {
  const numeric = Number(value);
  return numeric > 0 ? `+${value}` : String(value);
}

function winnerLabel(winner, engine1Name, engine2Name) {
  if (winner === "engine1") return engine1Name;
  if (winner === "engine2") return engine2Name;
  return "Draw";
}

function snapshotResult(result) {
  const snapshot = {
    engine1IsBlack: result.engine1_is_black,
    winner: result.winner,
    blackScore: result.black_score,
    engine1Score: result.engine1_score,
    blackCount: result.black_count,
    whiteCount: result.white_count,
    moves: result.moves,
  };
  result.free();
  return snapshot;
}

function addResult(stats, result) {
  stats.games += 1;
  stats.engine1Score += result.engine1Score;

  if (result.winner === "engine1") {
    stats.engine1Wins += 1;
  } else if (result.winner === "engine2") {
    stats.engine2Wins += 1;
  } else {
    stats.draws += 1;
  }
}

function decideLeader(stats, engine1Name, engine2Name) {
  if (stats.engine1Wins > stats.engine2Wins) return engine1Name;
  if (stats.engine2Wins > stats.engine1Wins) return engine2Name;
  if (stats.engine1Score > 0) return engine1Name;
  if (stats.engine1Score < 0) return engine2Name;
  return "Draw";
}

function printDetailsHeader(showMoves) {
  const movesColumn = showMoves ? " Moves |" : "";
  console.log(`| # | Opening | E1 color | Winner | B-W | E1 score | Discs |${movesColumn}`);
  console.log(
    `|--:|---------|----------|--------|----:|---------:|-------|${showMoves ? "-------|" : ""}`,
  );
}

function printGameRow(gameNumber, opening, result, engine1Name, engine2Name, showMoves) {
  const openingLabel = opening === "" ? "start" : opening;
  const color = result.engine1IsBlack ? "black" : "white";
  const discs = `${result.blackCount}-${result.whiteCount}`;
  const movesColumn = showMoves ? ` ${result.moves} |` : "";
  console.log(
    `| ${gameNumber} | ${openingLabel} | ${color} | ${winnerLabel(result.winner, engine1Name, engine2Name)} | ${formatSigned(result.blackScore)} | ${formatSigned(result.engine1Score)} | ${discs} |${movesColumn}`,
  );
}

async function main() {
  const [engine1Path, engine2Path] = positionals;
  const engine1Name = values.name1 ?? basename(engine1Path);
  const engine2Name = values.name2 ?? basename(engine2Path);
  const openings = values["opening-file"] ? readOpeningFile(values["opening-file"]) : [""];

  if (openings.length === 0) {
    throw new Error("opening file does not contain any playable openings");
  }

  const { module, relaxedSimd } = await importPreferredWasmModule({
    relaxedPath: "./pkg-node-relaxed/web.js",
    fallbackPath: "./pkg-node/web.js",
  });
  const { WeightMatchRunner } = module;

  if (!WeightMatchRunner) {
    throw new Error("WeightMatchRunner is missing; rebuild with `bun run build:wasm:node`");
  }

  console.log("Loading weight files...");
  const runner = new WeightMatchRunner(readWeightFile(engine1Path), readWeightFile(engine2Path));

  console.log(`Wasm SIMD: ${relaxedSimd ? "relaxed-simd" : "simd128"}`);
  console.log(`Engine 1: ${engine1Name}`);
  console.log(`Engine 2: ${engine2Name}`);
  console.log(`Openings: ${openings.length}`);
  console.log(`Games: ${openings.length * 2}\n`);

  if (values.details) {
    printDetailsHeader(values["show-moves"] === true);
  } else {
    console.log("Running matches...");
  }

  const stats = {
    games: 0,
    engine1Wins: 0,
    engine2Wins: 0,
    draws: 0,
    engine1Score: 0,
  };

  let gameNumber = 1;
  for (const opening of openings) {
    for (const engine1IsBlack of [true, false]) {
      const result = snapshotResult(runner.play_game(engine1IsBlack, opening));
      addResult(stats, result);

      if (values.details) {
        printGameRow(
          gameNumber,
          opening,
          result,
          engine1Name,
          engine2Name,
          values["show-moves"] === true,
        );
      }
      gameNumber += 1;
    }
  }

  const avgScore = stats.engine1Score / stats.games;
  const leader = decideLeader(stats, engine1Name, engine2Name);

  console.log("\n## Summary\n");
  console.log(`Openings: ${openings.length}`);
  console.log(`Games: ${stats.games}`);
  console.log(`${engine1Name}: ${stats.engine1Wins} wins`);
  console.log(`${engine2Name}: ${stats.engine2Wins} wins`);
  console.log(`Draws: ${stats.draws}`);
  console.log(
    `Engine 1 score: ${formatSigned(stats.engine1Score)} (${formatSigned(avgScore.toFixed(2))}/game)`,
  );
  console.log(`Stronger: ${leader}`);

  runner.free();
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
