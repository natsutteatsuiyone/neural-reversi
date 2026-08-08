#!/usr/bin/env bun
/**
 * Balanced strength tournament CLI for WebAssembly weight files.
 *
 * Usage:
 *   bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt>
 */

import { readdirSync, readFileSync } from "fs";
import { basename, join, resolve } from "path";
import { parseArgs } from "util";
import { importNodeWasm } from "../wasm-loader.js";

const CLI_OPTIONS = {
  "opening-file": { type: "string", short: "o" },
  jobs: { type: "string", short: "j" },
  help: { type: "boolean", short: "h" },
};

// Up to this many weights every pair is played. Larger fields run a sparse
// league instead, so the comparison count stays linear in the field size.
const FULL_ROUND_ROBIN_LIMIT = 8;
const SPARSE_ROUNDS = 4;

function parseCliArgs(args = process.argv.slice(2)) {
  return parseArgs({
    args,
    allowPositionals: true,
    options: CLI_OPTIONS,
  });
}

function usage(exitCode = 0) {
  console.log(`
WASM weight tournament CLI (1-ply)

Usage:
  bun scripts/weight-tournament.js <weights-dir> [options]

Options:
  -o, --opening-file  Opening file in match-runner format
  -j, --jobs          Parallel comparisons. Default: 1
  -h, --help          Show this help message

Every pair is played once for up to ${FULL_ROUND_ROBIN_LIMIT} weights; larger fields play
${SPARSE_ROUNDS} rounds, paired by current standing.

Examples:
  bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt>
  bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt> --jobs 4
  bun scripts/weight-tournament.js ../../weights --opening-file ../../openings.txt
`);
  process.exit(exitCode);
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
    }))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

function loadWeightBytes(weight) {
  if (weight.bytes !== undefined) {
    return weight;
  }

  weight.bytes = new Uint8Array(readFileSync(weight.path));
  return weight;
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
  const numeric = Number(value);
  return numeric > 0 ? `+${value}` : String(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function parsePositiveInteger(value, optionName) {
  if (value === undefined) {
    return undefined;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || String(parsed) !== value.trim()) {
    throw new Error(`${optionName} must be a positive integer`);
  }

  return parsed;
}

function compareMatchResult(stats) {
  if (stats.engine1Wins > stats.engine2Wins) return "engine1";
  if (stats.engine2Wins > stats.engine1Wins) return "engine2";
  if (stats.engine1Score > 0) return "engine1";
  if (stats.engine1Score < 0) return "engine2";
  return "draw";
}

function playMatch(WeightMatchRunner, engine1, engine2, openings) {
  const runner = new WeightMatchRunner(engine1.bytes, engine2.bytes);
  const stats = {
    games: 0,
    engine1Wins: 0,
    engine2Wins: 0,
    draws: 0,
    engine1Score: 0,
  };

  try {
    for (const opening of openings) {
      for (const engine1IsBlack of [true, false]) {
        const result = snapshotResult(runner.play_game(engine1IsBlack, opening));
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
    }
  } finally {
    runner.free();
  }

  return {
    ...stats,
    winner: compareMatchResult(stats),
  };
}

function winnerName(result, engine1, engine2) {
  if (result.winner === "engine1") return engine1.name;
  if (result.winner === "engine2") return engine2.name;
  return "Draw";
}

function createStanding() {
  return {
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    gamePoints: 0,
    discScore: 0,
  };
}

function scoreRate(standing) {
  return standing.games === 0 ? 0.5 : standing.gamePoints / standing.games;
}

function averageDiscScore(standing) {
  return standing.games === 0 ? 0 : standing.discScore / standing.games;
}

function compareStandings(a, b, standings) {
  const standingA = standings.get(a.name);
  const standingB = standings.get(b.name);
  return (
    scoreRate(standingB) - scoreRate(standingA) ||
    averageDiscScore(standingB) - averageDiscScore(standingA) ||
    a.name.localeCompare(b.name, undefined, { numeric: true })
  );
}

function pairKey(a, b) {
  return a.name < b.name ? `${a.name}\0${b.name}` : `${b.name}\0${a.name}`;
}

function rankedWeights(weights, standings) {
  return [...weights].sort((a, b) => compareStandings(a, b, standings));
}

/** Every pair once, in discovery order. */
function allPairings(weights) {
  const pairings = [];
  for (let i = 0; i < weights.length; i += 1) {
    for (let j = i + 1; j < weights.length; j += 1) {
      pairings.push([weights[i], weights[j]]);
    }
  }
  return pairings;
}

/**
 * Pairs each weight with the nearest-ranked opponent it has not met yet,
 * falling back to a repeat when every remaining opponent has been played.
 */
function pairByRank(weights, standings, playedPairs) {
  const order = rankedWeights(weights, standings);
  const pairings = [];

  if (order.length % 2 === 1) {
    // Bench whoever has played the most, so an odd field spreads the idle round
    // instead of starving the tail of the ranking every time.
    const idle = order.reduce((a, b) =>
      standings.get(b.name).games >= standings.get(a.name).games ? b : a,
    );
    order.splice(order.indexOf(idle), 1);
  }

  while (order.length >= 2) {
    const engine1 = order.shift();
    const unmet = order.findIndex((engine2) => !playedPairs.has(pairKey(engine1, engine2)));
    const [engine2] = order.splice(unmet === -1 ? 0 : unmet, 1);
    pairings.push([engine1, engine2]);
  }

  return pairings;
}

/** Splits pairings into batches so results print while the run progresses. */
function batchPairings(pairings, size) {
  const batches = [];
  for (let i = 0; i < pairings.length; i += size) {
    batches.push(pairings.slice(i, i + size));
  }
  return batches;
}

function addMatchToStandings(standings, engine1, engine2, result) {
  const standing1 = standings.get(engine1.name);
  const standing2 = standings.get(engine2.name);

  standing1.games += result.games;
  standing1.wins += result.engine1Wins;
  standing1.losses += result.engine2Wins;
  standing1.draws += result.draws;
  standing1.gamePoints += result.engine1Wins + result.draws * 0.5;
  standing1.discScore += result.engine1Score;

  standing2.games += result.games;
  standing2.wins += result.engine2Wins;
  standing2.losses += result.engine1Wins;
  standing2.draws += result.draws;
  standing2.gamePoints += result.engine2Wins + result.draws * 0.5;
  standing2.discScore -= result.engine1Score;
}

function printStandings(weights, standings) {
  console.log("\n## Standings\n");
  console.log("| # | Weight | Score | Games | W-L-D | Disc/game |");
  console.log("|--:|--------|------:|------:|------:|----------:|");

  rankedWeights(weights, standings).forEach((weight, index) => {
    const standing = standings.get(weight.name);
    console.log(
      `| ${index + 1} | ${weight.name} | ${formatPercent(scoreRate(standing))} | ` +
        `${standing.games} | ${standing.wins}-${standing.losses}-${standing.draws} | ` +
        `${formatSigned(averageDiscScore(standing).toFixed(2))} |`,
    );
  });
}

function workerWeight(weight) {
  return {
    name: weight.name,
    path: weight.path,
  };
}

function messageError(message) {
  return new Error(message.error ?? "worker failed");
}

function createMatchWorker(openings) {
  const worker = new Worker(new URL("./weight-tournament-worker.js", import.meta.url), {
    type: "module",
  });
  const pending = new Map();
  let nextMessageId = 1;

  const ready = new Promise((resolve, reject) => {
    worker.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "ready") {
        resolve(message);
        return;
      }

      if (message.type === "result") {
        pending.get(message.id)?.resolve(message.result);
        pending.delete(message.id);
        return;
      }

      if (message.type === "error") {
        const error = messageError(message);
        if (message.id !== undefined && pending.has(message.id)) {
          pending.get(message.id).reject(error);
          pending.delete(message.id);
        } else {
          reject(error);
        }
      }
    });
    worker.addEventListener("error", reject);
    worker.postMessage({ openings, type: "init" });
  });

  return {
    async ready() {
      return ready;
    },
    runMatch(engine1, engine2) {
      const id = nextMessageId;
      nextMessageId += 1;

      return new Promise((resolve, reject) => {
        pending.set(id, { reject, resolve });
        worker.postMessage({
          engine1: workerWeight(engine1),
          engine2: workerWeight(engine2),
          id,
          type: "match",
        });
      });
    },
    terminate() {
      worker.terminate();
    },
  };
}

async function createMatchExecutor(jobs, openings) {
  if (jobs === 1) {
    const { module, relaxedSimd } = await importNodeWasm();
    const { WeightMatchRunner } = module;
    if (!WeightMatchRunner) {
      throw new Error("WeightMatchRunner is missing; rebuild with `bun run build:wasm:node`");
    }

    return {
      jobs,
      relaxedSimd,
      async runRound(pairings) {
        return pairings.map(([engine1, engine2]) =>
          playMatch(
            WeightMatchRunner,
            loadWeightBytes(engine1),
            loadWeightBytes(engine2),
            openings,
          ),
        );
      },
      close() {},
    };
  }

  const workers = Array.from({ length: jobs }, () => createMatchWorker(openings));
  let readyMessages;
  try {
    readyMessages = await Promise.all(workers.map((worker) => worker.ready()));
  } catch (error) {
    for (const worker of workers) {
      worker.terminate();
    }
    throw error;
  }

  return {
    jobs,
    relaxedSimd: readyMessages.some((message) => message.relaxedSimd),
    async runRound(pairings) {
      const results = Array.from({ length: pairings.length });
      let nextPairingIndex = 0;

      await Promise.all(
        workers.map(async (worker) => {
          while (nextPairingIndex < pairings.length) {
            const pairingIndex = nextPairingIndex;
            nextPairingIndex += 1;

            const [engine1, engine2] = pairings[pairingIndex];
            results[pairingIndex] = await worker.runMatch(engine1, engine2);
          }
        }),
      );

      return results;
    },
    close() {
      for (const worker of workers) {
        worker.terminate();
      }
    },
  };
}

async function main() {
  const { values, positionals } = parseCliArgs();
  if (values.help) {
    usage(0);
  }

  if (positionals.length !== 1) {
    usage(1);
  }

  const weightsDir = positionals[0];
  const weights = readWeightFiles(weightsDir);
  if (weights.length < 2) {
    throw new Error("weights-dir must contain at least two .zst files");
  }

  const openings = values["opening-file"] ? readOpeningFile(values["opening-file"]) : [""];
  if (openings.length === 0) {
    throw new Error("no playable openings found");
  }

  const requestedJobs = parsePositiveInteger(values.jobs, "--jobs") ?? 1;
  const jobs = Math.min(requestedJobs, Math.max(1, Math.floor(weights.length / 2)));
  const matchExecutor = await createMatchExecutor(jobs, openings);
  const standings = new Map(weights.map((weight) => [weight.name, createStanding()]));
  const playedPairs = new Set();

  // The full round-robin schedule is fixed up front; the sparse league rebuilds
  // each round from the standings, so it is produced inside the loop.
  const schedule =
    weights.length <= FULL_ROUND_ROBIN_LIMIT
      ? batchPairings(allPairings(weights), matchExecutor.jobs)
      : null;
  const rounds = schedule ? schedule.length : SPARSE_ROUNDS;
  const totalComparisons = schedule
    ? (weights.length * (weights.length - 1)) / 2
    : rounds * Math.floor(weights.length / 2);
  const gamesPerComparison = openings.length * 2;

  console.log(`Wasm SIMD: ${matchExecutor.relaxedSimd ? "relaxed-simd" : "simd128"}`);
  console.log(`Weights: ${weights.length}`);
  console.log(`Openings: ${openings.length}`);
  console.log(`Mode: ${schedule ? "full round-robin" : `sparse league (${rounds} rounds)`}`);
  console.log(`Jobs: ${matchExecutor.jobs}`);
  console.log(`Comparisons: ${totalComparisons}`);
  console.log(`Games: ${totalComparisons * gamesPerComparison}\n`);

  let comparisonNumber = 1;
  try {
    for (let round = 0; round < rounds; round += 1) {
      const pairings = schedule?.[round] ?? pairByRank(weights, standings, playedPairs);
      const results = await matchExecutor.runRound(pairings);

      for (let pairingIndex = 0; pairingIndex < pairings.length; pairingIndex += 1) {
        const [engine1, engine2] = pairings[pairingIndex];
        const result = results[pairingIndex];
        playedPairs.add(pairKey(engine1, engine2));
        addMatchToStandings(standings, engine1, engine2, result);

        const line =
          `[${comparisonNumber}/${totalComparisons}] ${basename(engine1.name)} vs ${basename(engine2.name)}: ` +
          `${result.engine1Wins}-${result.engine2Wins}-${result.draws}, ` +
          `score ${formatSigned(result.engine1Score)}; winner ${winnerName(result, engine1, engine2)}`;
        console.log(line);
        comparisonNumber += 1;
      }
    }
  } finally {
    matchExecutor.close();
  }

  printStandings(weights, standings);
  console.log("\n## Result\n");
  console.log(`Strongest: ${rankedWeights(weights, standings)[0].name}`);
}

export { addMatchToStandings, createStanding, pairByRank, pairKey, playMatch };

if (import.meta.main) {
  main().catch((err) => {
    console.error("Error:", err);
    process.exit(1);
  });
}
