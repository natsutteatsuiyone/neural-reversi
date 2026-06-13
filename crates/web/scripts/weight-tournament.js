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
import { importPreferredWasmModule } from "../wasm-loader.js";

const CLI_OPTIONS = {
  "full-round-robin": { type: "boolean" },
  "opening-file": { type: "string", short: "o" },
  jobs: { type: "string", short: "j" },
  rounds: { type: "string", short: "r" },
  seed: { type: "string" },
  help: { type: "boolean", short: "h" },
};

const RANK_DISTANCE_PAIRING_PENALTY = 10;
const EXACT_SWISS_MATCHING_PLAYER_LIMIT = 20;

function parseCliArgs(args = process.argv.slice(2)) {
  return parseArgs({
    args,
    allowPositionals: true,
    options: CLI_OPTIONS,
  });
}

function usage(exitCode = 0) {
  console.log(`
WASM weight tournament CLI (1-ply, balanced Swiss/round-robin)

Usage:
  bun scripts/weight-tournament.js <weights-dir> [options]

Options:
  -o, --opening-file  Opening file in match-runner format
  -j, --jobs          Parallel comparisons per round. Default: 1
  -r, --rounds        Pairing rounds. Default: full round-robin for <= 8 weights, otherwise 4
      --full-round-robin
                      Play every pair once. Cannot be combined with --rounds
      --seed          Stable seed for the initial pairing order. Default: weight-tournament
  -h, --help          Show this help message

Examples:
  bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt>
  bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt> --rounds 6
  bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt> --full-round-robin --jobs 4
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

function hashText(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function seededWeightOrder(weights, seed) {
  return [...weights].sort((a, b) => {
    const hashA = hashText(`${seed}\0${a.name}`);
    const hashB = hashText(`${seed}\0${b.name}`);
    return hashA - hashB || a.name.localeCompare(b.name, undefined, { numeric: true });
  });
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

function fullRoundRobinRounds(weightCount) {
  return weightCount % 2 === 0 ? weightCount - 1 : weightCount;
}

function defaultRoundCount(weightCount) {
  if (weightCount <= 8) {
    return fullRoundRobinRounds(weightCount);
  }

  return 4;
}

function resolveRoundCount(weightCount, requestedRounds, fullRoundRobinRequested) {
  if (fullRoundRobinRequested && requestedRounds !== undefined) {
    throw new Error("--full-round-robin cannot be combined with --rounds");
  }

  const maxRounds = fullRoundRobinRounds(weightCount);
  const rounds = fullRoundRobinRequested
    ? maxRounds
    : Math.min(requestedRounds ?? defaultRoundCount(weightCount), maxRounds);

  return {
    fullRoundRobin: rounds === maxRounds,
    maxRounds,
    rounds,
  };
}

function createStanding() {
  return {
    games: 0,
    comparisons: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    gamePoints: 0,
    discScore: 0,
    byes: 0,
    opponents: [],
  };
}

function scoreRate(standing) {
  return standing.games === 0 ? 0.5 : standing.gamePoints / standing.games;
}

function averageDiscScore(standing) {
  return standing.games === 0 ? 0 : standing.discScore / standing.games;
}

function opponentScoreRate(standing, standings) {
  if (standing.opponents.length === 0) {
    return 0.5;
  }

  const total = standing.opponents.reduce(
    (sum, opponentName) => sum + scoreRate(standings.get(opponentName)),
    0,
  );
  return total / standing.opponents.length;
}

function compareStandings(a, b, standings) {
  const standingA = standings.get(a.name);
  const standingB = standings.get(b.name);
  return (
    scoreRate(standingB) - scoreRate(standingA) ||
    standingB.comparisons - standingA.comparisons ||
    opponentScoreRate(standingB, standings) - opponentScoreRate(standingA, standings) ||
    averageDiscScore(standingB) - averageDiscScore(standingA) ||
    a.name.localeCompare(b.name, undefined, { numeric: true })
  );
}

function pairKey(a, b) {
  return a.name < b.name ? `${a.name}\0${b.name}` : `${b.name}\0${a.name}`;
}

function buildRoundRobinSchedule(weights, rounds) {
  const players = [...weights];
  if (players.length % 2 === 1) {
    players.push(null);
  }

  const schedule = [];
  let rotated = players;
  for (let round = 0; round < rounds; round += 1) {
    const pairings = [];
    for (let i = 0; i < rotated.length / 2; i += 1) {
      const left = rotated[i];
      const right = rotated[rotated.length - 1 - i];
      if (left !== null && right !== null) {
        pairings.push(round % 2 === 0 ? [left, right] : [right, left]);
      }
    }
    schedule.push(pairings);

    rotated = [rotated[0], rotated[rotated.length - 1], ...rotated.slice(1, rotated.length - 1)];
  }

  return schedule;
}

function rankedWeights(weights, standings) {
  return [...weights].sort((a, b) => compareStandings(a, b, standings));
}

function chooseBye(order, standings) {
  const candidates = [...order].sort((a, b) => {
    const standingA = standings.get(a.name);
    const standingB = standings.get(b.name);
    return (
      standingA.byes - standingB.byes ||
      standingA.comparisons - standingB.comparisons ||
      scoreRate(standingA) - scoreRate(standingB) ||
      a.name.localeCompare(b.name, undefined, { numeric: true })
    );
  });

  return candidates[0];
}

function swissPairingCost(engine1, engine2, standings, playedPairs, orderIndexes) {
  const rankDistance =
    Math.abs(orderIndexes.get(engine1.name) - orderIndexes.get(engine2.name)) - 1;
  const balancePenalty = standings.get(engine2.name).comparisons;

  return {
    repeats: playedPairs.has(pairKey(engine1, engine2)) ? 1 : 0,
    score: rankDistance * RANK_DISTANCE_PAIRING_PENALTY + balancePenalty,
  };
}

function isWorseOrEqualPairing(repeats, score, bestRepeats, bestScore) {
  return repeats > bestRepeats || (repeats === bestRepeats && score >= bestScore);
}

function buildBestSwissMatching(unpaired, standings, playedPairs) {
  const orderIndexes = new Map(unpaired.map((engine, index) => [engine.name, index]));
  const lowestSeenScores = new Map();
  const stopOnRepeatFree = unpaired.length > EXACT_SWISS_MATCHING_PLAYER_LIMIT;
  let bestRepeats = Number.POSITIVE_INFINITY;
  let bestScore = Number.POSITIVE_INFINITY;
  let bestPairings = [];

  // Normal tournament sizes get exact secondary scoring; larger fields stop at repeat-free.
  function search(remaining, pairings, repeats, score) {
    if (isWorseOrEqualPairing(repeats, score, bestRepeats, bestScore)) {
      return false;
    }

    if (remaining.length === 0) {
      bestRepeats = repeats;
      bestScore = score;
      bestPairings = pairings;
      return stopOnRepeatFree && repeats === 0;
    }

    const key = remaining.map((engine) => engine.name).join("\0");
    const lowestSeenScore = lowestSeenScores.get(key);
    if (
      lowestSeenScore !== undefined &&
      (lowestSeenScore.repeats < repeats ||
        (lowestSeenScore.repeats === repeats && lowestSeenScore.score <= score))
    ) {
      return false;
    }
    lowestSeenScores.set(key, { repeats, score });

    const engine1 = remaining[0];
    const rest = remaining.slice(1);
    const candidates = rest
      .map((engine2, restIndex) => {
        const cost = swissPairingCost(engine1, engine2, standings, playedPairs, orderIndexes);
        return {
          engine2,
          restIndex,
          ...cost,
        };
      })
      .sort(
        (a, b) =>
          a.repeats - b.repeats ||
          a.score - b.score ||
          orderIndexes.get(a.engine2.name) - orderIndexes.get(b.engine2.name) ||
          a.engine2.name.localeCompare(b.engine2.name, undefined, { numeric: true }),
      );

    for (const candidate of candidates) {
      const nextRepeats = repeats + candidate.repeats;
      const nextScore = score + candidate.score;
      if (isWorseOrEqualPairing(nextRepeats, nextScore, bestRepeats, bestScore)) {
        continue;
      }

      const nextRemaining = rest
        .slice(0, candidate.restIndex)
        .concat(rest.slice(candidate.restIndex + 1));
      if (
        search(nextRemaining, [...pairings, [engine1, candidate.engine2]], nextRepeats, nextScore)
      ) {
        return true;
      }
    }

    return false;
  }

  search(unpaired, [], 0, 0);
  return bestPairings;
}

function buildSwissPairings(weights, standings, playedPairs, round, seed) {
  const order = round === 0 ? seededWeightOrder(weights, seed) : rankedWeights(weights, standings);
  const unpaired = [...order];

  if (unpaired.length % 2 === 1) {
    const bye = chooseBye(unpaired, standings);
    standings.get(bye.name).byes += 1;
    unpaired.splice(unpaired.indexOf(bye), 1);
  }

  return buildBestSwissMatching(unpaired, standings, playedPairs).map(([engine1, engine2]) =>
    round % 2 === 0 ? [engine1, engine2] : [engine2, engine1],
  );
}

function addMatchToStandings(standings, engine1, engine2, result) {
  const standing1 = standings.get(engine1.name);
  const standing2 = standings.get(engine2.name);
  const engine1GamePoints = result.engine1Wins + result.draws * 0.5;
  const engine2GamePoints = result.engine2Wins + result.draws * 0.5;

  standing1.games += result.games;
  standing1.comparisons += 1;
  standing1.wins += result.engine1Wins;
  standing1.losses += result.engine2Wins;
  standing1.draws += result.draws;
  standing1.gamePoints += engine1GamePoints;
  standing1.discScore += result.engine1Score;
  standing1.opponents.push(engine2.name);

  standing2.games += result.games;
  standing2.comparisons += 1;
  standing2.wins += result.engine2Wins;
  standing2.losses += result.engine1Wins;
  standing2.draws += result.draws;
  standing2.gamePoints += engine2GamePoints;
  standing2.discScore -= result.engine1Score;
  standing2.opponents.push(engine1.name);
}

function printStandings(weights, standings) {
  console.log("\n## Standings\n");
  console.log("| # | Weight | Score | Games | W-L-D | Disc/game | Opp score |");
  console.log("|--:|--------|------:|------:|------:|----------:|----------:|");

  rankedWeights(weights, standings).forEach((weight, index) => {
    const standing = standings.get(weight.name);
    console.log(
      `| ${index + 1} | ${weight.name} | ${formatPercent(scoreRate(standing))} | ` +
        `${standing.games} | ${standing.wins}-${standing.losses}-${standing.draws} | ` +
        `${formatSigned(averageDiscScore(standing).toFixed(2))} | ` +
        `${formatPercent(opponentScoreRate(standing, standings))} |`,
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
    const { module, relaxedSimd } = await importPreferredWasmModule({
      relaxedPath: "./pkg-node-relaxed/web.js",
      fallbackPath: "./pkg-node/web.js",
    });
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

  const requestedRounds = parsePositiveInteger(values.rounds, "--rounds");
  const { fullRoundRobin, rounds } = resolveRoundCount(
    weights.length,
    requestedRounds,
    values["full-round-robin"] === true,
  );
  const requestedJobs = parsePositiveInteger(values.jobs, "--jobs") ?? 1;
  const jobs = Math.min(requestedJobs, Math.max(1, Math.floor(weights.length / 2)));
  const matchExecutor = await createMatchExecutor(jobs, openings);
  const seed = values.seed ?? "weight-tournament";
  const openingOrder = seededWeightOrder(weights, seed);
  const standings = new Map(weights.map((weight) => [weight.name, createStanding()]));
  const playedPairs = new Set();
  const roundRobinSchedule = fullRoundRobin ? buildRoundRobinSchedule(openingOrder, rounds) : null;
  const totalComparisons = rounds * Math.floor(weights.length / 2);
  const gamesPerComparison = openings.length * 2;

  console.log(`Wasm SIMD: ${matchExecutor.relaxedSimd ? "relaxed-simd" : "simd128"}`);
  console.log(`Weights: ${weights.length}`);
  console.log(`Openings: ${openings.length}`);
  console.log(`Mode: ${fullRoundRobin ? "full round-robin" : "Swiss-style sparse league"}`);
  console.log(`Rounds: ${rounds}`);
  console.log(`Jobs: ${matchExecutor.jobs}`);
  console.log(`Comparisons: ${totalComparisons}`);
  console.log(`Games: ${totalComparisons * gamesPerComparison}`);
  console.log(`Seed: ${seed}\n`);

  let comparisonNumber = 1;
  try {
    for (let round = 0; round < rounds; round += 1) {
      const pairings =
        roundRobinSchedule?.[round] ??
        buildSwissPairings(weights, standings, playedPairs, round, seed);

      console.log(`Round ${round + 1}/${rounds}`);
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

export {
  addMatchToStandings,
  buildSwissPairings,
  createStanding,
  pairKey,
  playMatch,
  resolveRoundCount,
};

if (import.meta.main) {
  main().catch((err) => {
    console.error("Error:", err);
    process.exit(1);
  });
}
