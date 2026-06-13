import { expect, test } from "bun:test";
import {
  addMatchToStandings,
  buildSwissPairings,
  createStanding,
  pairKey,
  resolveRoundCount,
} from "./weight-tournament.js";

function createWeights(count) {
  return Array.from({ length: count }, (_, index) => ({
    name: `weight-${index + 1}.zst`,
  }));
}

function createStandings(weights) {
  return new Map(weights.map((weight) => [weight.name, createStanding()]));
}

function normalizePairings(pairings) {
  return pairings.map(([engine1, engine2]) => pairKey(engine1, engine2)).sort();
}

function addDrawnRoundToStandings(pairings, standings, playedPairs) {
  const drawnResult = {
    games: 2,
    engine1Wins: 0,
    engine2Wins: 0,
    draws: 2,
    engine1Score: 0,
  };

  for (const [engine1, engine2] of pairings) {
    playedPairs.add(pairKey(engine1, engine2));
    addMatchToStandings(standings, engine1, engine2, drawnResult);
  }
}

test("Swiss pairings avoid repeated pairs when a non-repeat matching exists", () => {
  const weights = createWeights(10);
  const standings = createStandings(weights);
  const playedPairs = new Set();

  const firstRoundPairings = buildSwissPairings(
    weights,
    standings,
    playedPairs,
    0,
    "weight-tournament",
  );
  addDrawnRoundToStandings(firstRoundPairings, standings, playedPairs);

  const secondRoundPairings = buildSwissPairings(
    weights,
    standings,
    playedPairs,
    1,
    "weight-tournament",
  );

  expect(secondRoundPairings).toHaveLength(5);
  expect(
    secondRoundPairings.filter(([engine1, engine2]) => playedPairs.has(pairKey(engine1, engine2))),
  ).toHaveLength(0);
});

test("Swiss pairings optimize rank distance among equally repeat-free matchings", () => {
  const weights = createWeights(8);
  const standings = createStandings(weights);
  const comparisons = [3, 2, 2, 0, 3, 0, 0, 1];
  const playedPairs = new Set([pairKey(weights[1], weights[4]), pairKey(weights[2], weights[3])]);

  weights.forEach((weight, index) => {
    const standing = standings.get(weight.name);
    standing.games = 100;
    standing.gamePoints = 100 - index;
    standing.comparisons = comparisons[index];
  });

  const pairings = buildSwissPairings(weights, standings, playedPairs, 1, "weight-tournament");

  expect(normalizePairings(pairings)).toEqual(
    normalizePairings([
      [weights[0], weights[2]],
      [weights[1], weights[3]],
      [weights[4], weights[5]],
      [weights[6], weights[7]],
    ]),
  );
});

test("round count resolver supports explicit full round-robin", () => {
  expect(resolveRoundCount(10, undefined, false)).toEqual({
    fullRoundRobin: false,
    maxRounds: 9,
    rounds: 4,
  });
  expect(resolveRoundCount(10, undefined, true)).toEqual({
    fullRoundRobin: true,
    maxRounds: 9,
    rounds: 9,
  });
  expect(resolveRoundCount(10, 99, false)).toEqual({
    fullRoundRobin: true,
    maxRounds: 9,
    rounds: 9,
  });
  expect(() => resolveRoundCount(10, 4, true)).toThrow(
    "--full-round-robin cannot be combined with --rounds",
  );
});
