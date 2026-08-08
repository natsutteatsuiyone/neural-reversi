import { expect, test } from "bun:test";
import { addMatchToStandings, createStanding, pairByRank, pairKey } from "./weight-tournament.js";

const DRAWN_MATCH = {
  games: 2,
  engine1Wins: 0,
  engine2Wins: 0,
  draws: 2,
  engine1Score: 0,
};

test("rank pairing prefers opponents that have not been played yet", () => {
  const weights = Array.from({ length: 4 }, (_, index) => ({
    name: `weight-${index + 1}.zst`,
  }));
  const standings = new Map(weights.map((weight) => [weight.name, createStanding()]));
  const playedPairs = new Set();

  // Every round is drawn, so the ranking never changes and only the played-pair
  // check can stop the pairing from handing back the same matchup every round.
  for (let round = 0; round < 3; round += 1) {
    for (const [engine1, engine2] of pairByRank(weights, standings, playedPairs)) {
      playedPairs.add(pairKey(engine1, engine2));
      addMatchToStandings(standings, engine1, engine2, DRAWN_MATCH);
    }
  }

  // 4 weights over 3 rounds of 2 pairings covers all 6 pairs exactly once.
  expect(playedPairs.size).toBe(6);
  for (const standing of standings.values()) {
    expect(standing.games).toBe(6);
  }
});
