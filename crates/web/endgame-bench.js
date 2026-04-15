#!/usr/bin/env bun
/**
 * Endgame solver benchmark CLI for WebAssembly.
 *
 * Usage:
 *   bun endgame-bench.js --problem fforum-40-59
 *   bun endgame-bench.js -p fforum-40-59 -t 64
 */

import { readFileSync } from 'fs';
import { parseArgs } from 'util';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const problemDir = resolve(__dirname, '../../problem');

const { values } = parseArgs({
  options: {
    problem: { type: 'string', short: 'p', default: 'fforum-40-59' },
    'tt-size': { type: 'string', short: 't', default: '32' },
    'max-empties': { type: 'string', short: 'e', default: '24' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (values.help) {
  console.log(`
Endgame solver benchmark CLI (WASM)

Usage:
  bun endgame-bench.js [options]

Options:
  -p, --problem       OBF file stem or path (default: fforum-40-59)
  -t, --tt-size       Transposition table size in MB (default: 32)
  -e, --max-empties   Max empty squares to include (default: 24)
  -h, --help          Show this help message

Examples:
  bun endgame-bench.js -p fforum-40-59
  bun endgame-bench.js -p fforum-60-79 -t 64
`);
  process.exit(0);
}

function parseObfFile(filePath) {
  const content = readFileSync(filePath, 'utf-8');
  const cases = [];
  for (const [index, rawLine] of content.split('\n').entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    const segments = line.split(';');
    const header = segments[0].trim();
    const boardStr = header.slice(0, 64);
    const side = header.slice(64).trim();

    const moves = [];
    for (const seg of segments.slice(1)) {
      const trimmed = seg.trim();
      if (!trimmed) continue;
      const colonIdx = trimmed.indexOf(':');
      if (colonIdx === -1) continue;
      const moveName = trimmed.slice(0, colonIdx).trim();
      const scoreStr = trimmed.slice(colonIdx + 1).trim();
      if (moveName === '--') continue;
      moves.push({
        move: moveName,
        score: parseInt(scoreStr.replace(/^\+/, ''), 10),
      });
    }

    if (moves.length === 0) continue;

    const expectedScore = moves[0].score;
    const bestMoves = moves.filter(m => m.score === expectedScore).map(m => m.move);
    const empties = (boardStr.match(/-/g) || []).length;

    cases.push({
      number: index + 1,
      boardStr,
      side,
      sideNum: side === 'X' ? 0 : 1,
      expectedScore,
      bestMoves,
      empties,
    });
  }
  return cases;
}

function resolveObfPath(problem) {
  if (problem.endsWith('.obf')) return resolve(problem);
  return resolve(problemDir, `${problem}.obf`);
}

function formatTime(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  const min = Math.floor(ms / 60000);
  const sec = ((ms % 60000) / 1000).toFixed(1);
  return `${min}m${sec}s`;
}

function formatNodes(n) {
  if (n < 1e6) return `${(n / 1e3).toFixed(1)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(2)}M`;
  return `${(n / 1e9).toFixed(2)}G`;
}

function formatNps(nps) {
  if (nps < 1e6) return `${(nps / 1e3).toFixed(0)}Knps`;
  return `${(nps / 1e6).toFixed(2)}Mnps`;
}

function pad(str, len, align = 'right') {
  str = String(str);
  if (align === 'right') return str.padStart(len);
  return str.padEnd(len);
}

async function main() {
  const { EndgameSolver } = await import('./pkg-node/web.js');

  const ttSize = parseInt(values['tt-size'], 10);
  const maxEmpties = parseInt(values['max-empties'], 10);
  const obfPath = resolveObfPath(values.problem);
  const problemName = values.problem.replace(/\.obf$/, '');

  console.log('Loading evaluation network...');
  const solver = new EndgameSolver(ttSize);

  console.log(`Reading problem file: ${obfPath}`);
  const allCases = parseObfFile(obfPath);
  const cases = allCases.filter(tc => tc.empties <= maxEmpties);
  const skipped = allCases.length - cases.length;
  console.log(`Found ${allCases.length} positions, running ${cases.length} (empties <= ${maxEmpties}${skipped > 0 ? `, skipped ${skipped}` : ''}, TT: ${ttSize}MB)\n`);

  console.log(`## ${problemName}\n`);
  console.log(
    `| ${'#'.padStart(3)} | Empties | ${'Time'.padStart(8)} | ${'Nodes'.padStart(9)} | ${'NPS'.padStart(9)} | ${'Score'.padStart(5)} | ${'Exp'.padStart(5)} | Move | Best     | Status |`
  );
  console.log(
    `|${'-'.repeat(4)}:|--------:|---------:|----------:|----------:|------:|------:|-----:|----------|--------|`
  );

  let totalTime = 0;
  let totalNodes = 0;
  let correct = 0;

  for (const tc of cases) {
    const start = performance.now();
    const result = solver.solve(tc.boardStr, tc.sideNum);
    const elapsed = performance.now() - start;

    const nodes = result.n_nodes;
    const nps = elapsed > 0 ? nodes / (elapsed / 1000) : 0;
    const move = result.best_move.toUpperCase();
    const score = Math.round(result.score);
    const isBest = tc.bestMoves.includes(move);

    totalTime += elapsed;
    totalNodes += nodes;
    if (isBest) correct++;

    console.log(
      `| ${pad(tc.number, 3)} | ${pad(tc.empties, 7)} | ${pad(formatTime(elapsed), 8)} | ${pad(formatNodes(nodes), 9)} | ${pad(formatNps(nps), 9)} | ${pad(score, 5)} | ${pad(tc.expectedScore, 5)} | ${pad(move, 4, 'left')} | ${pad(tc.bestMoves.join(','), 8, 'left')} | ${isBest ? '  OK  ' : ' MISS '} |`
    );

    result.free();
  }

  const overallNps = totalTime > 0 ? totalNodes / (totalTime / 1000) : 0;

  console.log(`\n### Statistics\n`);
  console.log(`- Total time: ${formatTime(totalTime)}`);
  console.log(`- Total nodes: ${formatNodes(totalNodes)}`);
  console.log(`- Overall NPS: ${formatNps(overallNps)}`);
  console.log(`- Best move: ${(correct / cases.length * 100).toFixed(1)}% (${correct}/${cases.length})`);

  solver.free();
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
