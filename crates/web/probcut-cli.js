#!/usr/bin/env node
/**
 * ProbCut training data generation CLI for WebAssembly.
 *
 * Usage:
 *   node probcut-cli.js --input games.txt --output probcut.csv
 *   node probcut-cli.js -i games.txt -o probcut.csv
 *
 * Input format: One game per line, moves concatenated (e.g., "D3C5F6F5E6C6D6...")
 * Output format: CSV with columns: ply,shallow_depth,shallow_score,deep_depth,deep_score,diff
 */

import { readFileSync, writeFileSync } from 'fs';
import { parseArgs } from 'util';

// Parse command line arguments
const { values } = parseArgs({
  options: {
    input: { type: 'string', short: 'i' },
    output: { type: 'string', short: 'o' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (values.help || !values.input || !values.output) {
  console.log(`
ProbCut training data generation CLI

Usage:
  node probcut-cli.js --input <file> --output <file>
  node probcut-cli.js -i <file> -o <file>

Options:
  -i, --input   Input file containing game sequences (one per line)
  -o, --output  Output CSV file for ProbCut training data
  -h, --help    Show this help message

Input format:
  One game per line, moves concatenated (e.g., "D3C5F6F5E6C6D6...")

Output format:
  CSV with columns: ply,shallow_depth,shallow_score,deep_depth,deep_score,diff
`);
  process.exit(values.help ? 0 : 1);
}

async function main() {
  // Dynamic import for WASM module
  const { ProbCutDatagen } = await import('./pkg-node/web.js');

  console.log('Loading evaluation network...');
  const datagen = new ProbCutDatagen();

  console.log(`Reading input file: ${values.input}`);
  const gamesText = readFileSync(values.input, 'utf-8');
  const lines = gamesText.split('\n').filter(line => line.trim().length > 0);
  console.log(`Found ${lines.length} games`);

  console.log('Processing games...');
  const result = datagen.process_games(gamesText, (progress) => {
    console.log(`Processed ${progress.game_index}/${progress.total_games} games (${progress.samples_so_far} samples)`);
  });
  console.log('');

  console.log(`Writing output file: ${values.output}`);
  writeFileSync(values.output, result.samples_csv);

  console.log(`
Statistics:
  Total samples: ${result.total_samples}
  Total positions: ${result.total_positions}
  Cache hit rate: ${(result.cache_hit_rate * 100).toFixed(1)}%
`);

  console.log('Done!');
  datagen.free();
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
