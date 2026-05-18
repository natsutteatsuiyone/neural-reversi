/**
 * Cell Key (CONTEXT.md → Cell Key): the one encoding of a board cell's
 * `(row, col)` into the string key of a per-cell overlay map.
 *
 * Every producer of a `Map<string, …>` keyed by a cell (hint-analysis
 * results, solver candidates, flip delays) and every consumer that probes
 * one goes through here, so the encoding can never drift between a producer
 * and a consumer.
 */
export type CellKey = string;

export function cellKey(row: number, col: number): CellKey {
  return `${row},${col}`;
}
