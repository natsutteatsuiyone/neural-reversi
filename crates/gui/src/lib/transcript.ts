import type { MoveRecord } from "@/domain/game/types";

/**
 * Build the exportable transcript for a sequence of moves: passes
 * (records with `row < 0`) are omitted, each played move contributes its
 * notation lowercased, with no separators (e.g. "f5d6c3").
 */
export function buildTranscript(moves: readonly MoveRecord[]): string {
  return moves
    .filter((m) => m.row >= 0)
    .map((m) => m.notation.toLowerCase())
    .join("");
}
