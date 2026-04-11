import type { Board, Player } from "@/types";

/**
 * Serializes a [Board] to the 64-character `X`/`O`/`-` string format expected
 * by the Rust backend commands (`ai_move_command`, `solver_search_command`,
 * etc.).
 *
 * The convention is **relative to the player to move**: `X` = the cell belongs
 * to `player`, `O` = the opponent, `-` = empty. This differs from the
 * absolute-color convention used by {@link "@/lib/board-parser".boardToString}.
 */
export function serializeBoardForAI(board: Board, player: Player): string {
  return board
    .flat()
    .map((cell) =>
      cell.color === player ? "X" : cell.color === null ? "-" : "O",
    )
    .join("");
}
