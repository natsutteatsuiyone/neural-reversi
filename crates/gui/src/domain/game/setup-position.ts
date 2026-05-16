import { parseBoardString, parseTranscript, validateBoard } from "@/domain/game/board-parser";
import type { Board, Player } from "@/domain/game/types";

export type SetupPositionSource = "manual" | "transcript" | "boardString";

export interface SetupPositionInput {
  source: SetupPositionSource;
  board: Board;
  currentPlayer: Player;
  transcriptInput: string;
  boardStringInput: string;
}

export type ResolvedSetupPosition =
  | { ok: true; board: Board; currentPlayer: Player }
  | { ok: false; error: string };

export function resolveSetupPosition(input: SetupPositionInput): ResolvedSetupPosition {
  if (input.source === "transcript") {
    const result = parseTranscript(input.transcriptInput);
    return result.ok
      ? { ok: true, board: result.board, currentPlayer: result.currentPlayer }
      : { ok: false, error: result.error };
  }

  if (input.source === "boardString") {
    const result = parseBoardString(input.boardStringInput);
    return result.ok
      ? { ok: true, board: result.board, currentPlayer: input.currentPlayer }
      : { ok: false, error: result.error };
  }

  return { ok: true, board: input.board, currentPlayer: input.currentPlayer };
}

export function resolveValidSetupPosition(input: SetupPositionInput): ResolvedSetupPosition {
  const resolved = resolveSetupPosition(input);
  if (!resolved.ok) {
    return resolved;
  }

  const error = validateBoard(resolved.board, resolved.currentPlayer);
  if (error) {
    return { ok: false, error };
  }

  return resolved;
}
