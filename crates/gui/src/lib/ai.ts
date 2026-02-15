import { invoke } from "@tauri-apps/api/core";
import { type Event, listen } from "@tauri-apps/api/event";
import type { Board, Player } from "@/types";
import { getValidMoves } from "./game-logic";

export type AIMoveResult = {
  row: number;
  col: number;
  score: number;
  depth: number;
  acc: number;
  timeTaken: number;
} | null;

export type AIMoveProgress = {
  bestMove: string;
  row: number;
  col: number;
  score: number;
  depth: number;
  targetDepth: number;
  acc: number;
  nodes: number;
  pvLine: string;
  isEndgame: boolean;
};

function serializeBoardForAI(board: Board, player: Player): string {
  return board
    .flat()
    .map((cell) =>
      cell.color === player ? "X" : cell.color === null ? "-" : "O"
    )
    .join("");
}

export async function getAIMove(
  board: Board,
  player: Player,
  level: number,
  timeLimit: number | undefined,
  remainingTime: number | undefined,
  callback: (event: Event<AIMoveProgress>) => void
): Promise<AIMoveResult> {
  const validMoves = getValidMoves(board, player);
  if (validMoves.length === 0) return null;

  const boardString = serializeBoardForAI(board, player);

  const unlisten = await listen<AIMoveProgress>("ai-move-progress", (data) => {
    callback(data);
  });

  try {
    return await invoke<AIMoveResult>("ai_move_command", {
      boardString,
      level,
      timeLimit,
      remainingTime,
    });
  } catch {
    return null;
  } finally {
    unlisten();
  }
}

export async function initializeAI(): Promise<void> {
  try {
    await invoke("init_ai_command");
  } catch (error) {
    console.error("Failed to initialize search:", error);
    throw error;
  }
}

export async function abortAISearch(): Promise<void> {
  try {
    await invoke("abort_ai_search_command");
  } catch (error) {
    console.error("Failed to abort search:", error);
  }
}

export async function analyze(
  board: Board,
  player: Player,
  level: number,
  callback: (event: Event<AIMoveProgress>) => void
): Promise<void> {
  const validMoves = getValidMoves(board, player);
  if (validMoves.length === 0) return;

  const boardString = serializeBoardForAI(board, player);

  const unlisten = await listen<AIMoveProgress>("ai-move-progress", (data) => {
    callback(data);
  });

  try {
    await invoke<AIMoveResult>("analyze_command", {
      boardString,
      level,
    });
  } finally {
    unlisten();
  }
}
