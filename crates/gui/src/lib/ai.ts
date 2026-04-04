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

async function withEventListener<T, R>(
  event: string,
  callback: (ev: Event<T>) => void,
  fn: () => Promise<R>
): Promise<R> {
  const unlisten = await listen<T>(event, callback);
  try {
    return await fn();
  } finally {
    unlisten();
  }
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

  try {
    return await withEventListener<AIMoveProgress, AIMoveResult>(
      "ai-move-progress",
      callback,
      () => invoke<AIMoveResult>("ai_move_command", {
        boardString,
        level,
        timeLimit,
        remainingTime,
      }),
    );
  } catch {
    return null;
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

export async function resizeTT(hashSize: number): Promise<void> {
  try {
    await invoke("resize_tt_command", { hashSize });
  } catch (error) {
    console.error("Failed to resize TT:", error);
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

  await withEventListener<AIMoveProgress, void>(
    "ai-move-progress",
    callback,
    () => invoke("analyze_command", { boardString, level }),
  );
}

export type GameAnalysisProgress = {
  moveIndex: number;
  bestMove: string;
  bestScore: number;
  playedScore: number;
  scoreLoss: number;
  depth: number;
};

export async function analyzeGame(
  board: Board,
  player: Player,
  moves: string[],
  level: number,
  callback: (event: Event<GameAnalysisProgress>) => void
): Promise<void> {
  const boardString = serializeBoardForAI(board, player);

  await withEventListener<GameAnalysisProgress, void>(
    "game-analysis-progress",
    callback,
    () => invoke("analyze_game_command", { boardString, moves, level }),
  );
}

export async function abortGameAnalysis(): Promise<void> {
  try {
    await invoke("abort_game_analysis_command");
  } catch (error) {
    console.error("Failed to abort game analysis:", error);
  }
}
