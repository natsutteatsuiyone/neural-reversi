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
} | null;

export type AIMoveProgress = {
  bestMove: string;
  row: number;
  col: number;
  score: number;
  depth: number;
  acc: number;
};

export async function getAIMove(
  board: Board,
  player: Player,
  level: number,
  selectivity: number,
  callback: ((arg0: Event<AIMoveProgress>) => void)
): Promise<AIMoveResult> {
  const validMoves = getValidMoves(board, player);
  if (validMoves.length === 0) return null;

  // ボード文字列を作成（プレイヤーをX、相手をOとして）
  const boardString = board
    .flat()
    .map((cell) =>
      cell.color === player ? "X" : cell.color === null ? "-" : "O"
    )
    .join("");

  const unlisten = await listen<AIMoveProgress>("ai-move-progress", (data) => {
    console.log(data);
    callback(data);
  });

  try {
    return await invoke<AIMoveResult>("ai_move_command", {
      boardString,
      level,
      selectivity,
    });
  } catch (error) {
    return null;
  } finally {
    unlisten();
  }
}

export async function initializeAI(): Promise<void> {
  try {
    await invoke('init_ai_command');
  } catch (error) {
    console.error('Failed to initialize search:', error);
  }
}

export async function abortAISearch(): Promise<void> {
  try {
    await invoke('abort_ai_search_command');
  } catch (error) {
    console.error('Failed to abort search:', error);
  }
}
