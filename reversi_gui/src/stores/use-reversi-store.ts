import { type AIMoveProgress, type AIMoveResult, abortAISearch, getAIMove, initializeAI } from "@/lib/ai";
import {
  calculateScores,
  getFlippedDiscs,
  getNotation,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import type { Board, GameMode, MoveRecord } from "@/types";
import { create } from "zustand";

interface Move {
  row: number;
  col: number;
  isAI: boolean;
  score?: number;
}

interface ReversiState {
  board: Board;
  moves: MoveRecord[];
  currentPlayer: "black" | "white";
  gameOver: boolean;
  gameMode: GameMode;
  gameStatus: "waiting" | "playing" | "finished";
  isPass: boolean;
  lastAIMove: AIMoveResult | null;
  lastMove: Move | null;
  validMoves: [number, number][];
  aiLevel: number;
  aiAccuracy: number;
  aiMoveProgress: AIMoveProgress | null;
  showPassNotification: boolean;
  isAIThinking: boolean;

  getScores: () => { black: number; white: number };
  isAITurn: () => boolean;
  isValidMove: (row: number, col: number) => boolean;
  makeAIMove: () => Promise<void>;
  abortAIMove: () => Promise<void>;
  makeMove: (move: Move) => void;
  makePass: () => void;
  resetGame: () => void;
  startGame: () => void;
  setAILevelChange: (level: number) => void;
  setAIAccuracyChange: (accuracy: number) => void;
  setGameMode: (mode: GameMode) => void;
  setGameStatus: (status: "waiting" | "playing" | "finished") => void;
  hidePassNotification: () => void;
}

export const useReversiStore = create<ReversiState>((set, get) => ({
  board: initializeBoard(),
  moves: [],
  currentPlayer: "black",
  gameOver: false,
  gameMode: "ai-white",
  gameStatus: "waiting",
  isPass: false,
  lastAIMove: null,
  lastMove: null,
  validMoves: [],
  aiLevel: 10,
  aiAccuracy: 1,
  aiMoveProgress: null,
  showPassNotification: false,
  isAIThinking: false,

  getScores: () => {
    return calculateScores(get().board);
  },

  isAITurn: () => {
    const { gameMode, gameOver, currentPlayer } = get();
    return (
      (!gameOver && gameMode === "ai-black" && currentPlayer === "black") ||
      (gameMode === "ai-white" && currentPlayer === "white")
    );
  },

  isValidMove: (row, col) => {
    const { validMoves, gameStatus } = get();
    if (gameStatus !== "playing") {
      return false;
    }
    return validMoves.some((move) => move[0] === row && move[1] === col);
  },

  makeAIMove: async () => {
    set({ isAIThinking: true });
    const player = get().currentPlayer;
    const board = get().board;
    const aiMove = await getAIMove(board, player, get().aiLevel, get().aiAccuracy, (ev) => {
      set({ aiMoveProgress: ev.payload });
    });
    set({ aiMoveProgress: null, isAIThinking: false });
    if (aiMove) {
      const move = {
        row: aiMove.row,
        col: aiMove.col,
        score: aiMove.score,
        isAI: true,
      };
      get().makeMove(move);
      set({
        lastAIMove: aiMove,
      });
    }
  },

  abortAIMove: async () => {
    if (get().isAIThinking) {
      await abortAISearch();
      set({ isAIThinking: false, aiMoveProgress: null });
    }
  },

  makeMove: (move: Move) => {
    set((state) => {
      const newBoard = state.board.map((row) =>
        row.map((cell) => ({ ...cell }))
      );
      const currentPlayer = get().currentPlayer;
      const flipped = getFlippedDiscs(
        state.board,
        move.row,
        move.col,
        currentPlayer
      );

      newBoard[move.row][move.col] = {
        color: currentPlayer,
        isNew: true,
      };

      for (const [r, c] of flipped) {
        newBoard[r][c] = { color: currentPlayer };
      }

      return {
        board: newBoard,
        moves: [
          ...state.moves,
          {
            id: state.moves.length,
            player: currentPlayer,
            row: move.row,
            col: move.col,
            notation: getNotation(move.row, move.col),
            score: move.score,
            isAI: move.isAI,
          },
        ],
        currentPlayer: nextPlayer(state.currentPlayer),
        isPass: false,
        lastMove: move,
        validMoves: getValidMoves(newBoard, nextPlayer(currentPlayer)),
      };
    });

    const { validMoves } = get();
    if (validMoves.length === 0) {
      const otherPlayer = nextPlayer(get().currentPlayer);
      const otherMoves = getValidMoves(get().board, otherPlayer);
      if (otherMoves.length === 0) {
        set(() => {
          return {
            gameOver: true,
            gameStatus: "finished",
          };
        });
      } else {
        set({ showPassNotification : true });
      }
    } else {
      if (get().isAITurn()) {
        void get().makeAIMove();
      }
    }
  },

  makePass: () => {
    set((state) => {
      const currentPlayer = state.currentPlayer;
      const passMove = {
        id: state.moves.length,
        player: currentPlayer, // 修正: 現在のプレイヤーを使用
        row: -1,
        col: -1,
        notation: "Pass",
      };

      const validMoves = getValidMoves(state.board, nextPlayer(currentPlayer));

      return {
        board: state.board.map((row) => row.map((cell) => ({ ...cell }))),
        moves: [...state.moves, passMove],
        currentPlayer: nextPlayer(currentPlayer),
        validMoves,
        isPass: true,
      };
    });
  },

  resetGame: async () => {
    // AIが思考中なら中断する
    if (get().isAIThinking) {
      await get().abortAIMove();
    }

    set({
      board: initializeBoard(),
      moves: [],
      currentPlayer: "black",
      gameOver: false,
      gameStatus: "waiting",
      isPass: false,
      lastMove: null,
      lastAIMove: null,
      showPassNotification: false,
      isAIThinking: false,
    });
  },

  startGame: async () => {
    await initializeAI();

    set((state) => {
      const validMoves = getValidMoves(state.board, state.currentPlayer);
      return {
        board: initializeBoard(),
        gameStatus: "playing",
        validMoves,
      };
    });

    if (get().gameMode === "ai-black") {
      void get().makeAIMove();
    }
  },

  setAILevelChange: (level) => set({ aiLevel: level }),

  setAIAccuracyChange: (accuracy) => set({ aiAccuracy: accuracy }),

  setGameMode: (mode) => set({ gameMode: mode }),

  setGameStatus: (status) => set({ gameStatus: status }),

  hidePassNotification: () => {
    set({ showPassNotification: false });
    get().makePass();
    if (get().isAITurn()) {
      void get().makeAIMove();
    }
  },
}));
