import type { Board, Player } from "@/types"
import { BOARD_SIZE, INITIAL_BOARD } from "./constants"

export function initializeBoard(): Board {
  const board = Array(BOARD_SIZE)
    .fill(null)
    .map(() =>
      Array(BOARD_SIZE)
        .fill(null)
        .map(() => ({ color: null as Player | null })),
    )

  for (const [row, col, color] of INITIAL_BOARD) {
    board[row][col] = { color: color }
  }

  return board
}

export function getValidMoves(board: Board, player: Player): [number, number][] {
  const moves: [number, number][] = []

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (!board[row][col].color && getFlippedDiscs(board, row, col, player).length > 0) {
        moves.push([row, col])
      }
    }
  }

  return moves
}

export function getFlippedDiscs(board: Board, row: number, col: number, player: Player): [number, number][] {
  const opponent = player === "black" ? "white" : "black"
  const flipped: [number, number][] = []
  const directions = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
  ]

  for (const [dx, dy] of directions) {
    let x = row + dx
    let y = col + dy
    const temp: [number, number][] = []

    while (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE && board[x][y].color === opponent) {
      temp.push([x, y])
      x += dx
      y += dy
    }

    if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE && board[x][y].color === player && temp.length > 0) {
      flipped.push(...temp)
    }
  }

  return flipped
}

export function calculateScores(board: Board) {
  let black = 0
  let white = 0

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (board[row][col].color === "black") black++
      if (board[row][col].color === "white") white++
    }
  }

  return { black, white }
}

export function getWinner(scores: { black: number; white: number }) {
  if (scores.black > scores.white) return "black"
  if (scores.white > scores.black) return "white"
  return "draw"
}

export function getNotation(row: number, col: number): string {
  const column = String.fromCharCode(97 + col) // a-h
  const rowNum = row + 1 // 1-8
  return `${column}${rowNum}`
}

export function opponentPlayer(player: Player): Player {
  return player === "black" ? "white" : "black"
}
