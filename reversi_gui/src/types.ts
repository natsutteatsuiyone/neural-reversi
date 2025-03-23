export type Player = "black" | "white"

export type Cell = {
  color: Player | null
  isNew?: boolean,
}

export type Board = Cell[][]

export type MoveRecord = {
  id: number
  player: Player
  row: number
  col: number
  notation: string
  score?: number
  isAI?: boolean
}

export type GameState = {
  board: Board
  currentPlayer: Player
  scores: {
    black: number
    white: number
  }
  moveCount: number
  gameOver: boolean
  lastMove: [number, number] | null
  validMoves: [number, number][]
  moves: MoveRecord[]
}

export type GameMode = "pvp" | "ai-black" | "ai-white"
export type GameStatus = "waiting" | "playing" | "finished"

