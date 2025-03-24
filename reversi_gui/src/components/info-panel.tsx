"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Trophy, RotateCcw, Play, Bot, CircleX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MoveHistory } from "./move-history";
import { GameModeSelector } from "./game-mode-selector";
import { cn } from "@/lib/utils";
import { useReversiStore } from "@/stores/use-reversi-store";
import { getNotation, getWinner } from "@/lib/game-logic";

export function InfoPanel() {
  const currentPlayer = useReversiStore((state) => state.currentPlayer);
  const gameOver = useReversiStore((state) => state.gameOver);
  const gameMode = useReversiStore((state) => state.gameMode);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const lastAIMove = useReversiStore((state) => state.lastAIMove);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress);
  const startGame = useReversiStore((state) => state.startGame);
  const resetGame = useReversiStore((state) => state.resetGame);
  const getScores = useReversiStore((state) => state.getScores);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);
  const isPlaying = gameStatus === "playing" || gameStatus === "finished";

  const scores = getScores();
  const winner = gameOver ? getWinner(scores) : null;

  const AIInfo = ({
    thinking,
    lastMove,
  }: {
    thinking: boolean;
    lastMove: typeof lastAIMove;
  }) => {
    if (!lastMove && !thinking) return null;
    let move = "";
    let score = "";
    let depth = "";
    if (lastMove) {
      move = getNotation(lastMove.row, lastMove.col);
      score =
        lastMove.score > 0 ? `+${lastMove.score}` : String(lastMove.score);
      depth =
        lastMove.acc === 100
          ? `${lastMove.depth}`
          : `${lastMove.depth}@${lastMove.acc}%`;
    } else if (aiMoveProgress) {
      move = aiMoveProgress.bestMove;
      score =
        aiMoveProgress.score > 0
          ? `+${aiMoveProgress.score}`
          : String(aiMoveProgress.score);
      depth =
        aiMoveProgress.acc === 100
          ? `${aiMoveProgress.depth}`
          : `${aiMoveProgress.depth}@${aiMoveProgress.acc}%`;
    }

    return (
      <div className="flex items-center grow">
        <div className="flex items-center gap-2 grow">
          <Bot
            className={cn(
              "w-4 h-4",
              thinking ? "text-emerald-300 animate-pulse" : "text-emerald-300"
            )}
          />
          {thinking && aiMoveProgress ? (
            <div className="text-sm text-emerald-200/70 font-mono">
              <div>
                <span>{move}</span>
                <span className="ml-1">({score})</span>
              </div>
              <div> {depth}</div>
            </div>
          ) : (
            lastMove && (
              <div className="text-sm text-emerald-200/70 font-mono">
                <div>
                  <span>{move}</span>
                  <span className="ml-1">({score})</span>
                </div>
                <div> {depth}</div>
              </div>
            )
          )}
        </div>
        {thinking && (
          <div className="text-sm text-emerald-200/70 flex flex-col items-center">
            <button type="button" onClick={async () => await abortAIMove()}>
              <CircleX className="w-5 h-5 text-amber-200 hover:text-amber-100 cursor-pointer" />
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full lg:w-80 h-full lg:h-[600px] bg-white/20 dark:bg-slate-900/30 rounded-xl shadow-lg p-6 shrink-0 backdrop-blur-md border border-white/20 flex flex-col">
      {isPlaying ? (
        <>
          {!gameOver && (
            <div className="mb-3">
              <h2 className="text-lg font-medium text-white/90 mb-2">
                Game Status
              </h2>
              <div className="flex flex-col gap-3">
                {/* Black */}
                <div
                  className={cn(
                    "rounded-lg h-14",
                    currentPlayer === "black" ? "bg-white/20" : "bg-white/10"
                  )}
                >
                  <div className="h-full px-3 flex items-center gap-3">
                    <div className="shrink-0">
                      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neutral-600 to-black shadow-md" />
                    </div>
                    <div className="flex-1 min-w-0 flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xl font-bold text-white/90">
                          {scores.black}
                        </span>
                        <span className="text-sm text-white/70">
                          {gameMode === "ai-black" ? (
                            <div className="flex flex-col">
                              <span>AI</span>
                              <span>Lv.{aiLevel}</span>
                            </div>
                          ) : (
                            <div>Player</div>
                          )}
                        </span>
                      </div>
                      {gameMode === "ai-black" && (
                        <AIInfo
                          thinking={currentPlayer === "black"}
                          lastMove={
                            currentPlayer !== "black" ? lastAIMove : null
                          }
                        />
                      )}
                    </div>
                    {currentPlayer === "black" && (
                      <div className="shrink-0">
                        <div className="w-2 h-2 rounded-full bg-emerald-400" />
                      </div>
                    )}
                    {currentPlayer === "white" && (
                      <div className="shrink-0">
                        <div className="w-2 h-2 rounded-full " />
                      </div>
                    )}
                  </div>
                </div>

                {/* White */}
                <div
                  className={cn(
                    "rounded-lg h-14",
                    currentPlayer === "white" ? "bg-white/20" : "bg-white/10"
                  )}
                >
                  <div className="h-full px-3 flex items-center gap-3">
                    <div className="shrink-0">
                      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-white to-neutral-200 border border-white/20 shadow-md" />
                    </div>
                    <div className="flex-1 min-w-0 flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xl font-bold text-white/90">
                          {scores.white}
                        </span>
                        <span className="text-sm text-white/70">
                          {gameMode === "ai-white" ? (
                            <div className="flex flex-col">
                              <span>AI</span>
                              <span>Lv.{aiLevel}</span>
                            </div>
                          ) : (
                            <div>Player</div>
                          )}
                        </span>
                      </div>
                      {gameMode === "ai-white" && (
                        <AIInfo
                          thinking={currentPlayer === "white"}
                          lastMove={
                            currentPlayer !== "white" ? lastAIMove : null
                          }
                        />
                      )}
                    </div>
                    {currentPlayer === "white" && (
                      <div className="shrink-0">
                        <div className="w-2 h-2 rounded-full bg-emerald-400" />
                      </div>
                    )}
                    {currentPlayer === "black" && (
                      <div className="shrink-0">
                        <div className="w-2 h-2 rounded-full " />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex-1 min-h-0 mt-4">
            <MoveHistory />
          </div>

          <div className="mt-4 shrink-0">
            <Button
              variant="outline"
              size="lg"
              className="w-full bg-white/10 hover:bg-white/20 text-white/90 border-white/20"
              onClick={async () => {
                await abortAIMove();
                await resetGame();
              }}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              New Game
            </Button>
          </div>

          <AnimatePresence>
            {gameOver && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="mt-4 shrink-0"
              >
                <div className="p-4 rounded-lg bg-white/10 text-center space-y-4 backdrop-blur-sm">
                  <Trophy className="w-8 h-8 mx-auto text-amber-300" />
                  <div className="space-y-4">
                    <p className="text-lg font-medium text-white/90">
                      {winner === "draw"
                        ? "Draw"
                        : `${winner === "black" ? "Black" : "White"} wins!`}
                    </p>

                    <div className="grid grid-cols-2 gap-3 pt-2">
                      <div className="space-y-1.5">
                        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-neutral-600 to-black shadow-md mx-auto" />
                        <motion.div
                          initial={{ opacity: 0, scale: 0.5 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.2 }}
                          className="text-2xl font-bold text-white/90"
                        >
                          {scores.black}
                        </motion.div>
                      </div>
                      <div className="space-y-1.5">
                        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-white to-neutral-200 border border-white/20 shadow-md mx-auto" />
                        <motion.div
                          initial={{ opacity: 0, scale: 0.5 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.3 }}
                          className="text-2xl font-bold text-white/90"
                        >
                          {scores.white}
                        </motion.div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      ) : (
        <div className="space-y-6">
          <GameModeSelector disabled={false} />
          <Button
            variant="outline"
            size="lg"
            className="w-full bg-white/10 hover:bg-white/20 text-white/90 border-white/20"
            onClick={startGame}
          >
            <Play className="w-4 h-4 mr-2" />
            Start Game
          </Button>
        </div>
      )}
    </div>
  );
}
