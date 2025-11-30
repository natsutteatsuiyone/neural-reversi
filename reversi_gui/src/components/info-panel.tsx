"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Trophy, RotateCcw, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MoveHistory } from "./move-history";
import { GameModeSelector } from "./game-mode-selector";
import { useReversiStore } from "@/stores/use-reversi-store";
import { getWinner } from "@/lib/game-logic";
import { AIEvaluationChart } from "./ai-evaluation-chart";
import { PlayerScoreCard } from "./info-panel/player-score-card";

export function InfoPanel() {
  const currentPlayer = useReversiStore((state) => state.currentPlayer);
  const gameOver = useReversiStore((state) => state.gameOver);
  const gameMode = useReversiStore((state) => state.gameMode);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const lastAIMove = useReversiStore((state) => state.lastAIMove);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const aiRemainingTime = useReversiStore((state) => state.aiRemainingTime);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress);
  const startGame = useReversiStore((state) => state.startGame);
  const resetGame = useReversiStore((state) => state.resetGame);
  const getScores = useReversiStore((state) => state.getScores);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);

  const isPlaying = gameStatus === "playing" || gameStatus === "finished";
  const scores = getScores();
  const winner = gameOver ? getWinner(scores) : null;
  const blackIsAI = gameMode === "ai-black";
  const whiteIsAI = gameMode === "ai-white";

  const handleReset = async () => {
    await abortAIMove();
    await resetGame();
  };

  return (
    <div className="w-full lg:w-96 h-full lg:h-[calc(100vh-2rem)] bg-white/20 dark:bg-slate-900/30 rounded-xl shadow-lg p-6 shrink-0 backdrop-blur-md border border-white/20 flex flex-col">
      {isPlaying ? (
        <>
          {!gameOver && (
            <div className="mb-3">
              <div className="flex flex-col gap-3">
                <PlayerScoreCard
                  color="black"
                  score={scores.black}
                  isCurrent={currentPlayer === "black"}
                  isAIControlled={blackIsAI}
                  aiLevel={aiLevel}
                  isThinking={blackIsAI && currentPlayer === "black"}
                  lastAIMove={blackIsAI ? lastAIMove : null}
                  aiMoveProgress={aiMoveProgress}
                  onAbort={abortAIMove}
                  aiMode={aiMode}
                  aiRemainingTime={aiRemainingTime}
                />
                <PlayerScoreCard
                  color="white"
                  score={scores.white}
                  isCurrent={currentPlayer === "white"}
                  isAIControlled={whiteIsAI}
                  aiLevel={aiLevel}
                  isThinking={whiteIsAI && currentPlayer === "white"}
                  lastAIMove={whiteIsAI ? lastAIMove : null}
                  aiMoveProgress={aiMoveProgress}
                  onAbort={abortAIMove}
                  aiMode={aiMode}
                  aiRemainingTime={aiRemainingTime}
                />
              </div>
            </div>
          )}

          <div className="">
            <AIEvaluationChart />
          </div>

          <div className="flex-1 min-h-0 mt-2">
            <MoveHistory />
          </div>

          <div className="mt-4 shrink-0">
            <Button
              variant="outline"
              size="lg"
              className="w-full bg-white/10 hover:bg-white/20 text-white/90 border-white/20"
              onClick={handleReset}
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
