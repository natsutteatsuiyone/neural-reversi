import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Play, Timer, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { AIMode, GameMode } from "@/types";
import { Stone } from "@/components/board/Stone";

export function NewGameModal() {
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const gameTimeLimit = useReversiStore((state) => state.gameTimeLimit);

  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIMode = useReversiStore((state) => state.setAIMode);
  const setGameTimeLimit = useReversiStore((state) => state.setGameTimeLimit);
  const isNewGameModalOpen = useReversiStore((state) => state.isNewGameModalOpen);
  const setNewGameModalOpen = useReversiStore((state) => state.setNewGameModalOpen);
  const startGame = useReversiStore((state) => state.startGame);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);

  // Local state for pending settings
  const [localGameMode, setLocalGameMode] = useState<GameMode>(gameMode);
  const [localAILevel, setLocalAILevel] = useState(aiLevel);
  const [localAIMode, setLocalAIMode] = useState<AIMode>(aiMode);
  const [localGameTimeLimit, setLocalGameTimeLimit] = useState(gameTimeLimit);

  // Sync local state with store when modal opens
  useEffect(() => {
    if (isNewGameModalOpen) {
      setLocalGameMode(gameMode);
      setLocalAILevel(aiLevel);
      setLocalAIMode(aiMode);
      setLocalGameTimeLimit(gameTimeLimit);
    }
  }, [isNewGameModalOpen, gameMode, aiLevel, aiMode, gameTimeLimit]);

  const handleStartGame = async () => {
    // Commit settings
    setGameMode(localGameMode);
    setAILevelChange(localAILevel);
    setAIMode(localAIMode);
    setGameTimeLimit(localGameTimeLimit);

    await abortAIMove();
    await startGame();
    setNewGameModalOpen(false);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (secs === 0) return `${mins}m`;
    return `${mins}m ${secs}s`;
  };

  return (
    <Dialog open={isNewGameModalOpen} onOpenChange={setNewGameModalOpen}>
      <DialogContent 
        className="sm:max-w-md bg-card border-white/10" 
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle className="text-xl text-foreground">New Game</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Game Mode Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium text-foreground-secondary">You play</Label>
            <div className="grid grid-cols-2 gap-3">
              <GameModeOption
                selected={localGameMode === "ai-white"}
                onClick={() => setLocalGameMode("ai-white")}
                playerColor="black"
              />
              <GameModeOption
                selected={localGameMode === "ai-black"}
                onClick={() => setLocalGameMode("ai-black")}
                playerColor="white"
              />
            </div>
          </div>

          {/* AI Mode Tabs */}
          <div className="space-y-3">
            <Label className="text-sm font-medium text-foreground-secondary">AI Mode</Label>
            <Tabs
              value={localAIMode}
              onValueChange={(v) => setLocalAIMode(v as AIMode)}
            >
              <TabsList className="w-full bg-white/10">
                <TabsTrigger value="game-time" className="flex-1 gap-2 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary">
                  <Timer className="w-4 h-4" />
                  Timed
                </TabsTrigger>
                <TabsTrigger value="level" className="flex-1 gap-2 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary">
                  <Zap className="w-4 h-4" />
                  Level
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {/* Settings based on AI mode */}
          {localAIMode === "level" ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-foreground-secondary">AI Level</Label>
                <span className="text-sm font-mono font-semibold text-primary">
                  {localAILevel}
                </span>
              </div>
              <Slider
                value={[localAILevel]}
                min={1}
                max={24}
                step={1}
                onValueChange={([value]) => setLocalAILevel(value)}
              />
              <div className="flex justify-between text-xs text-foreground-muted">
                <span>Easy</span>
                <span>Hard</span>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-foreground-secondary">Time per game</Label>
                <span className="text-sm font-mono font-semibold text-primary">
                  {formatTime(localGameTimeLimit)}
                </span>
              </div>
              <Slider
                value={[localGameTimeLimit]}
                min={30}
                max={600}
                step={30}
                onValueChange={([value]) => setLocalGameTimeLimit(value)}
              />
              <div className="flex justify-between text-xs text-foreground-muted">
                <span>30s</span>
                <span>10m</span>
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button
            variant="ghost"
            onClick={() => setNewGameModalOpen(false)}
            className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
          >
            Cancel
          </Button>
          <Button 
            onClick={() => void handleStartGame()} 
            className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
          >
            <Play className="w-4 h-4" />
            Start Game
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GameModeOption({
  selected,
  onClick,
  playerColor,
}: {
  selected: boolean;
  onClick: () => void;
  playerColor: "black" | "white";
}) {
  const label = playerColor === "black" ? "Black" : "White";

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all",
        selected
          ? "border-primary bg-primary/10"
          : "border-white/20 hover:border-white/30 hover:bg-white/5"
      )}
    >
      <Stone color={playerColor} size="lg" />
      <span className="text-sm font-semibold text-foreground">{label}</span>
    </button>
  );
}
