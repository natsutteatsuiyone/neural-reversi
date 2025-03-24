"use client";

import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import type { GameMode } from "@/types";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { BarChart } from "lucide-react";

interface GameModeSelectorProps {
  disabled?: boolean;
}

const Stone = ({ color }: { color: "black" | "white" }) => (
  <div
    className={cn(
      "w-5 h-5 rounded-full inline-flex shrink-0",
      color === "black"
        ? "bg-gradient-to-br from-neutral-600 to-black shadow-[inset_-1px_-1px_2px_rgba(255,255,255,0.1),inset_1px_1px_2px_rgba(0,0,0,0.4)]"
        : "bg-gradient-to-br from-white to-neutral-200 shadow-[inset_-1px_-1px_2px_rgba(0,0,0,0.05),inset_1px_1px_2px_rgba(255,255,255,0.8)] border border-white/20"
    )}
  />
);

export function GameModeSelector({ disabled }: GameModeSelectorProps) {
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiAccuracy = useReversiStore((state) => state.aiAccuracy);
  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIAccuracyChange = useReversiStore(
    (state) => state.setAIAccuracyChange
  );

  return (
    <div className="space-y-2">
      <h2 className="text-lg font-medium text-white/90">Game Mode</h2>
      <RadioGroup
        defaultValue={gameMode}
        onValueChange={(value) => setGameMode(value as GameMode)}
        className="grid grid-cols-1 gap-2"
        disabled={disabled}
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem
            value="ai-white"
            id="ai-white"
            className="border-white/20 text-white disabled:opacity-50"
            disabled={disabled}
          />
          <Label
            htmlFor="ai-white"
            className={`text-sm font-medium text-white/90 ${
              disabled ? "opacity-50" : ""
            }`}
          >
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1.5 w-16">
                <Stone color="black" />
                <span>Player</span>
              </div>
              <div className="flex items-center space-x-1">vs</div>
              <div className="flex items-center space-x-1.5">
                <Stone color="white" />
                <span>AI</span>
              </div>
            </div>
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem
            value="ai-black"
            id="ai-black"
            className="border-white/20 text-white disabled:opacity-50"
            disabled={disabled}
          />
          <Label
            htmlFor="ai-black"
            className={`text-sm font-medium text-white/90 ${
              disabled ? "opacity-50" : ""
            }`}
          >
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1.5 w-16">
                <Stone color="black" />
                <span>AI</span>
              </div>
              <div className="flex items-center space-x-1">vs</div>
              <div className="flex items-center justify-between space-x-1.5">
                <Stone color="white" />
                <span>Player</span>
              </div>
            </div>
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem
            value="analyze"
            id="analyze"
            className="border-white/20 text-white disabled:opacity-50"
            disabled={disabled}
          />
          <Label
            htmlFor="analyze"
            className={`text-sm font-medium text-white/90 ${
              disabled ? "opacity-50" : ""
            }`}
          >
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1.5">
                <BarChart size={16} className="" />
                <span>Analyze Mode</span>
              </div>
            </div>
          </Label>
        </div>
      </RadioGroup>
      <div className="space-y-3 pt-3 border-t border-white/10">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-white/90">AI Level</h3>
          <span className="text-sm font-mono text-white/90">{aiLevel}</span>
        </div>
        <Slider
          value={[aiLevel]}
          min={1}
          max={21}
          step={1}
          onValueChange={([value]) => setAILevelChange(value)}
          disabled={disabled}
          className="[&_[role=slider]]:bg-gradient-to-br [&_[role=slider]]:from-[#11936a] [&_[role=slider]]:to-[#0e7250]
              [&_[role=slider]]:border-0
              [&_[role=slider]]:shadow-[inset_0_1px_1px_rgba(255,255,255,0.25),0_1px_3px_rgba(0,0,0,0.3)]
              [&_[role=slider]]:hover:from-[#13a576] [&_[role=slider]]:hover:to-[#0f8259]
              [&_[role=slider]]:focus:from-[#13a576] [&_[role=slider]]:focus:to-[#0f8259]
              [&_[role=slider]]:focus:ring-2 [&_[role=slider]]:focus:ring-[#11936a]/30
              [&_[role=slider]]:transition-all [&_[role=slider]]:duration-150
              [&>.bg-primary]:bg-gradient-to-r [&>.bg-primary]:from-[#0e7250]/40 [&>.bg-primary]:to-[#11936a]/20"
        />
        <div className="flex justify-between text-xs font-medium">
          <span className="text-white/70">Easy</span>
          <span className="text-white/70">Hard</span>
        </div>
      </div>
      <div className="space-y-3 pt-3 border-white/10">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-white/90">
            Midgame Accuracy
          </h3>
          <span className="text-sm font-mono text-white/90">{aiAccuracy}</span>
        </div>
        <Slider
          value={[aiAccuracy]}
          min={1}
          max={6}
          step={1}
          onValueChange={([value]) => setAIAccuracyChange(value)}
          disabled={disabled}
          className="[&_[role=slider]]:bg-gradient-to-br [&_[role=slider]]:from-[#11936a] [&_[role=slider]]:to-[#0e7250]
              [&_[role=slider]]:border-0
              [&_[role=slider]]:shadow-[inset_0_1px_1px_rgba(255,255,255,0.25),0_1px_3px_rgba(0,0,0,0.3)]
              [&_[role=slider]]:hover:from-[#13a576] [&_[role=slider]]:hover:to-[#0f8259]
              [&_[role=slider]]:focus:from-[#13a576] [&_[role=slider]]:focus:to-[#0f8259]
              [&_[role=slider]]:focus:ring-2 [&_[role=slider]]:focus:ring-[#11936a]/30
              [&_[role=slider]]:transition-all [&_[role=slider]]:duration-150
              [&>.bg-primary]:bg-gradient-to-r [&>.bg-primary]:from-[#0e7250]/40 [&>.bg-primary]:to-[#11936a]/20"
        />
        <div className="flex justify-between text-xs font-medium">
          <span className="text-white/70">Low (Fast)</span>
          <span className="text-white/70">High (Slow)</span>
        </div>
      </div>
    </div>
  );
}
