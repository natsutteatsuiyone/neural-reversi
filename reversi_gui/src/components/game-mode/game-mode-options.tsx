import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { BarChart } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GameMode } from "@/types";

interface GameModeOptionsProps {
    gameMode: GameMode;
    setGameMode: (mode: GameMode) => void;
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

export function GameModeOptions({ gameMode, setGameMode, disabled }: GameModeOptionsProps) {
    return (
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
                    className={`text-sm font-medium text-white/90 ${disabled ? "opacity-50" : ""
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
                    className={`text-sm font-medium text-white/90 ${disabled ? "opacity-50" : ""
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
                    className={`text-sm font-medium text-white/90 ${disabled ? "opacity-50" : ""
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
    );
}
