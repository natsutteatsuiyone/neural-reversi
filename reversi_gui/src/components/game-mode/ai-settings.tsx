import { Slider } from "@/components/ui/slider";

interface AISettingsProps {
    aiLevel: number;
    setAILevelChange: (level: number) => void;
    disabled?: boolean;
}

export function AISettings({
    aiLevel,
    setAILevelChange,
    disabled,
}: AISettingsProps) {
    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-white/90">AI Level</h3>
                <span className="text-sm font-mono text-white/90">{aiLevel}</span>
            </div>
            <Slider
                value={[aiLevel]}
                min={1}
                max={24}
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
    );
}
