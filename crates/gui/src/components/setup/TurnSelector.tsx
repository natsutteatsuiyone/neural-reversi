import { Label } from "@/components/ui/label";
import { Stone } from "@/components/board/Stone";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import type { Player } from "@/types";

interface TurnSelectorProps {
  currentPlayer: Player;
  onPlayerChange?: (player: Player) => void;
  readOnly?: boolean;
}

export function TurnSelector({
  currentPlayer,
  onPlayerChange,
  readOnly = false,
}: TurnSelectorProps) {
  const { t } = useTranslation();

  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium text-foreground-secondary">
        {t("setup.turn")}
      </Label>
      <div className="flex gap-2" role="radiogroup" aria-label={t("setup.turn")}>
        {(["black", "white"] as const).map((color) => {
          const isSelected = currentPlayer === color;
          return (
            <button
              key={color}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => !readOnly && onPlayerChange?.(color)}
              disabled={readOnly}
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-lg border transition-all",
                readOnly
                  ? isSelected
                    ? "border-white/15 bg-white/5 text-foreground cursor-default"
                    : "border-white/10 text-foreground-secondary/50 cursor-default opacity-50"
                  : isSelected
                    ? "border-primary bg-primary/15 text-foreground cursor-pointer"
                    : "border-white/15 text-foreground-secondary hover:border-white/25 hover:bg-white/5 cursor-pointer"
              )}
            >
              <Stone color={color} size="sm" />
              <span className="text-sm font-medium">{t(`colors.${color}`)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
