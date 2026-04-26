import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SOLVER_MODES, type SolverMode } from "@/services/types";

interface SolverModeSelectorProps {
  idPrefix?: string;
  value?: SolverMode;
  onValueChange?: (value: SolverMode) => void;
}

export function SolverModeSelector({
  idPrefix = "solver-mode",
  value,
  onValueChange,
}: SolverModeSelectorProps) {
  const { t } = useTranslation();
  const storeValue = useReversiStore((s) => s.solverMode);
  const storeSetter = useReversiStore((s) => s.setSolverMode);
  const current = value ?? storeValue;
  const handleChange = onValueChange ?? storeSetter;

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-foreground-muted">
        {t("solver.mode")}
      </span>
      <RadioGroup
        value={current}
        onValueChange={(v) => void handleChange(v as SolverMode)}
        className="flex flex-row gap-3"
      >
        {SOLVER_MODES.map((mode) => {
          const id = `${idPrefix}-${mode}`;
          return (
            <label
              key={mode}
              htmlFor={id}
              className="flex items-center gap-1.5 text-sm text-foreground cursor-pointer"
            >
              <RadioGroupItem id={id} value={mode} />
              {t(`solver.modes.${mode}`)}
            </label>
          );
        })}
      </RadioGroup>
    </div>
  );
}
