import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SOLVER_SELECTIVITIES, type SolverSelectivity } from "@/services/types";

interface SolverSelectivitySelectorProps {
  idPrefix?: string;
}

export function SolverSelectivitySelector({
  idPrefix = "solver-selectivity",
}: SolverSelectivitySelectorProps) {
  const { t } = useTranslation();
  const targetSelectivity = useReversiStore((s) => s.targetSelectivity);
  const setTargetSelectivity = useReversiStore((s) => s.setTargetSelectivity);

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-foreground-muted">
        {t("solver.selectivity")}
      </span>
      <RadioGroup
        value={String(targetSelectivity)}
        onValueChange={(v) => void setTargetSelectivity(Number(v) as SolverSelectivity)}
        className="flex flex-row gap-3"
      >
        {SOLVER_SELECTIVITIES.map((sel) => {
          const id = `${idPrefix}-${sel}`;
          return (
            <label
              key={sel}
              htmlFor={id}
              className="flex items-center gap-1.5 text-sm text-foreground cursor-pointer"
            >
              <RadioGroupItem id={id} value={String(sel)} />
              {sel}%
            </label>
          );
        })}
      </RadioGroup>
    </div>
  );
}
