import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SOLVER_SELECTIVITIES, type SolverSelectivity } from "@/services/types";

interface SolverSelectivitySelectorProps {
  idPrefix?: string;
  value?: SolverSelectivity;
  onValueChange?: (value: SolverSelectivity) => void;
}

export function SolverSelectivitySelector({
  idPrefix = "solver-selectivity",
  value,
  onValueChange,
}: SolverSelectivitySelectorProps) {
  const { t } = useTranslation();
  const storeValue = useReversiStore((s) => s.targetSelectivity);
  const storeSetter = useReversiStore((s) => s.setTargetSelectivity);
  const current = value ?? storeValue;
  const handleChange = onValueChange ?? storeSetter;

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-foreground-muted">
        {t("solver.selectivity")}
      </span>
      <RadioGroup
        value={String(current)}
        onValueChange={(v) => void handleChange(Number(v) as SolverSelectivity)}
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
