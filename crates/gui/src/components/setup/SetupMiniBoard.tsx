import { memo } from "react";
import { cn } from "@/lib/utils";
import { Stone } from "@/components/board/Stone";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import type { Board } from "@/types";

interface SetupMiniBoardProps {
  board: Board;
  editable?: boolean;
  onCellClick?: (row: number, col: number) => void;
}

export const SetupMiniBoard = memo(function SetupMiniBoard({
  board,
  editable = false,
  onCellClick,
}: SetupMiniBoardProps) {
  return (
    <div className="flex flex-col">
      {/* Column labels */}
      <div className="flex ml-5 shrink-0 h-4">
        {COLUMN_LABELS.map((label) => (
          <div
            key={label}
            className="flex-1 text-center text-xs font-semibold text-foreground-muted uppercase"
          >
            {label}
          </div>
        ))}
      </div>

      <div className="flex">
        {/* Row labels */}
        <div className="flex flex-col justify-around w-5 shrink-0">
          {ROW_LABELS.map((label) => (
            <div
              key={label}
              className="text-center text-xs font-semibold text-foreground-muted"
            >
              {label}
            </div>
          ))}
        </div>

        {/* Board grid */}
        <div className="bg-board-surface p-1.5 rounded-md w-56 h-56">
          <div className="grid grid-cols-8 grid-rows-8 gap-px h-full w-full">
            {board.map((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <button
                  key={`${colIndex}-${rowIndex}`}
                  type="button"
                  className={cn(
                    "w-full h-full relative flex items-center justify-center",
                    "bg-board-cell",
                    editable && "hover:bg-board-cell-hover cursor-pointer",
                    !editable && "cursor-default"
                  )}
                  onClick={() => editable && onCellClick?.(rowIndex, colIndex)}
                  disabled={!editable}
                  aria-label={`${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]}`}
                >
                  {cell.color && <Stone color={cell.color} />}
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
});
