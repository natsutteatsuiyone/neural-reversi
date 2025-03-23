"use client";

import { GameBoard } from "@/components/game-board";
import { InfoPanel } from "@/components/info-panel";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useEffect } from "react";
import { toast } from "sonner";
import { Info } from "lucide-react";

export default function Reversi() {
  const showPassNotification = useReversiStore(
    (state) => state.showPassNotification
  );
  const hidePassNotification = useReversiStore(
    (state) => state.hidePassNotification
  );

  useEffect(() => {
    if (showPassNotification) {
      toast("No valid moves available. ", {
        description: "Passing turn.",
        icon: <Info className="w-4 h-4 text-blue-500" />,
        duration: 1500,
        onDismiss: () => {
          hidePassNotification();
        },
        onAutoClose: () => {
          hidePassNotification();
        },
      });
    }
  }, [showPassNotification, hidePassNotification]);

  return (
    <div className="flex flex-col lg:flex-row items-start justify-center gap-6 p-4 min-h-screen bg-[#0c513a]">
      <GameBoard />
      <InfoPanel />
    </div>
  );
}
