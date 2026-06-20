import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";
import { ManualSetupTab } from "@/components/setup/ManualSetupTab";
import { TranscriptTab } from "@/components/setup/TranscriptTab";
import { BoardStringTab } from "@/components/setup/BoardStringTab";
import type { SetupTab } from "@/stores/slices/types";

/**
 * The three-tab setup scaffold (manual / transcript / board-string) shared by
 * the New Game and Solver modals. Reads `setupTab` from the store directly so
 * a fourth setup tab is added in one place, not once per modal.
 */
export function SetupTabs() {
  const { t } = useTranslation();
  const setupTab = useReversiStore((state) => state.setupTab);
  const setSetupTab = useReversiStore((state) => state.setSetupTab);

  return (
    <Tabs value={setupTab} onValueChange={(v) => setSetupTab(v as SetupTab)} className="py-4">
      <TabsList className="w-full">
        <TabsTrigger value="manual" className="flex-1">
          {t("setup.tabs.manual")}
        </TabsTrigger>
        <TabsTrigger value="transcript" className="flex-1">
          {t("setup.tabs.transcript")}
        </TabsTrigger>
        <TabsTrigger value="boardString" className="flex-1">
          {t("setup.tabs.boardString")}
        </TabsTrigger>
      </TabsList>

      <div className="grid mt-4">
        <TabsContent
          keepMounted
          value="manual"
          className="col-start-1 row-start-1 block data-hidden:invisible"
        >
          <ManualSetupTab />
        </TabsContent>
        <TabsContent
          keepMounted
          value="transcript"
          className="col-start-1 row-start-1 block data-hidden:invisible"
        >
          <TranscriptTab />
        </TabsContent>
        <TabsContent
          keepMounted
          value="boardString"
          className="col-start-1 row-start-1 block data-hidden:invisible"
        >
          <BoardStringTab />
        </TabsContent>
      </div>
    </Tabs>
  );
}
