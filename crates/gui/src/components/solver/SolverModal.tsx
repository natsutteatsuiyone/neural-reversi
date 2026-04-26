import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Play } from "lucide-react";
import { useCallback } from "react";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";
import { ManualSetupTab } from "@/components/setup/ManualSetupTab";
import { TranscriptTab } from "@/components/setup/TranscriptTab";
import { BoardStringTab } from "@/components/setup/BoardStringTab";
import { SolverSelectivitySelector } from "./SolverSelectivitySelector";
import { SolverModeSelector } from "./SolverModeSelector";
import type { SetupTab } from "@/stores/slices/types";

export function SolverModal() {
    const { t } = useTranslation();

    const isOpen = useReversiStore((s) => s.isSolverModalOpen);
    const closeSolverModal = useReversiStore((s) => s.closeSolverModal);
    const setupTab = useReversiStore((s) => s.setupTab);
    const setSetupTab = useReversiStore((s) => s.setSetupTab);
    const setupError = useReversiStore((s) => s.setupError);
    const startSolverFromSetup = useReversiStore((s) => s.startSolverFromSetup);

    const handleOpenChange = useCallback(
        (open: boolean) => {
            if (!open) {
                closeSolverModal();
            }
        },
        [closeSolverModal],
    );

    const handleStart = async () => {
        try {
            const ok = await startSolverFromSetup();
            if (ok) {
                closeSolverModal();
            }
        } catch (error) {
            console.error("Failed to start solver:", error);
            toast.error(t("notification.startGameFailed"));
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={handleOpenChange}>
            <DialogContent
                aria-describedby={undefined}
                className="bg-card border-white/10 sm:max-w-lg"
                showCloseButton={false}
            >
                <DialogHeader>
                    <DialogTitle className="text-xl text-foreground">
                        {t("solver.title")}
                    </DialogTitle>
                </DialogHeader>

                <Tabs
                    value={setupTab}
                    onValueChange={(v) => setSetupTab(v as SetupTab)}
                    className="py-4"
                >
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
                            forceMount
                            value="manual"
                            className="col-start-1 row-start-1 data-[state=inactive]:invisible"
                        >
                            <ManualSetupTab />
                        </TabsContent>
                        <TabsContent
                            forceMount
                            value="transcript"
                            className="col-start-1 row-start-1 data-[state=inactive]:invisible"
                        >
                            <TranscriptTab />
                        </TabsContent>
                        <TabsContent
                            forceMount
                            value="boardString"
                            className="col-start-1 row-start-1 data-[state=inactive]:invisible"
                        >
                            <BoardStringTab />
                        </TabsContent>
                    </div>
                </Tabs>

                <div className="flex flex-row flex-wrap gap-6">
                    <SolverSelectivitySelector idPrefix="solver-modal-selectivity" />
                    <SolverModeSelector idPrefix="solver-modal-mode" />
                </div>

                <DialogFooter className="flex-row items-center gap-2">
                    <Button
                        variant="ghost"
                        onClick={closeSolverModal}
                        className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
                    >
                        {t("solver.cancel")}
                    </Button>
                    <div className="flex-1" />
                    <Button
                        onClick={() => void handleStart()}
                        disabled={!!setupError}
                        className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
                    >
                        <Play className="w-4 h-4" />
                        {t("solver.start")}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
