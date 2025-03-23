"use client"

import { useEffect, useState } from "react"
import { AlertDialog, AlertDialogContent, AlertDialogFooter } from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { SkipForward } from "lucide-react"
import type { Player } from "@/types"

interface PassDialogProps {
  isOpen: boolean
  onClose: () => void
  player: Player
}

export function PassDialog({ isOpen, onClose, player }: PassDialogProps) {
  const [open, setOpen] = useState(false)

  // isOpenプロップが変更されたときにダイアログの状態を更新
  useEffect(() => {
    setOpen(isOpen)
  }, [isOpen])

  // 3秒後に自動で閉じる
  useEffect(() => {
    if (open) {
      const timer = setTimeout(() => {
        setOpen(false)
        onClose()
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [open, onClose])

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent className="bg-white/10 border border-white/20 backdrop-blur-sm">
        <div className="flex flex-col items-center gap-4 py-4">
          <div className="p-3 rounded-full bg-amber-500/20">
            <SkipForward className="w-6 h-6 text-amber-300" />
          </div>
          <div className="text-center space-y-2">
            <h2 className="text-lg font-medium text-white/90">No Valid Moves</h2>
            <p className="text-sm text-white/70">
              {player === "black" ? "Black" : "White"} has no valid moves and must pass
            </p>
          </div>
        </div>
        <AlertDialogFooter>
          <Button
            variant="outline"
            className="w-full bg-white/10 hover:bg-white/20 text-white/90 border-white/20"
            onClick={() => {
              setOpen(false)
              onClose()
            }}
          >
            OK
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

