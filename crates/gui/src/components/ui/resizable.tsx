import type { ComponentProps } from "react";
import { GripVertical } from "lucide-react";
import { Group, Panel, Separator } from "react-resizable-panels";

import { cn } from "@/lib/utils";

type GroupProps = Omit<ComponentProps<typeof Group>, "orientation"> & {
  direction: "horizontal" | "vertical";
};

function ResizablePanelGroup({ direction, className, ...props }: GroupProps) {
  return (
    <Group
      orientation={direction}
      className={cn("h-full w-full", className)}
      {...props}
    />
  );
}

function ResizablePanel({
  style,
  ...props
}: ComponentProps<typeof Panel>) {
  return (
    <Panel
      {...props}
      style={{
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        ...style,
      }}
    />
  );
}

type HandleProps = ComponentProps<typeof Separator> & {
  withHandle?: boolean;
};

function ResizableHandle({ withHandle, className, children, ...props }: HandleProps) {
  return (
    <Separator
      className={cn(
        "group relative flex shrink-0 items-center justify-center bg-white/10 transition-colors hover:bg-accent-blue/60 focus-visible:outline-hidden focus-visible:ring-1 focus-visible:ring-ring",
        // separator between horizontally-laid panels -> vertical divider
        "aria-[orientation=vertical]:w-px aria-[orientation=vertical]:cursor-col-resize",
        // separator between vertically-laid panels -> horizontal divider
        "aria-[orientation=horizontal]:h-px aria-[orientation=horizontal]:cursor-row-resize",
        // larger hit region via ::after
        "after:absolute after:inset-0",
        "aria-[orientation=vertical]:after:-inset-x-1",
        "aria-[orientation=horizontal]:after:-inset-y-1",
        className,
      )}
      {...props}
    >
      {withHandle && (
        <div
          className={cn(
            "z-10 flex items-center justify-center rounded-sm border border-white/15 bg-white/20 text-foreground-muted transition-colors group-hover:text-foreground",
            "group-aria-[orientation=vertical]:h-5 group-aria-[orientation=vertical]:w-3",
            "group-aria-[orientation=horizontal]:h-3 group-aria-[orientation=horizontal]:w-5",
          )}
        >
          <GripVertical className="h-2.5 w-2.5 group-aria-[orientation=horizontal]:rotate-90" />
        </div>
      )}
      {children}
    </Separator>
  );
}

export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
