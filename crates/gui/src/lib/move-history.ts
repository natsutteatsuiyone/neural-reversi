import type { MoveRecord } from "@/types";

export class MoveHistory {
  private constructor(
    private readonly timeline: readonly MoveRecord[],
    private readonly cursor: number,
  ) {}

  static empty(): MoveHistory {
    return new MoveHistory([], 0);
  }

  append(record: MoveRecord): MoveHistory {
    const newTimeline = [...this.timeline.slice(0, this.cursor), record];
    return new MoveHistory(newTimeline, newTimeline.length);
  }

  undo(count: number): MoveHistory {
    if (count === 0) return this;
    const newCursor = Math.max(0, this.cursor - count);
    return new MoveHistory(this.timeline, newCursor);
  }

  redo(count: number): MoveHistory {
    if (count === 0) return this;
    const newCursor = Math.min(this.timeline.length, this.cursor + count);
    return new MoveHistory(this.timeline, newCursor);
  }

  get currentMoves(): readonly MoveRecord[] {
    return this.timeline.slice(0, this.cursor);
  }

  get canUndo(): boolean {
    return this.cursor > 0;
  }

  get canRedo(): boolean {
    return this.cursor < this.timeline.length;
  }

  get length(): number {
    return this.cursor;
  }

  get totalLength(): number {
    return this.timeline.length;
  }

  get lastMove(): MoveRecord | undefined {
    return this.cursor > 0 ? this.timeline[this.cursor - 1] : undefined;
  }

  get redoMoves(): readonly MoveRecord[] {
    return this.timeline.slice(this.cursor);
  }
}
