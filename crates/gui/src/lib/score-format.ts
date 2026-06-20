/**
 * Score Readout (CONTEXT.md → Score Readout): how an engine score is rendered
 * for a human.
 *
 * The sign rule (`+` when positive, the value's own `-` when negative, no sign
 * at zero) is universal and lives only here. The rounding is a per-surface
 * choice each caller names explicitly, so a new score display cannot silently
 * drift in sign or pick a one-off rounding.
 */

/** How the raw engine score is reduced before the sign rule is applied. */
export type ScoreRounding =
  /** Nearest whole disc — cell overlays, solver candidates. */
  | "whole"
  /** One decimal — the evaluation chart's trend line and the move history. */
  | "tenth"
  /** Verbatim — the thinking log / analysis header. */
  | "raw";

function reduce(score: number, rounding: ScoreRounding): string {
  switch (rounding) {
    case "whole":
      return String(Math.round(score));
    case "tenth":
      return score.toFixed(1);
    case "raw":
      return String(score);
  }
}

/**
 * Signed score string. `rounding` names how the raw score is reduced first;
 * pick the one the surface needs and never re-implement the sign rule.
 */
export function formatScore(score: number, rounding: ScoreRounding = "whole"): string {
  const reduced = reduce(score, rounding);
  // Sign the *reduced* value, not the raw score: a score that reduces to zero
  // (e.g. 0.4 with `whole`, or -0.04 with `tenth` → "-0.0") must read "0"/"0.0"
  // with no sign — positive *or* negative zero.
  const n = Number(reduced);
  if (n > 0) return `+${reduced}`;
  if (n < 0) return reduced;
  return reduce(0, rounding);
}

/** Sign → text-colour class for a signed-score readout. */
export function scoreToneClass(score: number): string {
  return score > 0 ? "text-primary" : score < 0 ? "text-destructive" : "text-foreground";
}

/** Search depth readout: bare depth at full accuracy, else `depth@acc%`. */
export function formatDepth(depth: number, acc: number): string {
  return acc === 100 ? `${depth}` : `${depth}@${acc}%`;
}
