/**
 * Board interaction timings, shared by three layers that must agree on them:
 * the 3D renderer (flip animation length), the Automation scheduler (when the
 * next auto-step may run — only after the flip/pass notice the user is
 * watching), and the pass-notification toast. Keeping them here is the single
 * home for "how long a board interaction takes"; previously the renderer, a
 * store slice, and the App toast each owned a copy and had to be kept in sync.
 */

/** Disc flip animation duration, in seconds (consumed by the 3D renderer). */
export const FLIP_DURATION_S = 0.4;

/** Disc flip animation duration, in milliseconds (consumed by the scheduler). */
export const FLIP_DURATION_MS = FLIP_DURATION_S * 1000;

/** How long the "turn passed" notice stays up before the next auto-step. */
export const PASS_NOTIFICATION_DURATION_MS = 1500;

/** How long a just-played AI move stays highlighted on the board. */
export const AI_MOVE_HIGHLIGHT_DURATION_MS = 1200;
