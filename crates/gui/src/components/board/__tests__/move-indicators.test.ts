import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LAST_MOVE_RING_COLOR } from "../board3d-utils";
import { MoveIndicators } from "../MoveIndicators";

describe("MoveIndicators", () => {
  it("renders one valid-move dot per valid move at the expected world coordinates", () => {
    const markup = renderToStaticMarkup(
      createElement(MoveIndicators, {
        validMoves: [
          { row: 2, col: 3 },
          { row: 4, col: 5 },
        ],
        lastMove: null,
      }),
    );

    expect(markup.match(/<circleGeometry/g)).toHaveLength(2);
    expect(markup).toContain('position="-0.5,0.002,-1.5"');
    expect(markup).toContain('position="1.5,0.002,0.5"');
    expect(markup).toContain('opacity="0.3"');
  });

  it("renders the last-move ring with the configured highlight color", () => {
    const markup = renderToStaticMarkup(
      createElement(MoveIndicators, {
        validMoves: [],
        lastMove: { row: 5, col: 4 },
      }),
    );

    expect(markup).toContain("<ringGeometry");
    expect(markup).toContain('position="0.5,0.002,1.5"');
    expect(markup).toContain(`color="${LAST_MOVE_RING_COLOR}"`);
  });
});
