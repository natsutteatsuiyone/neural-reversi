/**
 * Coerces a Base UI Slider `onValueChange` payload to a single number.
 *
 * Base UI passes a plain `number` at runtime for a single-thumb slider even
 * though the callback type looks like an array, so `(value as number[])[0]`
 * yields `undefined` -> `NaN`. This helper handles both shapes safely.
 */
export function sliderValueToNumber(value: number | readonly number[]): number {
  return typeof value === "number" ? value : value[0];
}
