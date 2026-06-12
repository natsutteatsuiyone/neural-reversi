const RELAXED_SIMD_DETECTOR = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15,
  65, 2, 253, 15, 253, 128, 2, 11,
]);

export function supportsRelaxedSimd() {
  return WebAssembly.validate(RELAXED_SIMD_DETECTOR);
}

export async function importPreferredWasmModule({ relaxedPath, fallbackPath }) {
  if (supportsRelaxedSimd()) {
    try {
      const module = await import(/* @vite-ignore */ relaxedPath);
      return { module, relaxedSimd: true };
    } catch (error) {
      console.warn(
        `Failed to load relaxed SIMD Wasm module at ${relaxedPath}; falling back to SIMD128.`,
        error,
      );
    }
  }

  const module = await import(/* @vite-ignore */ fallbackPath);
  return { module, relaxedSimd: false };
}
