const { readFileSync } = require("node:fs");

const moduleBytes = readFileSync(process.argv[2]);
const compiled = new WebAssembly.Module(moduleBytes);
const { memory, fixture_ptr, fixture_len } = new WebAssembly.Instance(compiled).exports;
const ptr = fixture_ptr();
const len = fixture_len();
process.stdout.write(Buffer.from(memory.buffer, ptr, len));
