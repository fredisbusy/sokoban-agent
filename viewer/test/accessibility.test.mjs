import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("styles provide a reduced-motion fallback", async () => {
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /transition:\s*none\s*!important/);
});
