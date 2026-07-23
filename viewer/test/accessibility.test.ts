import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("styles provide a reduced-motion fallback", async () => {
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /transition:\s*none\s*!important/);
});

test("interactive form does not ask users for an internal prompt commit", async () => {
  const component = await Promise.all([
    readFile(
      new URL("../components/live-viewer.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../components/run-form.tsx", import.meta.url), "utf8"),
  ]).then((parts) => parts.join("\n"));
  assert.doesNotMatch(component, /name="prompt_commit"/);
  assert.doesNotMatch(component, />Prompt commit</);
});

test("run form selects curated maps instead of requiring a level id", async () => {
  const component = await readFile(
    new URL("../components/run-form.tsx", import.meta.url),
    "utf8",
  );
  assert.match(component, /<select[\s\S]*name="level_id"/);
  assert.match(component, /<optgroup/);
  assert.doesNotMatch(component, /<input[^>]*name="level_id"/);
});
