import assert from "node:assert/strict";
import test from "node:test";

import { loadLevelCatalog } from "../lib/level-catalog-server.ts";

test("catalog exposes built-in and balanced official Boxoban difficulties", async () => {
  const levels = await loadLevelCatalog();

  assert.equal(levels.length, 17);
  assert.deepEqual(
    Object.fromEntries(
      ["builtin", "unfiltered", "medium", "hard"].map((difficulty) => [
        difficulty,
        levels.filter((level) => level.difficulty === difficulty).length,
      ]),
    ),
    { builtin: 2, unfiltered: 5, medium: 5, hard: 5 },
  );
  assert.equal(new Set(levels.map((level) => level.id)).size, levels.length);
  assert.equal(
    levels.find((level) => level.id === "tiny-walk")?.sourceType,
    "custom",
  );
  assert.ok(levels.every((level) => level.sha256.length === 64));
  for (const level of levels.filter((item) => item.difficulty !== "builtin")) {
    assert.equal(level.rows.length, 10);
    assert.ok(level.rows.every((row) => row.length === 10));
    assert.ok(level.reference);
    assert.ok(level.recommendedMaxSteps >= level.reference.actionCount);
  }
});
