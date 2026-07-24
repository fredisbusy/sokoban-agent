import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { parseLevelCatalog, type LevelOption } from "./levels.ts";

const CATALOG_PATH = resolve(
  process.cwd(),
  "../src/sokoban_agent/data/level_catalog.json",
);

export async function loadLevelCatalog(): Promise<LevelOption[]> {
  const payload = JSON.parse(await readFile(CATALOG_PATH, "utf8")) as unknown;
  return parseLevelCatalog(payload);
}
