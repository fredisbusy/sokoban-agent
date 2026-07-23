import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  BUILTIN_LEVELS,
  parseResearchLevels,
  type LevelOption,
} from "./levels.ts";

const MANIFEST_PATH = resolve(
  process.cwd(),
  "../benchmarks/boxoban_research_v1.json",
);

export async function loadLevelCatalog(): Promise<LevelOption[]> {
  const payload = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as unknown;
  return [...BUILTIN_LEVELS, ...parseResearchLevels(payload)];
}
