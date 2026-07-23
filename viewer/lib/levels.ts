export type LevelDifficulty =
  | "builtin"
  | "unfiltered"
  | "medium"
  | "hard";

export interface LevelReference {
  actionCount: number;
  pushCount: number;
  expandedStates: number;
}

export interface LevelOption {
  id: string;
  sourceLevelId: string;
  difficulty: LevelDifficulty;
  displayName: string;
  rows: string[];
  recommendedMaxSteps: number;
  reference: LevelReference | null;
}

export const DIFFICULTY_ORDER: LevelDifficulty[] = [
  "builtin",
  "unfiltered",
  "medium",
  "hard",
];

export const BUILTIN_LEVELS: LevelOption[] = [
  {
    id: "tiny-walk",
    sourceLevelId: "tiny-walk",
    difficulty: "builtin",
    displayName: "개발용 · tiny-walk",
    rows: ["#####", "#  .#", "# $ #", "#@  #", "#####"],
    recommendedMaxSteps: 15,
    reference: null,
  },
  {
    id: "tiny-push",
    sourceLevelId: "tiny-push",
    difficulty: "builtin",
    displayName: "개발용 · tiny-push",
    rows: ["#####", "# . #", "# $ #", "# @ #", "#####"],
    recommendedMaxSteps: 15,
    reference: null,
  },
];

export function parseResearchLevels(payload: unknown): LevelOption[] {
  const manifest = record(payload, "manifest");
  const levels = array(manifest.levels, "manifest levels");
  const parsed = levels.map((value) => {
    const level = record(value, "level");
    const difficulty = text(level.difficulty, "difficulty");
    if (!["unfiltered", "medium", "hard"].includes(difficulty)) {
      throw new Error(`지원하지 않는 Boxoban 난이도: ${difficulty}`);
    }
    const reference = record(level.reference, "reference");
    const actionCount = positiveInteger(reference.action_count, "action_count");
    const pushCount = positiveInteger(reference.push_count, "push_count");
    const rows = stringArray(level.rows, "rows");
    return {
      id: text(level.level_id, "level_id"),
      sourceLevelId: text(level.source_level_id, "source_level_id"),
      difficulty: difficulty as LevelDifficulty,
      displayName: `${difficultyLabel(difficulty as LevelDifficulty)} · ${
        text(level.source_level_id, "source_level_id")
      }`,
      rows,
      recommendedMaxSteps: Math.max(120, actionCount * 3),
      reference: {
        actionCount,
        pushCount,
        expandedStates: positiveInteger(
          reference.expanded_states,
          "expanded_states",
        ),
      },
    };
  });
  if (new Set(parsed.map((level) => level.id)).size !== parsed.length) {
    throw new Error("Boxoban level ID가 중복되었습니다");
  }
  return parsed;
}

export function difficultyLabel(difficulty: LevelDifficulty): string {
  const labels: Record<LevelDifficulty, string> = {
    builtin: "개발용",
    unfiltered: "기본 생성군 · unfiltered",
    medium: "중급 · medium",
    hard: "고급 · hard",
  };
  return labels[difficulty];
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} 값이 객체가 아닙니다`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} 값이 배열이 아닙니다`);
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} 값이 문자열이 아닙니다`);
  }
  return value;
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || Number(value) < 1) {
    throw new Error(`${label} 값이 양의 정수가 아닙니다`);
  }
  return Number(value);
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new Error(`${label} 값이 문자열 배열이 아닙니다`);
  }
  return value;
}
