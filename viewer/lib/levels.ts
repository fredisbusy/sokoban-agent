export type LevelDifficulty =
  | "builtin"
  | "custom"
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
  sha256: string;
  sourceType: "boxoban" | "custom";
  sourceLevelId: string;
  difficulty: LevelDifficulty;
  displayName: string;
  rows: string[];
  recommendedMaxSteps: number;
  reference: LevelReference | null;
}

export const DIFFICULTY_ORDER: LevelDifficulty[] = [
  "builtin",
  "custom",
  "unfiltered",
  "medium",
  "hard",
];

export function parseLevelCatalog(payload: unknown): LevelOption[] {
  const catalog = record(payload, "catalog");
  const levels = array(catalog.levels, "catalog levels");
  const parsed = levels.map((value) => {
    const level = record(value, "level");
    const difficulty = text(level.difficulty, "difficulty");
    if (
      !["builtin", "custom", "unfiltered", "medium", "hard"].includes(
        difficulty,
      )
    ) {
      throw new Error(`지원하지 않는 맵 난이도: ${difficulty}`);
    }
    const source = record(level.source, "source");
    const sourceType = text(source.type, "source type");
    if (!["boxoban", "custom"].includes(sourceType)) {
      throw new Error(`지원하지 않는 맵 출처: ${sourceType}`);
    }
    const reference = level.reference === null
      ? null
      : parseReference(record(level.reference, "reference"));
    const rows = stringArray(level.rows, "rows");
    const sourceLevelId = sourceType === "boxoban"
      ? text(source.source_level_id, "source_level_id")
      : text(level.id, "id");
    return {
      id: text(level.id, "id"),
      sha256: text(level.sha256, "sha256"),
      sourceType: sourceType as "boxoban" | "custom",
      sourceLevelId,
      difficulty: difficulty as LevelDifficulty,
      displayName: `${difficultyLabel(
        difficulty as LevelDifficulty,
      )} · ${sourceLevelId}`,
      rows,
      recommendedMaxSteps: positiveInteger(
        level.recommended_max_steps,
        "recommended_max_steps",
      ),
      reference,
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
    custom: "사용자 맵",
    unfiltered: "기본 생성군 · unfiltered",
    medium: "중급 · medium",
    hard: "고급 · hard",
  };
  return labels[difficulty];
}

function parseReference(
  reference: Record<string, unknown>,
): LevelReference {
  return {
    actionCount: positiveInteger(reference.action_count, "action_count"),
    pushCount: positiveInteger(reference.push_count, "push_count"),
    expandedStates: positiveInteger(
      reference.expanded_states,
      "expanded_states",
    ),
  };
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
