"use client";

import { type FormEvent, useState } from "react";

import {
  DIFFICULTY_ORDER,
  difficultyLabel,
  type LevelOption,
} from "../lib/levels";

interface RunFormProps {
  levels: LevelOption[];
  running: boolean;
  onLevelChange: (level: LevelOption) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

export function RunForm({
  levels,
  running,
  onLevelChange,
  onSubmit,
}: RunFormProps) {
  const initial = levels.find((level) => level.id === "tiny-walk") ?? levels[0];
  const [selectedId, setSelectedId] = useState(initial.id);
  const [maxSteps, setMaxSteps] = useState(initial.recommendedMaxSteps);

  function selectLevel(levelId: string) {
    const selected = levels.find((level) => level.id === levelId);
    if (!selected) return;
    setSelectedId(levelId);
    setMaxSteps(selected.recommendedMaxSteps);
    onLevelChange(selected);
  }

  return (
    <form className="run-form" onSubmit={onSubmit}>
      <Field
        label="Agent Server"
        name="api_url"
        defaultValue="http://localhost:2024"
        type="url"
      />
      <Field
        label="Assistant"
        name="assistant_id"
        defaultValue="sokoban_agent"
      />
      <label className="level-field">
        맵
        <select
          name="level_id"
          value={selectedId}
          onChange={(event) => selectLevel(event.target.value)}
        >
          {DIFFICULTY_ORDER.map((difficulty) => {
            const options = levels.filter(
              (level) => level.difficulty === difficulty,
            );
            return options.length === 0 ? null : (
              <optgroup
                label={difficultyLabel(difficulty)}
                key={difficulty}
              >
                {options.map((level) => (
                  <option value={level.id} key={level.id}>
                    {level.sourceLevelId}
                    {level.reference
                      ? ` · ${level.reference.actionCount}행동 / ${level.reference.pushCount}push`
                      : ""}
                  </option>
                ))}
              </optgroup>
            );
          })}
        </select>
      </label>
      <Field
        label="Seed"
        name="seed"
        defaultValue="0"
        type="number"
        required={false}
      />
      <label>
        Max steps
        <input
          name="max_steps"
          type="number"
          min="1"
          value={maxSteps}
          onChange={(event) => setMaxSteps(Number(event.target.value))}
          required
        />
      </label>
      <Field
        label="Prompt"
        name="prompt_name"
        defaultValue="sokoban-strategy"
      />
      <Field
        label="Model"
        name="model_name"
        defaultValue="qwen3.6:27b-mlx"
      />
      <SelectField
        label="Rationale"
        name="rationale_mode"
        options={["on", "off"]}
      />
      <SelectField
        label="Grounding"
        name="grounding_mode"
        options={["local-search", "direct"]}
      />
      <button className="primary" type="submit" disabled={running}>
        실행
      </button>
    </form>
  );
}

interface FieldProps {
  label: string;
  name: string;
  defaultValue: string;
  type?: string;
  min?: string;
  required?: boolean;
}

function Field({ label, required = true, ...inputProps }: FieldProps) {
  return (
    <label>
      {label}
      <input {...inputProps} required={required} />
    </label>
  );
}

interface SelectFieldProps {
  label: string;
  name: string;
  options: string[];
}

function SelectField({ label, name, options }: SelectFieldProps) {
  return (
    <label>
      {label}
      <select name={name} defaultValue={options[0]}>
        {options.map((option) => (
          <option value={option} key={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
