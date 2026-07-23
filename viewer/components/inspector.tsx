import type { ViewEvent } from "../lib/types";

interface InspectorProps {
  frame: ViewEvent | null;
}

export function Inspector({ frame }: InspectorProps) {
  const strategy = frame?.strategy;
  const tokenTotal = sumNullable(
    frame?.metrics.promptTokens ?? null,
    frame?.metrics.outputTokens ?? null,
  );

  return (
    <aside className="inspector">
      <section className="panel state-card">
        <p className="section-label">EXECUTION</p>
        <dl className="facts">
          <Fact label="Node" value={frame?.node} />
          <Fact label="Status" value={frame ? statusLabel(frame) : "대기"} />
          <Fact label="Action" value={frame?.action} />
          <Fact label="Step" value={`${frame?.step ?? 0} / ${frame?.maxSteps ?? "—"}`} />
          <Fact label="Push" value={frame?.pushCount} />
          <Fact label="Event" value={shorten(frame?.eventId)} />
        </dl>
      </section>

      <section className="panel detail-card">
        <p className="section-label">CURRENT STRATEGY</p>
        <Detail label="가설" value={strategy?.hypothesis} />
        <Detail label="배정" value={strategy?.assignment} />
        <Detail label="하위 목표" value={strategy?.subgoal} />
        <Detail label="보호 제약" value={strategy?.protectedCells.filter(Boolean).join(", ")} />
        <Detail label="위험" value={strategy?.risk} />
      </section>

      <section className="panel detail-card">
        <p className="section-label">REFLECTION</p>
        <Detail label="예상 효과" value={frame?.effect.expected} />
        <Detail label="관찰 결과" value={frame?.effect.observed} />
        <Detail label="최근 수정" value={frame?.revision} />
      </section>

      <section className="panel metrics-card">
        <p className="section-label">METRICS</p>
        <div className="metrics">
          <Metric label="LLM calls" value={frame?.metrics.llmCalls} />
          <Metric label="tokens" value={tokenTotal} />
          <Metric
            label="search"
            value={frame?.metrics.expandedStates ?? frame?.metrics.localSearchCalls}
          />
        </div>
      </section>
    </aside>
  );
}

interface ValueProps {
  label: string;
  value: unknown;
}

function Fact({ label, value }: ValueProps) {
  return <div><dt>{label}</dt><dd>{display(value)}</dd></div>;
}

function Detail({ label, value }: ValueProps) {
  return <div className="detail"><span>{label}</span><p>{display(value)}</p></div>;
}

function Metric({ label, value }: ValueProps) {
  return <div><strong>{display(value)}</strong><span>{label}</span></div>;
}

function display(value: unknown): string {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function statusLabel(frame: ViewEvent): string {
  if (frame.success) return "성공";
  if (frame.deadlock) return "데드락";
  if (frame.truncated) return "행동 제한";
  return frame.status;
}

function shorten(value: string | undefined): string {
  const text = value ?? "—";
  return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text;
}

function sumNullable(left: number | null, right: number | null): number | null {
  if (left === null && right === null) return null;
  return (left ?? 0) + (right ?? 0);
}
