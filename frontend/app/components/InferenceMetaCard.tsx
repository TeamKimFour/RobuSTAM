type Props = {
  generatedAt: string;
  modelVersion: string;
  latencyMs: number | null;
};

export default function InferenceMetaCard({ generatedAt, modelVersion, latencyMs }: Props) {
  const generatedLabel = formatDateTime(generatedAt);

  return (
    <section className="card">
      <h2 className="card-title">추론 배치 상태</h2>
      <div style={{ display: "grid", gap: 14 }}>
        <Row label="모델 버전" value={modelVersion} mono />
        <Row label="배치 생성 시각" value={generatedLabel} />
        <Row
          label="API 응답 시간"
          value={latencyMs != null ? `${latencyMs} ms` : "-"}
          tone={latencyMs != null && latencyMs < 200 ? "positive" : "muted"}
          hint={
            latencyMs != null
              ? latencyMs < 200
                ? "목표(<200ms) 충족"
                : "목표(<200ms) 초과"
              : undefined
          }
        />
      </div>
    </section>
  );
}

function Row({
  label,
  value,
  tone,
  mono,
  hint,
}: {
  label: string;
  value: string;
  tone?: "positive" | "muted";
  mono?: boolean;
  hint?: string;
}) {
  const color =
    tone === "positive" ? "var(--positive)" : tone === "muted" ? "var(--muted)" : "var(--text)";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
      <span style={{ color: "var(--muted)", fontSize: 13 }}>{label}</span>
      <div style={{ textAlign: "right" }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            color,
            fontFamily: mono ? "ui-monospace, SFMono-Regular, Menlo, monospace" : undefined,
            wordBreak: "break-all",
          }}
        >
          {value}
        </div>
        {hint ? (
          <div style={{ fontSize: 11, color: "var(--sub)", marginTop: 2 }}>{hint}</div>
        ) : null}
      </div>
    </div>
  );
}

function formatDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}
