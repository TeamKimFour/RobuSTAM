const FEATURES = [
  { label: "20일 변동성", value: 0.12, color: "#8b5cf6" },
  { label: "모멘텀 (Trend)", value: 0.85, color: "#10b981" },
  { label: "장단기 금리차", value: 0.45, color: "#525252" },
  { label: "시장 심리 지수", value: 0.72, color: "#dc2626" },
];

export default function FeaturesPanel() {
  return (
    <section className="card">
      <h2 className="card-title">권장 배분 근거 (입력 특징 벡터)</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 28,
          marginBottom: 24,
        }}
      >
        {FEATURES.map((f) => (
          <div key={f.label}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 8,
                fontSize: 13,
              }}
            >
              <span style={{ color: "var(--muted)" }}>{f.label}</span>
              <span style={{ fontWeight: 600, color: f.color }}>
                {f.value.toFixed(2)}
              </span>
            </div>
            <div
              style={{
                height: 4,
                background: "var(--surface-elev)",
                borderRadius: 999,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${f.value * 100}%`,
                  height: "100%",
                  background: f.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          padding: 16,
          background: "var(--surface-elev)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          fontSize: 13,
          lineHeight: 1.75,
          color: "var(--muted)",
        }}
      >
        <span
          style={{
            color: "var(--accent)",
            fontWeight: 600,
            marginRight: 6,
          }}
        >
          에이전트 판단:
        </span>
        시장의 모멘텀(0.85)이 강하게 지속되고 있으나, 20일 변동성 지표가
        안정적인 수준으로 유지됨에 따라 위험 자산인 SPY와 EWY의 비중을
        점진적으로 확대할 것을 권장합니다. 장단기 금리차의 축소세는 향후 안전
        자산(TLT)의 비중 축소 근거로 작용하고 있습니다.
      </div>
    </section>
  );
}
