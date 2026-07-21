const CHANGES = [
  { name: "SPY", pct: 3.2, color: "#8b5cf6" },
  { name: "EWY", pct: 1.1, color: "#10b981" },
  { name: "TLT", pct: -2.4, color: "#dc2626" },
  { name: "GLD", pct: 0.8, color: "#737373" },
  { name: "SHV", pct: -0.5, color: "#a5b4fc" },
];

// 스케일: |pct| = 4% 를 반쪽(50%) 폭으로 정규화 → 최대 폭도 100%(양쪽 합) 넘지 않음
const SCALE = 4;

export default function DailyChangeCard() {
  return (
    <section
      className="card"
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h2 className="card-title">전일 대비 변동성</h2>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-around" }}>
        {CHANGES.map((c) => {
          const positive = c.pct >= 0;
          const barColor = positive ? c.color : "#dc2626";
          const halfWidth = Math.min(Math.abs(c.pct) / SCALE, 1) * 50;
          return (
            <div
              key={c.name}
              style={{
                display: "grid",
                gridTemplateColumns: "48px 1fr 64px",
                alignItems: "center",
                gap: 16,
                padding: "9px 0",
              }}
            >
              <span style={{ fontSize: 13, color: "var(--muted)" }}>{c.name}</span>
              <div
                style={{
                  position: "relative",
                  height: 20,
                  background: "var(--surface-elev)",
                  borderRadius: 6,
                  overflow: "hidden",
                }}
              >
                {/* 중앙 0선 */}
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    bottom: 0,
                    width: 1,
                    background: "var(--border)",
                    transform: "translateX(-0.5px)",
                    zIndex: 1,
                  }}
                />
                {/* 값 막대: 중앙에서 좌/우로 뻗음 */}
                <div
                  style={{
                    position: "absolute",
                    left: positive ? "50%" : `${50 - halfWidth}%`,
                    width: `${halfWidth}%`,
                    top: 0,
                    bottom: 0,
                    background: barColor,
                    borderRadius: positive
                      ? "0 6px 6px 0"
                      : "6px 0 0 6px",
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: 13,
                  textAlign: "right",
                  color: positive ? "var(--positive)" : "var(--negative)",
                  fontWeight: 600,
                }}
              >
                {positive ? "+" : ""}
                {c.pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 12,
          fontSize: 11,
          color: "var(--sub)",
        }}
      >
        <span>-{SCALE}% ← 0 → +{SCALE}%</span>
        <span>최근 업데이트: 오전 09:15 (KST)</span>
      </div>
    </section>
  );
}
