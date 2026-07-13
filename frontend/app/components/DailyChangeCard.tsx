const CHANGES = [
  { name: "SPY", pct: 3.2, color: "#8b5cf6" },
  { name: "EWY", pct: 1.1, color: "#10b981" },
  { name: "TLT", pct: -2.4, color: "#dc2626" },
  { name: "GLD", pct: 0.8, color: "#737373" },
  { name: "SHV", pct: -0.5, color: "#a5b4fc" },
];

// 스케일: |pct| = 4% 를 100%로 정규화
const SCALE = 4;

export default function DailyChangeCard() {
  return (
    <section className="card">
      <h2 className="card-title">전일 대비 변동성</h2>
      <div>
        {CHANGES.map((c) => {
          const positive = c.pct >= 0;
          const barColor = positive ? c.color : "#dc2626";
          const width = Math.min(Math.abs(c.pct) / SCALE, 1) * 100;
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
                <div
                  style={{
                    position: "absolute",
                    left: positive ? 0 : "auto",
                    right: positive ? "auto" : 0,
                    width: `${width}%`,
                    height: "100%",
                    background: barColor,
                    borderRadius: 6,
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
          marginTop: 12,
          textAlign: "right",
          fontSize: 11,
          color: "var(--sub)",
        }}
      >
        최근 업데이트: 오전 09:15 (KST)
      </div>
    </section>
  );
}
