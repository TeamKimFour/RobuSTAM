type Props = {
  title: string;
  value: number;
  min: number;
  max: number;
  neutral: number;
  hint: string;
  leftLabel: string;
  rightLabel: string;
};

export default function MarketGaugeCard({
  title,
  value,
  min,
  max,
  neutral,
  hint,
  leftLabel,
  rightLabel,
}: Props) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const neutralPct = (neutral - min) / (max - min);

  return (
    <section className="card" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 className="card-title" style={{ margin: 0 }}>{title}</h2>
        <span style={{ fontSize: 22, fontWeight: 700, color: "var(--text)" }}>
          {value.toFixed(2)}
        </span>
      </div>
      <div
        style={{
          position: "relative",
          height: 12,
          borderRadius: 6,
          margin: "18px 0 8px",
          background:
            "linear-gradient(90deg, #dc2626 0%, #f59e0b 45%, #10b981 100%)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${neutralPct * 100}%`,
            top: -4,
            bottom: -4,
            width: 2,
            background: "var(--text)",
            transform: "translateX(-1px)",
            opacity: 0.7,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `calc(${pct * 100}% - 8px)`,
            top: -6,
            width: 16,
            height: 24,
            borderRadius: 4,
            background: "#fafafa",
            border: "2px solid #0f0f0f",
            boxShadow: "0 2px 6px rgba(0,0,0,0.5)",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--sub)",
        }}
      >
        <span>{leftLabel}</span>
        <span>기준 {neutral}</span>
        <span>{rightLabel}</span>
      </div>
      <p style={{ margin: "12px 0 0 0", fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
        {hint}
      </p>
    </section>
  );
}
