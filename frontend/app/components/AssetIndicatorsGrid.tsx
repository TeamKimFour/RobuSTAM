import type { AssetIndicators } from "../backtest/mock";
import { ASSET_COLOR } from "../lib/assets";

type Props = { rows: AssetIndicators[] };

export default function AssetIndicatorsGrid({ rows }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">자산별 기술 지표 스냅샷</h2>
      <div style={{ display: "grid", gap: 10 }}>
        <Header />
        {rows.map((r) => (
          <Row key={r.symbol} r={r} />
        ))}
      </div>
    </section>
  );
}

function Header() {
  const style = {
    fontSize: 11,
    color: "var(--sub)",
    letterSpacing: "0.05em",
    textTransform: "uppercase" as const,
  };
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "60px 1fr 90px 90px 90px 90px 90px",
        gap: 10,
        padding: "0 8px 6px 8px",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span style={style}>자산</span>
      <span style={style}>RSI 14</span>
      <span style={{ ...style, textAlign: "right" }}>MACD</span>
      <span style={{ ...style, textAlign: "right" }}>ROC10</span>
      <span style={{ ...style, textAlign: "right" }}>Vol20</span>
      <span style={{ ...style, textAlign: "right" }}>BB Width</span>
      <span style={{ ...style, textAlign: "right" }}>MA Cross</span>
    </div>
  );
}

function Row({ r }: { r: AssetIndicators }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "60px 1fr 90px 90px 90px 90px 90px",
        gap: 10,
        alignItems: "center",
        padding: "10px 8px",
        borderRadius: 6,
        background: "var(--surface-elev)",
      }}
    >
      <span style={{ fontWeight: 700, fontSize: 13, color: ASSET_COLOR[r.symbol] }}>{r.symbol}</span>
      <RsiBar value={r.rsi} />
      <span style={{ textAlign: "right", fontSize: 13, color: r.macdHist >= 0 ? "var(--positive)" : "var(--negative)" }}>
        {r.macdHist.toFixed(2)}
      </span>
      <span style={{ textAlign: "right", fontSize: 13, color: r.roc10 >= 0 ? "var(--positive)" : "var(--negative)" }}>
        {r.roc10 >= 0 ? "+" : ""}
        {r.roc10.toFixed(2)}%
      </span>
      <span style={{ textAlign: "right", fontSize: 13 }}>{(r.vol20 * 100).toFixed(2)}%</span>
      <span style={{ textAlign: "right", fontSize: 13 }}>{(r.bbWidth * 100).toFixed(2)}%</span>
      <CrossBadge kind={r.maCross} />
    </div>
  );
}

function RsiBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const color = value >= 70 ? "#dc2626" : value <= 30 ? "#10b981" : "#8b5cf6";
  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "relative",
          height: 10,
          background: "var(--surface)",
          borderRadius: 999,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${pct}%`,
            background: color,
          }}
        />
        <div style={{ position: "absolute", left: "30%", top: 0, bottom: 0, width: 1, background: "var(--sub)" }} />
        <div style={{ position: "absolute", left: "70%", top: 0, bottom: 0, width: 1, background: "var(--sub)" }} />
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{value.toFixed(1)}</div>
    </div>
  );
}

function CrossBadge({ kind }: { kind: "golden" | "dead" | "neutral" }) {
  const map = {
    golden: { text: "Golden", color: "var(--positive)", bg: "rgba(16,185,129,0.15)" },
    dead: { text: "Dead", color: "var(--negative)", bg: "rgba(220,38,38,0.15)" },
    neutral: { text: "Neutral", color: "var(--muted)", bg: "var(--surface)" },
  } as const;
  const it = map[kind];
  return (
    <span
      style={{
        textAlign: "center",
        fontSize: 11,
        padding: "3px 8px",
        borderRadius: 999,
        background: it.bg,
        color: it.color,
        fontWeight: 600,
      }}
    >
      {it.text}
    </span>
  );
}
