import type { BacktestMetrics } from "./mock";

type Props = { metrics: BacktestMetrics };

export default function MetricsRow({ metrics }: Props) {
  const items = [
    { label: "CAGR", value: pct(metrics.cagr), tone: metrics.cagr >= 0 ? "positive" : "negative" },
    { label: "Sharpe", value: metrics.sharpe.toFixed(2), tone: metrics.sharpe >= 1 ? "positive" : "muted" },
    { label: "Max Drawdown", value: pct(metrics.mdd), tone: "negative" },
    { label: "Volatility (연)", value: pct(metrics.vol), tone: "muted" },
  ] as const;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 20,
      }}
    >
      {items.map((it) => (
        <div key={it.label} className="card" style={{ textAlign: "center" }}>
          <div style={{ color: "var(--muted)", fontSize: 13 }}>{it.label}</div>
          <div
            style={{
              margin: "10px 0 0 0",
              fontSize: 30,
              fontWeight: 700,
              color:
                it.tone === "positive"
                  ? "var(--positive)"
                  : it.tone === "negative"
                  ? "var(--negative)"
                  : "var(--text)",
            }}
          >
            {it.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function pct(v: number): string {
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(2)}%`;
}
