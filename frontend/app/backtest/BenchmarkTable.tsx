import type { CSSProperties } from "react";
import type { BenchmarkRow } from "./mock";

const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "12px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};

const CELL: CSSProperties = {
  padding: "14px 8px",
  fontSize: 13,
};

type Props = { rows: BenchmarkRow[] };

export default function BenchmarkTable({ rows }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">전략별 지표 비교</h2>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD}>전략</th>
            <th style={{ ...HEAD, textAlign: "right" }}>CAGR</th>
            <th style={{ ...HEAD, textAlign: "right" }}>Sharpe</th>
            <th style={{ ...HEAD, textAlign: "right" }}>MDD</th>
            <th style={{ ...HEAD, textAlign: "right" }}>Volatility</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const highlight = i === 0;
            return (
              <tr
                key={r.name}
                style={{
                  borderBottom: i < rows.length - 1 ? "1px solid var(--border)" : "none",
                  background: highlight ? "rgba(139,92,246,0.08)" : undefined,
                }}
              >
                <td style={{ ...CELL, fontWeight: highlight ? 600 : 400 }}>{r.name}</td>
                <td style={{ ...CELL, textAlign: "right", color: r.cagr >= 0 ? "var(--positive)" : "var(--negative)" }}>
                  {pct(r.cagr)}
                </td>
                <td style={{ ...CELL, textAlign: "right" }}>{r.sharpe.toFixed(2)}</td>
                <td style={{ ...CELL, textAlign: "right", color: "var(--negative)" }}>{pct(r.mdd)}</td>
                <td style={{ ...CELL, textAlign: "right" }}>{pct(r.vol)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function pct(v: number): string {
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(2)}%`;
}
