import type { CSSProperties } from "react";
import type { FoldRow } from "./mock";

const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "10px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};

const CELL: CSSProperties = { padding: "12px 8px", fontSize: 13 };

type Props = { rows: FoldRow[] };

export default function FoldComparisonCard({ rows }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">Walk-Forward Fold별 성과</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        학습 구간을 확장해가며 각 test 블록에서 잰 out-of-sample 성과.
      </p>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD}>Fold</th>
            <th style={HEAD}>Test 기간</th>
            <th style={{ ...HEAD, textAlign: "right" }}>Sharpe</th>
            <th style={{ ...HEAD, textAlign: "right" }}>CAGR</th>
            <th style={{ ...HEAD, textAlign: "right" }}>MDD</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r.fold}
              style={{
                borderBottom: i < rows.length - 1 ? "1px solid var(--border)" : "none",
              }}
            >
              <td style={{ ...CELL, fontWeight: 600 }}>{r.fold}</td>
              <td style={{ ...CELL, color: "var(--muted)" }}>{r.period}</td>
              <td
                style={{
                  ...CELL,
                  textAlign: "right",
                  color: r.sharpe >= 1 ? "var(--positive)" : "var(--text)",
                }}
              >
                {r.sharpe.toFixed(2)}
              </td>
              <td style={{ ...CELL, textAlign: "right", color: r.cagr >= 0 ? "var(--positive)" : "var(--negative)" }}>
                {pct(r.cagr)}
              </td>
              <td style={{ ...CELL, textAlign: "right", color: "var(--negative)" }}>{pct(r.mdd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function pct(v: number): string {
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(2)}%`;
}
