import type { CSSProperties } from "react";
import type { ComboSnapshot } from "./mock";

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

type Props = { combos: ComboSnapshot[] };

export default function ComboComparisonTable({ combos }: Props) {
  // Sharpe 최고 콤보를 강조 (도현이 배포용 policy로 고를 후보).
  const bestSharpe = combos.reduce(
    (best, c) => (c.metrics.sharpe > best ? c.metrics.sharpe : best),
    -Infinity,
  );
  return (
    <section className="card">
      <h2 className="card-title">콤보별 성과 비교</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        각 콤보(피처 부분집합)를 동일 조건으로 학습·백테스트한 결과. 실데이터는
        <code style={{ margin: "0 4px" }}>python -m src.backtest.export --combo full=… --combo M0=…</code>
        로 생성된다.
      </p>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD}>콤보</th>
            <th style={{ ...HEAD, textAlign: "right" }}>Sharpe</th>
            <th style={{ ...HEAD, textAlign: "right" }}>CAGR</th>
            <th style={{ ...HEAD, textAlign: "right" }}>MDD</th>
            <th style={{ ...HEAD, textAlign: "right" }}>Vol</th>
            <th style={{ ...HEAD, textAlign: "right" }}>회전율 (평균)</th>
            <th style={{ ...HEAD, textAlign: "right" }}>거래비용</th>
          </tr>
        </thead>
        <tbody>
          {combos.map((c, i) => {
            const isBest = c.metrics.sharpe === bestSharpe;
            return (
              <tr
                key={c.combo}
                style={{
                  borderBottom: i < combos.length - 1 ? "1px solid var(--border)" : "none",
                  background: isBest ? "rgba(139,92,246,0.08)" : undefined,
                }}
              >
                <td style={{ ...CELL, fontWeight: isBest ? 600 : 400 }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: c.color,
                      marginRight: 8,
                      verticalAlign: "middle",
                    }}
                  />
                  {c.displayName}
                </td>
                <td
                  style={{
                    ...CELL,
                    textAlign: "right",
                    color: c.metrics.sharpe >= 1 ? "var(--positive)" : "var(--text)",
                    fontWeight: isBest ? 600 : 400,
                  }}
                >
                  {c.metrics.sharpe.toFixed(2)}
                </td>
                <td
                  style={{
                    ...CELL,
                    textAlign: "right",
                    color: c.metrics.cagr >= 0 ? "var(--positive)" : "var(--negative)",
                  }}
                >
                  {pct(c.metrics.cagr)}
                </td>
                <td style={{ ...CELL, textAlign: "right", color: "var(--negative)" }}>
                  {pct(c.metrics.mdd)}
                </td>
                <td style={{ ...CELL, textAlign: "right" }}>{pct(c.metrics.vol)}</td>
                <td style={{ ...CELL, textAlign: "right" }}>{c.metrics.avg_turnover.toFixed(2)}</td>
                <td style={{ ...CELL, textAlign: "right", color: "var(--sub)" }}>
                  {formatCost(c.metrics.total_cost)}
                </td>
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

function formatCost(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}
