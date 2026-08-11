import type { CSSProperties } from "react";
import type { ComboSnapshot } from "./mock";

const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "10px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};

const CELL: CSSProperties = { padding: "12px 8px", fontSize: 13, textAlign: "right" };

type Metric = "sharpe" | "cagr" | "mdd" | "avgTurnover";

const METRIC_LABEL: Record<Metric, string> = {
  sharpe: "Sharpe",
  cagr: "CAGR",
  mdd: "MDD",
  avgTurnover: "회전율",
};

type Props = {
  combos: ComboSnapshot[];
  metric?: Metric;
};

export default function ComboFoldMatrix({ combos, metric = "sharpe" }: Props) {
  const folds = combos[0]?.foldTable.map((r) => r.fold) ?? [];

  return (
    <section className="card">
      <h2 className="card-title">Fold × 콤보 {METRIC_LABEL[metric]}</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        walk-forward fold 3구간에서 각 콤보의 out-of-sample {METRIC_LABEL[metric]}.
      </p>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD}>콤보</th>
            {folds.map((f) => (
              <th key={f} style={{ ...HEAD, textAlign: "right" }}>{f}</th>
            ))}
            <th style={{ ...HEAD, textAlign: "right" }}>전 구간</th>
          </tr>
        </thead>
        <tbody>
          {combos.map((c, i) => {
            const overall = overallMetric(c, metric);
            return (
              <tr
                key={c.combo}
                style={{
                  borderBottom: i < combos.length - 1 ? "1px solid var(--border)" : "none",
                }}
              >
                <td style={{ padding: "12px 8px", fontSize: 13 }}>
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
                {c.foldTable.map((r) => (
                  <td key={r.fold} style={{ ...CELL, ...cellStyle(r, metric) }}>
                    {formatMetric(getValue(r, metric), metric)}
                  </td>
                ))}
                <td style={{ ...CELL, fontWeight: 600, color: "var(--text)" }}>
                  {formatMetric(overall, metric)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function getValue(row: ComboSnapshot["foldTable"][number], metric: Metric): number {
  return row[metric];
}

function overallMetric(combo: ComboSnapshot, metric: Metric): number {
  switch (metric) {
    case "sharpe":
      return combo.metrics.sharpe;
    case "cagr":
      return combo.metrics.cagr;
    case "mdd":
      return combo.metrics.mdd;
    case "avgTurnover":
      return combo.metrics.avg_turnover;
  }
}

function cellStyle(row: ComboSnapshot["foldTable"][number], metric: Metric): CSSProperties {
  const v = getValue(row, metric);
  if (metric === "sharpe") return { color: v >= 1 ? "var(--positive)" : "var(--text)" };
  if (metric === "cagr") return { color: v >= 0 ? "var(--positive)" : "var(--negative)" };
  if (metric === "mdd") return { color: "var(--negative)" };
  return {};
}

function formatMetric(v: number, metric: Metric): string {
  if (metric === "sharpe" || metric === "avgTurnover") return v.toFixed(2);
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(2)}%`;
}
