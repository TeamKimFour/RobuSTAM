"use client";

import dynamic from "next/dynamic";
import type { ComboSnapshot } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { combos: ComboSnapshot[] };

export default function ComboEquityOverlay({ combos }: Props) {
  const traces = combos.map((c) => ({
    x: c.points.map((p) => p.date),
    y: c.points.map((p) => p.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: c.displayName,
    line: { color: c.color, width: 2 },
    hovertemplate: "%{y:.3f}<extra>%{fullData.name}</extra>",
  }));

  return (
    <section className="card">
      <h2 className="card-title">콤보별 누적 NAV (초기 1.0)</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        같은 test 구간에서 콤보별 policy NAV를 겹쳐 보여준다. 벤치마크는 위 &ldquo;성과 vs 벤치마크&rdquo;
        섹션에서 별도 확인.
      </p>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 16, t: 8, b: 40 },
          height: 380,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", zerolinecolor: "#262626", tickformat: "%Y-%m" },
          yaxis: { gridcolor: "#262626", zerolinecolor: "#525252" },
          legend: { orientation: "h", y: -0.18, font: { color: "#e5e5e5", size: 11 } },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
