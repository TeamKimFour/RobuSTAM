"use client";

import dynamic from "next/dynamic";
import type { Series } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { series: Series[] };

export default function ContributionChart({ series }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">자산별 누적 P&amp;L 기여도</h2>
      <Plot
        data={series.map((s) => ({
          x: s.points.map((p) => p.date),
          y: s.points.map((p) => p.value * 100),
          name: s.name,
          type: "scatter",
          mode: "lines",
          stackgroup: "one",
          line: { color: s.color, width: 0.5 },
          hovertemplate: "%{y:.2f}%<extra>%{fullData.name}</extra>",
        }))}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 16, t: 8, b: 40 },
          height: 300,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", zerolinecolor: "#262626" },
          yaxis: {
            gridcolor: "#262626",
            zerolinecolor: "#525252",
            ticksuffix: "%",
          },
          legend: { orientation: "h", y: -0.18, font: { color: "#e5e5e5", size: 11 } },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
