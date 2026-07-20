"use client";

import dynamic from "next/dynamic";
import type { SeriesPoint } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  title: string;
  alpha: SeriesPoint[];
};

export default function AlphaChart({ title, alpha }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      <Plot
        data={[
          {
            x: alpha.map((p) => p.date),
            y: alpha.map((p) => p.value * 100),
            fill: "tozeroy",
            type: "scatter",
            mode: "lines",
            line: { color: "#10b981", width: 1.5 },
            fillcolor: "rgba(16,185,129,0.15)",
            hovertemplate: "%{y:.2f}%<extra>초과수익</extra>",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 16, t: 8, b: 40 },
          height: 220,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", zerolinecolor: "#262626" },
          yaxis: {
            gridcolor: "#262626",
            zerolinecolor: "#525252",
            ticksuffix: "%",
          },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
