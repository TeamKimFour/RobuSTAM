"use client";

import dynamic from "next/dynamic";
import type { Series } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  series: Series[];
  title: string;
  yFormat?: "nav" | "percent";
  height?: number;
};

export default function PerformanceChart({ series, title, yFormat = "nav", height = 380 }: Props) {
  const traces = series.map((s) => ({
    x: s.points.map((p) => p.date),
    y: s.points.map((p) => (yFormat === "percent" ? p.value * 100 : p.value)),
    type: "scatter" as const,
    mode: "lines" as const,
    name: s.name,
    line: { color: s.color, width: 2 },
    hovertemplate: yFormat === "percent" ? "%{y:.2f}%<extra>%{fullData.name}</extra>" : "%{y:.3f}<extra>%{fullData.name}</extra>",
  }));

  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 16, t: 8, b: 40 },
          height,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: {
            gridcolor: "#262626",
            zerolinecolor: "#262626",
            tickformat: "%Y-%m",
          },
          yaxis: {
            gridcolor: "#262626",
            zerolinecolor: "#525252",
            ticksuffix: yFormat === "percent" ? "%" : "",
          },
          legend: {
            orientation: "h",
            y: -0.18,
            font: { color: "#e5e5e5", size: 11 },
          },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
