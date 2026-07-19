"use client";

import dynamic from "next/dynamic";
import type { SeriesPoint } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  title: string;
  sharpe: SeriesPoint[];
  vol: SeriesPoint[];
};

export default function RollingMetricsChart({ title, sharpe, vol }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      <Plot
        data={[
          {
            x: sharpe.map((p) => p.date),
            y: sharpe.map((p) => p.value),
            name: "Rolling Sharpe (252d)",
            type: "scatter",
            mode: "lines",
            line: { color: "#8b5cf6", width: 2 },
            yaxis: "y",
          },
          {
            x: vol.map((p) => p.date),
            y: vol.map((p) => p.value * 100),
            name: "Rolling Vol (연, %)",
            type: "scatter",
            mode: "lines",
            line: { color: "#f59e0b", width: 2, dash: "dot" },
            yaxis: "y2",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 48, t: 8, b: 40 },
          height: 260,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", zerolinecolor: "#262626" },
          yaxis: {
            title: { text: "Sharpe", font: { color: "#8b5cf6", size: 11 } },
            gridcolor: "#262626",
            zerolinecolor: "#525252",
          },
          yaxis2: {
            title: { text: "Vol %", font: { color: "#f59e0b", size: 11 } },
            overlaying: "y",
            side: "right",
            gridcolor: "transparent",
            zerolinecolor: "transparent",
            ticksuffix: "%",
          },
          legend: { orientation: "h", y: -0.22, font: { color: "#e5e5e5", size: 11 } },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
