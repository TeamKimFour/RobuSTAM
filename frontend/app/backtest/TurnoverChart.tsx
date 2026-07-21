"use client";

import dynamic from "next/dynamic";
import type { SeriesPoint } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { turnover: SeriesPoint[]; cumCost: SeriesPoint[] };

export default function TurnoverChart({ turnover, cumCost }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">거래 강도(Turnover)와 누적 거래비용</h2>
      <Plot
        data={[
          {
            x: turnover.map((p) => p.date),
            y: turnover.map((p) => p.value * 100),
            name: "일일 Turnover (%)",
            type: "bar",
            marker: { color: "#8b5cf6" },
            yaxis: "y",
            hovertemplate: "%{y:.2f}%<extra>Turnover</extra>",
          },
          {
            x: cumCost.map((p) => p.date),
            y: cumCost.map((p) => p.value * 100),
            name: "누적 거래비용 (초기 NAV 대비 %)",
            type: "scatter",
            mode: "lines",
            line: { color: "#dc2626", width: 2 },
            yaxis: "y2",
            hovertemplate: "%{y:.2f}%<extra>누적 비용</extra>",
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
            title: { text: "Turnover %", font: { color: "#8b5cf6", size: 11 } },
            gridcolor: "#262626",
            ticksuffix: "%",
          },
          yaxis2: {
            title: { text: "누적 비용 %", font: { color: "#dc2626", size: 11 } },
            overlaying: "y",
            side: "right",
            gridcolor: "transparent",
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
