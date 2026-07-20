"use client";

import dynamic from "next/dynamic";
import { ASSET_ORDER, type AssetSymbol } from "../lib/assets";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  data: Record<AssetSymbol, number[]>;
  dates: string[];
};

export default function ReturnsHeatmap({ data, dates }: Props) {
  const symbols = [...ASSET_ORDER];
  const z = symbols.map((s) => data[s].map((v) => v * 100));

  return (
    <section className="card">
      <h2 className="card-title">최근 30 거래일 자산별 일간 수익률 히트맵</h2>
      <Plot
        data={[
          {
            z,
            x: dates,
            y: symbols,
            type: "heatmap",
            colorscale: [
              [0, "#7f1d1d"],
              [0.5, "#0f0f0f"],
              [1, "#065f46"],
            ],
            zmid: 0,
            hovertemplate: "%{y} · %{x}: %{z:.2f}%<extra></extra>",
            showscale: true,
            colorbar: {
              tickfont: { color: "#a3a3a3", size: 10 },
              thickness: 8,
              len: 0.9,
              ticksuffix: "%",
            },
          } as Plotly.Data,
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 8, t: 8, b: 40 },
          height: 240,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { tickangle: -30, tickfont: { size: 9 } },
          yaxis: { autorange: "reversed" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
