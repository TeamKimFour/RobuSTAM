"use client";

import dynamic from "next/dynamic";
import { ASSET_ORDER } from "../lib/assets";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { matrix: number[][] };

export default function CorrelationHeatmap({ matrix }: Props) {
  const text = matrix.map((row) => row.map((v) => v.toFixed(2)));
  const symbols = [...ASSET_ORDER];

  return (
    <section className="card">
      <h2 className="card-title">자산 간 상관계수</h2>
      <Plot
        data={[
          {
            z: matrix,
            x: symbols,
            y: symbols,
            type: "heatmap",
            colorscale: [
              [0, "#7f1d1d"],
              [0.5, "#0f0f0f"],
              [1, "#065f46"],
            ],
            zmin: -1,
            zmax: 1,
            text: text as unknown as string[],
            texttemplate: "%{text}",
            textfont: { size: 12, color: "#fafafa" },
            hovertemplate: "%{y} vs %{x}: %{z:.3f}<extra></extra>",
            showscale: true,
            colorbar: { tickfont: { color: "#a3a3a3", size: 10 }, thickness: 8, len: 0.8 },
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 8, t: 8, b: 32 },
          height: 280,
          font: { color: "#a3a3a3", size: 11 },
          yaxis: { autorange: "reversed" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
