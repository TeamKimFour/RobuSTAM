"use client";

import dynamic from "next/dynamic";
import type { MonthlyReturn } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { data: MonthlyReturn[] };

const MONTHS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"];

export default function MonthlyReturnsHeatmap({ data }: Props) {
  const years = Array.from(new Set(data.map((d) => d.year))).sort();
  const z: (number | null)[][] = years.map((y) =>
    MONTHS.map((_, mi) => {
      const found = data.find((d) => d.year === y && d.month === mi + 1);
      return found ? found.ret * 100 : null;
    }),
  );
  const text: string[][] = z.map((row) =>
    row.map((v) => (v == null ? "" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`)),
  );

  return (
    <section className="card">
      <h2 className="card-title">월별 수익률 히트맵</h2>
      <Plot
        data={[
          {
            z,
            x: MONTHS,
            y: years.map(String),
            type: "heatmap",
            colorscale: [
              [0, "#7f1d1d"],
              [0.5, "#0f0f0f"],
              [1, "#065f46"],
            ],
            zmid: 0,
            text: text as unknown as string[],
            texttemplate: "%{text}",
            textfont: { size: 10, color: "#fafafa" },
            hovertemplate: "%{y}년 %{x}월: %{z:.2f}%<extra></extra>",
            showscale: false,
          } as Plotly.Data,
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 40, r: 8, t: 8, b: 32 },
          height: 260,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { side: "top", tickangle: 0 },
          yaxis: { autorange: "reversed" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
