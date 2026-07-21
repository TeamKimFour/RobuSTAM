"use client";

import dynamic from "next/dynamic";
import { MLFLOW_RUNS } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function RunGridHeatmap() {
  const folds = [1, 2, 3];
  const configs = ["seed=42, 200k", "seed=42, 100k", "seed=7, 200k", "seed=7, 100k"];

  const z = folds.map((f) =>
    configs.map((c) => {
      const [seedStr, tsStr] = c.split(", ");
      const seed = Number(seedStr.split("=")[1]);
      const timesteps = tsStr === "200k" ? 200000 : 100000;
      const run = MLFLOW_RUNS.find(
        (r) => r.fold === f && r.seed === seed && r.timesteps === timesteps,
      );
      return run ? run.validSharpe : null;
    }),
  );
  const text = z.map((row) => row.map((v) => (v == null ? "" : v.toFixed(2))));

  return (
    <section className="card">
      <h2 className="card-title">Fold × Seed × Timesteps 그리드 (valid Sharpe)</h2>
      <Plot
        data={[
          {
            z,
            x: configs,
            y: folds.map((f) => `Fold ${f}`),
            type: "heatmap",
            colorscale: [
              [0, "#7f1d1d"],
              [0.5, "#0f0f0f"],
              [1, "#065f46"],
            ],
            zmid: 0,
            text: text as unknown as string[],
            texttemplate: "%{text}",
            textfont: { size: 12, color: "#fafafa" },
            hovertemplate: "%{y} · %{x}: valid Sharpe %{z:.3f}<extra></extra>",
            colorbar: { tickfont: { color: "#a3a3a3", size: 10 }, thickness: 8, len: 0.8 },
          } as Plotly.Data,
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 60, r: 8, t: 8, b: 40 },
          height: 240,
          font: { color: "#a3a3a3", size: 11 },
          yaxis: { autorange: "reversed" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
