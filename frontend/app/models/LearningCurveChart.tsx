"use client";

import dynamic from "next/dynamic";
import { LEARNING_CURVE } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function LearningCurveChart() {
  return (
    <section className="card">
      <h2 className="card-title">배포 모델 학습 곡선</h2>
      <Plot
        data={[
          {
            x: LEARNING_CURVE.map((d) => d.step),
            y: LEARNING_CURVE.map((d) => d.reward),
            name: "episode reward",
            type: "scatter",
            mode: "lines",
            line: { color: "#8b5cf6", width: 2 },
            yaxis: "y",
          },
          {
            x: LEARNING_CURVE.map((d) => d.step),
            y: LEARNING_CURVE.map((d) => d.entropy),
            name: "policy entropy",
            type: "scatter",
            mode: "lines",
            line: { color: "#10b981", width: 2, dash: "dot" },
            yaxis: "y2",
          },
          {
            x: LEARNING_CURVE.map((d) => d.step),
            y: LEARNING_CURVE.map((d) => d.klDiv),
            name: "KL divergence",
            type: "scatter",
            mode: "lines",
            line: { color: "#f59e0b", width: 1.5, dash: "dash" },
            yaxis: "y3",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 48, t: 8, b: 40 },
          height: 280,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", title: { text: "timesteps", font: { size: 11 } } },
          yaxis: {
            title: { text: "reward", font: { color: "#8b5cf6", size: 11 } },
            gridcolor: "#262626",
          },
          yaxis2: {
            title: { text: "entropy", font: { color: "#10b981", size: 11 } },
            overlaying: "y",
            side: "right",
            gridcolor: "transparent",
          },
          yaxis3: {
            overlaying: "y",
            side: "right",
            position: 0.95,
            gridcolor: "transparent",
            showticklabels: false,
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
