"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const ASSETS = [
  { name: "SPY", pct: 40, color: "#8b5cf6" },
  { name: "EWY", pct: 15, color: "#10b981" },
  { name: "TLT", pct: 20, color: "#dc2626" },
  { name: "GLD", pct: 15, color: "#737373" },
  { name: "SHV", pct: 10, color: "#a5b4fc" },
];

export default function AllocationCard() {
  return (
    <section className="card">
      <h2 className="card-title">오늘의 권장 자산 배분</h2>
      <Plot
        data={[
          {
            values: ASSETS.map((a) => a.pct),
            labels: ASSETS.map((a) => a.name),
            type: "pie",
            hole: 0.68,
            marker: {
              colors: ASSETS.map((a) => a.color),
              line: { color: "#0f0f0f", width: 2 },
            },
            textinfo: "none",
            hoverinfo: "label+percent",
            sort: false,
            direction: "clockwise",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          showlegend: false,
          margin: { l: 0, r: 0, t: 0, b: 0 },
          height: 240,
          annotations: [
            {
              text: "기준일",
              x: 0.5,
              y: 0.58,
              showarrow: false,
              font: { color: "#a3a3a3", size: 12 },
            },
            {
              text: "2026-06-28",
              x: 0.5,
              y: 0.42,
              showarrow: false,
              font: { color: "#fafafa", size: 16 },
            },
          ],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: 8,
          marginTop: 12,
        }}
      >
        {ASSETS.map((a) => (
          <div key={a.name} style={{ textAlign: "center" }}>
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: 999,
                background: a.color,
                marginBottom: 6,
              }}
            />
            <div style={{ fontSize: 12, color: "var(--text)", fontWeight: 600 }}>
              {a.name}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)" }}>{a.pct}%</div>
          </div>
        ))}
      </div>
    </section>
  );
}
