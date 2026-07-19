"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { durations: number[] };

export default function UnderwaterCard({ durations }: Props) {
  const max = Math.max(...durations, 1);
  const longest = Math.max(...durations, 0);
  const avg = durations.length
    ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length)
    : 0;

  return (
    <section className="card">
      <h2 className="card-title">Underwater 지속기간 (거래일)</h2>
      <div style={{ display: "flex", gap: 20, marginBottom: 12 }}>
        <Stat label="최장 수중" value={`${longest}일`} tone="negative" />
        <Stat label="평균 수중" value={`${avg}일`} tone="muted" />
        <Stat label="에피소드" value={`${durations.length}회`} tone="muted" />
      </div>
      <Plot
        data={[
          {
            x: durations.map((_, i) => `#${i + 1}`),
            y: durations,
            type: "bar",
            marker: {
              color: durations.map((d) => `rgba(220,38,38,${0.35 + 0.5 * (d / max)})`),
            },
            hovertemplate: "%{y}일<extra></extra>",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 40, r: 8, t: 8, b: 32 },
          height: 200,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { visible: false },
          yaxis: { gridcolor: "#262626", ticksuffix: "일" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: "negative" | "muted" }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ color: "var(--muted)", fontSize: 12 }}>{label}</div>
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: tone === "negative" ? "var(--negative)" : "var(--text)",
        }}
      >
        {value}
      </div>
    </div>
  );
}
