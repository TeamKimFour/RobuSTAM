"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LATENCY_MS = 142;
const LATENCY_MAX = 200;

const CALLS_24H = 1402;
const CALLS_HIST = [30, 45, 40, 55, 60, 50, 70, 65, 75, 68, 72, 80];

export default function KpiRow() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr 1fr",
        gap: 20,
      }}
    >
      {/* 평균 응답 속도 */}
      <div className="card">
        <div
          style={{ color: "var(--muted)", fontSize: 13, textAlign: "center" }}
        >
          평균 응답 속도
        </div>
        <div style={{ textAlign: "center", margin: "14px 0" }}>
          <span
            style={{
              fontSize: 40,
              fontWeight: 700,
              color: "var(--accent)",
            }}
          >
            {LATENCY_MS}
          </span>
          <span
            style={{ fontSize: 14, color: "var(--muted)", marginLeft: 4 }}
          >
            ms
          </span>
        </div>
        <div
          style={{
            height: 6,
            background: "var(--surface-elev)",
            borderRadius: 999,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${(LATENCY_MS / LATENCY_MAX) * 100}%`,
              height: "100%",
              background:
                "linear-gradient(90deg, #8b5cf6 0%, #10b981 100%)",
            }}
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 8,
            fontSize: 11,
            color: "var(--sub)",
          }}
        >
          <span>0ms</span>
          <span>정상 (&lt; 200ms)</span>
        </div>
      </div>

      {/* 지난 24시간 호출 */}
      <div className="card" style={{ textAlign: "center" }}>
        <div style={{ color: "var(--muted)", fontSize: 13 }}>
          지난 24시간 호출
        </div>
        <div style={{ fontSize: 40, fontWeight: 700, margin: "10px 0 4px" }}>
          {CALLS_24H.toLocaleString()}
        </div>
        <Plot
          data={[
            {
              x: CALLS_HIST.map((_, i) => i),
              y: CALLS_HIST,
              type: "bar",
              marker: { color: "#8b5cf6" },
              hoverinfo: "y",
            },
          ]}
          layout={{
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            margin: { l: 0, r: 0, t: 0, b: 0 },
            xaxis: { visible: false, fixedrange: true },
            yaxis: { visible: false, fixedrange: true },
            height: 44,
            bargap: 0.3,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        <span
          style={{
            display: "inline-block",
            fontSize: 11,
            padding: "3px 10px",
            background: "rgba(16,185,129,0.15)",
            color: "var(--positive)",
            borderRadius: 999,
            marginTop: 6,
          }}
        >
          Active Flow
        </span>
      </div>

      {/* 마지막 추론 시간 */}
      <div className="card" style={{ textAlign: "center" }}>
        <div style={{ color: "var(--muted)", fontSize: 13 }}>
          마지막 추론 시간
        </div>
        <div style={{ margin: "14px 0 8px", fontSize: 24, fontWeight: 700 }}>
          <span
            style={{ fontSize: 14, marginRight: 6, color: "var(--muted)" }}
          >
            오전
          </span>
          09:15:32
        </div>
        <div
          style={{
            fontSize: 11,
            color: "var(--positive)",
            marginBottom: 14,
          }}
        >
          ✓ 실시간 동기화 완료
        </div>
        <button
          style={{
            width: "100%",
            padding: "10px 12px",
            background: "var(--accent)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: 13,
            cursor: "pointer",
            fontWeight: 500,
          }}
        >
          수동 추론 요청
        </button>
      </div>
    </div>
  );
}
