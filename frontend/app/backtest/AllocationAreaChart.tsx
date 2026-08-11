"use client";

import dynamic from "next/dynamic";
import { ASSET_COLOR, ASSET_ORDER, type AssetSymbol } from "../lib/assets";
import type { WeightPoint } from "./mock";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = { data: WeightPoint[] };

/**
 * 자산 비중 스택 영역 차트 — 매 거래일 정책이 배정한 자산 비중의 시간축 변화.
 *
 * `stackgroup`으로 5개 자산을 누적 100% 영역으로 그린다. `groupnorm: "percent"`로
 * 합계가 정확히 100이 되도록 정규화(부동소수 누적 오차 방지).
 *
 * TODO: 콤보별로 비중 시계열이 다를 것이므로 콤보 선택 드롭다운을 붙일 수 있다.
 * 현재는 배포 policy(=active_combo=full) 하나만 표시한다.
 */
export default function AllocationAreaChart({ data }: Props) {
  const dates = data.map((d) => d.date);
  const traces = ASSET_ORDER.map((sym: AssetSymbol) => ({
    x: dates,
    y: data.map((d) => d[sym] * 100),
    type: "scatter" as const,
    mode: "none" as const,
    name: sym,
    stackgroup: "assets",
    groupnorm: "percent" as const,
    fillcolor: ASSET_COLOR[sym],
    line: { width: 0.5, color: ASSET_COLOR[sym] },
    hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>",
  }));

  return (
    <section className="card">
      <h2 className="card-title">자산 비중 변화 (시간축)</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        매 거래일 정책이 배정한 5자산 비중 스택. 국면별 이동이 여기서 드러난다.
      </p>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 48, r: 16, t: 8, b: 40 },
          height: 380,
          font: { color: "#a3a3a3", size: 11 },
          xaxis: { gridcolor: "#262626", zerolinecolor: "#262626", tickformat: "%Y-%m" },
          yaxis: {
            gridcolor: "#262626",
            zerolinecolor: "#525252",
            ticksuffix: "%",
            range: [0, 100],
          },
          legend: { orientation: "h", y: -0.18, font: { color: "#e5e5e5", size: 11 } },
          hovermode: "x unified",
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </section>
  );
}
