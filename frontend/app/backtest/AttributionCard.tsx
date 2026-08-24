import type { CSSProperties } from "react";
import type { AttributionEntry } from "./mock";
import type { Regime } from "./regime";

/**
 * 국면별 자산 attribution 해석 카드 (8주차).
 *
 * 국면 구간 동안 각 자산이 포트폴리오 수익에 얼마나 기여했는지 스칼라로 요약한다.
 * `ContributionChart`(시계열, 옆에 배치)와 상호보완 — 여기는 "그래서 이 국면의
 * 주역이 누구였나"라는 발표 문장을 만드는 데 쓴다.
 *
 * 값 규칙 (mock.ts::computeAttribution 계약과 일치):
 *   contribution: 국면 동안 자산 (일별 기여) 합 = 순 기여 pp
 *   avgWeight:    국면 동안 자산 평균 비중 (0~1)
 */

type Props = {
  entries: AttributionEntry[];
  regime: Regime;
};

export default function AttributionCard({ entries, regime }: Props) {
  // 정렬: |기여| 큰 순서대로 (주역·역주역이 위로 옴)
  const sorted = [...entries].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
  );
  const total = entries.reduce((s, e) => s + e.contribution, 0);
  const positives = entries.filter((e) => e.contribution > 0).length;
  const negatives = entries.filter((e) => e.contribution < 0).length;
  const maxAbs = Math.max(1e-9, ...entries.map((e) => Math.abs(e.contribution)));

  return (
    <section className="card">
      <h2 className="card-title">자산별 기여도 (attribution)</h2>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
        {regime.shortLabel} 구간에서 각 자산이 포트폴리오 수익에 얼마나 기여했는가 —
        비중 × 수익률을 국면 동안 누적. 옆의 시계열 차트와 짝을 이루는 스칼라 요약.
      </p>

      <div style={SUMMARY_ROW}>
        <Stat label="순 기여 합" value={pctStr(total)} color={total >= 0 ? "var(--positive)" : "var(--negative)"} />
        <Stat label="양(+) 기여 자산" value={`${positives} / ${entries.length}`} color="var(--text)" />
        <Stat label="음(−) 기여 자산" value={`${negatives} / ${entries.length}`} color="var(--text)" />
      </div>

      <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
        {sorted.map((e) => (
          <li key={e.symbol}>
            <AttributionRow entry={e} maxAbs={maxAbs} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function AttributionRow({ entry, maxAbs }: { entry: AttributionEntry; maxAbs: number }) {
  const positive = entry.contribution >= 0;
  const widthPct = Math.max(0.5, (Math.abs(entry.contribution) / maxAbs) * 100);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "auto 60px 1fr 90px",
        alignItems: "center",
        gap: 12,
        padding: "8px 4px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 60 }}>
        <span style={{ width: 10, height: 10, borderRadius: 2, background: entry.color, display: "inline-block" }} />
        <span style={{ fontSize: 13, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
          {entry.symbol}
        </span>
      </div>
      <span style={{ fontSize: 12, color: "var(--sub)", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        비중 {(entry.avgWeight * 100).toFixed(1)}%
      </span>
      <div style={BAR_WRAP}>
        <div
          style={{
            ...BAR_FILL,
            width: `${widthPct}%`,
            marginLeft: positive ? "50%" : `${50 - widthPct}%`,
            background: positive ? "rgba(16,185,129,0.65)" : "rgba(220,38,38,0.65)",
          }}
        />
        <div style={BAR_ZERO} />
      </div>
      <span
        style={{
          fontSize: 14,
          fontWeight: 700,
          textAlign: "right",
          color: positive ? "var(--positive)" : "var(--negative)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {pctStr(entry.contribution)}
      </span>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 10, color: "var(--sub)", letterSpacing: "0.03em", textTransform: "uppercase" }}>
        {label}
      </span>
      <span style={{ fontSize: 15, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

function pctStr(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(2)}%`;
}

const SUMMARY_ROW: CSSProperties = {
  display: "flex",
  gap: 24,
  marginBottom: 14,
  padding: "10px 12px",
  background: "rgba(255,255,255,0.02)",
  border: "1px solid var(--border)",
  borderRadius: 8,
};

const BAR_WRAP: CSSProperties = {
  position: "relative",
  height: 10,
  background: "rgba(255,255,255,0.03)",
  border: "1px solid var(--border)",
  borderRadius: 3,
  overflow: "hidden",
};

const BAR_FILL: CSSProperties = {
  position: "absolute",
  top: 0,
  bottom: 0,
  borderRadius: 2,
};

// 0 기준선 (중앙 세로선)
const BAR_ZERO: CSSProperties = {
  position: "absolute",
  left: "50%",
  top: 0,
  bottom: 0,
  width: 1,
  background: "var(--sub)",
  opacity: 0.5,
};
