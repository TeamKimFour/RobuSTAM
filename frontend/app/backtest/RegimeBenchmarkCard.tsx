import type { CSSProperties } from "react";
import type { BenchmarkComparisonEntry, FoldBenchmarkComparison } from "./mock";
import { REGIMES, type Regime } from "./regime";

/**
 * 국면별 RL vs 벤치마크 성과 카드 (8주차 발표 하이라이트).
 *
 * `backtest.json`의 `comparison[].vs_benchmark`(→ `snapshot.comparison`)를 국면
 * 컨텍스트와 함께 시각화한다. CLAUDE.md §1 판정(샤프 15%+ 개선 또는 MDD 20%+ 방어)을
 * "목표 달성" 배지로 명시한다.
 *
 * 국면 스위처 상태:
 *   - `all` → fold 3장 그리드 (각 국면에서 어떻게 이겼는지 나열)
 *   - `fold1/2/3` → 그 fold만 큰 카드 (해당 국면에 집중)
 *
 * fold ↔ 국면 매핑은 `regime.ts::REGIMES`가 SSOT.
 */

const SHARPE_TARGET_PCT = 0.15;   // CLAUDE.md §1 · runner.SHARPE_IMPROVEMENT_TARGET
const MDD_TARGET_PCT = 0.20;      // CLAUDE.md §1 · runner.MDD_DEFENSE_TARGET

type Props = {
  comparison: FoldBenchmarkComparison[];
  regime: Regime;
};

export default function RegimeBenchmarkCard({ comparison, regime }: Props) {
  const foldToRegime = new Map<number, Regime>();
  for (const r of REGIMES) {
    if (r.foldId != null) foldToRegime.set(r.foldId, r);
  }

  const shown =
    regime.foldId != null
      ? comparison.filter((c) => c.foldId === regime.foldId)
      : comparison;

  if (shown.length === 0) {
    return (
      <section className="card">
        <h2 className="card-title">국면별 벤치마크 대비 성과</h2>
        <p style={{ margin: "8px 0 0 0", fontSize: 13, color: "var(--muted)" }}>
          이 국면에 대한 비교 데이터가 없습니다.
        </p>
      </section>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gap: 16,
        gridTemplateColumns: shown.length === 1 ? "1fr" : "repeat(3, 1fr)",
      }}
    >
      {shown.map((fold) => (
        <FoldCard
          key={fold.foldId}
          fold={fold}
          regimeLabel={foldToRegime.get(fold.foldId)?.shortLabel ?? `Fold ${fold.foldId}`}
          regimeDescription={foldToRegime.get(fold.foldId)?.description ?? ""}
          expanded={shown.length === 1}
        />
      ))}
    </div>
  );
}

function FoldCard({
  fold,
  regimeLabel,
  regimeDescription,
  expanded,
}: {
  fold: FoldBenchmarkComparison;
  regimeLabel: string;
  regimeDescription: string;
  expanded: boolean;
}) {
  const passed = fold.entries.filter((e) => e.beatsTarget).length;
  return (
    <section className="card" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <header>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
          <h2 className="card-title" style={{ margin: 0 }}>
            Fold {fold.foldId} · {regimeLabel}
          </h2>
          <span style={{ fontSize: 12, color: "var(--sub)" }}>
            {passed}/{fold.entries.length} 벤치마크 통과
          </span>
        </div>
        {expanded && regimeDescription ? (
          <p style={{ margin: "4px 0 0 0", fontSize: 12, color: "var(--muted)" }}>
            {regimeDescription}
          </p>
        ) : null}
      </header>

      <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
        {fold.entries.map((e) => (
          <li key={e.benchmark}>
            <BenchmarkRow entry={e} />
          </li>
        ))}
      </ul>

      <footer style={{ marginTop: 4 }}>
        <p style={{ margin: 0, fontSize: 11, color: "var(--sub)", lineHeight: 1.4 }}>
          <b>판정 기준(CLAUDE.md §1):</b> 샤프 <b>+{Math.round(SHARPE_TARGET_PCT * 100)}%</b> 개선 <b>또는</b>{" "}
          MDD <b>{Math.round(MDD_TARGET_PCT * 100)}%</b> 방어 중 하나 이상.
        </p>
      </footer>
    </section>
  );
}

function BenchmarkRow({ entry }: { entry: BenchmarkComparisonEntry }) {
  const sharpeOk = entry.sharpeImprovementPct >= SHARPE_TARGET_PCT;
  const mddOk = entry.mddDefensePct >= MDD_TARGET_PCT;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1.3fr 1fr 1fr auto",
        gap: 10,
        alignItems: "center",
        padding: "10px 12px",
        borderRadius: 8,
        background: entry.beatsTarget ? "rgba(16,185,129,0.06)" : "rgba(255,255,255,0.02)",
        border: `1px solid ${entry.beatsTarget ? "rgba(16,185,129,0.28)" : "var(--border)"}`,
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 500 }}>vs {entry.benchmark}</div>
      <Metric
        label="샤프 개선"
        value={entry.sharpeImprovementPct}
        highlight={sharpeOk}
      />
      <Metric
        label="MDD 방어"
        value={entry.mddDefensePct}
        highlight={mddOk}
      />
      <VerdictBadge ok={entry.beatsTarget} />
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: number; highlight: boolean }) {
  const color = highlight ? "var(--positive)" : value >= 0 ? "var(--text)" : "var(--negative)";
  const sign = value >= 0 ? "+" : "";
  const style: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-end",
    gap: 2,
  };
  return (
    <div style={style}>
      <span style={{ fontSize: 10, color: "var(--sub)", letterSpacing: "0.03em", textTransform: "uppercase" }}>
        {label}
      </span>
      <span style={{ fontSize: 14, fontWeight: 600, color, fontVariantNumeric: "tabular-nums" }}>
        {sign}
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function VerdictBadge({ ok }: { ok: boolean }) {
  const label = ok ? "목표 달성" : "미달";
  const style: CSSProperties = {
    fontSize: 11,
    fontWeight: 600,
    padding: "4px 10px",
    borderRadius: 999,
    background: ok ? "rgba(16,185,129,0.16)" : "rgba(220,38,38,0.12)",
    color: ok ? "#8be6c0" : "#f5a5a5",
    whiteSpace: "nowrap",
  };
  return <span style={style}>{label}</span>;
}
