import TopNav from "../components/TopNav";
import AlphaChart from "./AlphaChart";
import BenchmarkTable from "./BenchmarkTable";
import BestWorstDaysCard from "./BestWorstDaysCard";
import ContributionChart from "./ContributionChart";
import CorrelationHeatmap from "./CorrelationHeatmap";
import ExtendedRiskCard from "./ExtendedRiskCard";
import FoldComparisonCard from "./FoldComparisonCard";
import { loadBacktestSnapshot } from "./loader";
import MetricsRow from "./MetricsRow";
import MonthlyReturnsHeatmap from "./MonthlyReturnsHeatmap";
import PerformanceChart from "./PerformanceChart";
import RegimeBenchmarkCard from "./RegimeBenchmarkCard";
import RegimeSwitcher from "./RegimeSwitcher";
import { filterByRegime, resolveRegime, type Regime } from "./regime";
import RollingMetricsChart from "./RollingMetricsChart";
import TurnoverChart from "./TurnoverChart";
import UnderwaterCard from "./UnderwaterCard";
import {
  ASSET_CONTRIBUTION_SERIES,
  CORRELATION_MATRIX,
  CUM_COST_SERIES,
  EXTENDED_RISK,
  TURNOVER_SERIES,
  alphaSeries,
  bestWorstDays,
  drawdownSeries,
  metricsFromPoints,
  monthlyReturns,
  rollingSharpe,
  rollingVol,
  underwaterDurations,
  type BenchmarkRow,
  type Series,
} from "./mock";

export const metadata = {
  title: "Backtest · RobuSTAM",
};

type PageProps = {
  searchParams?: { regime?: string };
};

export default async function BacktestPage({ searchParams }: PageProps) {
  const snapshot = await loadBacktestSnapshot();
  const regime = resolveRegime(searchParams?.regime);

  // ── 국면 필터: 지정된 국면 구간으로 잘라 지표를 재산출한다 ──
  const filteredSeries: Series[] = snapshot.series.map((s) => ({
    ...s,
    points: filterByRegime(s.points, regime),
  }));
  const rl = filteredSeries[0];
  const bench6040 = filteredSeries[1] ?? rl;

  const metrics = regime.id === "all" ? snapshot.metrics : metricsFromPoints(rl.points);
  const benchmarkTable: BenchmarkRow[] =
    regime.id === "all"
      ? snapshot.benchmarkTable
      : filteredSeries.map((s) => ({ name: s.name, ...metricsFromPoints(s.points) }));

  // fold 매핑이 있는 국면이면 해당 fold만, 없으면 전체를 보여준다.
  const foldTable =
    regime.foldId != null
      ? snapshot.foldTable.filter((r) => r.fold === `Fold ${regime.foldId}`)
      : snapshot.foldTable;

  const dd = [{ name: rl.name, color: rl.color, points: drawdownSeries(rl.points) }];
  const sharpe = rollingSharpe(rl.points);
  const vol = rollingVol(rl.points);
  const monthly = monthlyReturns(rl.points);
  const alpha = alphaSeries(rl, bench6040);
  const bw = bestWorstDays(10, regime.start, regime.end);
  const underwater = underwaterDurations(rl.points);

  // 자산 분해·거래비용 시계열도 국면으로 자른다 (mock 파생 시계열은 date 필드 보유).
  const turnover = filterByRegime(TURNOVER_SERIES, regime);
  const cumCost = filterByRegime(CUM_COST_SERIES, regime);
  const contribution = ASSET_CONTRIBUTION_SERIES.map((s) => ({
    ...s,
    points: filterByRegime(s.points, regime),
  }));

  return (
    <>
      <TopNav />
      <main
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "24px 32px 48px",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
              Backtest
            </h1>
            <p style={{ margin: "6px 0 0 0", color: "var(--muted)", fontSize: 13 }}>
              편도 수수료 0.1% · 슬리피지 포함 · walk-forward test 구간
            </p>
          </div>
          <DataSourceBadge source={snapshot.source} generatedAt={snapshot.generatedAt} />
        </header>

        <RegimeBar current={regime} />

        <SectionLabel>핵심 지표</SectionLabel>
        <MetricsRow metrics={metrics} />
        {regime.id === "all" ? (
          <ExtendedRiskCard risk={EXTENDED_RISK} />
        ) : (
          <FullRangeNotice label="확장 리스크 지표" />
        )}

        <SectionLabel>성과 vs 벤치마크</SectionLabel>
        <PerformanceChart title="누적 NAV (초기 1.0)" series={filteredSeries} yFormat="nav" />
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 20 }}>
          <PerformanceChart title="Drawdown (%)" series={dd} yFormat="percent" height={280} />
          <BenchmarkTable rows={benchmarkTable} />
        </div>

        <AlphaChart title={`벤치마크 대비 초과수익 (vs ${bench6040.name})`} alpha={alpha} />

        <SectionLabel>국면별 RL vs 벤치마크 판정</SectionLabel>
        <RegimeBenchmarkCard comparison={snapshot.comparison} regime={regime} />

        <SectionLabel>롤링·시즈널 분석</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 20 }}>
          <RollingMetricsChart title="Rolling Sharpe · Vol" sharpe={sharpe} vol={vol} />
          <MonthlyReturnsHeatmap data={monthly} />
        </div>

        <SectionLabel>거래·비용</SectionLabel>
        <TurnoverChart turnover={turnover} cumCost={cumCost} />

        <SectionLabel>자산 분해</SectionLabel>
        <ContributionChart series={contribution} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          {regime.id === "all" ? (
            <CorrelationHeatmap matrix={CORRELATION_MATRIX} />
          ) : (
            <FullRangeNotice label="자산 상관계수 히트맵" />
          )}
          <FoldComparisonCard rows={foldTable} />
        </div>

        <SectionLabel>극단·수중 분석</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <UnderwaterCard durations={underwater} />
          <BestWorstDaysCard best={bw.best} worst={bw.worst} />
        </div>

        <footer
          style={{
            textAlign: "center",
            color: "var(--sub)",
            fontSize: 12,
            marginTop: 24,
          }}
        >
          © 2026 RobuSTAM Quantitative RL System. All rights reserved.
        </footer>
      </main>
    </>
  );
}

function RegimeBar({ current }: { current: Regime }) {
  return (
    <section
      className="card"
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 16,
        padding: "14px 18px",
        flexWrap: "wrap",
      }}
    >
      <div>
        <div style={{ fontSize: 14, fontWeight: 600 }}>{current.fullLabel}</div>
        <div style={{ marginTop: 2, fontSize: 12, color: "var(--muted)" }}>
          {current.description}
        </div>
      </div>
      <RegimeSwitcher current={current.id} />
    </section>
  );
}

function FullRangeNotice({ label }: { label: string }) {
  return (
    <section className="card" style={{ padding: "20px 22px" }}>
      <h2 className="card-title" style={{ margin: 0 }}>{label}</h2>
      <p style={{ margin: "8px 0 0 0", fontSize: 13, color: "var(--muted)" }}>
        이 지표는 전 구간 기준입니다. 국면 필터 대응은 후속 작업(입력이 시계열이 아니라 스칼라라
        재산출 로직이 별도로 필요).
      </p>
    </section>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        margin: "16px 0 -4px 0",
        fontSize: 13,
        fontWeight: 600,
        color: "var(--muted)",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
      }}
    >
      {children}
    </h2>
  );
}

function DataSourceBadge({
  source,
  generatedAt,
}: {
  source: "real" | "mock";
  generatedAt: string | null;
}) {
  const isReal = source === "real";
  const colors = isReal
    ? { bg: "rgba(16,185,129,0.10)", border: "rgba(16,185,129,0.40)", text: "#8be6c0" }
    : { bg: "rgba(220,38,38,0.08)", border: "rgba(220,38,38,0.35)", text: "#f5a5a5" };
  const label = isReal
    ? "실데이터 (backtest.json)"
    : "합성 데이터 (mock — python -m src.backtest.export 실행 시 실데이터로 교체)";
  const suffix =
    isReal && generatedAt ? ` · 생성 ${generatedAt.replace("T", " ").slice(0, 19)}Z` : "";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 14px",
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        color: colors.text,
        borderRadius: 8,
        fontSize: 12,
        maxWidth: 480,
        lineHeight: 1.4,
      }}
    >
      <span aria-hidden style={{ fontSize: 14 }}>●</span>
      <span>
        <b>{label}</b>
        {suffix}
      </span>
    </div>
  );
}

