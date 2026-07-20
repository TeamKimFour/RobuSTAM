import TopNav from "../components/TopNav";
import AlphaChart from "./AlphaChart";
import BenchmarkTable from "./BenchmarkTable";
import BestWorstDaysCard from "./BestWorstDaysCard";
import ContributionChart from "./ContributionChart";
import CorrelationHeatmap from "./CorrelationHeatmap";
import ExtendedRiskCard from "./ExtendedRiskCard";
import FoldComparisonCard from "./FoldComparisonCard";
import MetricsRow from "./MetricsRow";
import MonthlyReturnsHeatmap from "./MonthlyReturnsHeatmap";
import PerformanceChart from "./PerformanceChart";
import RollingMetricsChart from "./RollingMetricsChart";
import TurnoverChart from "./TurnoverChart";
import UnderwaterCard from "./UnderwaterCard";
import {
  ASSET_CONTRIBUTION_SERIES,
  BACKTEST_METRICS,
  BACKTEST_SERIES,
  BENCHMARK_TABLE,
  CORRELATION_MATRIX,
  CUM_COST_SERIES,
  EXTENDED_RISK,
  FOLD_TABLE,
  TURNOVER_SERIES,
  alphaSeries,
  bestWorstDays,
  drawdownSeries,
  monthlyReturns,
  rollingSharpe,
  rollingVol,
  underwaterDurations,
} from "./mock";

export const metadata = {
  title: "Backtest · RobuSTAM",
};

export default function BacktestPage() {
  const rl = BACKTEST_SERIES[0];
  const bench6040 = BACKTEST_SERIES[1];

  const dd = [
    {
      name: rl.name,
      color: rl.color,
      points: drawdownSeries(rl.points),
    },
  ];
  const sharpe = rollingSharpe(rl.points);
  const vol = rollingVol(rl.points);
  const monthly = monthlyReturns(rl.points);
  const alpha = alphaSeries(rl, bench6040);
  const bw = bestWorstDays(10);
  const underwater = underwaterDurations(rl.points);

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
          <NotConnectedBadge />
        </header>

        <SectionLabel>핵심 지표</SectionLabel>
        <MetricsRow metrics={BACKTEST_METRICS} />
        <ExtendedRiskCard risk={EXTENDED_RISK} />

        <SectionLabel>성과 vs 벤치마크</SectionLabel>
        <PerformanceChart title="누적 NAV (초기 1.0)" series={BACKTEST_SERIES} yFormat="nav" />
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 20 }}>
          <PerformanceChart title="Drawdown (%)" series={dd} yFormat="percent" height={280} />
          <BenchmarkTable rows={BENCHMARK_TABLE} />
        </div>

        <AlphaChart title={`벤치마크 대비 초과수익 (vs ${bench6040.name})`} alpha={alpha} />

        <SectionLabel>롤링·시즈널 분석</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 20 }}>
          <RollingMetricsChart title="Rolling Sharpe · Vol" sharpe={sharpe} vol={vol} />
          <MonthlyReturnsHeatmap data={monthly} />
        </div>

        <SectionLabel>거래·비용</SectionLabel>
        <TurnoverChart turnover={TURNOVER_SERIES} cumCost={CUM_COST_SERIES} />

        <SectionLabel>자산 분해</SectionLabel>
        <ContributionChart series={ASSET_CONTRIBUTION_SERIES} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <CorrelationHeatmap matrix={CORRELATION_MATRIX} />
          <FoldComparisonCard rows={FOLD_TABLE} />
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

function NotConnectedBadge() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 14px",
        background: "rgba(220,38,38,0.08)",
        border: "1px solid rgba(220,38,38,0.35)",
        color: "#f5a5a5",
        borderRadius: 8,
        fontSize: 12,
        maxWidth: 420,
        lineHeight: 1.4,
      }}
    >
      <span aria-hidden style={{ fontSize: 14 }}>●</span>
      <span>
        <b>백엔드 백테스트 API 미연결.</b> 아래 값은 시연용 합성 데이터입니다.
        엔진(src/backtest)이 결과 파일 규격을 확정하면 실 데이터로 교체됩니다.
      </span>
    </div>
  );
}
