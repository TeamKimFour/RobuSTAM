import TopNav from "../components/TopNav";
import BenchmarkTable from "./BenchmarkTable";
import MetricsRow from "./MetricsRow";
import PerformanceChart from "./PerformanceChart";
import {
  BACKTEST_METRICS,
  BACKTEST_SERIES,
  BENCHMARK_TABLE,
  drawdownSeries,
} from "./mock";

export const metadata = {
  title: "Backtest · RobuSTAM",
};

export default function BacktestPage() {
  const drawdown = [
    {
      name: BACKTEST_SERIES[0].name,
      color: BACKTEST_SERIES[0].color,
      points: drawdownSeries(BACKTEST_SERIES[0].points),
    },
  ];

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
            <h1
              style={{
                margin: 0,
                fontSize: 24,
                fontWeight: 700,
                letterSpacing: "-0.02em",
              }}
            >
              Backtest
            </h1>
            <p
              style={{
                margin: "6px 0 0 0",
                color: "var(--muted)",
                fontSize: 13,
              }}
            >
              편도 수수료 0.1% · 슬리피지 포함 · walk-forward test 구간
            </p>
          </div>
          <NotConnectedBadge />
        </header>

        <MetricsRow metrics={BACKTEST_METRICS} />

        <PerformanceChart
          title="누적 성과 (NAV, 초기 1.0)"
          series={BACKTEST_SERIES}
          yFormat="nav"
        />

        <PerformanceChart
          title="Drawdown (%)"
          series={drawdown}
          yFormat="percent"
          height={220}
        />

        <BenchmarkTable rows={BENCHMARK_TABLE} />

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
