import type { CSSProperties } from "react";
import TopNav from "../components/TopNav";
import DemoBadge from "../components/DemoBadge";
import LearningCurveChart from "./LearningCurveChart";
import RunGridHeatmap from "./RunGridHeatmap";
import { DEPLOYED_HYPERPARAMS, MLFLOW_RUNS } from "./mock";

export const metadata = {
  title: "Models · RobuSTAM",
};

const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "10px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};
const CELL: CSSProperties = { padding: "12px 8px", fontSize: 13 };

export default function ModelsPage() {
  const sorted = [...MLFLOW_RUNS].sort((a, b) => b.validSharpe - a.validSharpe);
  const deployed = MLFLOW_RUNS.find((r) => r.deployed) ?? MLFLOW_RUNS[0];

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
        <header>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
            Models <DemoBadge>MLflow mock</DemoBadge>
          </h1>
          <p style={{ margin: "6px 0 0 0", color: "var(--muted)", fontSize: 13 }}>
            학습 run 결과와 배포 모델 상세. MLflow API 프록시 전이라 값은 mock이지만,
            <b> 필드 스키마는 train.py가 실제로 로깅하는 것과 1:1로 맞춰뒀다</b>
            (표시되지 않는 것 = 아직 로깅 안 되는 것).
          </p>
        </header>

        <WarningBanner />
        <MissingMetricsBanner />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 20 }}>
          <DeployedCard runId={deployed.runId} sharpe={deployed.validSharpe} />
          <HyperparamsCard />
        </div>

        <LearningCurveChart />
        <RunGridHeatmap />

        <section className="card">
          <h2 className="card-title">실험 랭킹 (valid Sharpe 내림차순)</h2>
          <p style={{ margin: "-8px 0 12px 0", fontSize: 12, color: "var(--muted)" }}>
            컬럼은 <code>train.py::evaluate(split=&quot;valid&quot;)</code>가 MLflow에 남기는 metric 그대로다
            (<code>valid_sharpe</code>·<code>valid_total_log_return</code>·
            <code>valid_total_txn_cost</code>·<code>valid_avg_turnover</code>).
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={HEAD}>#</th>
                <th style={HEAD}>Run ID</th>
                <th style={HEAD}>Feature Set</th>
                <th style={HEAD}>Fold</th>
                <th style={{ ...HEAD, textAlign: "right" }}>Seed</th>
                <th style={{ ...HEAD, textAlign: "right" }}>Timesteps</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid Sharpe</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid log return</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid turnover</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid 거래비용</th>
                <th style={HEAD}>배포</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => (
                <tr
                  key={r.runId}
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: r.deployed ? "rgba(139,92,246,0.08)" : undefined,
                  }}
                >
                  <td style={{ ...CELL, color: "var(--muted)" }}>{i + 1}</td>
                  <td style={{ ...CELL, fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                    {r.runId}
                  </td>
                  <td style={{ ...CELL, fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                    {r.featureSet}
                  </td>
                  <td style={CELL}>Fold {r.fold}</td>
                  <td style={{ ...CELL, textAlign: "right" }}>{r.seed}</td>
                  <td style={{ ...CELL, textAlign: "right" }}>{r.timesteps.toLocaleString()}</td>
                  <td
                    style={{
                      ...CELL,
                      textAlign: "right",
                      color: r.validSharpe >= 0 ? "var(--positive)" : "var(--negative)",
                      fontWeight: 600,
                    }}
                  >
                    {r.validSharpe.toFixed(2)}
                  </td>
                  <td
                    style={{
                      ...CELL,
                      textAlign: "right",
                      color: r.validTotalLogReturn >= 0 ? "var(--positive)" : "var(--negative)",
                    }}
                  >
                    {(r.validTotalLogReturn * 100).toFixed(2)}%
                  </td>
                  <td style={{ ...CELL, textAlign: "right" }}>{r.validAvgTurnover.toFixed(3)}</td>
                  <td style={{ ...CELL, textAlign: "right", color: "var(--sub)" }}>
                    ${r.validTotalTxnCost.toLocaleString()}
                  </td>
                  <td style={CELL}>
                    {r.deployed ? (
                      <span
                        style={{
                          fontSize: 11,
                          padding: "3px 8px",
                          borderRadius: 999,
                          background: "rgba(139,92,246,0.2)",
                          color: "var(--accent)",
                          fontWeight: 600,
                        }}
                      >
                        DEPLOYED
                      </span>
                    ) : (
                      <span style={{ color: "var(--sub)", fontSize: 12 }}>—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <footer style={{ textAlign: "center", color: "var(--sub)", fontSize: 12, marginTop: 24 }}>
          © 2026 RobuSTAM Quantitative RL System.
        </footer>
      </main>
    </>
  );
}

function WarningBanner() {
  return (
    <div
      style={{
        padding: "12px 16px",
        background: "rgba(245,158,11,0.08)",
        border: "1px solid rgba(245,158,11,0.35)",
        color: "#fbbf24",
        borderRadius: 8,
        fontSize: 13,
        lineHeight: 1.5,
      }}
    >
      <b>배선 검증 전용:</b> 현재 배포된 정책의 valid Sharpe가 <b>음수(-0.13)</b>입니다.
      과매매로 거래비용이 수익을 잠식 중(이슈 #34). precompute → API 배선이 도는지 확인하는
      용도로만 쓰이며, 실제 투자 판단 근거로 삼지 마세요.
    </div>
  );
}

/**
 * MLflow에 아직 로깅되지 않는 지표를 UI 소비자에게 명시적으로 알리는 배너.
 * 팀 회의(도현 담당 영역)에서 조율될 예정이며, 확정 시 이 배너를 제거하고 해당
 * 컬럼·차트를 되살린다.
 */
function MissingMetricsBanner() {
  return (
    <div
      style={{
        padding: "12px 16px",
        background: "rgba(148,163,184,0.06)",
        border: "1px dashed rgba(148,163,184,0.3)",
        color: "var(--sub)",
        borderRadius: 8,
        fontSize: 12,
        lineHeight: 1.6,
      }}
    >
      <b style={{ color: "var(--text)" }}>MLflow 로깅 미배선(도현 담당 · 회의 안건):</b> 아래 3종은
      현재 <code>train.py</code>가 남기지 않아 이 대시보드에도 표시하지 않습니다.
      <ul style={{ margin: "6px 0 0 0", paddingLeft: 18 }}>
        <li>
          <b>valid MDD</b> — <code>evaluate()</code>가 log_return/sharpe/txn_cost/turnover만
          계산합니다. MDD 계산 로직 자체가 없어요.
        </li>
        <li>
          <b>test Sharpe</b> — <code>evaluate()</code>가 학습 중 <code>split=&quot;valid&quot;</code>로만
          호출됩니다. test split 평가 코드가 없어요.
        </li>
        <li>
          <b>학습 곡선 (step별 reward/entropy/KL)</b> — SB3 <code>model.learn()</code>에 MLflow
          콜백이 안 걸려 있어 스텝별 학습 로그가 기록되지 않습니다.
        </li>
      </ul>
    </div>
  );
}

function DeployedCard({ runId, sharpe }: { runId: string; sharpe: number }) {
  return (
    <section className="card">
      <h2 className="card-title">현재 배포 모델</h2>
      <div style={{ display: "grid", gap: 12 }}>
        <Row label="Run ID" value={runId} mono />
        <Row label="model_version" value="ppo-fold1-50db7bc4" mono />
        <Row label="model_path" value="mlruns/models/ppo_fold1_b461...zip" mono small />
        <Row
          label="valid Sharpe"
          value={sharpe.toFixed(2)}
          tone={sharpe >= 0 ? "positive" : "negative"}
        />
        <Row label="feature_set" value="full" mono />
        <Row label="state_dim" value="187" />
        <Row label="scaler fold" value="Fold 1" />
        <Row label="feature_store_run_id" value="b461a86027e2" mono />
      </div>
    </section>
  );
}

function HyperparamsCard() {
  return (
    <section className="card">
      <h2 className="card-title">배포 모델 파라미터 (train.py MLflow log_params)</h2>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          {DEPLOYED_HYPERPARAMS.map((h) => (
            <tr key={h.key} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ ...CELL, color: "var(--muted)", width: 200, fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                {h.key}
              </td>
              <td
                style={{
                  ...CELL,
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {h.value}
              </td>
              <td style={{ ...CELL, color: "var(--sub)", fontSize: 12 }}>{h.note ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Row({
  label,
  value,
  mono,
  small,
  tone,
}: {
  label: string;
  value: string;
  mono?: boolean;
  small?: boolean;
  tone?: "positive" | "negative";
}) {
  const color =
    tone === "positive" ? "var(--positive)" : tone === "negative" ? "var(--negative)" : "var(--text)";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
      <span style={{ color: "var(--muted)", fontSize: 13 }}>{label}</span>
      <span
        style={{
          fontSize: small ? 12 : 14,
          fontWeight: 600,
          color,
          fontFamily: mono ? "ui-monospace, SFMono-Regular, Menlo, monospace" : undefined,
          wordBreak: "break-all",
          textAlign: "right",
        }}
      >
        {value}
      </span>
    </div>
  );
}
